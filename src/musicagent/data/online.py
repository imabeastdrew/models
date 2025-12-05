"""Interleaved melody and chord sequences.

The online model is trained on interleaved frame tokens:

    [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]

where y_t is a chord token and x_t is a melody token. Architecturally, a
decoder-only transformer over this sequence could model both
π(y_t | x<t, y<t) and π(x_t | x<t, y≤t), but **our training objective
only optimizes chord tokens y_t**; melody tokens serve purely as
conditioning context.

At inference, we alternate between sampling y_t and feeding in the given
melody token x_t, mirroring the training order.
"""

import logging
import random

import numpy as np
import torch

from musicagent.config import DataConfig
from musicagent.data.base import BaseDataset

logger = logging.getLogger(__name__)


class OnlineDataset(BaseDataset):
    """Dataset for online model (decoder-only with interleaved input).

    Returns interleaved sequences [SOS, y₁, x₁, y₂, x₂, ...] where the yₜ / xₜ
    pairs correspond to frame-aligned chord and melody tokens from the
    preprocessed arrays:
    - We strip the dataset-level SOS/EOS tokens from the underlying sequences
      so that t indexes *musical time* rather than bookkeeping markers.
    - Position 0: SOS token (chord, in unified chord space)
    - Odd positions (1, 3, 5, ...): chord tokens (y₁, y₂, y₃, ...)
    - Even positions (2, 4, 6, ...): melody tokens (x₁, x₂, x₃, ...)

    The model predicts each token given all previous tokens (causal).
    At inference, we only use chord predictions (odd positions).
    """

    def __init__(self, config: DataConfig, split: str = "train"):
        super().__init__(config, split)

        src_path = config.data_processed / f"{split}_src.npy"
        tgt_path = config.data_processed / f"{split}_tgt.npy"

        if not src_path.exists():
            raise FileNotFoundError(f"Data not found at {src_path}")

        self.src_data = np.load(src_path, mmap_mode="r")
        self.tgt_data = np.load(tgt_path, mmap_mode="r")

        if len(self.src_data) != len(self.tgt_data):
            raise ValueError(f"Mismatch: src={len(self.src_data)}, tgt={len(self.tgt_data)}")

        # ------------------------------------------------------------------
        # Filter out sequences with zero usable frames in "frame space".
        #
        # Frames are defined over the region [1, eos_idx) where:
        #   index 0      : SOS
        #   indices 1..k : frame-aligned tokens
        #   index k+1    : EOS
        #   >k+1         : PAD
        #
        # Any sequence with end_frame == start_frame has no musical content
        # for the online model and is excluded from this dataset.
        # ------------------------------------------------------------------
        self._indices: list[int] = []
        for i in range(len(self.src_data)):
            src_arr = np.array(self.src_data[i])
            try:
                eos_idx = np.where(src_arr == self.config.eos_id)[0][0]
            except IndexError:
                pad_indices = np.where(src_arr == self.config.pad_id)[0]
                eos_idx = pad_indices[0] if len(pad_indices) > 0 else len(src_arr) - 1

            start_frame = 1
            end_frame = max(start_frame, eos_idx)
            if end_frame - start_frame > 0:
                self._indices.append(i)

        if not self._indices:
            raise ValueError(
                "OnlineDataset contains no sequences with at least one frame; "
                "check preprocessing and DataConfig settings."
            )

        # `BaseDataset` has already loaded the separate melody and chord
        # vocabularies, exposing `melody_vocab_size` and `chord_vocab_size`
        # computed from the separate vocab files. We use those inherited values
        # directly rather than recalculating from the unified vocab views.

    def _interleave(
        self,
        melody_seq: np.ndarray,
        chord_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interleave melody and chord sequences: [SOS, y₁, x₁, y₂, x₂, ...].

        We prepend an SOS token (as a chord) to the sequence, then alternate
        chord (y) and melody (x). At this point:

        - ``melody_seq`` is in melody vocab ID space.
        - ``chord_seq`` is in chord vocab ID space.

        Sequence layout:
        0: SOS (Chord)  - chord vocab space
        1: y₁ (Chord)   - chord vocab space
        2: x₁ (Melody)  - melody vocab space
        3: y₂ (Chord)   - chord vocab space
        4: x₂ (Melody)  - melody vocab space
        ...

        Returns:
            interleaved: Token IDs where chord positions use chord vocab IDs
                        and melody positions use melody vocab IDs.
            is_melody: Boolean mask, True for melody positions (even indices > 0).
        """
        seq_len = len(melody_seq)
        total_len = seq_len * 2 + 1

        # Add space for SOS at the beginning
        interleaved = np.zeros(total_len, dtype=np.int64)
        is_melody = np.zeros(total_len, dtype=np.bool_)

        # Prepend SOS (as chord token in chord vocab space)
        # SOS has same ID (1) in both vocab spaces
        interleaved[0] = self.config.sos_id
        is_melody[0] = False  # SOS is treated as chord position

        for t in range(seq_len):
            # y_t (chord) goes to position 2*t + 1, already in chord vocab ID
            interleaved[2 * t + 1] = int(chord_seq[t])
            is_melody[2 * t + 1] = False

            # x_t (melody) goes to position 2*t + 2, already in melody vocab ID
            interleaved[2 * t + 2] = int(melody_seq[t])
            is_melody[2 * t + 2] = True

        return interleaved, is_melody

    def __len__(self) -> int:
        """Number of sequences with at least one usable frame."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        real_idx = self._indices[idx]
        src_arr = np.array(self.src_data[real_idx])
        tgt_arr = np.array(self.tgt_data[real_idx])

        # The paper constrains melodies and chords to at most `max_len` frames.
        # For the online model, which operates on interleaved frame tokens
        # [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T], this means:
        #
        #   T ≤ config.max_len   (frame horizon)
        #   len(interleaved) = 1 + 2T  ≤ 1 + 2 * config.max_len
        #
        # We therefore cap the *frame* sequence at `config.max_len`, and allow
        # the interleaved token sequence to grow up to ~2× longer than the
        # offline model's token limit.
        effective_max_len = self.config.max_len

        # ------------------------------------------------------------------
        # Strip dataset-level SOS/EOS and work in "frame space".
        #
        # On disk we have:
        #   index 0      : SOS
        #   indices 1..k : frame-aligned tokens
        #   index k+1    : EOS
        #   >k+1         : PAD
        #
        # For the online model we operate only on the frame region
        # [1, eos_idx), so that t indexes musical time directly.
        # ------------------------------------------------------------------
        try:
            eos_idx = np.where(src_arr == self.config.eos_id)[0][0]
        except IndexError:
            # If EOS is missing, fall back to the first PAD token if present,
            # otherwise use the full sequence minus SOS. This avoids treating
            # trailing PAD as meaningful "frame" tokens.
            pad_indices = np.where(src_arr == self.config.pad_id)[0]
            eos_idx = pad_indices[0] if len(pad_indices) > 0 else len(src_arr) - 1

        # Frames live between position 1 (after SOS) and eos_idx (exclusive).
        start_frame = 1
        end_frame = max(start_frame, eos_idx)

        melody_frames = src_arr[start_frame:end_frame]
        chord_frames = tgt_arr[start_frame:end_frame]

        if self.split == "train":
            # Cap the usable number of frames to effective_max_len and apply
            # random cropping when sequences are longer.
            frames_len = len(melody_frames)
            if frames_len == 0:
                # This should have been filtered out in __init__.
                raise RuntimeError(
                    "OnlineDataset encountered a zero-frame sequence after filtering."
                )

            usable_len = min(frames_len, effective_max_len)

            if frames_len > usable_len:
                max_start = frames_len - usable_len
                frame_start = random.randint(0, max_start)
            else:
                frame_start = 0
            frame_end = frame_start + usable_len

            src_seq = melody_frames[frame_start:frame_end]
            tgt_seq = chord_frames[frame_start:frame_end]

            # Sample a semitone shift that safely maps all chord tokens in this
            # cropped slice. If no non-zero shift is safe, this returns 0 and
            # the sequence is left untransposed.
            semitones = self._sample_safe_semitones(tgt_seq)
            if semitones != 0:
                src_seq = self._transpose_melody(src_seq, semitones)
                tgt_seq = self._transpose_chord(tgt_seq, semitones)
        else:
            # Validation/Test: no augmentation, just truncate in frame space.
            frames_len = len(melody_frames)
            if frames_len == 0:
                # This should have been filtered out in __init__.
                raise RuntimeError(
                    "OnlineDataset encountered a zero-frame sequence after filtering."
                )

            usable_len = min(frames_len, effective_max_len)
            src_seq = melody_frames[:usable_len]
            tgt_seq = chord_frames[:usable_len]

        # Interleave: [SOS, y₁, x₁, y₂, x₂, ...]
        # Returns (interleaved_ids, is_melody_mask)
        interleaved, is_melody = self._interleave(src_seq, tgt_seq)

        # For language modeling, input is [:-1] and target is [1:]
        # But we return the full sequence; the training loop handles the shift
        return {
            "input_ids": torch.tensor(interleaved, dtype=torch.long),
            "is_melody": torch.tensor(is_melody, dtype=torch.bool),
        }


def make_online_collate_fn(pad_id: int = 0):
    """Create a collate function for online dataset with the given pad_id.

    Args:
        pad_id: Token ID used for padding (should match DataConfig.pad_id).

    Returns:
        A collate function that pads sequences to uniform length and returns
        a dictionary with both input_ids and is_melody tensors.
    """

    def collate_fn(batch: list) -> dict[str, torch.Tensor]:
        """Collate function for online dataset (interleaved sequence + mask).

        Online sequences are variable-length (due to cropping at the frame level),
        so we right-pad them to a common length for batching. The is_melody mask
        is padded with False (padding positions are treated as chord positions
        but will be ignored via attention mask).
        """
        sequences = [x["input_ids"] for x in batch]
        masks = [x["is_melody"] for x in batch]

        if not sequences:
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "is_melody": torch.empty(0, 0, dtype=torch.bool),
            }

        max_len = max(seq.size(0) for seq in sequences)
        device = sequences[0].device

        # Pad input_ids with pad_id
        padded_ids = torch.full(
            (len(sequences), max_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=device,
        )

        # Pad is_melody with False (pad positions treated as chord for embedding,
        # but will be masked out in attention anyway)
        padded_mask = torch.zeros(
            (len(sequences), max_len),
            dtype=torch.bool,
            device=device,
        )

        for i, (seq, mask) in enumerate(zip(sequences, masks)):
            length = seq.size(0)
            padded_ids[i, :length] = seq
            padded_mask[i, :length] = mask

        return {
            "input_ids": padded_ids,
            "is_melody": padded_mask,
        }

    return collate_fn
