"""Separate melody and chord sequences."""

import random

import numpy as np
import torch

from musicagent.config import DataConfig
from musicagent.data.base import BaseDataset


class OfflineDataset(BaseDataset):
    """Dataset for offline model (encoder-decoder).

    Returns separate melody and chord sequences for the offline
    encoder-decoder transformer.
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

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src_arr = np.array(self.src_data[idx])
        tgt_arr = np.array(self.tgt_data[idx])

        # ------------------------------------------------------------------
        # Find EOS position and extract frames (strip SOS/EOS bookkeeping).
        #
        # On disk we have:
        #   index 0      : SOS
        #   indices 1..k : frame-aligned tokens
        #   index k+1    : EOS
        #   >k+1         : PAD
        #
        # We extract only the frame region [1, eos_idx) so that random
        # cropping operates on pure musical content. SOS and EOS are then
        # re-added to ensure every sequence has proper start/end markers,
        # matching inference behavior where generation always starts with
        # SOS and terminates on EOS.
        # ------------------------------------------------------------------
        try:
            eos_idx = np.where(src_arr == self.config.eos_id)[0][0]
        except IndexError:
            # If EOS is missing, fall back to the first PAD token if present,
            # otherwise use the full sequence length.
            pad_indices = np.where(src_arr == self.config.pad_id)[0]
            eos_idx = pad_indices[0] if len(pad_indices) > 0 else len(src_arr)

        # Extract frames only (skip SOS at index 0, stop before EOS)
        melody_frames = src_arr[1:eos_idx]
        chord_frames = tgt_arr[1:eos_idx]

        frames_len = len(melody_frames)

        # Reserve 2 slots for SOS and EOS in the final sequence
        max_frames = self.config.max_len - 2

        if self.split == "train":
            # Random cropping for long sequences (in frame space)
            if frames_len > max_frames:
                start = random.randint(0, frames_len - max_frames)
                melody_frames = melody_frames[start : start + max_frames]
                chord_frames = chord_frames[start : start + max_frames]

            # On-the-fly random transposition in [-max_transpose, max_transpose].
            semitones = random.randint(
                -self.config.max_transpose, self.config.max_transpose
            )
            melody_frames = self._transpose_melody(melody_frames, semitones)
            chord_frames = self._transpose_chord(chord_frames, semitones)
        else:
            # Validation/Test: Just truncate (no augmentation)
            melody_frames = melody_frames[:max_frames]
            chord_frames = chord_frames[:max_frames]

        # Re-add SOS and EOS to ensure proper sequence structure
        src_seq = np.concatenate(
            [[self.config.sos_id], melody_frames, [self.config.eos_id]]
        )
        tgt_seq = np.concatenate(
            [[self.config.sos_id], chord_frames, [self.config.eos_id]]
        )

        return {
            "src": torch.tensor(src_seq, dtype=torch.long),
            "tgt": torch.tensor(tgt_seq, dtype=torch.long),
        }
