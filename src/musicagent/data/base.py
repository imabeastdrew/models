"""Base dataset for online and offline."""

import json
import logging
import random
from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import torch
from torch.utils.data import Dataset

from musicagent.config import DataConfig

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """Abstract base class with shared vocabulary and transposition logic."""

    # Root names must match those used in preprocessing.ChordMapper.
    ROOT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    # Precompute mapping for faster lookup during transposition-table construction.
    ROOT_NAME_TO_INDEX = {name: idx for idx, name in enumerate(ROOT_NAMES)}

    def __init__(self, config: DataConfig, split: str = "train"):
        self.config = config
        self.split = split

        # Load separate melody and chord vocabularies for models that use
        # separate embedding tables. The preprocessing pipeline saves
        # ``vocab_melody.json`` and ``vocab_chord.json``; sequences on disk are
        # already stored in these native ID spaces (melody IDs for ``*_src``,
        # chord IDs for ``*_tgt``). We no longer depend on a unified on-disk
        # vocabulary for runtime loading.
        if not config.vocab_melody.exists():
            raise FileNotFoundError(
                f"Melody vocabulary not found at {config.vocab_melody}. "
                "Please preprocess the dataset with scripts/preprocess.py."
            )
        if not config.vocab_chord.exists():
            raise FileNotFoundError(
                f"Chord vocabulary not found at {config.vocab_chord}. "
                "Please preprocess the dataset with scripts/preprocess.py."
            )

        with open(config.vocab_melody) as f:
            melody_vocab = json.load(f)
        with open(config.vocab_chord) as f:
            chord_vocab = json.load(f)

        self.melody_token_to_id: dict[str, int] = melody_vocab.get("token_to_id", {})
        self.chord_token_to_id: dict[str, int] = chord_vocab.get("token_to_id", {})

        self.melody_id_to_token: dict[int, str] = {
            idx: tok for tok, idx in self.melody_token_to_id.items()
        }
        self.chord_id_to_token: dict[int, str] = {
            idx: tok for tok, idx in self.chord_token_to_id.items()
        }

        # Compute vocab sizes dynamically from max ID + 1 to handle sparse layouts.
        if self.melody_token_to_id:
            self.melody_vocab_size: int = max(self.melody_token_to_id.values()) + 1
        else:
            self.melody_vocab_size = 0

        if self.chord_token_to_id:
            self.chord_vocab_size: int = max(self.chord_token_to_id.values()) + 1
        else:
            self.chord_vocab_size = 0

        # Derive melody / chord views (string → ID) for transposition table
        # construction. These simply mirror the token_to_id mappings.
        self.vocab_melody: dict[str, int] = dict(self.melody_token_to_id)
        self.vocab_chord: dict[str, int] = dict(self.chord_token_to_id)

        # Precompute the set of chord vocab IDs (excluding specials) for fast
        # membership checks in transposition helpers. We treat only tokens that
        # end with ``_on`` / ``_hold`` as actual chord symbols.
        self._chord_vocab_ids: set[int] = {
            idx
            for tok, idx in self.vocab_chord.items()
            if tok.endswith("_on") or tok.endswith("_hold")
        }

        self._build_transposition_tables()

    def _build_transposition_tables(self) -> None:
        """Pre-compute transposition tables for melody and chords."""
        max_transpose = self.config.max_transpose
        num_offsets = 2 * max_transpose + 1

        # Initialize with identity (default to original token ID) in each
        # native vocab space.
        self.melody_transpose_table = np.arange(self.melody_vocab_size, dtype=np.int32)
        self.melody_transpose_table = np.tile(self.melody_transpose_table, (num_offsets, 1))

        self.chord_transpose_table = np.arange(self.chord_vocab_size, dtype=np.int32)
        self.chord_transpose_table = np.tile(self.chord_transpose_table, (num_offsets, 1))

        # Fill tables
        for semitones in range(-max_transpose, max_transpose + 1):
            row_idx = semitones + max_transpose

            # --- Melody ---
            for token, token_id in self.vocab_melody.items():
                if not token.startswith("pitch_"):
                    continue
                try:
                    _, pitch_str, kind = token.split("_", 2)
                    pitch_val = int(pitch_str)

                    new_pitch = max(0, min(127, pitch_val + semitones))
                    new_token = f"pitch_{new_pitch}_{kind}"
                    new_id = self.vocab_melody.get(new_token)

                    if new_id is not None:
                        self.melody_transpose_table[row_idx, token_id] = new_id
                except ValueError:
                    continue

            # --- Chord ---
            for token, token_id in self.vocab_chord.items():
                suffix = None
                if token.endswith("_on"):
                    base = token[:-3]
                    suffix = "_on"
                elif token.endswith("_hold"):
                    base = token[:-5]
                    suffix = "_hold"
                else:
                    continue

                try:
                    root, rest = base.split(":", 1)
                except ValueError:
                    # Malformed chord token; skip.
                    continue

                root_idx = self.ROOT_NAME_TO_INDEX.get(root)
                if root_idx is None:
                    # Unknown root name; skip.
                    continue

                new_root = self.ROOT_NAMES[(root_idx + semitones) % 12]
                new_token = f"{new_root}:{rest}{suffix}"
                new_id = self.vocab_chord.get(new_token)

                if new_id is not None:
                    self.chord_transpose_table[row_idx, token_id] = new_id

    def _transpose_melody(self, seq: np.ndarray, semitones: int) -> np.ndarray:
        """Transpose a melody token sequence by a given number of semitones.

        Special tokens (pad/sos/eos/rest) are left unchanged.
        """
        if semitones == 0:
            return seq

        row_idx = semitones + self.config.max_transpose
        if row_idx < 0 or row_idx >= self.melody_transpose_table.shape[0]:
            logger.warning(f"Semitones {semitones} out of bounds for pre-computed table.")
            return seq

        # NumPy advanced indexing is typed as returning Any in older stubs; cast
        # to satisfy static type checkers while preserving the ndarray return.
        return cast(np.ndarray, self.melody_transpose_table[row_idx, seq])

    def _transpose_chord(self, seq: np.ndarray, semitones: int) -> np.ndarray:
        """Transpose a chord token sequence by a given number of semitones.

        Special tokens (pad/sos/eos/rest) are left unchanged.
        """
        if semitones == 0:
            return seq

        row_idx = semitones + self.config.max_transpose
        if row_idx < 0 or row_idx >= self.chord_transpose_table.shape[0]:
            logger.warning(f"Semitones {semitones} out of bounds for pre-computed table.")
            return seq

        # See note in _transpose_melody about NumPy typings.
        return cast(np.ndarray, self.chord_transpose_table[row_idx, seq])

    # ------------------------------------------------------------------
    # Transposition helpers used by train-time augmentation.
    # ------------------------------------------------------------------

    def _sample_safe_semitones(self, chord_frames: np.ndarray) -> int:
        """Sample a non-zero semitone shift that safely maps all chord tokens.

        Scheme B with rejection sampling:

        - Consider only non-zero shifts in [-max_transpose, max_transpose].
        - A shift ``k`` is safe for this sequence if *every* chord token ID in
          ``chord_frames`` is mapped to a different ID by ``chord_transpose_table``
          (i.e. there exists a corresponding transposed chord token).
        - We sample uniformly from the set of safe non-zero shifts using
          rejection sampling without replacement.
        - If no such non-zero shift exists, return 0 (no transposition).
        """
        max_t = self.config.max_transpose
        if max_t <= 0:
            return 0

        # Unique chord IDs present in this slice (ignore non-chord tokens).
        unique_ids = np.unique(chord_frames)
        chord_ids: set[int] = {int(uid) for uid in unique_ids if int(uid) in self._chord_vocab_ids}
        if not chord_ids:
            # No chord tokens to constrain transposition.
            return 0

        # Candidate non-zero shifts.
        candidates = [k for k in range(-max_t, max_t + 1) if k != 0]
        if not candidates:
            return 0

        # Rejection sampling without replacement so each safe k is equally likely.
        while candidates:
            k = random.choice(candidates)
            candidates.remove(k)
            row_idx = k + max_t

            all_ok = True
            for cid in chord_ids:
                trans_id = int(self.chord_transpose_table[row_idx, cid])
                # If the ID stays the same, we treat this as "no mapped chord
                # token exists" for this semitone shift.
                if trans_id == cid:
                    all_ok = False
                    break

            if all_ok:
                return k

        # No non-zero shift can safely transpose all chord tokens.
        return 0

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a sample from the dataset."""
        ...


def _collate_offline(
    batch: list,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Core collate implementation for offline dataset (separate src/tgt).

    This implementation mirrors the online collate behavior by supporting
    variable-length sequences via right-padding with the PAD token (ID 0).
    In typical usage, ``OfflineDataset`` returns fixed-length sequences
    (all length ``config.max_len``), so this reduces to a simple stack.
    """
    if not batch:
        empty = torch.empty(0, 0, dtype=torch.long)
        return empty, empty

    src_seqs = [x["src"] for x in batch]
    tgt_seqs = [x["tgt"] for x in batch]

    max_src_len = max(seq.size(0) for seq in src_seqs)
    max_tgt_len = max(seq.size(0) for seq in tgt_seqs)

    device = src_seqs[0].device
    dtype = src_seqs[0].dtype

    src_batch = torch.full(
        (len(src_seqs), max_src_len),
        fill_value=pad_id,
        dtype=dtype,
        device=device,
    )
    tgt_batch = torch.full(
        (len(tgt_seqs), max_tgt_len),
        fill_value=pad_id,
        dtype=tgt_seqs[0].dtype,
        device=tgt_seqs[0].device,
    )

    for i, seq in enumerate(src_seqs):
        length = seq.size(0)
        src_batch[i, :length] = seq

    for i, seq in enumerate(tgt_seqs):
        length = seq.size(0)
        tgt_batch[i, :length] = seq

    return src_batch, tgt_batch


def make_offline_collate_fn(pad_id: int = 0):
    """Create a collate function for offline dataset with the given pad_id."""

    def _fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        return _collate_offline(batch, pad_id=pad_id)

    return _fn
