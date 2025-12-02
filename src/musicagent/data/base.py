"""Base dataset for online and offline."""

import json
import logging
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

        # In the current pipeline we require a unified vocabulary produced by
        # ``scripts/preprocess.py``.
        if not config.vocab_unified.exists():
            raise FileNotFoundError(
                f"Unified vocabulary not found at {config.vocab_unified}. "
                "Please preprocess the dataset with scripts/preprocess.py "
                "so that all sequences and vocabularies share a unified ID space."
            )

        with open(config.vocab_unified) as f:
            unified = json.load(f)
        token_to_id: dict[str, int] = unified.get("token_to_id", {})

        # For any legacy code paths that previously branched on this flag, make
        # it explicit that we are always operating in unified-ID mode now.
        self.uses_unified_ids_on_disk: bool = True

        self.unified_token_to_id: dict[str, int] = token_to_id
        self.unified_id_to_token: dict[int, str] = {idx: tok for tok, idx in token_to_id.items()}

        # Expose unified vocab size for models that operate directly in this
        # space. We use 1 + max(ID) rather than len(token_to_id) to remain
        # robust to any sparse ID layouts created at preprocessing time.
        if token_to_id:
            self.unified_vocab_size: int = max(token_to_id.values()) + 1
        else:
            self.unified_vocab_size = 0

        # Derive melody / chord views from the unified vocabulary based on
        # token naming conventions used in preprocessing.
        #
        # - Melody tokens: "pitch_{midi}_on" / "pitch_{midi}_hold"
        # - Chord tokens: "{Root}:{quality}/{inv}_on" / "_hold"
        #
        # Special tokens (pad/sos/eos/rest) are left out of these views and
        # handled separately via their numeric IDs in DataConfig.
        self.vocab_melody: dict[str, int] = {}
        self.vocab_chord: dict[str, int] = {}

        for tok, idx in token_to_id.items():
            if tok.startswith("pitch_"):
                self.vocab_melody[tok] = idx
            elif tok.endswith("_on") or tok.endswith("_hold"):
                self.vocab_chord[tok] = idx

        # Expose per-track vocab sizes for convenience (used in training,
        # evaluation utilities and tests). These counts intentionally exclude
        # special tokens such as <pad>/<sos>/<eos>/rest.
        self.melody_vocab_size: int = len(self.vocab_melody)
        self.chord_vocab_size: int = len(self.vocab_chord)

        # Reverse mappings for on-the-fly transposition.
        self.id_to_melody = {v: k for k, v in self.vocab_melody.items()}
        self.id_to_chord = {v: k for k, v in self.vocab_chord.items()}

        self._build_transposition_tables()

    def _build_transposition_tables(self) -> None:
        """Pre-compute transposition tables for melody and chords."""
        max_transpose = self.config.max_transpose
        vocab_size = self.unified_vocab_size
        num_offsets = 2 * max_transpose + 1

        # Initialize with identity (default to original token ID)
        self.melody_transpose_table = np.arange(vocab_size, dtype=np.int32)
        self.melody_transpose_table = np.tile(self.melody_transpose_table, (num_offsets, 1))

        self.chord_transpose_table = np.arange(vocab_size, dtype=np.int32)
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


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Default collate function for offline dataset.

    This mirrors the original behavior and assumes PAD has ID 0. It is kept
    for backward compatibility in tests and simple usage.
    """
    return _collate_offline(batch, pad_id=0)


def make_offline_collate_fn(pad_id: int = 0):
    """Create a collate function for offline dataset with the given pad_id."""

    def _fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        return _collate_offline(batch, pad_id=pad_id)

    return _fn
