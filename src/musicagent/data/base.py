"""Base dataset for online and offline."""

import json
import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset

from musicagent.config import DataConfig

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """Abstract base class with shared vocabulary and transposition logic."""

    # Root names must match those used in preprocessing.ChordMapper.
    ROOT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    def __init__(self, config: DataConfig, split: str = 'train'):
        self.config = config
        self.split = split

        # Load vocabularies
        with open(config.vocab_melody) as f:
            self.vocab_melody: dict[str, int] = json.load(f)['token_to_id']
        with open(config.vocab_chord) as f:
            self.vocab_chord: dict[str, int] = json.load(f)['token_to_id']

        # Build reverse mappings for on-the-fly transposition.
        self.id_to_melody = {v: k for k, v in self.vocab_melody.items()}
        self.id_to_chord = {v: k for k, v in self.vocab_chord.items()}

    def _transpose_melody(self, seq: np.ndarray, semitones: int) -> np.ndarray:
        """Transpose a melody token sequence by a given number of semitones.

        Special tokens (pad/sos/eos/rest) are left unchanged.
        """
        if semitones == 0:
            return seq

        result = seq.copy()

        for i, token_id in enumerate(seq):
            # Leave special tokens untouched.
            if token_id in (
                self.config.pad_id,
                self.config.sos_id,
                self.config.eos_id,
                self.config.rest_id,
            ):
                continue

            token = self.id_to_melody.get(int(token_id))
            if token is None:
                continue

            # Melody tokens: "pitch_{midi}_on" / "pitch_{midi}_hold"
            if not token.startswith("pitch_"):
                continue
            try:
                _, pitch_str, kind = token.split("_", 2)
                pitch_val = int(pitch_str)
            except ValueError:
                continue

            new_pitch = max(0, min(127, pitch_val + semitones))
            new_token = f"pitch_{new_pitch}_{kind}"
            new_id = self.vocab_melody.get(new_token)
            if new_id is None:
                logger.warning(
                    f"Token {new_token} not found in vocab after transposition "
                    f"(original: {token}, semitones: {semitones})"
                )
                continue
            result[i] = new_id

        return result

    def _transpose_chord(self, seq: np.ndarray, semitones: int) -> np.ndarray:
        """Transpose a chord token sequence by a given number of semitones.

        Special tokens (pad/sos/eos/rest) are left unchanged.
        """
        if semitones == 0:
            return seq

        result = seq.copy()

        for i, token_id in enumerate(seq):
            # Leave special tokens untouched.
            if token_id in (
                self.config.pad_id,
                self.config.sos_id,
                self.config.eos_id,
                self.config.rest_id,
            ):
                continue

            token = self.id_to_chord.get(int(token_id))
            if token is None:
                continue

            # Chord tokens: "{Root}:{quality}/{inv}_on" or "_hold"
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
                continue

            try:
                root_idx = self.ROOT_NAMES.index(root)
            except ValueError:
                continue

            new_root = self.ROOT_NAMES[(root_idx + semitones) % 12]
            new_token = f"{new_root}:{rest}{suffix}"
            new_id = self.vocab_chord.get(new_token)
            if new_id is None:
                logger.warning(
                    f"Token {new_token} not found in vocab after transposition "
                    f"(original: {token}, semitones: {semitones})"
                )
                continue
            result[i] = new_id

        return result

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

