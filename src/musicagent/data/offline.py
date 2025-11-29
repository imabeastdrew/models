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

    def __init__(self, config: DataConfig, split: str = 'train'):
        super().__init__(config, split)

        src_path = config.data_processed / f"{split}_src.npy"
        tgt_path = config.data_processed / f"{split}_tgt.npy"

        if not src_path.exists():
            raise FileNotFoundError(f"Data not found at {src_path}")

        self.src_data = np.load(src_path, mmap_mode='r')
        self.tgt_data = np.load(tgt_path, mmap_mode='r')

        if len(self.src_data) != len(self.tgt_data):
            raise ValueError(f"Mismatch: src={len(self.src_data)}, tgt={len(self.tgt_data)}")

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src_arr = np.array(self.src_data[idx])
        tgt_arr = np.array(self.tgt_data[idx])

        if self.split == 'train':
            # Find effective sequence length up to EOS (inclusive).
            try:
                eos_idx = np.where(src_arr == self.config.eos_id)[0][0]
                valid_len = eos_idx + 1
            except IndexError:
                valid_len = len(src_arr)

            # Random cropping for long sequences.
            if valid_len > self.config.max_len:
                max_start = valid_len - self.config.max_len
                start_idx = random.randint(0, max_start)
                end_idx = start_idx + self.config.max_len
                src_seq = src_arr[start_idx:end_idx]
                tgt_seq = tgt_arr[start_idx:end_idx]
            else:
                src_seq = src_arr[:self.config.max_len]
                tgt_seq = tgt_arr[:self.config.max_len]

            # On-the-fly random transposition in [-max_transpose, max_transpose].
            semitones = random.randint(-self.config.max_transpose, self.config.max_transpose)
            src_seq = self._transpose_melody(src_seq, semitones)
            tgt_seq = self._transpose_chord(tgt_seq, semitones)

            return {
                'src': torch.tensor(src_seq, dtype=torch.long),
                'tgt': torch.tensor(tgt_seq, dtype=torch.long),
            }

        # Validation/Test: Just take beginning (no augmentation)
        return {
            'src': torch.tensor(src_arr[:self.config.max_len], dtype=torch.long),
            'tgt': torch.tensor(tgt_arr[:self.config.max_len], dtype=torch.long)
        }

