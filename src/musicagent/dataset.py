import json
import logging
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from musicagent.config import DataConfig

logger = logging.getLogger(__name__)



class MusicAgentDataset(Dataset):
    def __init__(self, config: DataConfig, split: str = 'train'):
        self.config = config
        self.split = split

        src_path = config.data_processed / f"{split}_src.npy"
        tgt_path = config.data_processed / f"{split}_tgt.npy"

        if not src_path.exists():
            raise FileNotFoundError(f"Data not found at {src_path}")

        self.src_data = np.load(src_path, mmap_mode='r')
        self.tgt_data = np.load(tgt_path, mmap_mode='r')

        if len(self.src_data) != len(self.tgt_data):
            raise ValueError(f"Mismatch: src={len(self.src_data)}, tgt={len(self.tgt_data)}")

        with open(config.vocab_melody) as f:
            self.vocab_melody: dict[str, int] = json.load(f)['token_to_id']
        with open(config.vocab_chord) as f:
            self.vocab_chord: dict[str, int] = json.load(f)['token_to_id']

        # Build reverse mappings for on-the-fly transposition.
        self.id_to_melody = {v: k for k, v in self.vocab_melody.items()}
        self.id_to_chord = {v: k for k, v in self.vocab_chord.items()}

        # Root names must match those used in preprocessing.ChordMapper.
        self.root_names = ['C', 'Db', 'D', 'Eb', 'E', 'F',
                           'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    def _transpose_sequence(self, seq: np.ndarray, semitones: int, is_src: bool) -> np.ndarray:
        """Transpose a token sequence by a given number of semitones.

        Special tokens (pad/sos/eos/rest) are left unchanged.
        """
        if semitones == 0:
            return seq

        result = seq.copy()
        vocab = self.vocab_melody if is_src else self.vocab_chord
        id_to_token = self.id_to_melody if is_src else self.id_to_chord

        for i, token_id in enumerate(seq):
            # Leave special tokens untouched.
            if token_id in (
                self.config.pad_id,
                self.config.sos_id,
                self.config.eos_id,
                self.config.rest_id,
            ):
                continue

            token = id_to_token.get(int(token_id))
            if token is None:
                continue

            if is_src:
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
                new_id = vocab.get(new_token)
                if new_id is None:
                    logger.warning(
                        f"Token {new_token} not found in vocab after transposition (original: {token}, semitones: {semitones})"
                    )
                    continue
                result[i] = new_id
            else:
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
                    root_idx = self.root_names.index(root)
                except ValueError:
                    continue

                new_root = self.root_names[(root_idx + semitones) % 12]
                new_token = f"{new_root}:{rest}{suffix}"
                new_id = vocab.get(new_token)
                if new_id is None:
                    logger.warning(
                        f"Token {new_token} not found in vocab after transposition (original: {token}, semitones: {semitones})"
                    )
                    continue
                result[i] = new_id

        return result

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
            src_seq = self._transpose_sequence(src_seq, semitones, is_src=True)
            tgt_seq = self._transpose_sequence(tgt_seq, semitones, is_src=False)

            return {
                'src': torch.tensor(src_seq, dtype=torch.long),
                'tgt': torch.tensor(tgt_seq, dtype=torch.long),
            }

        # Validation/Test: Just take beginning (no augmentation)
        return {
            'src': torch.tensor(src_arr[:self.config.max_len], dtype=torch.long),
            'tgt': torch.tensor(tgt_arr[:self.config.max_len], dtype=torch.long)
        }


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    srcs = torch.stack([x['src'] for x in batch])
    tgts = torch.stack([x['tgt'] for x in batch])
    return srcs, tgts

