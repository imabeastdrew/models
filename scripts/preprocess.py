#!/usr/bin/env python3
"""Preprocess HookTheory JSON directly to numpy arrays."""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import ijson
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from musicagent.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self, name: str, config: DataConfig):
        self.name = name
        self.config = config
        self.token_to_id: Dict[str, int] = {
            config.pad_token: config.pad_id,
            config.sos_token: config.sos_id,
            config.eos_token: config.eos_id,
            config.rest_token: config.rest_id,
        }
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self.next_id = max(self.token_to_id.values()) + 1
        self.counts: Counter = Counter()

    def add(self, token: str) -> None:
        self.counts[token] += 1
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
            
    def get_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.config.rest_id)
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'counts': dict(self.counts.most_common())
            }, f, indent=2)
        logger.info(f"Saved {self.name} vocabulary to {path} (Size: {len(self.token_to_id)})")


class ChordMapper:
    def __init__(self):
        self.root_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        
    def get_token(self, root_pc: int, intervals: List[int], inversion: int) -> str:
        quality = "-".join(map(str, intervals))
        root = self.root_names[root_pc % 12]
        return f"{root}:{quality}/{inversion}"


def get_transposed_pitch(pitch, semitones):
    new_pitch = pitch + semitones
    return max(0, min(127, new_pitch))


def process_dataset(config: DataConfig) -> None:
    if not config.data_raw.exists():
        raise FileNotFoundError(f"{config.data_raw} does not exist")

    config.data_processed.mkdir(exist_ok=True, parents=True)
    logger.info(f"Processing {config.data_raw}...")
    
    melody_vocab = Vocabulary("melody", config)
    chord_vocab = Vocabulary("chord", config)
    chord_mapper = ChordMapper()
    
    # ==========================================
    # Pass 1: Build Vocabulary
    # ==========================================
    logger.info("Pass 1: Building Vocabulary...")
    with open(config.data_raw, 'rb') as f:
        for _, data in tqdm(ijson.kvitems(f, ''), desc="Building Vocab"):
            annot = data.get('annotations', {})
            
            # Augment the vocab by including all transpositions in [-max_transpose, max_transpose].
            for transpose in range(-config.max_transpose, config.max_transpose + 1):
                if annot.get('melody'):
                    for n in annot['melody']:
                        pitch = n['pitch_class'] + (n['octave'] * 12) + config.center_midi
                        tp_pitch = get_transposed_pitch(pitch, transpose)
                        melody_vocab.add(f"pitch_{tp_pitch}_on")
                        melody_vocab.add(f"pitch_{tp_pitch}_hold")

                if annot.get('harmony'):
                    for c in annot['harmony']:
                        tp_root = (c['root_pitch_class'] + transpose) % 12
                        token = chord_mapper.get_token(tp_root, c['root_position_intervals'], c['inversion'])
                        chord_vocab.add(f"{token}_on")
                        chord_vocab.add(f"{token}_hold")
    
    melody_vocab.save(config.vocab_melody)
    chord_vocab.save(config.vocab_chord)

    # ==========================================
    # Pass 2: Tokenize and Save to NPY
    # ==========================================
    logger.info("Pass 2: Tokenizing to Numpy...")
    
    buffers = defaultdict(lambda: {'src': [], 'tgt': []})
    
    with open(config.data_raw, 'rb') as f:
        for _, data in tqdm(ijson.kvitems(f, ''), desc="Tokenizing"):
            split = data.get('split', 'TRAIN').lower()
            annot = data.get('annotations', {})
            num_beats = annot.get('num_beats', 0)
            if num_beats <= 0:
                continue
            
            total_frames = int(num_beats * config.frame_rate)

            # No on-disk augmentation: we store sequences in their original key.
            # Random transposition in [-max_transpose, max_transpose] is applied
            # on-the-fly in the Dataset for the train split.
            transpose = 0

            arr_len = config.storage_len
            # Default to silence (rest) for all frames; padding is applied after EOS.
            m_row = np.full(arr_len, config.rest_id, dtype=np.int32)
            c_row = np.full(arr_len, config.rest_id, dtype=np.int32)

            m_row[0] = config.sos_id
            c_row[0] = config.sos_id

            # Melody
            if annot.get('melody'):
                for n in annot['melody']:
                    start = int(round(n['onset'] * config.frame_rate))
                    end = int(round(n['offset'] * config.frame_rate))

                    pitch = n['pitch_class'] + (n['octave'] * 12) + config.center_midi
                    tp_pitch = get_transposed_pitch(pitch, transpose)
                    token_on = f"pitch_{tp_pitch}_on"
                    token_hold = f"pitch_{tp_pitch}_hold"
                    id_on = melody_vocab.get_id(token_on)
                    id_hold = melody_vocab.get_id(token_hold)

                    for idx in range(start, min(end, total_frames)):
                        arr_idx = idx + 1
                        if arr_idx >= arr_len - 1:
                            break
                        m_row[arr_idx] = id_on if idx == start else id_hold

            # Harmony
            if annot.get('harmony'):
                for c in annot['harmony']:
                    start = int(round(c['onset'] * config.frame_rate))
                    end = int(round(c['offset'] * config.frame_rate))

                    tp_root = (c['root_pitch_class'] + transpose) % 12
                    base_token = chord_mapper.get_token(tp_root, c['root_position_intervals'], c['inversion'])
                    token_on = f"{base_token}_on"
                    token_hold = f"{base_token}_hold"
                    id_on = chord_vocab.get_id(token_on)
                    id_hold = chord_vocab.get_id(token_hold)

                    for idx in range(start, min(end, total_frames)):
                        arr_idx = idx + 1
                        if arr_idx >= arr_len - 1:
                            break
                        c_row[arr_idx] = id_on if idx == start else id_hold

            # EOS
            eos_idx = min(total_frames + 1, arr_len - 1)
            m_row[eos_idx] = config.eos_id
            c_row[eos_idx] = config.eos_id
            # Everything after EOS is padding, not silence.
            if eos_idx + 1 < arr_len:
                m_row[eos_idx + 1 :] = config.pad_id
                c_row[eos_idx + 1 :] = config.pad_id

            buffers[split]['src'].append(m_row)
            buffers[split]['tgt'].append(c_row)

    # Save buffers to disk
    for split, data in buffers.items():
        if not data['src']:
            continue
        
        src_arr = np.stack(data['src'])
        tgt_arr = np.stack(data['tgt'])
        
        out_src = config.data_processed / f"{split}_src.npy"
        out_tgt = config.data_processed / f"{split}_tgt.npy"
        
        logger.info(f"Saving {split} src: {src_arr.shape} to {out_src}")
        np.save(out_src, src_arr)
        logger.info(f"Saving {split} tgt: {tgt_arr.shape} to {out_tgt}")
        np.save(out_tgt, tgt_arr)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HookTheory data to numpy arrays.")
    parser.add_argument("--input", type=Path, help="Input JSON path")
    parser.add_argument("--output", type=Path, help="Output directory")
    args = parser.parse_args()
    
    cfg = DataConfig()
    if args.input:
        cfg.data_raw = args.input
    if args.output:
        cfg.data_processed = args.output
    
    process_dataset(cfg)

