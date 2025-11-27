
import json
import numpy as np
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from musicagent.config import DataConfig
# Import the module correctly. Since it's in scripts/, we might need to add it to path or import by path
# But the user's project structure shows scripts/preprocess.py.
# Standard python import from scripts usually requires it to be a package or adding to sys.path.
# I'll use importlib to load it or just modify sys.path in the test.

import sys
sys.path.append("scripts")
# Assuming we can import it now, or I will copy the relevant logic if it's hard to import.
# Ideally, reusable logic should be in src/, but for now I'll try to import it.

# Re-reading the file content of preprocess.py shows it has `if __name__ == "__main__":`
# and defines classes Vocabulary, ChordMapper and function process_dataset.
# So I can import it if I add scripts to path.

from preprocess import process_dataset, Vocabulary, ChordMapper

def test_chord_mapper():
    mapper = ChordMapper()
    # Test C Major: Root 0, intervals [4, 3], inversion 0
    assert mapper.get_token(0, [4, 3], 0) == "C:4-3/0"
    
    # Test G Major: Root 7, intervals [4, 3], inversion 0
    assert mapper.get_token(7, [4, 3], 0) == "G:4-3/0"
    
    # Test wraparound: B Major -> Root 11
    assert mapper.get_token(11, [4, 3], 0) == "B:4-3/0"

def test_vocabulary(tmp_path):
    cfg = DataConfig()
    vocab = Vocabulary("test", cfg)
    
    # Add tokens
    vocab.add("token1")
    vocab.add("token2")
    vocab.add("token1") # Duplicate
    
    assert vocab.get_id("token1") == cfg.rest_id + 1
    assert vocab.get_id("token2") == cfg.rest_id + 2
    assert vocab.get_id("unknown") == cfg.rest_id # Default to rest
    
    # Save and reload verification could be added but simple logic is tested

def test_process_dataset_end_to_end(tmp_path):
    """
    Test the full preprocessing pipeline with a minimal JSON input.
    """
    # Create a dummy HookTheory-style JSON
    input_json = {
        "id1": {
            "split": "TRAIN",
            "annotations": {
                "num_beats": 4,
                "melody": [
                    {"pitch_class": 0, "octave": 5, "onset": 0.0, "offset": 1.0}, # C5
                    {"pitch_class": 2, "octave": 5, "onset": 1.0, "offset": 2.0}, # D5
                ],
                "harmony": [
                    {"root_pitch_class": 0, "root_position_intervals": [4, 3], "inversion": 0, "onset": 0.0, "offset": 2.0}, # C Major
                ]
            }
        },
        "id2": {
            "split": "VALID",
            "annotations": {
                "num_beats": 4,
                "melody": [],
                "harmony": []
            }
        }
    }
    
    input_path = tmp_path / "input.json"
    with open(input_path, "w") as f:
        json.dump(input_json, f)
        
    output_dir = tmp_path / "output"
    cfg = DataConfig(data_raw=input_path, data_processed=output_dir, frame_rate=1) # 1 frame per beat for simplicity
    
    # Run processing
    process_dataset(cfg)
    
    # Check files exist
    assert (output_dir / "vocab_melody.json").exists()
    assert (output_dir / "vocab_chord.json").exists()
    assert (output_dir / "train_src.npy").exists()
    assert (output_dir / "train_tgt.npy").exists()
    assert (output_dir / "valid_src.npy").exists()
    
    # Load and verify content
    train_src = np.load(output_dir / "train_src.npy")
    train_tgt = np.load(output_dir / "train_tgt.npy")
    
    # Check shapes
    assert len(train_src) == 1 # 1 train sample
    
    # Check content (approximate)
    # SOS at start
    assert train_src[0, 0] == cfg.sos_id
    assert train_tgt[0, 0] == cfg.sos_id
    
    # We expect some melody tokens. 
    # Since we don't know exact IDs without loading vocab, we just check for non-special tokens.
    # Frame rate is 1, so C5 is at t=0 (array idx 1 because of SOS), D5 at t=1 (array idx 2)
    # C5 onset at 0.0 -> frame 0. SOS at frame 0 of array?
    # The script does: m_row[0] = sos_id. Loop starts at idx=start (0), writes to arr_idx = idx + 1 (1).
    # So C5 at index 1.
    
    assert train_src[0, 1] != cfg.rest_id
    assert train_src[0, 2] != cfg.rest_id
    
    # Harmony C Major from 0.0 to 2.0 -> frames 0, 1. Array indices 1, 2.
    assert train_tgt[0, 1] != cfg.rest_id
    assert train_tgt[0, 2] != cfg.rest_id

