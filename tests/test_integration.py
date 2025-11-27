import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from musicagent.config import DataConfig, ModelConfig
from musicagent.model import OfflineTransformer
from musicagent.train import main


def _create_dummy_data(data_dir: Path, cfg: DataConfig):
    """Creates minimal dummy data for integration testing."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create vocabularies
    melody_vocab = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "pitch_60_on": 4,
        "pitch_60_hold": 5,
    }
    chord_vocab = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "C:4-3/0_on": 4,
        "C:4-3/0_hold": 5,
    }

    with open(data_dir / "vocab_melody.json", "w") as f:
        json.dump({"token_to_id": melody_vocab}, f)
    with open(data_dir / "vocab_chord.json", "w") as f:
        json.dump({"token_to_id": chord_vocab}, f)

    # Create dummy numpy arrays (3 samples, length 16)
    src = np.full((3, 16), cfg.rest_id, dtype=np.int32)
    tgt = np.full((3, 16), cfg.rest_id, dtype=np.int32)

    # Add some content
    src[:, 0] = cfg.sos_id
    src[:, 1] = 4 # pitch_60_on
    src[:, 2] = cfg.eos_id

    tgt[:, 0] = cfg.sos_id
    tgt[:, 1] = 4 # C:4-3/0_on
    tgt[:, 2] = cfg.eos_id

    np.save(data_dir / "train_src.npy", src)
    np.save(data_dir / "train_tgt.npy", tgt)
    np.save(data_dir / "valid_src.npy", src)
    np.save(data_dir / "valid_tgt.npy", tgt)

def test_train_main_integration(tmp_path):
    """
    End-to-end integration test for train.main().
    Runs 1 epoch with minimal data to ensure no crashes.
    """
    # Setup dummy data
    data_dir = tmp_path / "realchords_data"
    checkpoints_dir = tmp_path / "checkpoints"
    cfg = DataConfig()
    _create_dummy_data(data_dir, cfg)

    # Mock sys.argv
    test_args = [
        "musicagent-train",
        "--epochs", "1",
        "--save-dir", str(checkpoints_dir),
        "--no-wandb",
        "--batch-size", "2",
        "--d-model", "32",
        "--n-heads", "4",
        "--n-layers", "2",
        "--max-len", "16",
        "--storage-len", "16",
        "--device", "cpu", # Force CPU for test
        "--warmup-steps", "1"
    ]

    # Patch DataConfig in train.py to return a config pointing to our tmp dir
    with patch("musicagent.train.DataConfig") as mock_data_config:
        # Configure the mock instance to behave like a real DataConfig
        # but with our paths
        mock_cfg = DataConfig(data_processed=data_dir)
        # Ensure overrides in main() still work by returning a modifiable object
        mock_data_config.return_value = mock_cfg

        with patch.object(sys, "argv", test_args):
            # Run main
            main()

    # Assertions
    assert (checkpoints_dir / "best_model.pt").exists()

def test_checkpoint_save_load(tmp_path):
    """Verify that a model can be saved and loaded correctly."""
    checkpoint_path = tmp_path / "test_model.pt"

    d_cfg = DataConfig(max_len=16)
    m_cfg = ModelConfig(d_model=32, n_heads=4, n_layers=2)
    vocab_src = 10
    vocab_tgt = 10

    # Create original model
    model = OfflineTransformer(m_cfg, d_cfg, vocab_src, vocab_tgt)

    # Save state dict
    torch.save(model.state_dict(), checkpoint_path)

    # Load new model
    loaded_model = OfflineTransformer(m_cfg, d_cfg, vocab_src, vocab_tgt)
    loaded_model.load_state_dict(torch.load(checkpoint_path))

    # Compare parameters
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2)

