"""Tests for offline evaluation module."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from musicagent.config import DataConfig, OfflineConfig
from musicagent.eval import offline as eval_offline
from musicagent.models import OfflineTransformer


def _write_vocab(path: Path, token_to_id: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"token_to_id": token_to_id}, f)


def _create_offline_test_data(
    tmp_path: Path, cfg: DataConfig, n_samples: int = 3
) -> tuple[dict[str, int], dict[str, int]]:
    """Create a dummy dataset that matches the real preprocessing format."""
    # Melody vocab: special tokens + a single pitch
    melody_token_to_id: dict[str, int] = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "pitch_60_on": 4,
        "pitch_60_hold": 5,
    }

    # Chord vocab: special tokens + a single chord token in the real format
    chord_token_to_id: dict[str, int] = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "C:4-3/0_on": 4,
        "C:4-3/0_hold": 5,
    }

    _write_vocab(cfg.vocab_melody, melody_token_to_id)
    _write_vocab(cfg.vocab_chord, chord_token_to_id)

    # Create test split only, matching the on-disk layout from preprocessing.
    src = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)
    tgt = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)

    for i in range(n_samples):
        # SOS at position 0
        src[i, 0] = cfg.sos_id
        tgt[i, 0] = cfg.sos_id
        # A short melody/chord pair
        src[i, 1] = melody_token_to_id["pitch_60_on"]
        src[i, 2] = melody_token_to_id["pitch_60_hold"]
        tgt[i, 1] = chord_token_to_id["C:4-3/0_on"]
        tgt[i, 2] = chord_token_to_id["C:4-3/0_hold"]
        # EOS at position 3
        eos_idx = 3
        src[i, eos_idx] = cfg.eos_id
        tgt[i, eos_idx] = cfg.eos_id
        # Everything after EOS is padding, mirroring preprocessing.
        if eos_idx + 1 < cfg.storage_len:
            src[i, eos_idx + 1 :] = cfg.pad_id
            tgt[i, eos_idx + 1 :] = cfg.pad_id

    np.save(cfg.data_processed / "test_src.npy", src)
    np.save(cfg.data_processed / "test_tgt.npy", tgt)

    return melody_token_to_id, chord_token_to_id


def test_offline_main_smoke(tmp_path) -> None:
    """End-to-end test for musicagent.eval.offline.main() on a tiny dummy dataset.

    The dummy dataset and vocabs match the real preprocessing format
    (pitch and chord token naming, SOS/EOS/pad/rest handling, and storage layout),
    so this exercises the full evaluation loop without relying on real data files.
    """
    data_dir = tmp_path / "realchords_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Small configs for a lightweight test run.
    d_cfg = DataConfig(data_processed=data_dir, max_len=16, storage_len=32)
    m_cfg = OfflineConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        batch_size=2,
        lr=1e-3,
        warmup_steps=10,
    )

    melody_vocab, chord_vocab = _create_offline_test_data(data_dir, d_cfg, n_samples=3)

    # Build and save a tiny model checkpoint matching the configs above.
    # The evaluation code now operates in unified ID space, so we instantiate
    # the model with the unified vocab size derived from the composed vocab.
    vocab_size = max(
        max(melody_vocab.values(), default=0),
        max(chord_vocab.values(), default=0),
    ) + 1
    model = OfflineTransformer(m_cfg, d_cfg, vocab_size=vocab_size)
    checkpoint_path = tmp_path / "offline_test_model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # Simulate CLI args to point offline.main() at our dummy checkpoint and CPU.
    test_args = [
        "musicagent-eval-offline",
        "--checkpoint",
        str(checkpoint_path),
        "--batch-size",
        "2",
        "--device",
        "cpu",
    ]

    with (
        patch("musicagent.eval.offline.DataConfig") as mock_data_config,
        patch("musicagent.eval.offline.OfflineConfig") as mock_model_config,
        patch.object(sys, "argv", test_args),
    ):
        mock_data_config.return_value = d_cfg
        mock_model_config.return_value = m_cfg

        # The test passes if main() runs without raising and
        # completes a full evaluation pass over the dummy test set.
        eval_offline.main()
