"""Tests for online evaluation module."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from musicagent.config import DataConfig, OnlineConfig
from musicagent.eval import online as eval_online
from musicagent.models import OnlineTransformer


def _write_vocab(path: Path, token_to_id: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"token_to_id": token_to_id}, f)


def _create_online_test_data(
    tmp_path: Path, cfg: DataConfig, n_samples: int = 3
) -> tuple[dict[str, int], dict[str, int]]:
    """Create a dummy dataset that matches the real preprocessing format.

    This mirrors the unified‑vocab layout used by the preprocessing pipeline:
    sequences on disk are stored in a single unified ID space with:

        index 0      : SOS
        indices 1..k : frame‑aligned tokens
        index k+1    : EOS
        >k+1         : PAD
    """
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

    # Unified vocabulary mirroring scripts/preprocess.py.
    unified_token_to_id: dict[str, int] = {}
    unified_token_to_id.update(melody_token_to_id)
    melody_size = len(melody_token_to_id)

    special_ids = {
        cfg.pad_id,
        cfg.sos_id,
        cfg.eos_id,
        cfg.rest_id,
    }
    for token, cid in chord_token_to_id.items():
        if cid in special_ids:
            unified_token_to_id[token] = cid
        else:
            unified_token_to_id[token] = melody_size + cid

    _write_vocab(cfg.vocab_unified, unified_token_to_id)

    # Create test split only, matching the unified on-disk layout from preprocessing.
    src = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)
    tgt = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)

    for i in range(n_samples):
        # SOS at position 0
        src[i, 0] = cfg.sos_id
        tgt[i, 0] = cfg.sos_id
        # A short melody/chord pair in unified ID space
        src[i, 1] = unified_token_to_id["pitch_60_on"]
        src[i, 2] = unified_token_to_id["pitch_60_hold"]
        tgt[i, 1] = unified_token_to_id["C:4-3/0_on"]
        tgt[i, 2] = unified_token_to_id["C:4-3/0_hold"]
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


def test_online_main_smoke(tmp_path: Path) -> None:
    """End-to-end test for musicagent.eval.online.main() on a tiny dummy dataset.

    The dummy dataset and vocabs match the real preprocessing format
    (pitch and chord token naming, SOS/EOS/pad/rest handling, and storage layout),
    so this exercises the full evaluation loop without relying on real data files.
    """
    data_dir = tmp_path / "realchords_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Small configs for a lightweight test run.
    d_cfg = DataConfig(data_processed=data_dir, max_len=16, storage_len=32)
    m_cfg = OnlineConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        batch_size=2,
        lr=1e-3,
        warmup_steps=10,
    )

    _create_online_test_data(data_dir, d_cfg, n_samples=3)

    # Build and save a tiny model checkpoint matching the configs above.
    # The evaluation code operates in unified ID space, so we instantiate the
    # model with the unified vocab size derived from the unified vocab file.
    with (data_dir / "vocab_unified.json").open() as f:
        unified = json.load(f)["token_to_id"]
    vocab_size = max(unified.values(), default=0) + 1
    model = OnlineTransformer(m_cfg, d_cfg, vocab_size=vocab_size)
    checkpoint_path = tmp_path / "online_test_model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # Simulate CLI args to point online.main() at our dummy checkpoint and CPU.
    test_args = [
        "musicagent-eval-online",
        "--checkpoint",
        str(checkpoint_path),
        "--batch-size",
        "2",
        "--device",
        "cpu",
    ]

    with (
        patch("musicagent.eval.online.DataConfig") as mock_data_config,
        patch("musicagent.eval.online.OnlineConfig") as mock_model_config,
        patch.object(sys, "argv", test_args),
    ):
        mock_data_config.return_value = d_cfg
        mock_model_config.return_value = m_cfg

        # The test passes if main() runs without raising and
        # completes a full evaluation pass over the dummy test set.
        eval_online.main()
