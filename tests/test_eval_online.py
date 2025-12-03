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

    Sequences on disk are stored directly in separate vocab ID spaces:
    ``*_src.npy`` uses melody vocab IDs and ``*_tgt.npy`` uses chord vocab IDs.
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

    # Write separate vocab files; unified vocab is no longer required at runtime.
    _write_vocab(cfg.vocab_melody, melody_token_to_id)
    _write_vocab(cfg.vocab_chord, chord_token_to_id)

    # Create test split only, matching the unified on-disk layout from preprocessing.
    src = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)
    tgt = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)

    for i in range(n_samples):
        # SOS at position 0
        src[i, 0] = cfg.sos_id
        tgt[i, 0] = cfg.sos_id
        # A short melody/chord pair in native vocab ID spaces
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


def _is_valid_chord_token(token: str) -> bool:
    """Check if a token looks like a chord token (not a melody token)."""
    # Special tokens are valid in any position
    if token in ("<pad>", "<sos>", "<eos>", "rest"):
        return True
    # Chord tokens have format "Root:quality/inversion_on" or "_hold"
    # e.g., "C:4-3/0_on", "Db:4-3-7/1_hold"
    if ":" in token and "/" in token:
        return True
    # If it starts with "pitch_", it's a melody token - NOT valid as chord
    if token.startswith("pitch_"):
        return False
    return False


def _is_valid_melody_token(token: str) -> bool:
    """Check if a token looks like a melody token (not a chord token)."""
    # Special tokens are valid in any position
    if token in ("<pad>", "<sos>", "<eos>", "rest"):
        return True
    # Melody tokens have format "pitch_{midi}_{on|hold}"
    if token.startswith("pitch_"):
        return True
    # If it has chord format, it's NOT valid as melody
    if ":" in token and "/" in token:
        return False
    return False


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

    melody_vocab, chord_vocab = _create_online_test_data(data_dir, d_cfg, n_samples=3)

    # Build and save a tiny model checkpoint matching the configs above.
    # The model uses separate vocab sizes for melody and chord embeddings.
    melody_vocab_size = max(melody_vocab.values(), default=0) + 1
    chord_vocab_size = max(chord_vocab.values(), default=0) + 1
    model = OnlineTransformer(
        m_cfg, d_cfg, melody_vocab_size=melody_vocab_size, chord_vocab_size=chord_vocab_size
    )
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


def test_online_eval_decodes_tokens_correctly(tmp_path: Path) -> None:
    """Verify that evaluate_online decodes tokens with correct vocabulary mappings.

    This test catches bugs where chord IDs are decoded using the wrong vocabulary
    (e.g., unified vocab instead of chord vocab), which would produce melody tokens
    instead of chord tokens.
    """
    from torch.utils.data import DataLoader

    from musicagent.data import OnlineDataset, make_online_collate_fn

    data_dir = tmp_path / "realchords_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    d_cfg = DataConfig(data_processed=data_dir, max_len=16, storage_len=32)
    m_cfg = OnlineConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )

    melody_vocab, chord_vocab = _create_online_test_data(data_dir, d_cfg, n_samples=3)

    melody_vocab_size = max(melody_vocab.values(), default=0) + 1
    chord_vocab_size = max(chord_vocab.values(), default=0) + 1

    model = OnlineTransformer(
        m_cfg, d_cfg, melody_vocab_size=melody_vocab_size, chord_vocab_size=chord_vocab_size
    )
    model.eval()

    test_ds = OnlineDataset(d_cfg, split="test")
    collate = make_online_collate_fn(pad_id=d_cfg.pad_id)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate)

    result = eval_online.evaluate_online(
        model=model,
        test_loader=test_loader,
        d_cfg=d_cfg,
        melody_vocab_size=melody_vocab_size,
        device=torch.device("cpu"),
        log_progress=False,
    )

    # Verify we got predictions
    assert result.num_sequences > 0, "Expected at least one sequence"
    assert len(result.cached_predictions) > 0, "Expected cached predictions"

    # Semantic validation: chord tokens should look like chords, not melody tokens
    for idx, (mel_tokens, pred_tokens, ref_tokens) in result.cached_predictions.items():
        # Melody tokens should be valid melody tokens
        for token in mel_tokens:
            assert _is_valid_melody_token(token), (
                f"Melody token decoded incorrectly as chord-like token: {token!r} (sequence {idx})"
            )

        # Predicted chord tokens should be valid chord tokens
        for token in pred_tokens:
            assert _is_valid_chord_token(token), (
                f"Predicted chord decoded as melody token: {token!r} "
                f"(sequence {idx}). This indicates a vocab mapping bug."
            )

        # Reference chord tokens should be valid chord tokens
        for token in ref_tokens:
            assert _is_valid_chord_token(token), (
                f"Reference chord decoded as melody token: {token!r} "
                f"(sequence {idx}). This indicates a vocab mapping bug."
            )
