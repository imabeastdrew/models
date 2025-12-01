"""Tests for dataset modules."""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from musicagent.config import DataConfig
from musicagent.data import (
    OfflineDataset,
    OnlineDataset,
    collate_fn,
    make_online_collate_fn,
)


def _write_vocab(path: Path, token_to_id: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"token_to_id": token_to_id}, f)


def _write_unified_vocab(
    cfg: DataConfig,
    melody_token_to_id: dict[str, int],
    chord_token_to_id: dict[str, int],
) -> dict[str, int]:
    """Create and write a unified vocabulary mirroring scripts/preprocess.py.

    The unified mapping:
    - Shares IDs for special tokens (<pad>/<sos>/<eos>/rest) across tracks.
    - Keeps melody IDs as-is.
    - Offsets non-special chord IDs above the melody range so that a single
      embedding table can cover both tracks.
    """
    unified_token_to_id: dict[str, int] = {}

    # 1) Copy melody tokens directly.
    unified_token_to_id.update(melody_token_to_id)
    melody_size = len(melody_token_to_id)

    special_ids = {
        cfg.pad_id,
        cfg.sos_id,
        cfg.eos_id,
        cfg.rest_id,
    }

    # 2) Add chord tokens, sharing special IDs and offsetting the rest.
    for token, cid in chord_token_to_id.items():
        if cid in special_ids:
            unified_token_to_id[token] = cid
        else:
            unified_token_to_id[token] = melody_size + cid

    _write_vocab(cfg.vocab_unified, unified_token_to_id)
    return unified_token_to_id


def _create_test_dataset(
    tmp_path: Path,
    cfg: DataConfig,
    n_samples: int = 5,
) -> tuple[dict[str, int], dict[str, int]]:
    """Helper to create minimal dataset files for testing."""
    # Create melody vocab with special tokens + some pitch tokens
    melody_token_to_id = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
    }
    # Add pitch tokens for transposition testing
    for midi in range(48, 84):  # Range that allows +/- 12 semitones
        melody_token_to_id[f"pitch_{midi}_on"] = len(melody_token_to_id)
        melody_token_to_id[f"pitch_{midi}_hold"] = len(melody_token_to_id)

    # Create chord vocab with special tokens + some chord tokens.
    chord_token_to_id = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
    }
    roots = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    qualities = ["4-3", "3-4", "4-3-3"]  # common triad / 7th-style intervals
    for root in roots:
        for quality in qualities:
            base = f"{root}:{quality}/0"
            chord_token_to_id[f"{base}_on"] = len(chord_token_to_id)
            chord_token_to_id[f"{base}_hold"] = len(chord_token_to_id)

    _write_vocab(cfg.vocab_melody, melody_token_to_id)
    _write_vocab(cfg.vocab_chord, chord_token_to_id)

    # Unified vocabulary matching the real preprocessing pipeline.
    unified_token_to_id = _write_unified_vocab(cfg, melody_token_to_id, chord_token_to_id)

    # Create sample sequences
    src = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)
    tgt = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)

    for i in range(n_samples):
        seq_len = min(10 + i * 5, cfg.storage_len - 2)
        src[i, 0] = cfg.sos_id
        tgt[i, 0] = cfg.sos_id
        for j in range(1, seq_len):
            src[i, j] = unified_token_to_id["pitch_60_on"]
            tgt[i, j] = unified_token_to_id["C:4-3/0_on"]
        src[i, seq_len] = cfg.eos_id
        tgt[i, seq_len] = cfg.eos_id
        if seq_len + 1 < cfg.storage_len:
            src[i, seq_len + 1 :] = cfg.pad_id
            tgt[i, seq_len + 1 :] = cfg.pad_id

    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)
    np.save(cfg.data_processed / "valid_src.npy", src[:2])
    np.save(cfg.data_processed / "valid_tgt.npy", tgt[:2])

    return melody_token_to_id, chord_token_to_id


def test_dataset_len(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    _create_test_dataset(tmp_path, cfg, n_samples=7)
    ds = OfflineDataset(cfg, split="train")
    assert len(ds) == 7


def test_dataset_getitem_returns_tensors(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    _create_test_dataset(tmp_path, cfg)
    ds = OfflineDataset(cfg, split="train")
    sample = ds[0]
    assert "src" in sample
    assert "tgt" in sample
    assert isinstance(sample["src"], torch.Tensor)
    assert isinstance(sample["tgt"], torch.Tensor)


def test_dataset_getitem_respects_max_len(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=64, max_len=16)
    _create_test_dataset(tmp_path, cfg)
    ds = OfflineDataset(cfg, split="train")
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["src"].shape[0] <= cfg.max_len
        assert sample["tgt"].shape[0] <= cfg.max_len


def test_dataset_valid_split_no_augmentation(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16, max_transpose=6)
    _create_test_dataset(tmp_path, cfg)
    ds = OfflineDataset(cfg, split="valid")
    samples = [ds[0] for _ in range(5)]
    for s in samples[1:]:
        assert torch.equal(samples[0]["src"], s["src"])
        assert torch.equal(samples[0]["tgt"], s["tgt"])


def test_transpose_melody(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=4, max_len=4, max_transpose=12)
    melody_token_to_id = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "pitch_60_on": 4,
        "pitch_62_on": 5,
    }
    _write_vocab(cfg.vocab_melody, melody_token_to_id)
    chord_token_to_id = {cfg.pad_token: cfg.pad_id}
    _write_vocab(cfg.vocab_chord, chord_token_to_id)
    unified_ids = _write_unified_vocab(cfg, melody_token_to_id, chord_token_to_id)

    src = np.array(
        [[cfg.sos_id, unified_ids["pitch_60_on"], cfg.eos_id, cfg.pad_id]], dtype=np.int32
    )
    tgt = np.array([[cfg.sos_id, cfg.rest_id, cfg.eos_id, cfg.pad_id]], dtype=np.int32)
    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)

    ds = OfflineDataset(cfg, split="train")
    original_seq = src[0]
    transposed = ds._transpose_melody(original_seq, semitones=2)
    assert transposed[1] == melody_token_to_id["pitch_62_on"]
    assert transposed[0] == cfg.sos_id
    assert transposed[2] == cfg.eos_id


def test_transpose_chord(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=4, max_len=4, max_transpose=12)
    melody_token_to_id = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
    }
    _write_vocab(cfg.vocab_melody, melody_token_to_id)
    chord_token_to_id = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "C:4-3/0_on": 4,
        "D:4-3/0_on": 5,
        "C:4-3/0_hold": 6,
        "D:4-3/0_hold": 7,
    }
    _write_vocab(cfg.vocab_chord, chord_token_to_id)

    unified_ids = _write_unified_vocab(cfg, melody_token_to_id, chord_token_to_id)

    src = np.array([[cfg.sos_id, cfg.rest_id, cfg.eos_id, cfg.pad_id]], dtype=np.int32)
    tgt = np.array(
        [[cfg.sos_id, unified_ids["C:4-3/0_on"], cfg.eos_id, cfg.pad_id]], dtype=np.int32
    )
    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)

    ds = OfflineDataset(cfg, split="train")
    original_seq = tgt[0]
    transposed = ds._transpose_chord(original_seq, semitones=2)
    assert transposed[1] == unified_ids["D:4-3/0_on"]
    assert transposed[0] == cfg.sos_id
    assert transposed[2] == cfg.eos_id


def test_transpose_zero_semitones_unchanged(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=16, max_len=16)
    melody_vocab, _ = _create_test_dataset(tmp_path, cfg)
    ds = OfflineDataset(cfg, split="train")
    original = np.array([cfg.sos_id, melody_vocab["pitch_60_on"], cfg.eos_id], dtype=np.int32)
    transposed = ds._transpose_melody(original, semitones=0)
    assert np.array_equal(original, transposed)


def test_collate_fn_stacks_batches(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    _create_test_dataset(tmp_path, cfg, n_samples=4)
    ds = OfflineDataset(cfg, split="valid")
    batch = [ds[i] for i in range(2)]
    src_batch, tgt_batch = collate_fn(batch)
    assert src_batch.shape == (2, cfg.max_len)
    assert tgt_batch.shape == (2, cfg.max_len)


def test_dataset_file_not_found(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=16, max_len=16)
    with pytest.raises(FileNotFoundError):
        OfflineDataset(cfg, split="train")


def test_dataset_length_mismatch(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=16, max_len=16)
    melody_token_to_id = {cfg.pad_token: cfg.pad_id}
    chord_token_to_id = {cfg.pad_token: cfg.pad_id}
    _write_vocab(cfg.vocab_melody, melody_token_to_id)
    _write_vocab(cfg.vocab_chord, chord_token_to_id)
    _write_unified_vocab(cfg, melody_token_to_id, chord_token_to_id)
    np.save(cfg.data_processed / "train_src.npy", np.zeros((5, 16), dtype=np.int32))
    np.save(cfg.data_processed / "train_tgt.npy", np.zeros((3, 16), dtype=np.int32))
    with pytest.raises(ValueError, match="Mismatch"):
        OfflineDataset(cfg, split="train")


def test_transpose_missing_token_logs_warning(tmp_path: Path, caplog) -> None:
    """_transpose_melody should log a warning when a token is not in vocab."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=4, max_len=4, max_transpose=12)

    # Create vocab with pitch 60 but NOT pitch 62
    melody_token_to_id = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "pitch_60_on": 4,
    }
    _write_vocab(cfg.vocab_melody, melody_token_to_id)
    chord_token_to_id = {cfg.pad_token: cfg.pad_id}
    _write_vocab(cfg.vocab_chord, chord_token_to_id)
    _write_unified_vocab(cfg, melody_token_to_id, chord_token_to_id)

    src = np.array(
        [[cfg.sos_id, melody_token_to_id["pitch_60_on"], cfg.eos_id, cfg.pad_id]], dtype=np.int32
    )
    tgt = np.array([[cfg.sos_id, cfg.rest_id, cfg.eos_id, cfg.pad_id]], dtype=np.int32)
    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)

    ds = OfflineDataset(cfg, split="train")

    # Transpose +2 semitones (pitch_60_on -> pitch_62_on).
    # pitch_62_on is NOT in vocab, so it should log a warning and keep original.

    with caplog.at_level(logging.WARNING):
        transposed = ds._transpose_melody(src[0], semitones=2)

    assert "Token pitch_62_on not found in vocab" in caplog.text
    # The token should remain unchanged because we `continue`d the loop
    assert transposed[1] == melody_token_to_id["pitch_60_on"]


def test_online_dataset_len(tmp_path: Path) -> None:
    """OnlineDataset should have same length as OfflineDataset."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    _create_test_dataset(tmp_path, cfg, n_samples=7)
    ds = OnlineDataset(cfg, split="train")
    assert len(ds) == 7


def test_online_dataset_returns_interleaved(tmp_path: Path) -> None:
    """OnlineDataset should return interleaved input_ids."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    _create_test_dataset(tmp_path, cfg)
    ds = OnlineDataset(cfg, split="train")
    sample = ds[0]

    assert "input_ids" in sample
    assert isinstance(sample["input_ids"], torch.Tensor)
    # Interleaved sequence length is of the form 1 + 2T where T is the number
    # of frame steps retained (capped at cfg.max_len). Thus, the token length
    # should be at most 1 + 2 * cfg.max_len.
    assert 0 < sample["input_ids"].shape[0] <= 1 + 2 * cfg.max_len


def test_online_dataset_unified_vocab_size(tmp_path: Path) -> None:
    """OnlineDataset should compute correct unified vocab size."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    melody_vocab, chord_vocab = _create_test_dataset(tmp_path, cfg)
    ds = OnlineDataset(cfg, split="train")

    assert ds.melody_vocab_size == len(melody_vocab)
    assert ds.chord_vocab_size == len(chord_vocab)
    assert ds.unified_vocab_size == len(melody_vocab) + len(chord_vocab)


def test_online_dataset_interleave_format(tmp_path: Path) -> None:
    """Interleaved sequence should alternate chord, melody tokens."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    melody_vocab, chord_vocab = _create_test_dataset(tmp_path, cfg)
    ds = OnlineDataset(cfg, split="valid")  # No augmentation
    sample = ds[0]

    input_ids = sample["input_ids"]
    melody_vocab_size = ds.melody_vocab_size

    # Position 0 is SOS (chord), odd positions are chords, even positions (>=2)
    # are melody tokens.

    # Check position 0 is a chord token in unified vocab
    sos_token = input_ids[0].item()
    assert sos_token >= melody_vocab_size or sos_token < 4, (
        f"Position 0 should be SOS chord token, got {sos_token}"
    )

    # Odd positions should be chord tokens (offset by melody_vocab_size
    # or special tokens)
    for i in range(1, len(input_ids), 2):
        chord_token = input_ids[i].item()
        assert chord_token >= melody_vocab_size or chord_token < 4, (
            f"Odd position {i} should be chord token, got {chord_token}"
        )

    # Even positions >= 2 should be melody tokens (no offset)
    for i in range(2, len(input_ids), 2):
        melody_token = input_ids[i].item()
        assert melody_token < melody_vocab_size, (
            f"Even position {i} should be melody token, got {melody_token}"
        )


def test_online_collate_fn_stacks_batches(tmp_path: Path) -> None:
    """make_online_collate_fn should stack input_ids into a batch tensor."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    _create_test_dataset(tmp_path, cfg, n_samples=4)
    ds = OnlineDataset(cfg, split="valid")
    batch = [ds[i] for i in range(2)]
    collate = make_online_collate_fn(pad_id=cfg.pad_id)
    input_ids = collate(batch)

    # Batch dimension should be the number of samples; sequence length should
    # match the interleaved length from the dataset and not exceed the
    # theoretical maximum 1 + 2 * max_len for frames.
    assert input_ids.shape[0] == 2
    assert 0 < input_ids.shape[1] <= 1 + 2 * cfg.max_len


def test_online_dataset_filters_zero_frame_sequences(tmp_path: Path) -> None:
    """OnlineDataset should drop sequences with zero usable frames."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=8, max_len=4)

    # Minimal vocabs with one melodic and one chord token.
    melody_token_to_id = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "pitch_60_on": 4,
    }
    chord_token_to_id = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "C:4-3/0_on": 4,
    }
    _write_vocab(cfg.vocab_melody, melody_token_to_id)
    _write_vocab(cfg.vocab_chord, chord_token_to_id)
    unified_ids = _write_unified_vocab(cfg, melody_token_to_id, chord_token_to_id)

    # Build two sequences:
    #   - index 0: zero-frame sequence -> [SOS, EOS, PAD...]
    #   - index 1: one-frame sequence  -> [SOS, frame, EOS, PAD...]
    src = np.full((2, cfg.storage_len), cfg.rest_id, dtype=np.int32)
    tgt = np.full((2, cfg.storage_len), cfg.rest_id, dtype=np.int32)

    # Zero-frame sample
    src[0, 0] = cfg.sos_id
    src[0, 1] = cfg.eos_id
    src[0, 2:] = cfg.pad_id
    tgt[0, 0] = cfg.sos_id
    tgt[0, 1] = cfg.eos_id
    tgt[0, 2:] = cfg.pad_id

    # One-frame sample
    src[1, 0] = cfg.sos_id
    src[1, 1] = unified_ids["pitch_60_on"]
    src[1, 2] = cfg.eos_id
    src[1, 3:] = cfg.pad_id
    tgt[1, 0] = cfg.sos_id
    tgt[1, 1] = unified_ids["C:4-3/0_on"]
    tgt[1, 2] = cfg.eos_id
    tgt[1, 3:] = cfg.pad_id

    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)

    ds = OnlineDataset(cfg, split="train")

    # Only the one-frame sequence should remain after filtering.
    assert len(ds) == 1

    sample = ds[0]
    input_ids = sample["input_ids"]
    # Interleaved sequence for a single frame should have length 3:
    # [SOS_chord, y1_chord, x1_melody]
    assert input_ids.shape[0] == 3
