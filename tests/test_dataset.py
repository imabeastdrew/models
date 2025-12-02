"""Tests for dataset modules."""

import json
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
    # Create melody vocab with special tokens + some pitch tokens (in-memory only;
    # the runtime pipeline uses a unified vocab file on disk).
    melody_token_to_id: dict[str, int] = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
    }
    # Add pitch tokens for transposition testing
    for midi in range(48, 84):  # Range that allows +/- 12 semitones
        melody_token_to_id[f"pitch_{midi}_on"] = len(melody_token_to_id)
        melody_token_to_id[f"pitch_{midi}_hold"] = len(melody_token_to_id)

    # Create chord vocab with special tokens + some chord tokens (in-memory only).
    chord_token_to_id: dict[str, int] = {
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
    melody_token_to_id: dict[str, int] = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "pitch_60_on": 4,
        "pitch_62_on": 5,
    }
    chord_token_to_id: dict[str, int] = {cfg.pad_token: cfg.pad_id}
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
    melody_token_to_id: dict[str, int] = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
    }
    chord_token_to_id: dict[str, int] = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "C:4-3/0_on": 4,
        "D:4-3/0_on": 5,
        "C:4-3/0_hold": 6,
        "D:4-3/0_hold": 7,
    }
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
    melody_token_to_id: dict[str, int] = {cfg.pad_token: cfg.pad_id}
    chord_token_to_id: dict[str, int] = {cfg.pad_token: cfg.pad_id}
    _write_unified_vocab(cfg, melody_token_to_id, chord_token_to_id)
    np.save(cfg.data_processed / "train_src.npy", np.zeros((5, 16), dtype=np.int32))
    np.save(cfg.data_processed / "train_tgt.npy", np.zeros((3, 16), dtype=np.int32))
    with pytest.raises(ValueError, match="Mismatch"):
        OfflineDataset(cfg, split="train")


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


def test_offline_dataset_always_starts_with_sos(tmp_path: Path) -> None:
    """OfflineDataset should always return sequences starting with SOS."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=64, max_len=16)
    _create_test_dataset(tmp_path, cfg, n_samples=10)

    # Test train split (with random cropping)
    ds_train = OfflineDataset(cfg, split="train")
    for i in range(len(ds_train)):
        for _ in range(5):  # Multiple samples to test random cropping
            sample = ds_train[i]
            assert sample["src"][0].item() == cfg.sos_id, (
                f"Train src should start with SOS, got {sample['src'][0].item()}"
            )
            assert sample["tgt"][0].item() == cfg.sos_id, (
                f"Train tgt should start with SOS, got {sample['tgt'][0].item()}"
            )

    # Test valid split (no augmentation)
    ds_valid = OfflineDataset(cfg, split="valid")
    for i in range(len(ds_valid)):
        sample = ds_valid[i]
        assert sample["src"][0].item() == cfg.sos_id, (
            f"Valid src should start with SOS, got {sample['src'][0].item()}"
        )
        assert sample["tgt"][0].item() == cfg.sos_id, (
            f"Valid tgt should start with SOS, got {sample['tgt'][0].item()}"
        )


def test_offline_dataset_always_ends_with_eos(tmp_path: Path) -> None:
    """OfflineDataset should always return sequences ending with EOS."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=64, max_len=16)
    _create_test_dataset(tmp_path, cfg, n_samples=10)

    # Test train split (with random cropping)
    ds_train = OfflineDataset(cfg, split="train")
    for i in range(len(ds_train)):
        for _ in range(5):  # Multiple samples to test random cropping
            sample = ds_train[i]
            assert sample["src"][-1].item() == cfg.eos_id, (
                f"Train src should end with EOS, got {sample['src'][-1].item()}"
            )
            assert sample["tgt"][-1].item() == cfg.eos_id, (
                f"Train tgt should end with EOS, got {sample['tgt'][-1].item()}"
            )

    # Test valid split (no augmentation)
    ds_valid = OfflineDataset(cfg, split="valid")
    for i in range(len(ds_valid)):
        sample = ds_valid[i]
        assert sample["src"][-1].item() == cfg.eos_id, (
            f"Valid src should end with EOS, got {sample['src'][-1].item()}"
        )
        assert sample["tgt"][-1].item() == cfg.eos_id, (
            f"Valid tgt should end with EOS, got {sample['tgt'][-1].item()}"
        )


def test_offline_dataset_frame_alignment_after_crop(tmp_path: Path) -> None:
    """After random cropping, src and tgt should remain frame-aligned."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=64, max_len=16, max_transpose=0)

    # Create dataset with distinct tokens at each frame position
    melody_token_to_id: dict[str, int] = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
    }
    for midi in range(48, 84):
        melody_token_to_id[f"pitch_{midi}_on"] = len(melody_token_to_id)
        melody_token_to_id[f"pitch_{midi}_hold"] = len(melody_token_to_id)

    chord_token_to_id: dict[str, int] = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
    }
    roots = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    for root in roots:
        chord_token_to_id[f"{root}:4-3/0_on"] = len(chord_token_to_id)
        chord_token_to_id[f"{root}:4-3/0_hold"] = len(chord_token_to_id)

    unified_ids = _write_unified_vocab(cfg, melody_token_to_id, chord_token_to_id)

    # Create a sequence where frame i has pitch_(48+i) and chord root i%12
    # This lets us verify alignment by checking that frame indices match
    n_frames = 30  # More than max_len to force cropping
    src = np.full((1, cfg.storage_len), cfg.pad_id, dtype=np.int32)
    tgt = np.full((1, cfg.storage_len), cfg.pad_id, dtype=np.int32)

    src[0, 0] = cfg.sos_id
    tgt[0, 0] = cfg.sos_id

    for frame_idx in range(n_frames):
        pitch = 48 + frame_idx
        root = roots[frame_idx % 12]
        src[0, frame_idx + 1] = unified_ids[f"pitch_{pitch}_on"]
        tgt[0, frame_idx + 1] = unified_ids[f"{root}:4-3/0_on"]

    src[0, n_frames + 1] = cfg.eos_id
    tgt[0, n_frames + 1] = cfg.eos_id

    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)

    ds = OfflineDataset(cfg, split="train")

    # Run multiple times to test different random crops
    for _ in range(20):
        sample = ds[0]
        src_seq = sample["src"].numpy()
        tgt_seq = sample["tgt"].numpy()

        # Skip SOS at position 0, stop before EOS at the end
        # Frames are at positions 1 to len-2
        for pos in range(1, len(src_seq) - 1):
            src_token = src_seq[pos]
            tgt_token = tgt_seq[pos]

            # Find which frame index this corresponds to by parsing the pitch
            src_token_str = None
            for tok, tok_id in unified_ids.items():
                if tok_id == src_token:
                    src_token_str = tok
                    break

            tgt_token_str = None
            for tok, tok_id in unified_ids.items():
                if tok_id == tgt_token:
                    tgt_token_str = tok
                    break

            if src_token_str and src_token_str.startswith("pitch_"):
                # Extract pitch and compute expected frame index
                pitch = int(src_token_str.split("_")[1])
                expected_frame_idx = pitch - 48
                expected_root = roots[expected_frame_idx % 12]
                expected_chord = f"{expected_root}:4-3/0_on"

                assert tgt_token_str == expected_chord, (
                    f"Frame alignment broken at pos {pos}: "
                    f"src has pitch {pitch} (frame {expected_frame_idx}), "
                    f"expected chord {expected_chord}, got {tgt_token_str}"
                )


def test_offline_dataset_sequence_length_with_sos_eos(tmp_path: Path) -> None:
    """Sequence length should be at most max_len and include SOS + EOS."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=64, max_len=16)
    _create_test_dataset(tmp_path, cfg, n_samples=5)

    ds = OfflineDataset(cfg, split="train")
    for i in range(len(ds)):
        for _ in range(5):
            sample = ds[i]
            # Length should be at most max_len
            assert sample["src"].shape[0] <= cfg.max_len
            assert sample["tgt"].shape[0] <= cfg.max_len

            # Should have at least SOS + EOS (length >= 2)
            assert sample["src"].shape[0] >= 2
            assert sample["tgt"].shape[0] >= 2
