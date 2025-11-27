import json
from pathlib import Path

import numpy as np
import pytest
import torch

from musicagent.config import DataConfig
from musicagent.dataset import MusicAgentDataset, collate_fn


def _write_vocab(path: Path, token_to_id: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"token_to_id": token_to_id}, f)


def _create_test_dataset(tmp_path: Path, cfg: DataConfig, n_samples: int = 5) -> None:
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
    # ChordMapper emits tokens like "C:4-3/0" before suffixing "_on"/"_hold",
    # where "4-3" are root-position intervals and "0" is the inversion.
    chord_token_to_id = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
    }
    roots = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    # Use interval-based qualities to reflect real tokens, e.g. "C:4-3/0_on".
    qualities = ['4-3', '3-4', '4-3-3']  # common triad / 7th-style intervals
    for root in roots:
        for quality in qualities:
            base = f"{root}:{quality}/0"
            chord_token_to_id[f"{base}_on"] = len(chord_token_to_id)
            chord_token_to_id[f"{base}_hold"] = len(chord_token_to_id)

    _write_vocab(cfg.vocab_melody, melody_token_to_id)
    _write_vocab(cfg.vocab_chord, chord_token_to_id)

    # Create sample sequences similar to process_dataset:
    # start filled with REST, SOS at position 0, EOS near the end,
    # and PAD after EOS.
    src = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)
    tgt = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)

    for i in range(n_samples):
        seq_len = min(10 + i * 5, cfg.storage_len - 2)  # Variable lengths
        src[i, 0] = cfg.sos_id
        tgt[i, 0] = cfg.sos_id
        # Fill with some melody/chord tokens
        for j in range(1, seq_len):
            src[i, j] = melody_token_to_id["pitch_60_on"]
            tgt[i, j] = chord_token_to_id["C:4-3/0_on"]
        src[i, seq_len] = cfg.eos_id
        tgt[i, seq_len] = cfg.eos_id
        # Everything after EOS is padding.
        if seq_len + 1 < cfg.storage_len:
            src[i, seq_len + 1 :] = cfg.pad_id
            tgt[i, seq_len + 1 :] = cfg.pad_id

    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)
    np.save(cfg.data_processed / "valid_src.npy", src[:2])
    np.save(cfg.data_processed / "valid_tgt.npy", tgt[:2])

    return melody_token_to_id, chord_token_to_id


def test_dataset_len(tmp_path: Path) -> None:
    """Dataset __len__ should return correct number of samples."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    _create_test_dataset(tmp_path, cfg, n_samples=7)

    ds = MusicAgentDataset(cfg, split="train")
    assert len(ds) == 7


def test_dataset_getitem_returns_tensors(tmp_path: Path) -> None:
    """__getitem__ should return dict with 'src' and 'tgt' tensors."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    _create_test_dataset(tmp_path, cfg)

    ds = MusicAgentDataset(cfg, split="train")
    sample = ds[0]

    assert "src" in sample
    assert "tgt" in sample
    assert isinstance(sample["src"], torch.Tensor)
    assert isinstance(sample["tgt"], torch.Tensor)
    assert sample["src"].dtype == torch.long
    assert sample["tgt"].dtype == torch.long


def test_dataset_getitem_respects_max_len(tmp_path: Path) -> None:
    """Returned sequences should not exceed max_len."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=64, max_len=16)
    _create_test_dataset(tmp_path, cfg)

    ds = MusicAgentDataset(cfg, split="train")

    for i in range(len(ds)):
        sample = ds[i]
        assert sample["src"].shape[0] <= cfg.max_len
        assert sample["tgt"].shape[0] <= cfg.max_len


def test_dataset_valid_split_no_augmentation(tmp_path: Path) -> None:
    """Validation split should return deterministic results (no augmentation)."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16, max_transpose=6)
    _create_test_dataset(tmp_path, cfg)

    ds = MusicAgentDataset(cfg, split="valid")

    # Get same sample multiple times
    samples = [ds[0] for _ in range(5)]

    # All should be identical (no random augmentation)
    for s in samples[1:]:
        assert torch.equal(samples[0]["src"], s["src"])
        assert torch.equal(samples[0]["tgt"], s["tgt"])


def test_transpose_sequence_melody(tmp_path: Path) -> None:
    """_transpose_sequence should correctly shift melody pitch tokens."""
    cfg = DataConfig(
        data_processed=tmp_path,
        storage_len=4,
        max_len=4,
        max_transpose=12,
    )

    # Minimal vocab with special tokens plus two melody pitches.
    melody_token_to_id = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
        "pitch_60_on": 4,
        "pitch_62_on": 5,
    }
    _write_vocab(cfg.vocab_melody, melody_token_to_id)

    # Chord vocab is not used in this test but required by the dataset.
    chord_token_to_id = {
        cfg.pad_token: cfg.pad_id,
    }
    _write_vocab(cfg.vocab_chord, chord_token_to_id)

    # One simple example sequence: SOS, pitch_60_on, EOS, PAD
    src = np.array(
        [[cfg.sos_id, melody_token_to_id["pitch_60_on"], cfg.eos_id, cfg.pad_id]],
        dtype=np.int32,
    )
    tgt = np.array(
        [[cfg.sos_id, cfg.rest_id, cfg.eos_id, cfg.pad_id]],
        dtype=np.int32,
    )

    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)

    ds = MusicAgentDataset(cfg, split="train")

    original_seq = src[0]
    transposed = ds._transpose_sequence(original_seq, semitones=2, is_src=True)

    # The melody token should transpose from pitch_60_on -> pitch_62_on (by +2 semitones).
    assert transposed[1] == melody_token_to_id["pitch_62_on"]
    # Special tokens should be unchanged.
    assert transposed[0] == cfg.sos_id
    assert transposed[2] == cfg.eos_id


def test_transpose_sequence_chord(tmp_path: Path) -> None:
    """_transpose_sequence should correctly shift chord root tokens."""
    cfg = DataConfig(
        data_processed=tmp_path,
        storage_len=4,
        max_len=4,
        max_transpose=12,
    )

    # Minimal melody vocab (not used but required)
    melody_token_to_id = {
        cfg.pad_token: cfg.pad_id,
        cfg.sos_token: cfg.sos_id,
        cfg.eos_token: cfg.eos_id,
        cfg.rest_token: cfg.rest_id,
    }
    _write_vocab(cfg.vocab_melody, melody_token_to_id)

    # Chord vocab with C and D major triads, matching ChordMapper format.
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

    # Dummy data files
    src = np.array([[cfg.sos_id, cfg.rest_id, cfg.eos_id, cfg.pad_id]], dtype=np.int32)
    tgt = np.array(
        [[cfg.sos_id, chord_token_to_id["C:4-3/0_on"], cfg.eos_id, cfg.pad_id]],
        dtype=np.int32,
    )

    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)

    ds = MusicAgentDataset(cfg, split="train")

    original_seq = tgt[0]
    transposed = ds._transpose_sequence(original_seq, semitones=2, is_src=False)

    # C:4-3/0_on + 2 semitones -> D:4-3/0_on
    assert transposed[1] == chord_token_to_id["D:4-3/0_on"]
    # Special tokens should be unchanged
    assert transposed[0] == cfg.sos_id
    assert transposed[2] == cfg.eos_id


def test_transpose_zero_semitones_unchanged(tmp_path: Path) -> None:
    """Transposing by 0 semitones should return identical sequence."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=16, max_len=16)
    melody_vocab, _ = _create_test_dataset(tmp_path, cfg)

    ds = MusicAgentDataset(cfg, split="train")

    original = np.array([cfg.sos_id, melody_vocab["pitch_60_on"], cfg.eos_id], dtype=np.int32)
    transposed = ds._transpose_sequence(original, semitones=0, is_src=True)

    assert np.array_equal(original, transposed)


def test_collate_fn_stacks_batches(tmp_path: Path) -> None:
    """collate_fn should stack samples into batch tensors."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    _create_test_dataset(tmp_path, cfg, n_samples=4)

    ds = MusicAgentDataset(cfg, split="valid")  # Use valid for determinism

    batch = [ds[i] for i in range(2)]  # Valid split has 2 samples
    src_batch, tgt_batch = collate_fn(batch)

    assert src_batch.shape == (2, cfg.max_len)
    assert tgt_batch.shape == (2, cfg.max_len)
    assert src_batch.dtype == torch.long
    assert tgt_batch.dtype == torch.long


def test_dataset_file_not_found(tmp_path: Path) -> None:
    """Dataset should raise FileNotFoundError for missing data files."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=16, max_len=16)
    # Don't create any files

    with pytest.raises(FileNotFoundError):
        MusicAgentDataset(cfg, split="train")


def test_dataset_length_mismatch(tmp_path: Path) -> None:
    """Dataset should raise ValueError if src and tgt have different lengths."""
    cfg = DataConfig(data_processed=tmp_path, storage_len=16, max_len=16)

    # Create vocab files
    _write_vocab(cfg.vocab_melody, {cfg.pad_token: 0})
    _write_vocab(cfg.vocab_chord, {cfg.pad_token: 0})

    # Create mismatched data files
    np.save(cfg.data_processed / "train_src.npy", np.zeros((5, 16), dtype=np.int32))
    np.save(cfg.data_processed / "train_tgt.npy", np.zeros((3, 16), dtype=np.int32))

    with pytest.raises(ValueError, match="Mismatch"):
        MusicAgentDataset(cfg, split="train")
