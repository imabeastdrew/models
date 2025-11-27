import json
from pathlib import Path
import logging
import numpy as np
import pytest
import torch

from musicagent.config import DataConfig
from musicagent.dataset import MusicAgentDataset, collate_fn

def _write_vocab(path: Path, token_to_id: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"token_to_id": token_to_id}, f)

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
    roots = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    qualities = ['4-3', '3-4', '4-3-3']  # common triad / 7th-style intervals
    for root in roots:
        for quality in qualities:
            base = f"{root}:{quality}/0"
            chord_token_to_id[f"{base}_on"] = len(chord_token_to_id)
            chord_token_to_id[f"{base}_hold"] = len(chord_token_to_id)

    _write_vocab(cfg.vocab_melody, melody_token_to_id)
    _write_vocab(cfg.vocab_chord, chord_token_to_id)

    # Create sample sequences
    src = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)
    tgt = np.full((n_samples, cfg.storage_len), cfg.rest_id, dtype=np.int32)

    for i in range(n_samples):
        seq_len = min(10 + i * 5, cfg.storage_len - 2)
        src[i, 0] = cfg.sos_id
        tgt[i, 0] = cfg.sos_id
        for j in range(1, seq_len):
            src[i, j] = melody_token_to_id["pitch_60_on"]
            tgt[i, j] = chord_token_to_id["C:4-3/0_on"]
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
    ds = MusicAgentDataset(cfg, split="train")
    assert len(ds) == 7

def test_dataset_getitem_returns_tensors(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    _create_test_dataset(tmp_path, cfg)
    ds = MusicAgentDataset(cfg, split="train")
    sample = ds[0]
    assert "src" in sample
    assert "tgt" in sample
    assert isinstance(sample["src"], torch.Tensor)
    assert isinstance(sample["tgt"], torch.Tensor)

def test_dataset_getitem_respects_max_len(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=64, max_len=16)
    _create_test_dataset(tmp_path, cfg)
    ds = MusicAgentDataset(cfg, split="train")
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["src"].shape[0] <= cfg.max_len
        assert sample["tgt"].shape[0] <= cfg.max_len

def test_dataset_valid_split_no_augmentation(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16, max_transpose=6)
    _create_test_dataset(tmp_path, cfg)
    ds = MusicAgentDataset(cfg, split="valid")
    samples = [ds[0] for _ in range(5)]
    for s in samples[1:]:
        assert torch.equal(samples[0]["src"], s["src"])
        assert torch.equal(samples[0]["tgt"], s["tgt"])

def test_transpose_sequence_melody(tmp_path: Path) -> None:
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

    src = np.array([[cfg.sos_id, melody_token_to_id["pitch_60_on"], cfg.eos_id, cfg.pad_id]], dtype=np.int32)
    tgt = np.array([[cfg.sos_id, cfg.rest_id, cfg.eos_id, cfg.pad_id]], dtype=np.int32)
    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)

    ds = MusicAgentDataset(cfg, split="train")
    original_seq = src[0]
    transposed = ds._transpose_sequence(original_seq, semitones=2, is_src=True)
    assert transposed[1] == melody_token_to_id["pitch_62_on"]
    assert transposed[0] == cfg.sos_id
    assert transposed[2] == cfg.eos_id

def test_transpose_sequence_chord(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=4, max_len=4, max_transpose=12)
    melody_token_to_id = {cfg.pad_token: cfg.pad_id, cfg.sos_token: cfg.sos_id, cfg.eos_token: cfg.eos_id, cfg.rest_token: cfg.rest_id}
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

    src = np.array([[cfg.sos_id, cfg.rest_id, cfg.eos_id, cfg.pad_id]], dtype=np.int32)
    tgt = np.array([[cfg.sos_id, chord_token_to_id["C:4-3/0_on"], cfg.eos_id, cfg.pad_id]], dtype=np.int32)
    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)

    ds = MusicAgentDataset(cfg, split="train")
    original_seq = tgt[0]
    transposed = ds._transpose_sequence(original_seq, semitones=2, is_src=False)
    assert transposed[1] == chord_token_to_id["D:4-3/0_on"]
    assert transposed[0] == cfg.sos_id
    assert transposed[2] == cfg.eos_id

def test_transpose_zero_semitones_unchanged(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=16, max_len=16)
    melody_vocab, _ = _create_test_dataset(tmp_path, cfg)
    ds = MusicAgentDataset(cfg, split="train")
    original = np.array([cfg.sos_id, melody_vocab["pitch_60_on"], cfg.eos_id], dtype=np.int32)
    transposed = ds._transpose_sequence(original, semitones=0, is_src=True)
    assert np.array_equal(original, transposed)

def test_collate_fn_stacks_batches(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=32, max_len=16)
    _create_test_dataset(tmp_path, cfg, n_samples=4)
    ds = MusicAgentDataset(cfg, split="valid")
    batch = [ds[i] for i in range(2)]
    src_batch, tgt_batch = collate_fn(batch)
    assert src_batch.shape == (2, cfg.max_len)
    assert tgt_batch.shape == (2, cfg.max_len)

def test_dataset_file_not_found(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=16, max_len=16)
    with pytest.raises(FileNotFoundError):
        MusicAgentDataset(cfg, split="train")

def test_dataset_length_mismatch(tmp_path: Path) -> None:
    cfg = DataConfig(data_processed=tmp_path, storage_len=16, max_len=16)
    _write_vocab(cfg.vocab_melody, {cfg.pad_token: 0})
    _write_vocab(cfg.vocab_chord, {cfg.pad_token: 0})
    np.save(cfg.data_processed / "train_src.npy", np.zeros((5, 16), dtype=np.int32))
    np.save(cfg.data_processed / "train_tgt.npy", np.zeros((3, 16), dtype=np.int32))
    with pytest.raises(ValueError, match="Mismatch"):
        MusicAgentDataset(cfg, split="train")

def test_transpose_missing_token_logs_warning(tmp_path: Path, caplog) -> None:
    """_transpose_sequence should log a warning when a token is not in vocab."""
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
    _write_vocab(cfg.vocab_chord, {cfg.pad_token: 0})
    
    src = np.array([[cfg.sos_id, melody_token_to_id["pitch_60_on"], cfg.eos_id, cfg.pad_id]], dtype=np.int32)
    tgt = np.array([[cfg.sos_id, cfg.rest_id, cfg.eos_id, cfg.pad_id]], dtype=np.int32)
    np.save(cfg.data_processed / "train_src.npy", src)
    np.save(cfg.data_processed / "train_tgt.npy", tgt)
    
    ds = MusicAgentDataset(cfg, split="train")
    
    # Transpose +2 semitones (pitch_60_on -> pitch_62_on).
    # pitch_62_on is NOT in vocab, so it should log a warning and keep original? 
    # Or just keep whatever the method does (currently skips, leaving original value from copy? No, seq is copied at start).
    # The method does: result = seq.copy(), then iterate. If missing, continue (skipping assignment).
    # So the result at that index will remain 4 (pitch_60_on).
    # Wait, if we skip, we leave the OLD token ID there.
    
    with caplog.at_level(logging.WARNING):
        transposed = ds._transpose_sequence(src[0], semitones=2, is_src=True)
        
    assert "Token pitch_62_on not found in vocab" in caplog.text
    # The token should remain unchanged because we `continue`d the loop
    assert transposed[1] == melody_token_to_id["pitch_60_on"]
