"""Tests for model modules."""

import torch

from musicagent.config import DataConfig, OfflineConfig, OnlineConfig
from musicagent.models import OfflineTransformer, OnlineTransformer, PositionalEncoding


def test_positional_encoding_shape() -> None:
    """PositionalEncoding should add positional info without changing shape."""
    d_model = 32
    max_len = 100
    pe = PositionalEncoding(d_model=d_model, max_len=max_len)

    batch_size = 2
    seq_len = 16
    x = torch.zeros((batch_size, seq_len, d_model))
    out = pe(x)

    assert out.shape == x.shape
    # Output should differ from input (positional encoding was added)
    assert not torch.allclose(out, x)


def test_positional_encoding_deterministic() -> None:
    """Same input should produce same output (no randomness in PE)."""
    pe = PositionalEncoding(d_model=16, max_len=50)
    x = torch.randn(2, 10, 16)

    out1 = pe(x)
    out2 = pe(x)

    assert torch.allclose(out1, out2)


def test_offline_transformer_forward_shapes() -> None:
    """Offline model forward pass should return [batch, seq_len, vocab_tgt]."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OfflineConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        batch_size=2,
        lr=1e-3,
        warmup_steps=10,
    )

    vocab_src_size = 10
    vocab_tgt_size = 12
    model = OfflineTransformer(m_cfg, d_cfg, vocab_src_size, vocab_tgt_size)

    batch_size = 2
    seq_len = 8
    src = torch.zeros((batch_size, seq_len), dtype=torch.long)
    tgt = torch.zeros((batch_size, seq_len), dtype=torch.long)

    out = model(src, tgt)

    assert out.shape == (batch_size, seq_len, vocab_tgt_size)


def test_create_mask_shapes() -> None:
    """create_mask should produce masks with correct shapes."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OfflineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OfflineTransformer(m_cfg, d_cfg, vocab_src_size=10, vocab_tgt_size=12)

    batch_size = 2
    src_len = 8
    tgt_len = 6

    src = torch.zeros((batch_size, src_len), dtype=torch.long)
    tgt = torch.zeros((batch_size, tgt_len), dtype=torch.long)

    src_key_mask, tgt_key_mask, tgt_mask = model.create_mask(src, tgt)

    assert src_key_mask.shape == (batch_size, src_len)
    assert tgt_key_mask.shape == (batch_size, tgt_len)
    assert tgt_mask.shape == (tgt_len, tgt_len)


def test_create_mask_padding() -> None:
    """Padding positions should be masked (True) in key padding masks."""
    d_cfg = DataConfig(max_len=16, pad_id=0)
    m_cfg = OfflineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OfflineTransformer(m_cfg, d_cfg, vocab_src_size=10, vocab_tgt_size=12)

    # src: [1, 2, 0, 0] where 0 is pad
    src = torch.tensor([[1, 2, 0, 0]], dtype=torch.long)
    tgt = torch.tensor([[1, 0, 0, 0]], dtype=torch.long)

    src_key_mask, tgt_key_mask, _ = model.create_mask(src, tgt)

    # Padding positions should be True (masked)
    assert src_key_mask[0, 0].item() is False  # token 1, not pad
    assert src_key_mask[0, 1].item() is False  # token 2, not pad
    assert src_key_mask[0, 2].item() is True   # pad
    assert src_key_mask[0, 3].item() is True   # pad

    assert tgt_key_mask[0, 0].item() is False  # token 1, not pad
    assert tgt_key_mask[0, 1].item() is True   # pad


def test_causal_mask_is_lower_triangular() -> None:
    """Target mask should be causal (bool mask: True above diagonal, False elsewhere)."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OfflineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OfflineTransformer(m_cfg, d_cfg, vocab_src_size=10, vocab_tgt_size=12)

    src = torch.ones((1, 4), dtype=torch.long)
    tgt = torch.ones((1, 4), dtype=torch.long)

    _, _, tgt_mask = model.create_mask(src, tgt)

    assert tgt_mask.dtype == torch.bool

    # Upper triangle (j > i) should be True (masked), diagonal and below False.
    for i in range(4):
        for j in range(4):
            if j > i:
                assert tgt_mask[i, j].item() is True
            else:
                assert tgt_mask[i, j].item() is False


def test_offline_model_gradient_flow() -> None:
    """Verify gradients flow back through the offline model."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OfflineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OfflineTransformer(m_cfg, d_cfg, vocab_src_size=10, vocab_tgt_size=12)

    src = torch.randint(1, 10, (2, 8))
    tgt = torch.randint(1, 12, (2, 8))

    out = model(src, tgt)
    loss = out.sum()
    loss.backward()

    # Check that gradients exist for key parameters
    assert model.src_embed.weight.grad is not None
    assert model.tgt_embed.weight.grad is not None
    assert model.fc_out.weight.grad is not None


# ============================================================================
# Online Model Tests
# ============================================================================


def test_online_transformer_forward_shapes() -> None:
    """Online model forward pass should return [batch, seq_len, unified_vocab]."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(
        d_model=32,
        n_heads=4,  # Must divide d_model
        n_layers=2,
        dropout=0.1,
    )

    melody_vocab_size = 10
    chord_vocab_size = 12
    model = OnlineTransformer(m_cfg, d_cfg, melody_vocab_size, chord_vocab_size)

    batch_size = 2
    seq_len = 8  # Interleaved sequence length
    # Input is unified vocab: melody tokens (0-9) + chord tokens (10-21)
    input_ids = torch.randint(0, melody_vocab_size + chord_vocab_size, (batch_size, seq_len))

    out = model(input_ids)

    assert out.shape == (batch_size, seq_len, melody_vocab_size + chord_vocab_size)


def test_online_model_gradient_flow() -> None:
    """Verify gradients flow back through the online model."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OnlineTransformer(m_cfg, d_cfg, melody_vocab_size=10, chord_vocab_size=12)

    input_ids = torch.randint(1, 22, (2, 8))

    out = model(input_ids)
    loss = out.sum()
    loss.backward()

    # Check that gradients exist for key parameters
    assert model.embed.weight.grad is not None
    assert model.fc_out.weight.grad is not None


def test_online_causal_mask() -> None:
    """Online model should use causal (lower triangular) attention."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OnlineTransformer(m_cfg, d_cfg, melody_vocab_size=10, chord_vocab_size=12)

    seq_len = 6
    mask = model._create_causal_mask(seq_len, torch.device('cpu'))

    assert mask.shape == (seq_len, seq_len)
    assert mask.dtype == torch.bool

    # Upper triangle should be True (masked), lower triangle + diagonal False
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert mask[i, j].item() is True
            else:
                assert mask[i, j].item() is False


def test_online_vocab_offset_conversion() -> None:
    """Vocab offset methods should correctly convert between spaces."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    melody_vocab_size = 10
    chord_vocab_size = 12
    model = OnlineTransformer(m_cfg, d_cfg, melody_vocab_size, chord_vocab_size)

    # Melody tokens stay the same (0-9)
    assert model._melody_to_unified(5) == 5

    # Chord PAD token should map to unified PAD (shared pad_id)
    assert model._chord_to_unified(d_cfg.pad_id) == d_cfg.pad_id

    # Non-pad chord tokens are offset by melody_vocab_size
    assert model._chord_to_unified(5) == melody_vocab_size + 5  # 15

def test_online_unified_vocab_size() -> None:
    """Unified vocab size should be sum of melody and chord vocab sizes."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OnlineTransformer(m_cfg, d_cfg, melody_vocab_size=10, chord_vocab_size=12)

    assert model.unified_vocab_size == 22
    assert model.melody_vocab_size == 10
    assert model.chord_vocab_size == 12


def test_online_generate_output_shape() -> None:
    """Generate should return chord sequence matching melody length."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OnlineTransformer(m_cfg, d_cfg, melody_vocab_size=10, chord_vocab_size=12)
    model.eval()

    batch_size = 2
    melody_len = 8
    melody = torch.randint(1, 10, (batch_size, melody_len))

    chords = model.generate(melody, sos_id=1, eos_id=2)

    # Output has one chord per melody token (SOS is internal, not returned)
    assert chords.shape == (batch_size, melody_len)
