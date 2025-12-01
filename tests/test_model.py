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
    """Offline model forward pass should return [batch, seq_len, vocab]."""
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

    vocab_size = 32
    model = OfflineTransformer(m_cfg, d_cfg, vocab_size=vocab_size)

    batch_size = 2
    seq_len = 8
    src = torch.zeros((batch_size, seq_len), dtype=torch.long)
    tgt = torch.zeros((batch_size, seq_len), dtype=torch.long)

    out = model(src, tgt)

    assert out.shape == (batch_size, seq_len, vocab_size)


def test_offline_model_gradient_flow() -> None:
    """Verify gradients flow back through the offline model."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OfflineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OfflineTransformer(m_cfg, d_cfg, vocab_size=32)

    src = torch.randint(1, 10, (2, 8))
    tgt = torch.randint(1, 10, (2, 8))

    out = model(src, tgt)
    loss = out.sum()
    loss.backward()

    # Check that gradients exist for at least one parameter
    assert any(p.grad is not None for p in model.parameters())


def test_offline_forward_and_generate_toy() -> None:
    """Smoke-test forward + generate on a tiny offline model."""
    d_cfg = DataConfig(max_len=8)
    m_cfg = OfflineConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        batch_size=2,
        lr=1e-3,
        warmup_steps=10,
    )

    vocab_size = 32
    chord_token_ids = list(range(4, vocab_size))
    model = OfflineTransformer(
        m_cfg,
        d_cfg,
        vocab_size=vocab_size,
        chord_token_ids=chord_token_ids,
    )
    model.eval()

    batch_size = 2
    src_len = 6
    tgt_len = 6

    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))

    # Forward pass logits over unified vocabulary.
    out = model(src, tgt)
    assert out.shape == (batch_size, tgt_len, vocab_size)

    # Generation should return IDs in unified vocab space.
    chords = model.generate(
        src,
        max_len=tgt_len,
        sos_id=d_cfg.sos_id,
        eos_id=d_cfg.eos_id,
        temperature=1.0,
        sample=False,
    )

    assert chords.shape[0] == batch_size
    assert 1 <= chords.shape[1] <= tgt_len
    assert chords.max().item() < vocab_size


# ============================================================================
# Online Model Tests
# ============================================================================


def test_online_transformer_forward_shapes() -> None:
    """Online model forward pass should return [batch, seq_len, vocab_size]."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(
        d_model=32,
        n_heads=4,  # Must divide d_model
        n_layers=2,
        dropout=0.1,
    )

    vocab_size = 22
    model = OnlineTransformer(m_cfg, d_cfg, vocab_size=vocab_size)

    batch_size = 2
    seq_len = 8  # Interleaved sequence length
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    out = model(input_ids)

    assert out.shape == (batch_size, seq_len, vocab_size)


def test_online_model_gradient_flow() -> None:
    """Verify gradients flow back through the online model."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OnlineTransformer(m_cfg, d_cfg, vocab_size=22)

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
    model = OnlineTransformer(m_cfg, d_cfg, vocab_size=22)

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

def test_online_unified_vocab_size() -> None:
    """Model should expose the configured unified vocab size."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    vocab_size = 22
    model = OnlineTransformer(m_cfg, d_cfg, vocab_size=vocab_size)

    assert model.vocab_size == vocab_size


def test_online_generate_output_shape() -> None:
    """Generate should return chord sequence matching melody length."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    vocab_size = 22
    chord_token_ids = list(range(4, vocab_size))
    model = OnlineTransformer(
        m_cfg,
        d_cfg,
        vocab_size=vocab_size,
        chord_token_ids=chord_token_ids,
    )
    model.eval()

    batch_size = 2
    melody_len = 8
    melody = torch.randint(1, vocab_size, (batch_size, melody_len))

    chords = model.generate(melody, sos_id=1, eos_id=2)

    # Output has one chord per melody token (SOS is internal, not returned)
    assert chords.shape == (batch_size, melody_len)
