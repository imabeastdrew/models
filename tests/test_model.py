import torch

from musicagent.config import DataConfig, ModelConfig
from musicagent.model import OfflineTransformer, PositionalEncoding


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
    """Model forward pass should return [batch, seq_len, vocab_tgt]."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = ModelConfig(
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
    m_cfg = ModelConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
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
    m_cfg = ModelConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
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
    m_cfg = ModelConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
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


def test_model_gradient_flow() -> None:
    """Verify gradients flow back through the model."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = ModelConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
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
