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
    """Offline model forward pass should return [batch, seq_len, chord_vocab]."""
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

    melody_vocab_size = 20
    chord_vocab_size = 32
    model = OfflineTransformer(
        m_cfg, d_cfg, melody_vocab_size=melody_vocab_size, chord_vocab_size=chord_vocab_size
    )

    batch_size = 2
    seq_len = 8
    src = torch.zeros((batch_size, seq_len), dtype=torch.long)
    tgt = torch.zeros((batch_size, seq_len), dtype=torch.long)

    out = model(src, tgt)

    # Output logits are over chord vocab only
    assert out.shape == (batch_size, seq_len, chord_vocab_size)


def test_offline_model_gradient_flow() -> None:
    """Verify gradients flow back through the offline model."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OfflineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OfflineTransformer(m_cfg, d_cfg, melody_vocab_size=20, chord_vocab_size=32)

    src = torch.randint(1, 10, (2, 8))
    tgt = torch.randint(1, 10, (2, 8))

    out = model(src, tgt)
    loss = out.sum()
    loss.backward()

    # Check that gradients exist for key parameters
    assert model.encoder_embed.weight.grad is not None
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

    melody_vocab_size = 20
    chord_vocab_size = 32
    model = OfflineTransformer(
        m_cfg,
        d_cfg,
        melody_vocab_size=melody_vocab_size,
        chord_vocab_size=chord_vocab_size,
    )
    model.eval()

    batch_size = 2
    src_len = 6
    tgt_len = 6

    # src is in melody vocab space, tgt is in chord vocab space
    src = torch.randint(1, melody_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, chord_vocab_size, (batch_size, tgt_len))

    # Forward pass logits over chord vocabulary only.
    out = model(src, tgt)
    assert out.shape == (batch_size, tgt_len, chord_vocab_size)

    # Generation should return IDs in chord vocab space.
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
    assert chords.max().item() < chord_vocab_size


# ============================================================================
# Online Model Tests
# ============================================================================


def test_online_transformer_forward_shapes() -> None:
    """Online model forward pass should return [batch, seq_len, chord_vocab_size]."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(
        d_model=32,
        n_heads=4,  # Must divide d_model
        n_layers=2,
        dropout=0.1,
    )

    melody_vocab_size = 12
    chord_vocab_size = 22
    model = OnlineTransformer(
        m_cfg, d_cfg, melody_vocab_size=melody_vocab_size, chord_vocab_size=chord_vocab_size
    )

    batch_size = 2
    seq_len = 8  # Interleaved sequence length

    # Create interleaved input_ids with is_melody mask
    # Pattern: [SOS(chord), y1(chord), x1(melody), y2(chord), x2(melody), ...]
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    is_melody = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    for i in range(seq_len):
        if i == 0 or i % 2 == 1:  # Chord positions (SOS at 0, then odd)
            input_ids[:, i] = torch.randint(0, chord_vocab_size, (batch_size,))
            is_melody[:, i] = False
        else:  # Melody positions (even >= 2)
            input_ids[:, i] = torch.randint(0, melody_vocab_size, (batch_size,))
            is_melody[:, i] = True

    out = model(input_ids, is_melody)

    # Output logits are over chord vocab only
    assert out.shape == (batch_size, seq_len, chord_vocab_size)


def test_online_model_gradient_flow() -> None:
    """Verify gradients flow back through the online model."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OnlineTransformer(m_cfg, d_cfg, melody_vocab_size=12, chord_vocab_size=22)

    # Create input with proper is_melody mask
    batch_size = 2
    seq_len = 8
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    is_melody = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    for i in range(seq_len):
        if i == 0 or i % 2 == 1:  # Chord positions
            input_ids[:, i] = torch.randint(1, 22, (batch_size,))
            is_melody[:, i] = False
        else:  # Melody positions
            input_ids[:, i] = torch.randint(1, 12, (batch_size,))
            is_melody[:, i] = True

    out = model(input_ids, is_melody)
    loss = out.sum()
    loss.backward()

    # Check that gradients exist for key parameters (separate embeddings)
    assert model.melody_embed.weight.grad is not None
    assert model.chord_embed.weight.grad is not None
    assert model.fc_out.weight.grad is not None


def test_online_causal_behavior() -> None:
    """Online model should exhibit causal behavior (future tokens don't affect past)."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = OnlineTransformer(m_cfg, d_cfg, melody_vocab_size=12, chord_vocab_size=22)
    model.eval()

    # Create two inputs that differ only in the last position
    batch_size = 1
    seq_len = 6
    input_ids_1 = torch.randint(1, 10, (batch_size, seq_len))
    input_ids_2 = input_ids_1.clone()
    input_ids_2[:, -1] = (input_ids_1[:, -1] + 5) % 10  # Modify last token

    # Create is_melody mask (alternating pattern: SOS, chord, melody, chord, melody, ...)
    is_melody = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for i in range(seq_len):
        if i > 0 and i % 2 == 0:  # Even positions after 0 are melody
            is_melody[:, i] = True

    # Forward pass for both inputs
    with torch.no_grad():
        out_1 = model(input_ids_1, is_melody)
        out_2 = model(input_ids_2, is_melody)

    # Outputs at positions before the last should be identical (causal behavior)
    # Due to T5's causal attention, changing the last token should not affect
    # the outputs at earlier positions
    assert torch.allclose(out_1[:, :-1, :], out_2[:, :-1, :], atol=1e-5)


def test_online_vocab_sizes() -> None:
    """Model should expose the configured vocab sizes."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    melody_vocab_size = 12
    chord_vocab_size = 22
    model = OnlineTransformer(
        m_cfg, d_cfg, melody_vocab_size=melody_vocab_size, chord_vocab_size=chord_vocab_size
    )

    assert model.melody_vocab_size == melody_vocab_size
    assert model.chord_vocab_size == chord_vocab_size


def test_online_generate_output_shape() -> None:
    """Generate should return chord sequence matching melody length."""
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    melody_vocab_size = 12
    chord_vocab_size = 22
    model = OnlineTransformer(
        m_cfg,
        d_cfg,
        melody_vocab_size=melody_vocab_size,
        chord_vocab_size=chord_vocab_size,
    )
    model.eval()

    batch_size = 2
    melody_len = 8
    # Melody input is in melody vocab space
    melody = torch.randint(1, melody_vocab_size, (batch_size, melody_len))

    chords = model.generate(melody, sos_id=1, eos_id=2)

    # Output has one chord per melody token (SOS is internal, not returned)
    # Output is in chord vocab space
    assert chords.shape == (batch_size, melody_len)
    assert chords.max().item() < chord_vocab_size


def test_online_generate_with_variable_length_batch() -> None:
    """Test that batched generation handles padding correctly for variable-length melodies.

    This verifies the cumulative attention mask fix: padding tokens in past positions
    should remain masked throughout generation, not be incorrectly marked as valid.
    """
    d_cfg = DataConfig(max_len=16)
    m_cfg = OnlineConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    melody_vocab_size = 12
    chord_vocab_size = 22
    model = OnlineTransformer(
        m_cfg,
        d_cfg,
        melody_vocab_size=melody_vocab_size,
        chord_vocab_size=chord_vocab_size,
    )
    model.eval()

    # Create batch with variable-length melodies:
    # - Melody 0: length 6 (full, no padding)
    # - Melody 1: length 4 (padded to length 6 with pad_id=0)
    max_len = 6
    melody_0 = torch.randint(1, melody_vocab_size, (1, max_len))  # No padding
    melody_1_real = torch.randint(1, melody_vocab_size, (1, 4))  # Real tokens
    melody_1_pad = torch.full((1, 2), d_cfg.pad_id, dtype=torch.long)  # Padding
    melody_1 = torch.cat([melody_1_real, melody_1_pad], dim=1)  # [real, real, real, real, pad, pad]

    # Batch them together
    batched_melody = torch.cat([melody_0, melody_1], dim=0)  # (2, 6)

    # Generate chords for the batch
    chords = model.generate(batched_melody, sos_id=d_cfg.sos_id, eos_id=d_cfg.eos_id)

    # Basic shape and validity checks
    assert chords.shape == (2, max_len)
    assert chords.max().item() < chord_vocab_size
    assert chords.min().item() >= 0

    # Generate melody_0 individually (unbatched) and verify it matches the batched result
    # This confirms that padding in other batch items doesn't affect generation
    chords_0_individual = model.generate(melody_0, sos_id=d_cfg.sos_id, eos_id=d_cfg.eos_id)

    # The first melody's chords should be identical whether batched or individual
    # (since it has no padding and shouldn't be affected by other batch items)
    assert torch.equal(chords[0], chords_0_individual[0])
