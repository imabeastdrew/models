"""Online transformer.

The online model generates chords without seeing the current or future melody.
It is trained on interleaved sequences of the form:

    [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]

where y_t are chord tokens (in chord vocab space) and x_t are melody tokens
(in melody vocab space). The model uses separate embedding tables for each
modality, selected via the is_melody mask.

The model is trained as a causal language model over this sequence and, at
inference time, we only use the chord predictions while feeding in the given
melody tokens.
"""

import math
from typing import cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from musicagent.config import DataConfig, OnlineConfig


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm without mean subtraction."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)


class RelativePositionBias(nn.Module):
    """Relative position bias for self-attention."""

    def __init__(self, num_buckets: int, max_distance: int, num_heads: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Translate relative position to a bucket number (T5-style)."""
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        n = -relative_position
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            (
                (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact))
                * (num_buckets - max_exact)
            ).to(torch.long)
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        # torch.where is typed as returning Any in some torch stubs; cast for mypy.
        ret = torch.where(is_small, n, val_if_large)
        return cast(torch.Tensor, ret)

    def forward(
        self, seq_len: int, device: torch.device, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """Return bias tensor of shape (1, num_heads, seq_len, seq_len + past_key_values_length)."""
        total_len = seq_len + past_key_values_length
        context_position = torch.arange(
            past_key_values_length, total_len, dtype=torch.long, device=device
        )[:, None]
        memory_position = torch.arange(total_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # (S, S)
        rp_bucket = self._relative_position_bucket(relative_position)
        # (S, S, H)
        values = self.relative_attention_bias(rp_bucket)
        # (1, H, S, S)
        bias = values.permute(2, 0, 1).unsqueeze(0)
        return cast(torch.Tensor, bias)


class SelfAttentionWithRelPos(nn.Module):
    """Multi-head self-attention with relative position bias and causal mask."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        rel_pos_bias: RelativePositionBias,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rel_pos_bias = rel_pos_bias

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            x: (batch, seq_len, d_model)
            causal_mask: (seq_len, seq_len + past_len) bool, True where masked
            padding_mask: (batch, seq_len + past_len) bool, True for PAD tokens
            past_key_value: Tuple of (key, value) tensors from previous step
            use_cache: Whether to return new key/value states
        """
        bsz, seq_len, _ = x.size()
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (batch, n_heads, seq_len, head_dim)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_key_value = (k, v) if use_cache else None

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Relative position bias: (1, n_heads, seq_len, seq_len + past_len)
        rel_bias = self.rel_pos_bias(seq_len, device, past_key_values_length=past_len)
        attn_scores = attn_scores + rel_bias

        # Causal mask
        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(
                causal_mask.to(device=device)[None, None, :, :], float("-inf")
            )

        # Key padding mask
        if padding_mask is not None:
            attn_scores = attn_scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        return self.o_proj(attn_output), current_key_value


class DecoderBlock(nn.Module):
    """Single decoder block with self-attention and feed-forward network."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        rel_pos_bias: RelativePositionBias,
    ):
        super().__init__()
        self.self_attn = SelfAttentionWithRelPos(d_model, n_heads, dropout, rel_pos_bias)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        d_ff = d_model * 4
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Self-attention block
        h, present_kv = self.self_attn(
            self.norm1(x),
            causal_mask,
            padding_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + self.dropout(h)

        # Feed-forward block
        h = self.ff(self.norm2(x))
        x = x + self.dropout(h)
        return x, present_kv


class OnlineTransformer(nn.Module):
    """Causal transformer over interleaved melody/chord tokens with separate embeddings.

    Architecture:
    - Separate embedding layers for melody and chord vocabularies
    - Stack of transformer *encoder* layers with a causal self-attention mask
    - Output projection to chord vocabulary only

    Training input format (from OnlineDataset):
        [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]

    - Position 0: SOS token (chord vocab space)
    - Odd positions  (1, 3, 5, ...): chord tokens (y₁, y₂, ...) in chord vocab space
    - Even positions (2, 4, 6, ...): melody tokens (x₁, x₂, ...) in melody vocab space

    The model uses an is_melody mask to select the appropriate embedding for each
    position. Output logits are over chord vocabulary only.
    """

    def __init__(
        self,
        model_config: OnlineConfig,
        data_config: DataConfig,
        melody_vocab_size: int,
        chord_vocab_size: int,
    ):
        super().__init__()
        self.cfg = model_config
        self.data_cfg = data_config

        self.melody_vocab_size = melody_vocab_size
        self.chord_vocab_size = chord_vocab_size
        self.d_model = model_config.d_model
        self.pad_id = data_config.pad_id
        # Gradient checkpointing flag (applied to decoder blocks in training).
        self.gradient_checkpointing: bool = False

        # Separate embeddings for melody and chord tokens
        self.melody_embed = nn.Embedding(
            melody_vocab_size,
            model_config.d_model,
            padding_idx=data_config.pad_id,
        )
        self.chord_embed = nn.Embedding(
            chord_vocab_size,
            model_config.d_model,
            padding_idx=data_config.pad_id,
        )

        # Relative position bias shared across all decoder layers. The maximum
        # sequence length seen during training is 1 + 2 * max_len.
        self.rel_pos_bias = RelativePositionBias(
            num_buckets=32,
            max_distance=128,
            num_heads=model_config.n_heads,
        )

        # Stack of decoder blocks
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=model_config.d_model,
                    n_heads=model_config.n_heads,
                    dropout=model_config.dropout,
                    rel_pos_bias=self.rel_pos_bias,
                )
                for _ in range(model_config.n_layers)
            ]
        )

        # Final normalization and projection to chord vocabulary only
        self.final_norm = RMSNorm(model_config.d_model)
        self.fc_out = nn.Linear(model_config.d_model, chord_vocab_size)

    def _create_causal_mask(
        self, seq_len: int, device: torch.device, past_len: int = 0
    ) -> torch.Tensor:
        """Create causal attention mask.

        Args:
            seq_len: Sequence length
            device: Device to create tensor on
            past_len: Length of past key values

        Returns:
            Boolean mask of shape (seq_len, seq_len + past_len), True where masked
        """
        full_len = seq_len + past_len
        mask = torch.triu(
            torch.ones(full_len, full_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask[past_len:, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        is_melody: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for training or generation.

        Args:
            input_ids: Interleaved tokens (batch, seq_len). Chord positions contain
                       chord vocab IDs, melody positions contain melody vocab IDs.
                       Format: [SOS, y₁, x₁, y₂, x₂, ...]
            is_melody: Boolean mask (batch, seq_len). True for melody positions,
                       False for chord positions (including SOS).
            past_key_values: List of tuples of (key, value) tensors from previous step.
            use_cache: Whether to return new key/value states.

        Returns:
            Logits over chord vocabulary (batch, seq_len, chord_vocab_size)
            If use_cache is True, also returns list of new (key, value) tuples.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.melody_embed.weight.dtype

        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values[0][0].size(2)

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, device, past_len=past_len)

        # Create padding mask
        padding_mask = input_ids == self.pad_id
        if past_len > 0:
            # Assume past is not padded (valid context)
            past_mask = torch.zeros((batch_size, past_len), dtype=torch.bool, device=device)
            padding_mask = torch.cat([past_mask, padding_mask], dim=1)

        # Embed tokens using separate embeddings based on is_melody mask.
        # We use boolean indexing to avoid IndexError from out-of-range IDs.
        x = torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=dtype)

        # Embed melody positions (is_melody == True)
        if is_melody.any():
            x[is_melody] = self.melody_embed(input_ids[is_melody])

        # Embed chord positions (is_melody == False)
        chord_mask = ~is_melody
        if chord_mask.any():
            x[chord_mask] = self.chord_embed(input_ids[chord_mask])

        next_decoder_cache = []

        # Apply stacked decoder blocks with causal + padding masks.
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            use_checkpoint = (
                self.gradient_checkpointing and self.training and not use_cache and past_kv is None
            )

            if use_checkpoint:

                def _layer_forward(
                    x_in: torch.Tensor,
                    *,
                    _layer=layer,
                ) -> torch.Tensor:
                    out, _ = _layer(
                        x_in,
                        causal_mask,
                        padding_mask,
                        past_key_value=None,
                        use_cache=False,
                    )
                    from typing import cast as _cast

                    return _cast(torch.Tensor, out)

                x = checkpoint(_layer_forward, x)
                present_kv = None
            else:
                x, present_kv = layer(
                    x,
                    causal_mask,
                    padding_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )
                if use_cache:
                    next_decoder_cache.append(present_kv)

        out = self.final_norm(x)
        logits: torch.Tensor = self.fc_out(out)

        if use_cache:
            return logits, next_decoder_cache
        return logits

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for decoder blocks.

        This trades additional compute for reduced activation memory during
        training. It is automatically unused during generation, which runs
        under ``torch.no_grad()`` with ``use_cache=True``.
        """
        self.gradient_checkpointing = True

    @torch.no_grad()
    def generate(
        self,
        melody: torch.Tensor,
        sos_id: int,
        eos_id: int,
        temperature: float = 1.0,
        sample: bool = False,
    ) -> torch.Tensor:
        """Generate chord accompaniment for a given melody.

        This implements the online generation process matching the MLE
        training order. Training sequences are interleaved as:

            [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]

        For each timestep t we:
            1. Predict y_t given [SOS, y₁, x₁, ..., y_{t-1}, x_{t-1}]
            2. Append y_t to the sequence (chord first)
            3. Append x_t to the sequence (melody second)

        Args:
            melody:
                Frame-space melody token IDs in melody vocab space,
                shape (batch, melody_len). This tensor should NOT include
                any sequence-level SOS/EOS/PAD bookkeeping; tokens are
                passed through unchanged.
            sos_id:
                Start-of-sequence token ID (same ID in both vocab spaces).
            eos_id:
                Ignored in the current implementation. Generation runs for
                one chord per available melody frame (up to the configured
                frame horizon) rather than stopping on an EOS token.
            temperature:
                Softmax temperature for sampling (used only when
                ``sample=True``).
            sample:
                If True, sample from the chord distribution; otherwise
                use greedy argmax decoding.

        Returns:
            Tensor of generated token IDs in chord vocab space,
            shape (batch, num_steps), where
            ``num_steps = min(melody_len, data_cfg.max_len)``.
        """
        device = melody.device
        batch_size = melody.size(0)
        melody_len = melody.size(1)

        # Mirror the training horizon: OnlineDataset constrains the number of
        # *frames* such that T ≤ data_cfg.max_len and constructs interleaved
        # sequences of length 1 + 2T. To avoid extrapolating far beyond the
        # training context (and to stay within positional encoding bounds), we
        # cap the number of melody frames we respond to using the same frame
        # limit.
        max_frames = self.data_cfg.max_len
        if melody_len > max_frames:
            melody = melody[:, :max_frames]
            melody_len = melody.size(1)

        # Initialize with SOS chord token (in chord vocab space)
        # Sequence format: [SOS, y₁, x₁, y₂, x₂, ...]
        input_ids = torch.full(
            (batch_size, 1),
            sos_id,
            dtype=torch.long,
            device=device,
        )
        # SOS is a chord position
        is_melody = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)

        generated_chords: list[torch.Tensor] = []
        past_key_values = None

        for t in range(melody_len):
            # Predict y_{t+1} given current context
            # Context is [SOS] for t=0, or [SOS, y₁, x₁, ..., y_t, x_t] for t>0
            logits, past_key_values = self.forward(
                input_ids, is_melody, past_key_values=past_key_values, use_cache=True
            )  # (batch, seq_len, chord_vocab_size)

            # Get logits for next position (chord prediction)
            # Output is over chord vocab only, no masking needed
            next_logits = logits[:, -1, :]  # (batch, chord_vocab_size)

            if sample:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_chord = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_chord = next_logits.argmax(dim=-1)

            generated_chords.append(next_chord)

            # Get current melody token (already in melody vocab space)
            mel_token = melody[:, t]

            # Prepare input for next step: [chord, melody]
            # chord is in chord vocab space, melody is in melody vocab space
            input_ids = torch.stack([next_chord, mel_token], dim=1)
            # is_melody mask: [False, True] for [chord, melody]
            is_melody = torch.tensor([[False, True]], dtype=torch.bool, device=device).expand(
                batch_size, 2
            )

        # Stack generated chords: (batch, num_chords) in chord vocab space.
        return torch.stack(generated_chords, dim=1)
