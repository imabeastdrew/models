"""Online transformer using x-transformers decoder stack.

The online model generates chords without seeing the current or future melody.
It is trained on interleaved sequences of the form:

    [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]

where y_t are chord tokens and x_t are melody tokens. The model uses a single
embedding table shared across both modalities; an is_melody mask is still used
for loss masking and positional semantics.

The model is trained as a causal language model over this sequence and, at
inference time, we only use the chord predictions while feeding in the given
melody tokens. Implemented with x-transformers decoder stack.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from x_transformers import Decoder

from musicagent.config import DataConfig, OnlineConfig


class OnlineTransformer(nn.Module):
    """x-transformers causal transformer over interleaved melody/chord tokens.

    Architecture:
    - Single embedding layer over a unified vocabulary
    - T5Stack configured as decoder-only (causal self-attention)
    - Output projection to the unified vocabulary

    Training input format (from OnlineDataset):
        [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]

    - Position 0: SOS token (treated as chord position)
    - Odd positions  (1, 3, 5, ...): chord tokens (y₁, y₂, ...)
    - Even positions (2, 4, 6, ...): melody tokens (x₁, x₂, ...)

    The model keeps an is_melody mask to preserve positional semantics. Output
    logits are over the unified vocabulary.
    """

    def __init__(
        self,
        model_config: OnlineConfig,
        data_config: DataConfig,
        vocab_size: int,
    ):
        super().__init__()
        self.cfg = model_config
        self.data_cfg = data_config

        self.vocab_size = vocab_size
        self.d_model = model_config.d_model
        self.pad_id = data_config.pad_id

        # Single embedding table + tied LM head
        self.embed = nn.Embedding(
            vocab_size,
            model_config.d_model,
            padding_idx=data_config.pad_id,
        )

        self.decoder = Decoder(
            dim=model_config.d_model,
            depth=model_config.n_layers,
            heads=model_config.n_heads,
            ff_mult=4,
            dropout=model_config.dropout,
            causal=True,
            rotary_pos_emb=True,
        )
        self.norm = nn.LayerNorm(model_config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        is_melody: torch.Tensor,  # retained for API compatibility
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Forward pass for training or generation.

        Args:
            input_ids: Interleaved tokens (batch, seq_len). Format: [SOS, y₁, x₁, ...]
            is_melody: Boolean mask (unused but retained for signature stability).
            attention_mask: Optional (batch, seq_len) mask; if None, computed from input_ids.
            past_key_values: Unused (x-transformers path does full-sequence decoding).
            use_cache: Ignored; full decoding each call.

        Returns:
            Logits over the unified vocabulary (batch, seq_len, vocab_size).
        """
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_id)

        emb = self.embed(input_ids)
        dec_out = self.decoder(emb, mask=attention_mask)
        dec_out = self.norm(dec_out)
        logits = cast(torch.Tensor, dec_out @ self.embed.weight.T)
        return logits

    def enable_gradient_checkpointing(self) -> None:
        """Placeholder for API compatibility; x-transformers path does full decoding."""
        return None

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for decoder layers.

        This trades additional compute for reduced activation memory during
        training. It is automatically unused during generation, which runs
        under ``torch.no_grad()`` with ``use_cache=True``.
        """
        self.decoder.gradient_checkpointing_enable()

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
            Tensor of generated token IDs, shape (batch, num_steps), where
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

        # Initialize with SOS token (treated as chord position)
        input_ids = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        generated_chords: list[torch.Tensor] = []

        for t in range(melody_len):
            logits = self.forward(
                input_ids,
                is_melody=torch.zeros_like(input_ids, dtype=torch.bool),
                attention_mask=None,
            )
            next_logits = logits[:, -1, :]

            if sample:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_chord = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_chord = next_logits.argmax(dim=-1)

            generated_chords.append(next_chord)

            mel_token = melody[:, t]
            step_tokens = torch.stack([next_chord, mel_token], dim=1)
            input_ids = torch.cat([input_ids, step_tokens], dim=1)

        return torch.stack(generated_chords, dim=1)
