"""Online transformer using T5 decoder stack.

The online model generates chords without seeing the current or future melody.
It is trained on interleaved sequences of the form:

    [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]

where y_t are chord tokens (in chord vocab space) and x_t are melody tokens
(in melody vocab space). The model uses separate embedding tables for each
modality, selected via the is_melody mask.

The model is trained as a causal language model over this sequence and, at
inference time, we only use the chord predictions while feeding in the given
melody tokens.

This implementation uses HuggingFace's T5Stack configured as a decoder-only
model (no encoder), providing T5-specific features:
- T5-style feed-forward network (ReLU by default; set feed_forward_proj="gated-gelu"
  in T5Config for gated variant used in T5 1.1)
- T5-style layer normalization (pre-norm architecture)
- T5 relative position bias
- T5 weight initialization
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack

from musicagent.config import DataConfig, OnlineConfig


class OnlineTransformer(nn.Module):
    """T5-based causal transformer over interleaved melody/chord tokens.

    Architecture:
    - Separate embedding layers for melody and chord vocabularies
    - T5Stack configured as decoder-only (causal self-attention)
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

        # Configure T5Stack as decoder-only (no encoder)
        # Using is_decoder=True enables causal attention mask
        # Using is_encoder_decoder=False removes cross-attention layers
        t5_config = T5Config(
            vocab_size=chord_vocab_size,  # Not used directly since we provide inputs_embeds
            d_model=model_config.d_model,
            d_kv=model_config.d_model // model_config.n_heads,
            d_ff=model_config.d_model * 4,
            num_heads=model_config.n_heads,
            num_layers=model_config.n_layers,
            dropout_rate=model_config.dropout,
            is_decoder=True,
            is_encoder_decoder=False,  # Decoder-only mode (no cross-attention)
            use_cache=True,
            pad_token_id=data_config.pad_id,
        )

        # T5Stack requires an embed_tokens argument, but we use inputs_embeds
        # in forward(), so we create a dummy embedding that won't be used.
        # This is similar to how the offline model handles encoder embeddings.
        self._dummy_embed = nn.Embedding(1, model_config.d_model)
        self.decoder = T5Stack(t5_config, embed_tokens=self._dummy_embed)

        # Output projection to chord vocabulary only
        self.fc_out = nn.Linear(model_config.d_model, chord_vocab_size, bias=False)

    def _compute_embeddings(
        self,
        input_ids: torch.Tensor,
        is_melody: torch.Tensor,
    ) -> torch.Tensor:
        """Compute embeddings using separate melody/chord embedding tables.

        Args:
            input_ids: Token IDs (batch, seq_len). Chord positions contain
                      chord vocab IDs, melody positions contain melody vocab IDs.
            is_melody: Boolean mask (batch, seq_len). True for melody positions.

        Returns:
            Embedded tensor (batch, seq_len, d_model).
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.melody_embed.weight.dtype

        x = torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=dtype)

        # Embed melody positions (is_melody == True)
        if is_melody.any():
            x[is_melody] = self.melody_embed(input_ids[is_melody])

        # Embed chord positions (is_melody == False)
        chord_mask = ~is_melody
        if chord_mask.any():
            x[chord_mask] = self.chord_embed(input_ids[chord_mask])

        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        is_melody: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Forward pass for training or generation.

        Args:
            input_ids: Interleaved tokens (batch, seq_len). Chord positions contain
                       chord vocab IDs, melody positions contain melody vocab IDs.
                       Format: [SOS, y₁, x₁, y₂, x₂, ...]
            is_melody: Boolean mask (batch, seq_len). True for melody positions,
                       False for chord positions (including SOS).
            attention_mask: Optional attention mask (batch, total_seq_len) where
                           total_seq_len = past_len + seq_len. Values are 1 for valid
                           positions, 0 for padding. If None, computed from input_ids.
                           When using KV-cache during generation, pass the full
                           cumulative mask to correctly handle padding in past positions.
            past_key_values: Tuple of tuples of (key, value) tensors from previous step.
                            Shape per layer: ((batch, n_heads, past_len, head_dim), ...).
            use_cache: Whether to return new key/value states.

        Returns:
            Logits over chord vocabulary (batch, seq_len, chord_vocab_size).
            If use_cache is True, also returns tuple of (key, value) tuples.
        """
        # Compute embeddings using separate embedding tables
        inputs_embeds = self._compute_embeddings(input_ids, is_melody)

        # Create attention mask if not provided
        if attention_mask is None:
            # Default: compute from input_ids (1 for valid, 0 for padding)
            attention_mask = input_ids.ne(self.pad_id).to(dtype=inputs_embeds.dtype)

            # If we have past key values, prepend mask for past positions
            # NOTE: This assumes all past positions are valid. For correct handling
            # of padding in batched generation, pass an explicit attention_mask.
            if past_key_values is not None:
                past_len = past_key_values[0][0].size(2)
                past_mask = torch.ones(
                    (attention_mask.size(0), past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=1)

        # Forward through T5Stack decoder
        # T5Stack with is_decoder=True automatically applies causal mask
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.fc_out(hidden_states)

        if use_cache:
            return logits, outputs.past_key_values
        return logits

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

        # Track cumulative attention mask to correctly handle padding in batched
        # generation with variable-length melodies. SOS is always valid.
        cumulative_mask = torch.ones((batch_size, 1), dtype=torch.float, device=device)

        generated_chords: list[torch.Tensor] = []
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None

        for t in range(melody_len):
            # Predict y_{t+1} given current context
            # Context is [SOS] for t=0, or [SOS, y₁, x₁, ..., y_t, x_t] for t>0
            result = self.forward(
                input_ids,
                is_melody,
                attention_mask=cumulative_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = cast(
                tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]],
                result,
            )

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

            # Update cumulative attention mask for next step:
            # - Generated chord is always valid (we just generated it)
            # - Melody token validity depends on whether it's padding
            chord_valid = torch.ones((batch_size, 1), dtype=torch.float, device=device)
            melody_valid = mel_token.ne(self.pad_id).unsqueeze(1).to(dtype=torch.float)
            cumulative_mask = torch.cat([cumulative_mask, chord_valid, melody_valid], dim=1)

        # Stack generated chords: (batch, num_chords) in chord vocab space.
        return torch.stack(generated_chords, dim=1)
