"""Online transformer.

The online model generates chords without seeing the current or future melody.
It is trained on interleaved sequences of the form:

    [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]

where y_t are chord tokens (in unified chord space) and x_t are melody tokens.
The model is trained as a causal language model over this sequence and, at
inference time, we only use the chord predictions while feeding in the given
melody tokens.
"""

from typing import cast

import torch
import torch.nn as nn

from musicagent.config import DataConfig, OnlineConfig
from musicagent.models.components import PositionalEncoding


class OnlineTransformer(nn.Module):
    """Causal transformer over interleaved melody/chord tokens in unified ID space.

    Architecture:
    - Single embedding layer for unified vocabulary (melody + chord tokens)
    - Stack of transformer *encoder* layers with a causal self-attention mask
    - Output projection to unified vocabulary

    Training input format (from OnlineDataset):
        [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]

    - Position 0: SOS token (unified vocab)
    - Odd positions  (1, 3, 5, ...): chord tokens (y₁, y₂, ...)
    - Even positions (2, 4, 6, ...): melody tokens (x₁, x₂, ...)

    The model is optimized to predict the next token at every position. During
    training we mask out loss contributions from melody tokens so that only
    chord predictions are learned, mirroring the paper's setup.
    """

    def __init__(
        self,
        model_config: OnlineConfig,
        data_config: DataConfig,
        vocab_size: int,
        chord_token_ids: list[int] | None = None,
    ):
        super().__init__()
        self.cfg = model_config
        self.data_cfg = data_config

        self.vocab_size = vocab_size

        # Unified embedding for both melody and chord tokens
        self.embed = nn.Embedding(
            self.vocab_size,
            model_config.d_model,
            padding_idx=data_config.pad_id,
        )

        # Positional encoding length must accommodate the full interleaved
        # sequence produced by OnlineDataset. With a frame horizon of
        # `data_config.max_len` we have at most:
        #
        #   len(interleaved) = 1 + 2T  where T ≤ data_config.max_len
        #
        # so we allocate positional embeddings for up to `1 + 2 * max_len`
        # positions.
        max_seq_len = 1 + 2 * data_config.max_len
        self.pos_enc = PositionalEncoding(model_config.d_model, max_seq_len)

        # Causal encoder: stack of transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config.d_model,
            nhead=model_config.n_heads,
            dim_feedforward=model_config.d_model * 4,
            dropout=model_config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config.n_layers,
        )

        # Final layer norm (for norm_first architecture)
        self.final_norm = nn.LayerNorm(model_config.d_model)

        # Output projection to unified vocabulary
        self.fc_out = nn.Linear(model_config.d_model, self.vocab_size)

        self.pad_id = data_config.pad_id

        # Optional logits mask to constrain generation to chord tokens plus a
        # small set of special symbols (pad/sos/eos/rest). This mirrors the
        # paper's setup where the online model predicts chords conditioned on
        # melody, not new melody tokens.
        logits_mask = torch.zeros(self.vocab_size, dtype=torch.float32)
        if chord_token_ids is not None:
            mask = torch.full((self.vocab_size,), float("-inf"), dtype=torch.float32)
            special_ids = {
                self.pad_id,
                data_config.sos_id,
                data_config.eos_id,
                data_config.rest_id,
            }
            for idx in special_ids:
                if 0 <= idx < self.vocab_size:
                    mask[idx] = 0.0
            for idx in chord_token_ids:
                if 0 <= idx < self.vocab_size:
                    mask[idx] = 0.0
            logits_mask = mask

        self.register_buffer("logits_mask", logits_mask, persistent=False)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask.

        Args:
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            Boolean mask of shape (seq_len, seq_len), True where masked
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for training.

        Args:
            input_ids: Interleaved tokens (batch, seq_len) in unified ID space.
                       Format: [SOS, y₁, x₁, y₂, x₂, ...]

        Returns:
            Logits over unified vocabulary (batch, seq_len, vocab_size)
        """
        device = input_ids.device
        seq_len = input_ids.size(1)

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, device)

        # Create padding mask
        padding_mask = (input_ids == self.pad_id)

        # Embed and add positional encoding
        x = self.pos_enc(self.embed(input_ids))

        # Causal encoder: self-attention with a lower-triangular (causal) mask
        out = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        out = self.final_norm(out)
        logits: torch.Tensor = self.fc_out(out)
        return logits

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
                Frame-space melody token IDs in the unified vocab space,
                shape (batch, melody_len). This tensor should NOT include
                any sequence-level SOS/EOS/PAD bookkeeping; tokens are
                passed through unchanged.
            sos_id:
                Start-of-sequence token ID in the unified vocabulary.
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
            Tensor of generated token IDs in the *unified vocabulary*,
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

        # Use SOS directly in unified vocab space.
        sos_unified = sos_id

        # Initialize with SOS chord token
        # Sequence format: [SOS, y₁, x₁, y₂, x₂, ...]
        sequence = torch.full(
            (batch_size, 1),
            sos_unified,
            dtype=torch.long,
            device=device,
        )

        generated_chords: list[torch.Tensor] = []

        for t in range(melody_len):
            # Predict y_{t+1} given current context
            # Context is [SOS] for t=0, or [SOS, y₁, x₁, ..., y_t, x_t] for t>0
            logits = self.forward(sequence)  # (batch, seq_len, vocab)

            # Get logits for next position (chord prediction)
            next_logits = logits[:, -1, :]  # (batch, vocab)

            # Optionally mask to chord tokens only.
            mask = cast(torch.Tensor, self.logits_mask)
            if mask.numel() == next_logits.size(-1):
                next_logits = next_logits + mask

            if sample:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_unified = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_unified = next_logits.argmax(dim=-1)

            generated_chords.append(next_unified)

            # Get current melody token
            mel_unified = melody[:, t]

            # Append in CORRECT order matching training: chord (y_t) THEN melody (x_t)
            # This builds: [SOS, y₁, x₁, y₂, x₂, ...]
            sequence = torch.cat([
                sequence,
                next_unified.unsqueeze(1),
                mel_unified.unsqueeze(1),
            ], dim=1)

        # Stack generated chords: (batch, num_chords) in unified ID space.
        return torch.stack(generated_chords, dim=1)

