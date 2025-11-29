"""Online transformer.

The online model generates chords without seeing the current or future melody.
It is trained on interleaved sequences of the form:

    [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]

where y_t are chord tokens (in unified chord space) and x_t are melody tokens.
The model is trained as a causal language model over this sequence and, at
inference time, we only use the chord predictions while feeding in the given
melody tokens.
"""

import torch
import torch.nn as nn

from musicagent.config import DataConfig, OnlineConfig
from musicagent.models.components import PositionalEncoding


class OnlineTransformer(nn.Module):
    """Causal transformer over interleaved melody/chord tokens.

    Architecture:
    - Single embedding layer for unified vocabulary (melody + chord tokens)
    - Stack of transformer *encoder* layers with a causal self-attention mask
    - Output projection to unified vocabulary

    Training input format (from OnlineDataset):
        [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]

    - Position 0: SOS chord token (in unified chord space)
    - Odd positions  (1, 3, 5, ...): chord tokens (y₁, y₂, ...)
    - Even positions (2, 4, 6, ...): melody tokens (x₁, x₂, ...)

    The model is optimized to predict the next token at every position. At
    inference, we follow the online process: at each step we predict a chord
    token given past melody/chord context, then append the observed melody
    token from the user.
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
        self.unified_vocab_size = melody_vocab_size + chord_vocab_size

        # Unified embedding for both melody and chord tokens
        self.embed = nn.Embedding(
            self.unified_vocab_size,
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
        self.fc_out = nn.Linear(model_config.d_model, self.unified_vocab_size)

        self.pad_id = data_config.pad_id

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
            input_ids: Interleaved tokens (batch, seq_len)
                       Format: [SOS, y₁, x₁, y₂, x₂, ...]

        Returns:
            Logits over unified vocabulary (batch, seq_len, unified_vocab_size)
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

    def _melody_to_unified(self, token_id: int) -> int:
        """Convert melody token ID to unified vocab ID."""
        return token_id

    def _chord_to_unified(self, token_id: int) -> int:
        """Convert chord token ID to unified vocab ID.

        Special handling for PAD: map chord pad to unified pad (0).
        """
        if token_id == self.pad_id:
            return self.pad_id
        return token_id + self.melody_vocab_size

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
                Frame-space melody token IDs in *melody vocab space*,
                shape (batch, melody_len). This tensor should NOT include
                any sequence-level SOS/EOS/PAD bookkeeping; tokens are
                passed through to the unified vocab unchanged.
            sos_id:
                Start-of-sequence token ID in the chord vocabulary.
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
            Tensor of generated chord token IDs in *chord vocab space*,
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

        # Convert SOS to unified vocab
        sos_unified = self._chord_to_unified(sos_id)

        # Precompute a mask that disables melody tokens in the unified
        # vocabulary so that chord predictions cannot select them. We also
        # disable chord-side SOS/EOS tokens, which are used only for
        # bookkeeping and never appear as frame-level chord targets.
        melody_mask = torch.zeros(self.unified_vocab_size, device=device)
        melody_mask[:self.melody_vocab_size] = float("-inf")

        sos_unified = self._chord_to_unified(sos_id)
        eos_unified = self._chord_to_unified(eos_id)
        melody_mask[sos_unified] = float("-inf")
        melody_mask[eos_unified] = float("-inf")

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
            next_logits = logits[:, -1, :]  # (batch, unified_vocab)

            # Mask out melody tokens - only allow chord predictions
            # Chord tokens are in range [melody_vocab_size, unified_vocab_size)
            next_logits = next_logits + melody_mask

            if sample:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_unified = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_unified = next_logits.argmax(dim=-1)

            # Convert back to chord vocab
            next_chord = next_unified - self.melody_vocab_size

            generated_chords.append(next_chord)

            # Get current melody token (no offset needed for melody)
            mel_token = melody[:, t]
            mel_unified = mel_token

            # Append in CORRECT order matching training: chord (y_t) THEN melody (x_t)
            # This builds: [SOS, y₁, x₁, y₂, x₂, ...]
            sequence = torch.cat([
                sequence,
                next_unified.unsqueeze(1),
                mel_unified.unsqueeze(1),
            ], dim=1)

        # Stack generated chords: (batch, num_chords)
        return torch.stack(generated_chords, dim=1)

