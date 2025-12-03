"""Offline encoder–decoder transformer for chord generation.

The offline model sees the complete melody sequence before generating chords.

This module provides an ``OfflineTransformer`` with a simple interface:

- ``forward(src, tgt) -> logits`` where ``src`` contains melody token IDs in
  melody vocab space and ``tgt`` contains chord token IDs in chord vocab space.
- ``generate(src, max_len, sos_id, eos_id, temperature, sample)`` which
  returns generated token IDs in chord vocab space.

The model uses separate embedding tables for melody (encoder) and chord (decoder)
tokens, allowing each modality to develop task-appropriate representations.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from transformers import T5Config as Seq2SeqConfig
from transformers import T5ForConditionalGeneration as Seq2SeqForConditionalGeneration

from musicagent.config import DataConfig, OfflineConfig


class OfflineTransformer(nn.Module):
    """Encoder–decoder transformer for offline chord generation.

    The model uses separate vocabularies for melody (encoder) and chord (decoder):
    - Encoder input: melody token IDs in melody vocab space
    - Decoder input/output: chord token IDs in chord vocab space

    This separation allows cleaner gradient flow and task-appropriate embeddings
    for each modality.
    """

    def __init__(
        self,
        model_config: OfflineConfig,
        data_config: DataConfig,
        melody_vocab_size: int,
        chord_vocab_size: int,
    ):
        """Create an offline encoder–decoder model with separate vocabularies.

        Args:
            model_config: OfflineConfig with architecture hyperparameters.
            data_config: DataConfig with token IDs (pad/sos/eos, max_len, ...).
            melody_vocab_size: Size of the melody token vocabulary.
            chord_vocab_size: Size of the chord token vocabulary.
        """
        super().__init__()
        self.cfg = model_config
        self.data_cfg = data_config

        self.melody_vocab_size = melody_vocab_size
        self.chord_vocab_size = chord_vocab_size
        self.pad_id = data_config.pad_id

        # Custom encoder embedding for melody tokens (separate from T5's internal
        # embedding which is used for decoder/chord tokens).
        self.encoder_embed = nn.Embedding(
            melody_vocab_size,
            model_config.d_model,
            padding_idx=data_config.pad_id,
        )

        # Configure T5 with chord vocab size. T5's internal embedding will be
        # used for decoder (chord tokens). We override encoder embedding manually.
        model_cfg = Seq2SeqConfig(
            vocab_size=chord_vocab_size,
            d_model=model_config.d_model,
            d_ff=model_config.d_model * 4,
            num_heads=model_config.n_heads,
            num_layers=model_config.n_layers,
            num_decoder_layers=model_config.n_layers,
            dropout_rate=model_config.dropout,
            pad_token_id=self.pad_id,
            eos_token_id=data_config.eos_id,
            decoder_start_token_id=data_config.sos_id,
        )

        self.model = Seq2SeqForConditionalGeneration(model_cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            src: Source melody tokens (batch, src_len) in melody vocab space.
            tgt: Target chord tokens (batch, tgt_len) in chord vocab space.

        Returns:
            Logits over the chord vocabulary (batch, tgt_len, chord_vocab_size).
        """
        # Embed melody with our custom encoder embedding
        encoder_embeds = self.encoder_embed(src)

        outputs = self.model(
            inputs_embeds=encoder_embeds,
            attention_mask=src.ne(self.pad_id),
            decoder_input_ids=tgt,
            use_cache=False,
            return_dict=True,
        )

        logits = cast(torch.Tensor, outputs.logits)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        sos_id: int,
        eos_id: int,
        temperature: float = 1.0,
        sample: bool = False,
    ) -> torch.Tensor:
        """Autoregressively generate a chord sequence given a melody.

        This uses custom encoder embeddings for melody, then delegates decoding
        to HuggingFace's generate() which uses T5's internal chord embeddings.

        Args:
            src: (batch, src_len) melody token IDs in melody vocab space.
            max_len: Maximum number of chord tokens to generate.
            sos_id: Start-of-sequence token ID (same in both vocab spaces).
            eos_id: End-of-sequence token ID (same in both vocab spaces).
            temperature: Softmax temperature (used only when ``sample=True``).
            sample: If True, sample from the distribution; otherwise greedy.

        Returns:
            (batch, generated_len) token IDs in chord vocab space.
        """
        # Step 1: Embed melody with our custom encoder embedding
        encoder_embeds = self.encoder_embed(src)
        encoder_attention_mask = src.ne(self.pad_id)

        # Step 2: Manually run T5 encoder to get encoder_outputs
        encoder_outputs = self.model.encoder(
            inputs_embeds=encoder_embeds,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        # Step 3: Generate using encoder_outputs (NOT input_ids)
        # This way, decoder uses T5's chord embedding correctly
        generated = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_new_tokens=max_len,
            eos_token_id=eos_id,
            pad_token_id=self.pad_id,
            decoder_start_token_id=sos_id,
            do_sample=sample,
            num_beams=1,
            use_cache=True,
            temperature=float(temperature),
        )

        # ``generated`` has shape (batch, seq_len_dec) including the initial
        # decoder start token. We drop that first token so the returned tensor
        # contains only the generated sequence.
        tensor_generated = cast(torch.Tensor, generated)

        if tensor_generated.size(1) > 0:
            tensor_generated = tensor_generated[:, 1:]

        # Ensure we never exceed ``max_len`` in the returned sequence.
        return tensor_generated[:, :max_len]
