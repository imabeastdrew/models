"""Offline encoder–decoder transformer for chord generation.

The offline model sees the complete melody sequence before generating chords.

This module provides an ``OfflineTransformer`` with a simple interface:

- ``forward(src, tgt) -> logits`` where both ``src`` and ``tgt`` are integer
  IDs in a unified vocabulary shared between melody and chord tokens.
- ``generate(src, max_len, sos_id, eos_id, temperature, sample)`` which
  returns generated token IDs in this same unified vocabulary.

Internally we operate purely in this unified token space (mirroring the online
model); higher-level utilities such as datasets and decoders are responsible
for mapping between unified IDs and human-readable melody/chord tokens.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    T5Config as Seq2SeqConfig,
    T5ForConditionalGeneration as Seq2SeqForConditionalGeneration,
)

from musicagent.config import DataConfig, OfflineConfig


class _ChordOnlyLogitsProcessor(LogitsProcessor):
    """Logits processor that masks out non-chord tokens during generation.

    The processor keeps a CPU copy of the base mask and lazily materializes
    per-(device, dtype) views to avoid device mismatches when the model is
    moved (e.g., via ``model.to("cuda")``) and to reduce repeated transfers.
    """

    def __init__(self, mask: torch.Tensor):
        # Store a canonical float32 CPU copy; per-device views are created
        # on demand in ``_mask_for``.
        self._base_mask = mask.detach().clone().to("cpu", dtype=torch.float32)
        self._cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _mask_for(self, scores: torch.FloatTensor) -> torch.Tensor:
        key = (scores.device, scores.dtype)
        if key not in self._cache:
            self._cache[key] = self._base_mask.to(scores.device, dtype=scores.dtype)
        return self._cache[key]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        mask = self._mask_for(scores)
        if mask.numel() == scores.size(-1):
            return scores + mask
        return scores


class OfflineTransformer(nn.Module):
    """Encoder–decoder transformer for offline chord generation.

    The model operates purely in the unified token ID space emitted by
    preprocessing. Both melody (encoder input) and chords (decoder input /
    output) are represented as integer IDs in this shared vocabulary.
    """

    def __init__(
        self,
        model_config: OfflineConfig,
        data_config: DataConfig,
        vocab_size: int,
        chord_token_ids: list[int] | None = None,
    ):
        """Create an offline encoder–decoder model.

        Args:
            model_config: OfflineConfig with architecture hyperparameters.
            data_config: DataConfig with token IDs (pad/sos/eos, max_len, ...).
            vocab_size: Size of the unified token vocabulary.
            chord_token_ids: Optional list of token IDs that correspond to
                chord symbols. When provided, generation will be constrained
                to this set (plus special tokens) by masking logits.
        """
        super().__init__()
        self.cfg = model_config
        self.data_cfg = data_config

        self.vocab_size = vocab_size
        self.pad_id = data_config.pad_id

        # ------------------------------------------------------------------
        # Logits mask for generation (optional chord-only decoding)
        # ------------------------------------------------------------------
        logits_mask = torch.zeros(self.vocab_size, dtype=torch.float32)
        if chord_token_ids is not None:
            # Disallow everything except chord tokens and a few special tokens.
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

        # Configure a compact encoder–decoder transformer matching our
        # OfflineConfig dimensions. We keep the architecture close to the
        # original paper: d_model=512, n_layers=8, etc., but allow overrides
        # via OfflineConfig.
        model_cfg = Seq2SeqConfig(
            vocab_size=self.vocab_size,
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
            src: Source melody tokens (batch, src_len) in unified vocab space.
            tgt: Target chord tokens (batch, tgt_len) in unified vocab space.

        Returns:
            Logits over the unified vocabulary (batch, tgt_len, vocab_size).
        """
        outputs = self.model(
            input_ids=src,
            attention_mask=src.ne(self.pad_id),
            decoder_input_ids=tgt,
            use_cache=False,
            return_dict=True,
        )

        return outputs.logits

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

        This mirrors the original CLI/eval interface while delegating the heavy
        lifting to :meth:`transformers.T5ForConditionalGeneration.generate`,
        which uses the model's built-in KV cache for efficient decoding.

        Args:
            src: (batch, src_len) melody token IDs in unified vocab space.
            max_len: Maximum number of chord tokens to generate.
            sos_id: Start-of-sequence token ID in the unified vocabulary.
            eos_id: End-of-sequence token ID in the unified vocabulary.
            temperature: Softmax temperature (used only when ``sample=True``).
            sample: If True, sample from the distribution; otherwise greedy.

        Returns:
            (batch, generated_len) token IDs in the unified vocabulary.
        """
        device = src.device
        _, _ = src.size()

        encoder_attention_mask = src.ne(self.pad_id)

        # Logits processor for chord-only decoding (optional).
        mask = cast(torch.Tensor, self.logits_mask)
        logits_processors = LogitsProcessorList()
        if mask.numel() == self.vocab_size:
            logits_processors.append(_ChordOnlyLogitsProcessor(mask))

        # Delegate to Hugging Face's generate(), which uses KV caching.
        # We explicitly pass ``decoder_start_token_id`` so that the caller's
        # ``sos_id`` argument continues to control the start token and the
        # behavior matches the previous manual loop.
        gen_kwargs: dict[str, object] = {
            "input_ids": src,
            "attention_mask": encoder_attention_mask,
            "max_new_tokens": max_len,
            "eos_token_id": eos_id,
            "pad_token_id": self.pad_id,
            "decoder_start_token_id": sos_id,
            "do_sample": sample,
            "num_beams": 1,
            "use_cache": True,
        }
        if sample and temperature != 1.0:
            gen_kwargs["temperature"] = float(temperature)

        generated = self.model.generate(
            logits_processor=logits_processors if len(logits_processors) > 0 else None,
            **gen_kwargs,
        )

        # ``generated`` has shape (batch, seq_len_dec) including the initial
        # decoder start token. We drop that first token so the returned tensor
        # contains only the generated sequence, as in the previous
        # implementation.
        if generated.size(1) > 0:
            generated = generated[:, 1:]

        # Ensure we never exceed ``max_len`` in the returned sequence.
        return generated[:, :max_len]

