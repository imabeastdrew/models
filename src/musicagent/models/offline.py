"""Offline encoder–decoder transformer for chord generation (x-transformers).

Uses a single shared embedding + tied LM head over the unified vocabulary.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from x_transformers import Decoder, Encoder

from musicagent.config import DataConfig, OfflineConfig


class OfflineTransformer(nn.Module):
    """Encoder–decoder transformer for offline chord generation (x-transformers)."""

    def __init__(
        self,
        model_config: OfflineConfig,
        data_config: DataConfig,
        vocab_size: int,
    ):
        """Create an offline encoder–decoder model with a single vocabulary."""
        super().__init__()
        self.cfg = model_config
        self.data_cfg = data_config

        self.vocab_size = vocab_size
        self.pad_id = data_config.pad_id

        # Shared embedding and tied LM head
        self.token_emb = nn.Embedding(vocab_size, model_config.d_model, padding_idx=self.pad_id)

        self.encoder = Encoder(
            dim=model_config.d_model,
            depth=model_config.n_layers,
            heads=model_config.n_heads,
            ff_mult=4,
            dropout=model_config.dropout,
            rotary_pos_emb=True,
        )
        self.decoder = Decoder(
            dim=model_config.d_model,
            depth=model_config.n_layers,
            heads=model_config.n_heads,
            ff_mult=4,
            dropout=model_config.dropout,
            cross_attend=True,
            rotary_pos_emb=True,
        )

        # Optional normalization before logits
        self.norm = nn.LayerNorm(model_config.d_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            src: Source melody tokens (batch, src_len).
            tgt: Target chord tokens (batch, tgt_len).

        Returns:
            Logits over the unified vocabulary (batch, tgt_len, vocab_size).
        """
        src_mask = src.ne(self.pad_id)
        tgt_mask = tgt.ne(self.pad_id)

        src_emb = self.token_emb(src)
        tgt_emb = self.token_emb(tgt)

        enc_out = self.encoder(src_emb, mask=src_mask)
        dec_out = self.decoder(tgt_emb, context=enc_out, mask=tgt_mask, context_mask=src_mask)
        dec_out = self.norm(dec_out)
        logits = cast(torch.Tensor, dec_out @ self.token_emb.weight.T)
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
            (batch, generated_len) token IDs in unified vocab space.
        """
        device = src.device
        batch = src.size(0)
        src_mask = src.ne(self.pad_id)
        src_emb = self.token_emb(src)
        enc_out = self.encoder(src_emb, mask=src_mask)

        generated: list[torch.Tensor] = []
        cur = torch.full((batch, 1), sos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = cur.ne(self.pad_id)
            tgt_emb = self.token_emb(cur)
            dec_out = self.decoder(tgt_emb, context=enc_out, mask=tgt_mask, context_mask=src_mask)
            dec_out = self.norm(dec_out)
            next_logits = dec_out[:, -1, :] @ self.token_emb.weight.T

            if sample:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tok = next_logits.argmax(dim=-1)

            cur = torch.cat([cur, next_tok.unsqueeze(1)], dim=1)
            generated.append(next_tok)

            if not sample:
                if bool((next_tok == eos_id).all()):
                    break

        return torch.stack(generated, dim=1) if generated else torch.empty(
            (batch, 0), dtype=torch.long, device=device
        )
