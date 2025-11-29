"""Offline transformer.

The offline model sees the complete melody before generating chords.
It uses a standard encoder-decoder transformer architecture.
"""

import torch
import torch.nn as nn

from musicagent.config import DataConfig, OfflineConfig
from musicagent.models.components import PositionalEncoding


class OfflineTransformer(nn.Module):
    """Encoder-decoder transformer for offline chord generation.

    Given a complete melody sequence, generates a chord sequence
    autoregressively using teacher forcing during training.
    """

    def __init__(
        self,
        model_config: OfflineConfig,
        data_config: DataConfig,
        vocab_src_size: int,
        vocab_tgt_size: int
    ):
        super().__init__()
        self.cfg = model_config

        self.src_embed = nn.Embedding(
            vocab_src_size,
            model_config.d_model,
            padding_idx=data_config.pad_id,
        )
        self.tgt_embed = nn.Embedding(
            vocab_tgt_size,
            model_config.d_model,
            padding_idx=data_config.pad_id,
        )
        self.pos_enc = PositionalEncoding(model_config.d_model, data_config.max_len)

        self.transformer = nn.Transformer(
            d_model=model_config.d_model,
            nhead=model_config.n_heads,
            num_encoder_layers=model_config.n_layers,
            num_decoder_layers=model_config.n_layers,
            dim_feedforward=model_config.d_model * 4,
            dropout=model_config.dropout,
            batch_first=True,
            norm_first=True
        )

        self.fc_out = nn.Linear(model_config.d_model, vocab_tgt_size)
        self.pad_id = data_config.pad_id

    def create_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        """Create attention masks for transformer.

        Args:
            src: Source tensor (batch, src_len)
            tgt: Target tensor (batch, tgt_len)

        Returns:
            Tuple of (src_key_padding_mask, tgt_key_padding_mask, tgt_mask)
        """
        src_key_padding_mask = (src == self.pad_id)
        tgt_key_padding_mask = (tgt == self.pad_id)

        seq_len = tgt.size(1)
        # Causal attention mask: True where positions should be masked.
        # Shape: [tgt_len, tgt_len], with True above the main diagonal.
        tgt_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device),
            diagonal=1,
        )

        return src_key_padding_mask, tgt_key_padding_mask, tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            src: Source melody tokens (batch, src_len)
            tgt: Target chord tokens (batch, tgt_len)

        Returns:
            Logits over target vocabulary (batch, tgt_len, vocab_tgt_size)
        """
        src_key_mask, tgt_key_mask, tgt_mask = self.create_mask(src, tgt)

        src_emb = self.pos_enc(self.src_embed(src))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_mask,
            tgt_key_padding_mask=tgt_key_mask,
            memory_key_padding_mask=src_key_mask,
            tgt_mask=tgt_mask
        )

        logits: torch.Tensor = self.fc_out(out)
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
        """Autoregressively generate chord sequence given melody.

        Args:
            src: (batch, src_len) melody token IDs
            max_len: maximum output length
            sos_id: start-of-sequence token ID
            eos_id: end-of-sequence token ID
            temperature: softmax temperature (ignored if sample=False)
            sample: if True, sample from distribution; else greedy argmax

        Returns:
            (batch, generated_len) chord token IDs (including SOS)
        """
        device = src.device
        batch_size = src.size(0)

        # Pre-encode the source sequence once
        src_key_padding_mask = (src == self.pad_id)
        src_emb = self.pos_enc(self.src_embed(src))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        # Start with SOS token
        generated = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_emb = self.pos_enc(self.tgt_embed(generated))

            seq_len = generated.size(1)
            tgt_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )

            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            logits = self.fc_out(out[:, -1, :])  # (batch, vocab)

            if sample:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            # For finished sequences, keep generating pad
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, self.pad_id),
                next_token,
            )

            generated = torch.cat([generated, next_token], dim=1)

            # Mark sequences that just produced EOS as finished
            finished = finished | (next_token.squeeze(1) == eos_id)

            if finished.all():
                break

        return generated

