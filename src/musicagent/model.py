import math

import torch
import torch.nn as nn

from musicagent.config import DataConfig, ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class OfflineTransformer(nn.Module):
    def __init__(
        self, 
        model_config: ModelConfig, 
        data_config: DataConfig, 
        vocab_src_size: int, 
        vocab_tgt_size: int
    ):
        super().__init__()
        self.cfg = model_config
        
        self.src_embed = nn.Embedding(vocab_src_size, model_config.d_model, padding_idx=data_config.pad_id)
        self.tgt_embed = nn.Embedding(vocab_tgt_size, model_config.d_model, padding_idx=data_config.pad_id)
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
        
        return self.fc_out(out)
