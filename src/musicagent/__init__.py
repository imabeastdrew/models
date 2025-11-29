"""MusicAgent: Melody to Chord prediction.

This package supports both offline and online chord accompaniment:

- Offline: Encoder-decoder transformer that sees the complete melody
- Online: Decoder-only transformer that generates chords in real-time
  without seeing the current or future melody
"""

from musicagent.config import DataConfig, OfflineConfig, OnlineConfig
from musicagent.data import OfflineDataset, OnlineDataset
from musicagent.models import OfflineTransformer, OnlineTransformer

__all__ = [
    # Configs
    "DataConfig",
    "OfflineConfig",
    "OnlineConfig",
    # Datasets
    "OfflineDataset",
    "OnlineDataset",
    # Models
    "OfflineTransformer",
    "OnlineTransformer",
]
