"""MusicAgent: Melody to Chord prediction."""

from musicagent.config import DataConfig, ModelConfig
from musicagent.dataset import MusicAgentDataset
from musicagent.model import OfflineTransformer

__all__ = ["DataConfig", "ModelConfig", "MusicAgentDataset", "OfflineTransformer"]
