"""Training modules for MusicAgent."""

from musicagent.training.offline import train_offline
from musicagent.training.online import train_online

__all__ = [
    "train_offline",
    "train_online",
]

