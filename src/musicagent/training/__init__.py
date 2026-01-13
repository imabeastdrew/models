"""Training components for MusicAgent.

This package exposes LightningModules and helper constructors for offline and
online training. There is no CLI entrypoint; training is expected to be driven
via configs and external scripts (e.g., W&B sweeps).
"""

from musicagent.training.offline import (
    OfflineLightningModule,
    build_offline_module_and_loaders,
)
from musicagent.training.online import (
    OnlineLightningModule,
    build_online_module_and_loaders,
)

__all__ = [
    "OfflineLightningModule",
    "OnlineLightningModule",
    "build_offline_module_and_loaders",
    "build_online_module_and_loaders",
]

