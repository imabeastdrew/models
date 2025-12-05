"""Configuration dataclasses for data and models."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class DataConfig:
    """For preprocessing and loading."""

    # Paths
    data_raw: Path = Path("sheetsage-data/hooktheory/Hooktheory.json")
    data_processed: Path = Path("realchords_data")

    # Constants
    frame_rate: int = 4  # 16th notes
    center_midi: int = 60
    max_len: int = 256
    storage_len: int = 1024  # For random cropping
    max_transpose: int = 6  # For augmentation

    # Tokens
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"
    rest_token: str = "rest"

    # Token IDs
    pad_id: int = 0
    sos_id: int = 1
    eos_id: int = 2
    rest_id: int = 3

    @property
    def vocab_unified(self) -> Path:
        """Path to unified (melody + chord) vocabulary."""
        return self.data_processed / "vocab_unified.json"

    @property
    def vocab_melody(self) -> Path:
        """Path to melody-only vocabulary."""
        return self.data_processed / "vocab_melody.json"

    @property
    def vocab_chord(self) -> Path:
        """Path to chord-only vocabulary."""
        return self.data_processed / "vocab_chord.json"


@dataclass
class OfflineConfig:
    """Config for offline transformer."""

    # Architecture.
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1

    # Training
    batch_size: int = 256
    lr: float = 1e-3
    warmup_steps: int = 1000
    label_smoothing: float = 0.1
    grad_clip: float = 0.5
    weight_decay: float = 0.0

    device: str = field(default_factory=lambda: ("cuda" if torch.cuda.is_available() else "cpu"))


@dataclass
class OnlineConfig:
    """Config for online transformer."""

    # Architecture.
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1

    # Training
    batch_size: int = 256
    lr: float = 1e-3
    warmup_steps: int = 1000
    label_smoothing: float = 0.1
    grad_clip: float = 0.5
    weight_decay: float = 0.0

    device: str = field(default_factory=lambda: ("cuda" if torch.cuda.is_available() else "cpu"))
