"""Configuration dataclasses for data and models."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class DataConfig:
    """Configuration for data preprocessing and loading."""

    # Paths
    data_raw: Path = Path("sheetsage-data/hooktheory/Hooktheory.json")
    data_processed: Path = Path("realchords_data")

    # Constants
    frame_rate: int = 4       # 16th notes
    center_midi: int = 60
    max_len: int = 256        # Model Input Length
    storage_len: int = 1024   # Disk Storage Length (to allow random cropping)
    max_transpose: int = 6    # Maximum semitone shift for data augmentation and vocab building

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
    def vocab_melody(self) -> Path:
        return self.data_processed / "vocab_melody.json"

    @property
    def vocab_chord(self) -> Path:
        return self.data_processed / "vocab_chord.json"


@dataclass
class OfflineConfig:
    """Config for offline transformer.

    The offline model sees the complete melody before generating chords.
    Encoder-decoder architecture.
    """

    # Architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1

    # Training
    batch_size: int = 256
    lr: float = 1e-3
    warmup_steps: int = 1000

    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    ))


@dataclass
class OnlineConfig:
    """Configuration for online decoder-only transformer.

    The online model generates chords without seeing the current or future
    melody. It uses a decoder-only architecture with interleaved input
    [SOS, y₁, x₁, y₂, x₂, ...] and causal attention. During training we
    only optimize chord positions (y_t); melody tokens are used purely as
    conditioning context and are not included in the loss.
    """

    # Architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1

    # Training
    batch_size: int = 256
    lr: float = 1e-3
    warmup_steps: int = 1000

    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    ))


# Backward compatibility alias
ModelConfig = OfflineConfig
