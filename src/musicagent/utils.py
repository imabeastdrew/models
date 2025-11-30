import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeVar

import numpy as np
import torch

from musicagent.config import DataConfig

# Generic type for model configs (OfflineConfig or OnlineConfig)
ModelConfigT = TypeVar("ModelConfigT")


def load_configs_from_dir(
    cfg_dir: Path,
    model_config_class: type[ModelConfigT],
) -> tuple[DataConfig, ModelConfigT]:
    """Load DataConfig and model config from a directory containing JSON files.

    This is useful for loading configs saved alongside model checkpoints,
    ensuring evaluation uses the same settings as training.

    Args:
        cfg_dir: Directory containing data_config.json and model_config.json
        model_config_class: The model config class to instantiate (OnlineConfig or OfflineConfig)

    Returns:
        Tuple of (DataConfig, model_config)
    """
    data_cfg_path = cfg_dir / "data_config.json"
    model_cfg_path = cfg_dir / "model_config.json"

    # Load data config
    with data_cfg_path.open() as f:
        data_dict = json.load(f)
    # Coerce path-like fields back to Path objects
    if "data_raw" in data_dict:
        data_dict["data_raw"] = Path(data_dict["data_raw"])
    if "data_processed" in data_dict:
        data_dict["data_processed"] = Path(data_dict["data_processed"])
    d_cfg = DataConfig(**data_dict)

    # Load model config
    with model_cfg_path.open() as f:
        m_cfg = model_config_class(**json.load(f))

    return d_cfg, m_cfg


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def seed_everything(
    seed: int,
    *,
    deterministic: bool = False,
    extra_info: Optional[dict] = None,
) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments.

    Args:
        seed: Base random seed.
        deterministic: If True, enables deterministic algorithms in PyTorch/CuDNN
            (can be slower and may raise warnings if some ops are non-deterministic).
        extra_info: Optional dict to log alongside the seed settings.
    """
    logging.getLogger(__name__).info(
        "Seeding experiment",
        extra={"seed": seed, **(extra_info or {})},
    )

    # Core RNGs
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Prefer deterministic algorithms when available.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class AdaptationDynamicsResult:
    """Result of per-beat NiC analysis."""

    valid_beats: list[int]
    """Beat indices that had enough samples."""
    beat_means: list[float]
    """Mean NiC ratio per valid beat."""
    beat_stds: list[float]
    """Standard deviation of NiC ratio per valid beat."""
    samples_per_beat: dict[int, int]
    """Number of samples for each beat (for all beats, not just valid ones)."""


def compute_adaptation_dynamics(
    cached_predictions: dict[int, tuple[list[str], list[str], list[str]]],
    *,
    max_beats: int = 64,
    frame_rate: int = 4,
    min_samples: int = 10,
) -> AdaptationDynamicsResult:
    """Compute per-beat NiC statistics for adaptation dynamics analysis.

    Per paper Section K: excludes beats that are entirely silent.

    Args:
        cached_predictions: Dict mapping index -> (melody_tokens, pred_tokens, ref_tokens)
        max_beats: Maximum number of beats to analyze
        frame_rate: Frames per beat (default 4 for 16th notes)
        min_samples: Minimum samples required per beat to include in results

    Returns:
        AdaptationDynamicsResult with per-beat statistics
    """
    # Import here to avoid circular dependency (eval imports utils)
    from musicagent.eval import note_in_chord_at_beat

    beat_nic_all: dict[int, list[float]] = {b: [] for b in range(max_beats)}

    for mel_tokens, pred_tokens, _ in cached_predictions.values():
        # note_in_chord_at_beat returns None for silent beats
        beat_nic = note_in_chord_at_beat(mel_tokens, pred_tokens, frame_rate=frame_rate)

        for beat, nic in beat_nic.items():
            # Only include non-silent beats (nic is None for silent beats per paper)
            if beat < max_beats and nic is not None:
                beat_nic_all[beat].append(nic)

    # Compute mean and std per beat
    valid_beats: list[int] = []
    beat_means: list[float] = []
    beat_stds: list[float] = []

    for beat in range(max_beats):
        if len(beat_nic_all[beat]) >= min_samples:
            valid_beats.append(beat)
            beat_means.append(float(np.mean(beat_nic_all[beat])))
            beat_stds.append(float(np.std(beat_nic_all[beat])))

    samples_per_beat = {beat: len(beat_nic_all[beat]) for beat in range(max_beats)}

    return AdaptationDynamicsResult(
        valid_beats=valid_beats,
        beat_means=beat_means,
        beat_stds=beat_stds,
        samples_per_beat=samples_per_beat,
    )

