import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, TypeVar

import numpy as np
import torch

from musicagent.config import DataConfig

if TYPE_CHECKING:
    from torch.utils.data import Dataset

# Generic type for model configs (OfflineConfig or OnlineConfig)
ModelConfigT = TypeVar("ModelConfigT")
ModelType = Literal["online", "offline"]


@dataclass
class ModelRegistry:
    """Registry entry containing all classes needed for a model type."""

    config_class: type
    dataset_class: type["Dataset[Any]"]
    collate_fn_maker: Callable[..., Callable[..., Any]]
    model_class: type[torch.nn.Module]
    eval_func: Callable[..., Any]
    eval_result_class: type


def get_model_registry(model_type: ModelType) -> ModelRegistry:
    """Get the registry entry for a model type.

    Args:
        model_type: Either "online" or "offline"

    Returns:
        ModelRegistry with all classes needed for that model type
    """
    # Lazy imports to avoid circular dependencies
    if model_type == "online":
        from musicagent.config import OnlineConfig
        from musicagent.data import OnlineDataset, make_online_collate_fn
        from musicagent.eval import OnlineEvalResult, evaluate_online
        from musicagent.models import OnlineTransformer

        return ModelRegistry(
            config_class=OnlineConfig,
            dataset_class=OnlineDataset,
            collate_fn_maker=make_online_collate_fn,
            model_class=OnlineTransformer,
            eval_func=evaluate_online,
            eval_result_class=OnlineEvalResult,
        )
    elif model_type == "offline":
        from musicagent.config import OfflineConfig
        from musicagent.data import OfflineDataset, make_offline_collate_fn
        from musicagent.eval import OfflineEvalResult, evaluate_offline
        from musicagent.models import OfflineTransformer

        return ModelRegistry(
            config_class=OfflineConfig,
            dataset_class=OfflineDataset,
            collate_fn_maker=make_offline_collate_fn,
            model_class=OfflineTransformer,
            eval_func=evaluate_offline,
            eval_result_class=OfflineEvalResult,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@dataclass
class ArtifactPaths:
    """Paths returned from downloading a wandb model artifact."""

    artifact_dir: Path
    """Directory where artifact was downloaded."""
    checkpoint_path: Path
    """Path to the model checkpoint file."""


def download_wandb_artifact(
    artifact_ref: str,
    download_dir: Path | str = "checkpoints",
) -> ArtifactPaths:
    """Download a model artifact from wandb and return paths.

    Args:
        artifact_ref: Wandb artifact reference (e.g., "org/project/artifact:version")
        download_dir: Directory to download artifact to (created if needed)

    Returns:
        ArtifactPaths with artifact_dir and checkpoint_path
    """
    import wandb

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    artifact = api.artifact(artifact_ref, type="model")
    artifact_dir = Path(artifact.download(root=str(download_dir)))
    checkpoint_path = artifact_dir / "best_model.pt"

    return ArtifactPaths(artifact_dir=artifact_dir, checkpoint_path=checkpoint_path)


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


@dataclass
class LoadedModel:
    """Result of loading a model from an artifact."""

    model: torch.nn.Module
    """The loaded model in eval mode."""
    d_cfg: DataConfig
    """Data configuration from the artifact."""
    m_cfg: Any
    """Model configuration (OnlineConfig or OfflineConfig)."""
    device: torch.device
    """Device the model is on."""


def load_model_from_artifact(
    artifact_dir: Path,
    checkpoint_path: Path,
    model_type: ModelType,
    device: str | torch.device | None = None,
) -> LoadedModel:
    """Load a model and configs from a downloaded artifact.

    Args:
        artifact_dir: Directory containing config files
        checkpoint_path: Path to the model checkpoint
        model_type: Either "online" or "offline"
        device: Device to load model on (default: cuda if available)

    Returns:
        LoadedModel with model, configs, and device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    reg = get_model_registry(model_type)
    d_cfg: DataConfig
    m_cfg: Any
    d_cfg, m_cfg = load_configs_from_dir(artifact_dir, reg.config_class)
    m_cfg.device = str(device)

    # Get vocab sizes from a temporary dataset
    temp_ds: Any = reg.dataset_class(d_cfg, split="test")  # type: ignore[call-arg]

    if model_type == "online":
        model = reg.model_class(
            m_cfg, d_cfg, temp_ds.melody_vocab_size, temp_ds.chord_vocab_size
        )
    else:
        model = reg.model_class(
            m_cfg, d_cfg, len(temp_ds.vocab_melody), len(temp_ds.vocab_chord)
        )

    model = model.to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return LoadedModel(model=model, d_cfg=d_cfg, m_cfg=m_cfg, device=device)


@dataclass
class TestLoaderResult:
    """Result of creating a test dataloader."""

    test_loader: Any  # DataLoader
    """The test DataLoader."""
    test_dataset: Any  # Dataset
    """The test dataset."""
    id_to_melody: dict[int, str]
    """Mapping from melody token ID to string."""
    id_to_chord: dict[int, str]
    """Mapping from chord token ID to string."""
    melody_vocab_size: int
    """Size of melody vocabulary."""
    chord_vocab_size: int
    """Size of chord vocabulary."""


def create_test_loader(
    d_cfg: DataConfig,
    model_type: ModelType,
    batch_size: int = 32,
) -> TestLoaderResult:
    """Create a test dataloader with vocab mappings.

    Args:
        d_cfg: Data configuration
        model_type: Either "online" or "offline"
        batch_size: Batch size for the dataloader

    Returns:
        TestLoaderResult with loader, dataset, and vocab mappings
    """
    from torch.utils.data import DataLoader

    reg = get_model_registry(model_type)

    test_ds: Any = reg.dataset_class(d_cfg, split="test")  # type: ignore[call-arg]
    collate_fn = reg.collate_fn_maker(pad_id=d_cfg.pad_id)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    id_to_melody: dict[int, str] = {v: k for k, v in test_ds.vocab_melody.items()}
    id_to_chord: dict[int, str] = {v: k for k, v in test_ds.vocab_chord.items()}

    return TestLoaderResult(
        test_loader=test_loader,
        test_dataset=test_ds,
        id_to_melody=id_to_melody,
        id_to_chord=id_to_chord,
        melody_vocab_size=getattr(test_ds, "melody_vocab_size", len(id_to_melody)),
        chord_vocab_size=getattr(test_ds, "chord_vocab_size", len(id_to_chord)),
    )


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

