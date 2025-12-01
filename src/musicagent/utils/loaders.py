"""Model and data loading utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from musicagent.config import DataConfig

from .config import load_configs_from_dir
from .core import safe_load_state_dict
from .registry import ModelType, get_model_registry


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
        vocab_size = temp_ds.unified_vocab_size
        chord_token_ids = sorted(temp_ds.vocab_chord.values())
        model = reg.model_class(
            m_cfg,
            d_cfg,
            vocab_size=vocab_size,
            chord_token_ids=chord_token_ids,
        )
    else:
        vocab_size = temp_ds.unified_vocab_size
        chord_token_ids = sorted(temp_ds.vocab_chord.values())
        model = reg.model_class(
            m_cfg,
            d_cfg,
            vocab_size=vocab_size,
            chord_token_ids=chord_token_ids,
        )

    model = model.to(device)
    state = safe_load_state_dict(checkpoint_path, map_location=device)
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

