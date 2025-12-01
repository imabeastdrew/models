"""Utilities for musicagent."""

from .analysis import AdaptationDynamicsResult, compute_adaptation_dynamics
from .artifacts import ArtifactPaths, download_wandb_artifact
from .config import load_configs_from_dir
from .core import safe_load_state_dict, seed_everything, setup_logging
from .loaders import (
    LoadedModel,
    TestLoaderResult,
    create_test_loader,
    load_model_from_artifact,
)
from .registry import ModelConfigT, ModelRegistry, ModelType, get_model_registry
from .visualization import (
    format_tokens_for_display,
    get_examples_by_nic_quality,
    show_example,
    show_examples_by_quality,
)

__all__ = [
    # Core
    "setup_logging",
    "seed_everything",
    "safe_load_state_dict",
    # Registry
    "ModelRegistry",
    "ModelType",
    "ModelConfigT",
    "get_model_registry",
    # Config
    "load_configs_from_dir",
    # Artifacts
    "ArtifactPaths",
    "download_wandb_artifact",
    # Loaders
    "LoadedModel",
    "load_model_from_artifact",
    "TestLoaderResult",
    "create_test_loader",
    # Analysis
    "AdaptationDynamicsResult",
    "compute_adaptation_dynamics",
    # Visualization
    "format_tokens_for_display",
    "show_example",
    "get_examples_by_nic_quality",
    "show_examples_by_quality",
]
