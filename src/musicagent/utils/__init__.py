"""Utilities for musicagent."""

from .analysis import AdaptationDynamicsResult, compute_adaptation_dynamics
from .artifacts import ArtifactPaths, download_wandb_artifact
from .config import load_configs_from_dir
from .core import seed_everything, setup_logging
from .loaders import (
    LoadedModel,
    TestLoaderResult,
    create_test_loader,
    load_model_from_artifact,
)
from .registry import ModelConfigT, ModelRegistry, ModelType, get_model_registry

__all__ = [
    # Core
    "setup_logging",
    "seed_everything",
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
]

