"""Configuration loading utilities."""

import json
from pathlib import Path
from typing import TypeVar

from musicagent.config import DataConfig

# Generic type for model configs (OfflineConfig or OnlineConfig)
ModelConfigT = TypeVar("ModelConfigT")


def load_configs_from_dir(
    cfg_dir: Path,
    model_config_class: type[ModelConfigT],
) -> tuple[DataConfig, ModelConfigT]:
    """Load DataConfig and model config from a directory containing JSON files.

    Useful for loading configs saved alongside checkpoints.

    Args:
        cfg_dir: Directory containing data_config.json and model_config.json
        model_config_class: Class for OnlineConfig or OfflineConfig

    Returns:
        DataConfig and model config
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
