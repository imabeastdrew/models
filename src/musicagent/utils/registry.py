"""Model registry."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar

import torch

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
