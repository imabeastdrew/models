"""Model and data loading utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from musicagent.config import DataConfig
from musicagent.data import WeightedJointOfflineDataset, WeightedJointOnlineDataset

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

    # Get vocab size from a temporary dataset
    dataset_class = reg.dataset_class
    if model_type == "online" and d_cfg.data_processed_list:
        dataset_class = WeightedJointOnlineDataset  # type: ignore[assignment]
    if model_type == "offline" and d_cfg.data_processed_list:
        dataset_class = WeightedJointOfflineDataset  # type: ignore[assignment]

    temp_ds: Any = dataset_class(d_cfg, split="test")  # type: ignore[call-arg]
    vocab_size = getattr(temp_ds, "vocab_size", None)

    if vocab_size is None:
        raise AttributeError("Dataset does not expose vocab_size required for model construction.")

    model = reg.model_class(
        m_cfg,
        d_cfg,
        vocab_size=vocab_size,
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
    id_to_token: dict[int, str]
    """Mapping from token ID to string."""
    vocab_size: int
    """Size of vocabulary."""


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

    dataset_class = reg.dataset_class
    if model_type == "online" and d_cfg.data_processed_list:
        dataset_class = WeightedJointOnlineDataset  # type: ignore[assignment]
    if model_type == "offline" and d_cfg.data_processed_list:
        dataset_class = WeightedJointOfflineDataset  # type: ignore[assignment]

    test_ds: Any = dataset_class(d_cfg, split="test")  # type: ignore[call-arg]
    collate_fn = reg.collate_fn_maker(pad_id=d_cfg.pad_id)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    id_to_token: dict[int, str] = getattr(test_ds, "id_to_token", {})

    return TestLoaderResult(
        test_loader=test_loader,
        test_dataset=test_ds,
        id_to_token=id_to_token,
        vocab_size=getattr(test_ds, "vocab_size", 0),
    )


@dataclass
class EvaluationRun:
    """Bundle containing everything needed from a single evaluation run.

    This is a thin convenience wrapper used by high-level helpers such as
    :func:`evaluate_checkpoint` and :func:`evaluate_artifact`. It keeps the
    low-level pieces (loaded model, configs, dataloader) together with the
    computed evaluation result object.
    """

    model_type: ModelType
    """Either ``\"online\"`` or ``\"offline\"``."""

    loaded: LoadedModel
    """Loaded model, configs, and device information."""

    loader: TestLoaderResult
    """Test dataloader, dataset, and vocab mappings."""

    result: Any
    """The evaluation result (``OfflineEvalResult`` or ``OnlineEvalResult``)."""


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    model_type: ModelType,
    *,
    artifact_dir: Path | None = None,
    batch_size: int = 32,
    temperature: float = 1.0,
    sample: bool = False,
    device: str | torch.device | None = None,
    log_progress: bool = True,
) -> EvaluationRun:
    """Run a full evaluation loop for a local checkpoint.

    This helper wires together ``load_model_from_artifact``,
    ``create_test_loader``, and the registered evaluation function for the
    given ``model_type``. It is the easiest way to evaluate a trained model
    from Python or a notebook.

    Args:
        checkpoint_path: Path to the model checkpoint (e.g. ``best_model.pt``).
        model_type: Either ``\"online\"`` or ``\"offline\"``.
        artifact_dir: Directory containing the saved configs. If ``None``, this
            defaults to ``checkpoint_path.parent``.
        batch_size: Batch size for evaluation dataloader.
        temperature: Sampling temperature (used when ``sample=True``).
        sample: If ``True``, sample from the distribution instead of greedy
            decoding.
        device: Device string or ``torch.device``. If ``None``, defaults to
            CUDA if available.
        log_progress: Whether to log progress from the underlying evaluation
            function.

    Returns:
        EvaluationRun with the evaluation result and all supporting objects.
    """
    checkpoint_path = Path(checkpoint_path)
    if artifact_dir is None:
        artifact_dir = checkpoint_path.parent

    reg = get_model_registry(model_type)
    loaded = load_model_from_artifact(
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
    )

    loader = create_test_loader(loaded.d_cfg, model_type=model_type, batch_size=batch_size)

    # Dispatch to the appropriate evaluation function with the right signature.
    if model_type == "online":
        result = reg.eval_func(
            model=loaded.model,
            test_loader=loader.test_loader,
            d_cfg=loaded.d_cfg,
            device=loaded.device,
            id_to_token=loader.id_to_token,
            temperature=temperature,
            sample=sample,
            log_progress=log_progress,
        )
    else:  # "offline"
        result = reg.eval_func(
            model=loaded.model,
            test_loader=loader.test_loader,
            d_cfg=loaded.d_cfg,
            device=loaded.device,
            id_to_token=loader.id_to_token,
            temperature=temperature,
            sample=sample,
            log_progress=log_progress,
        )

    return EvaluationRun(
        model_type=model_type,
        loaded=loaded,
        loader=loader,
        result=result,
    )


def evaluate_artifact(
    artifact_ref: str,
    model_type: ModelType,
    *,
    download_dir: Path | str = "checkpoints",
    batch_size: int = 32,
    temperature: float = 1.0,
    sample: bool = False,
    device: str | torch.device | None = None,
    log_progress: bool = True,
) -> EvaluationRun:
    """Download a wandb model artifact and run evaluation in one call.

    This is the highest-level convenience wrapper intended for notebooks:

    .. code-block:: python

        from musicagent.utils import evaluate_artifact

        run = evaluate_artifact(
            \"marty1ai/musicagent/best-model:v50\",
            model_type=\"offline\",
        )
        print(run.result.nic_ratio)

    Args:
        artifact_ref: Wandb artifact reference (``\"org/project/artifact:version\"``).
        model_type: Either ``\"online\"`` or ``\"offline\"``.
        download_dir: Directory where the artifact will be downloaded.
        batch_size: Batch size for evaluation dataloader.
        temperature: Sampling temperature (used when ``sample=True``).
        sample: If ``True``, sample from the distribution instead of greedy
            decoding.
        device: Device string or ``torch.device``. If ``None``, defaults to
            CUDA if available.
        log_progress: Whether to log progress from the underlying evaluation
            function.

    Returns:
        EvaluationRun with the evaluation result and all supporting objects.
    """
    from .artifacts import download_wandb_artifact

    paths = download_wandb_artifact(artifact_ref, download_dir=download_dir)

    return evaluate_checkpoint(
        checkpoint_path=paths.checkpoint_path,
        model_type=model_type,
        artifact_dir=paths.artifact_dir,
        batch_size=batch_size,
        temperature=temperature,
        sample=sample,
        device=device,
        log_progress=log_progress,
    )
