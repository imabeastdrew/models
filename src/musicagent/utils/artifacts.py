"""Wandb artifact download utilities."""

from dataclasses import dataclass
from pathlib import Path


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

