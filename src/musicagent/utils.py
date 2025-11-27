import logging
import os
import random
import sys
from typing import Optional

import numpy as np
import torch


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def seed_everything(seed: int, *, deterministic: bool = False, extra_info: Optional[dict] = None) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments.

    Args:
        seed: Base random seed.
        deterministic: If True, enables deterministic algorithms in PyTorch/CuDNN
            (can be slower and may raise warnings if some ops are non-deterministic).
        extra_info: Optional dict to log alongside the seed settings.
    """
    logging.getLogger(__name__).info("Seeding experiment", extra={"seed": seed, **(extra_info or {})})

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

