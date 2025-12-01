import random

import numpy as np
import torch

from musicagent.utils import seed_everything


def test_seed_everything_reproducible() -> None:
    """seed_everything should make random operations reproducible."""
    seed_everything(42)
    rand1_py = random.random()
    rand1_np = np.random.rand()
    rand1_torch = torch.rand(1).item()

    seed_everything(42)
    rand2_py = random.random()
    rand2_np = np.random.rand()
    rand2_torch = torch.rand(1).item()

    assert rand1_py == rand2_py
    assert rand1_np == rand2_np
    assert rand1_torch == rand2_torch


def test_seed_everything_different_seeds() -> None:
    """Different seeds should produce different random values."""
    seed_everything(42)
    rand1 = random.random()

    seed_everything(123)
    rand2 = random.random()

    assert rand1 != rand2
