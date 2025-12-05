"""Shared training utilities for MusicAgent models."""

from __future__ import annotations

import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps: int):
    """Linear warmup followed by a constant learning rate.

    This mirrors the schedule described in the ReaLchords paper: a fixed base
    learning rate (1e-3 in our configs) with a short warmup phase and no
    decay over the remaining training steps.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
