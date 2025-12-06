"""Shared training utilities for MusicAgent models."""

from __future__ import annotations

import math

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


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Linear warmup followed by linear decay to zero.

    Args:
        optimizer:
            Optimizer whose learning rate will be scheduled.
        num_warmup_steps:
            Number of steps to linearly increase LR from 0 → base LR.
        num_training_steps:
            Total number of training steps; after warmup the LR decays
            linearly from base LR → 0 over the remaining steps.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Warmup phase: linearly scale from 0 to 1.
            return float(current_step) / float(max(1, num_warmup_steps))

        if current_step >= num_training_steps:
            # After the planned training horizon, keep LR at 0.
            return 0.0

        # Decay phase: linearly scale from 1 → 0 over the remaining steps.
        decay_steps = max(1, num_training_steps - num_warmup_steps)
        remaining = num_training_steps - current_step
        return max(0.0, float(remaining) / float(decay_steps))

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
):
    """Linear warmup followed by cosine decay to zero.

    This mirrors HuggingFace's get_cosine_schedule_with_warmup:
    - LR ramps linearly from 0 → base LR over ``num_warmup_steps``
    - Then follows a cosine decay curve back to 0 over the remaining steps.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Warmup phase: linearly scale from 0 to 1.
            return float(current_step) / float(max(1, num_warmup_steps))

        if current_step >= num_training_steps:
            # After the planned training horizon, keep LR at 0.
            return 0.0

        # Cosine decay phase.
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)),
        )

    return LambdaLR(optimizer, lr_lambda)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
