"""Tests for training utilities."""

import torch.nn as nn
import torch.optim as optim

from musicagent.utils.train import count_parameters, get_constant_schedule_with_warmup


def test_count_parameters() -> None:
    """count_parameters should return total trainable parameters."""
    # Simple linear layer: 10 * 5 weights + 5 bias = 55 params
    model = nn.Linear(10, 5)
    assert count_parameters(model) == 55


def test_count_parameters_frozen() -> None:
    """Frozen parameters should not be counted."""
    model = nn.Sequential(
        nn.Linear(10, 5),  # 55 params
        nn.Linear(5, 2),  # 12 params
    )
    # Freeze first layer
    for param in model[0].parameters():
        param.requires_grad = False

    # Only second layer should be counted
    assert count_parameters(model) == 12


def test_constant_schedule_warmup_phase() -> None:
    """LR should increase linearly during warmup and stay constant afterwards."""
    model = nn.Linear(10, 5)
    optimizer = optim.SGD(model.parameters(), lr=1.0)

    warmup_steps = 100
    scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)

    # Initial LR from scheduler should be 0.0 (no steps taken yet).
    assert abs(scheduler.get_last_lr()[0] - 0.0) < 1e-6

    # Simulate stepping through warmup with the same order used in training:
    # optimizer.step() followed by scheduler.step().
    for step in range(1, warmup_steps + 1):
        optimizer.step()
        scheduler.step()
        expected_lr = step / warmup_steps
        assert abs(scheduler.get_last_lr()[0] - expected_lr) < 1e-6

    # After warmup, LR should remain constant at 1.0.
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        assert abs(scheduler.get_last_lr()[0] - 1.0) < 1e-6
