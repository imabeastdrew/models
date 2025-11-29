"""Tests for training utilities."""

import torch.nn as nn
import torch.optim as optim

from musicagent.training.offline import count_parameters, get_linear_schedule_with_warmup


def test_count_parameters() -> None:
    """count_parameters should return total trainable parameters."""
    # Simple linear layer: 10 * 5 weights + 5 bias = 55 params
    model = nn.Linear(10, 5)
    assert count_parameters(model) == 55


def test_count_parameters_frozen() -> None:
    """Frozen parameters should not be counted."""
    model = nn.Sequential(
        nn.Linear(10, 5),  # 55 params
        nn.Linear(5, 2),   # 12 params
    )
    # Freeze first layer
    for param in model[0].parameters():
        param.requires_grad = False

    # Only second layer should be counted
    assert count_parameters(model) == 12


def test_linear_schedule_warmup_phase() -> None:
    """LR should increase linearly during warmup."""
    model = nn.Linear(10, 5)
    optimizer = optim.SGD(model.parameters(), lr=1.0)

    warmup_steps = 100
    total_steps = 1000
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Initial LR from scheduler should be 0.0 (no steps taken yet).
    assert abs(scheduler.get_last_lr()[0] - 0.0) < 1e-6

    # Simulate stepping through warmup with the same order used in training:
    # optimizer.step() followed by scheduler.step().
    for step in range(1, warmup_steps + 1):
        optimizer.step()
        scheduler.step()
        expected_lr = step / warmup_steps
        assert abs(scheduler.get_last_lr()[0] - expected_lr) < 1e-6


def test_linear_schedule_decay_phase() -> None:
    """LR should decrease linearly after warmup."""
    model = nn.Linear(10, 5)
    optimizer = optim.SGD(model.parameters(), lr=1.0)

    warmup_steps = 10
    total_steps = 100
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    lrs = []
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    warmup_lrs = lrs[:warmup_steps]
    decay_lrs = lrs[warmup_steps:]

    # Warmup region should be non-decreasing starting from 1 / warmup_steps
    assert abs(warmup_lrs[0] - (1.0 / warmup_steps)) < 1e-6
    assert all(b >= a for a, b in zip(warmup_lrs, warmup_lrs[1:]))

    # Decay region should be non-increasing
    assert all(b <= a for a, b in zip(decay_lrs, decay_lrs[1:]))


def test_linear_schedule_reaches_zero() -> None:
    """LR should reach 0 at the end of training."""
    model = nn.Linear(10, 5)
    optimizer = optim.SGD(model.parameters(), lr=1.0)

    warmup_steps = 10
    total_steps = 50
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Step through entire training (and a bit beyond to hit the floor),
    # mirroring the optimizer/scheduler order used in training.
    for _ in range(total_steps + 5):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr()[0] == 0.0
