"""Lightning-based components for online (decoder-only) training.

This module defines:
- ``OnlineLightningModule``: wraps ``OnlineTransformer`` in a LightningModule.
- ``build_online_module_and_loaders``: constructs datasets, dataloaders, and module
  from ``DataConfig`` and ``OnlineConfig``.

There is intentionally **no CLI** here; training should be driven by configs and an
external script that instantiates a ``lightning.Trainer``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Tuple

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import Adafactor

from musicagent.config import DataConfig, OnlineConfig
from musicagent.data import (
    WeightedJointOnlineDataset,
    make_online_collate_fn,
)
from musicagent.data.online import OnlineDataset
from musicagent.models import OnlineTransformer
from musicagent.utils.train import (
    count_parameters,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


def _normalize_weights(weights: list[float], n: int) -> list[float]:
    if not weights:
        return [1.0 / n for _ in range(n)]
    if len(weights) != n:
        if len(weights) == 1:
            weights = [weights[0]] * n
        else:
            weights = weights[:n] + [1.0] * max(0, n - len(weights))
    total = sum(weights)
    if total <= 0:
        return [1.0 / n for _ in range(n)]
    return [w / total for w in weights]


def _resolve_sources(cfg: DataConfig) -> list[tuple[str, Path, float]]:
    paths = cfg.data_processed_list or [cfg.data_processed]
    weights = _normalize_weights(cfg.data_weights, len(paths))
    sources: list[tuple[str, Path, float]] = []
    for path, w in zip(paths, weights):
        name = Path(path).name or str(path)
        sources.append((name, Path(path), w))
    return sources


class OnlineLightningModule(L.LightningModule):
    """LightningModule wrapping the online decoder-only transformer."""

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: OnlineConfig,
        vocab_size: int,
        max_steps: int,
        lr_schedule: str = "constant",
        save_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.vocab_size = vocab_size
        self.max_steps = max_steps
        self.lr_schedule = lr_schedule
        self.save_dir = save_dir or Path("checkpoints/online")

        self.model = OnlineTransformer(
            model_cfg,
            data_cfg,
            vocab_size=vocab_size,
        )
        # Gradient checkpointing for memory efficiency
        self.model.enable_gradient_checkpointing()

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=data_cfg.pad_id,
            label_smoothing=model_cfg.label_smoothing,
        )
        self.best_val_loss: float = float("inf")

        self.save_hyperparameters(
            {
                "data_cfg": asdict(data_cfg),
                "model_cfg": asdict(model_cfg),
                "max_steps": max_steps,
                "lr_schedule": lr_schedule,
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        is_melody: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, is_melody)

    def _shift_and_mask(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"]
        is_melody = batch["is_melody"]

        inputs = input_ids[:, :-1]
        is_melody_inputs = is_melody[:, :-1]
        targets = input_ids[:, 1:].clone()

        # Mask out melody positions so we only optimize chords.
        pad_id = self.model.pad_id
        targets[:, 1::2] = pad_id
        return (inputs, is_melody_inputs), targets

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        (inputs, is_melody_inputs), targets = self._shift_and_mask(batch)
        output = self(inputs, is_melody_inputs)
        loss = self.criterion(output.reshape(-1, output.size(-1)), targets.reshape(-1))

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        (inputs, is_melody_inputs), targets = self._shift_and_mask(batch)
        output = self(inputs, is_melody_inputs)

        flat_output = output.reshape(-1, output.size(-1))
        flat_targets = targets.reshape(-1)
        loss = self.criterion(flat_output, flat_targets)

        self.log(
            "valid/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        with torch.no_grad():
            val = float(loss.detach().cpu())
            ppl = math.exp(min(val, 100.0))
        self.log(
            "valid/perplexity",
            ppl,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        val_loss = metrics.get("valid/loss")
        if val_loss is None:
            return
        loss_value = float(val_loss.detach().cpu())
        if loss_value < self.best_val_loss:
            self.best_val_loss = loss_value
            self._save_best_checkpoint(loss_value)

    def _save_best_checkpoint(self, val_loss: float) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.save_dir / "best_model.pt"
        torch.save(self.model.state_dict(), checkpoint_path)

        data_cfg_path = self.save_dir / "data_config.json"
        model_cfg_path = self.save_dir / "model_config.json"
        with data_cfg_path.open("w") as f:
            json.dump(asdict(self.data_cfg), f, default=str, indent=2)
        with model_cfg_path.open("w") as f:
            json.dump(asdict(self.model_cfg), f, default=str, indent=2)

        logger.info(
            "Saved best online model to %s (val_loss=%.4f)",
            checkpoint_path,
            val_loss,
        )

    def configure_optimizers(self):
        optimizer = Adafactor(
            self.parameters(),
            lr=self.model_cfg.lr,
            weight_decay=self.model_cfg.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

        if self.lr_schedule == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.model_cfg.warmup_steps,
                num_training_steps=self.max_steps,
            )
        elif self.lr_schedule == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.model_cfg.warmup_steps,
                num_training_steps=self.max_steps,
            )
        else:
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.model_cfg.warmup_steps,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def build_online_module_and_loaders(
    data_cfg: DataConfig,
    model_cfg: OnlineConfig,
    *,
    max_steps: int,
    lr_schedule: str = "constant",
    save_dir: Path | None = None,
    seed: int = 42,
) -> Tuple[OnlineLightningModule, DataLoader, DataLoader]:
    """Create OnlineLightningModule and train/valid DataLoaders from configs.

    This is the preferred entry point for config-driven online MLE training. Example:

    .. code-block:: python

        data_cfg = DataConfig(data_processed=Path("realchords_data"))
        model_cfg = OnlineConfig()
        module, train_loader, valid_loader = build_online_module_and_loaders(
            data_cfg, model_cfg, max_steps=100_000
        )
        trainer = L.Trainer(max_steps=100_000, gradient_clip_val=model_cfg.grad_clip)
        trainer.fit(module, train_loader, valid_loader)
    """
    del seed  # reserved for future use

    if model_cfg.d_model % model_cfg.n_heads != 0:
        raise ValueError(
            f"d_model ({model_cfg.d_model}) must be divisible by n_heads ({model_cfg.n_heads})"
        )

    sources = _resolve_sources(data_cfg)
    use_joint = len(sources) > 1

    try:
        if use_joint:
            logger.info(
                "Using weighted joint online datasets: %s",
                [(n, str(p), w) for n, p, w in sources],
            )
            train_ds = WeightedJointOnlineDataset(data_cfg, sources, split="train")
            valid_ds = WeightedJointOnlineDataset(
                data_cfg, sources, split="valid", eval_mode=True
            )
        else:
            train_ds = OnlineDataset(data_cfg, split="train")
            valid_ds = OnlineDataset(data_cfg, split="valid")
    except FileNotFoundError as e:
        logger.error(e)
        raise

    collate_fn = make_online_collate_fn(pad_id=data_cfg.pad_id)
    if use_joint:
        num_samples = int(math.ceil(len(train_ds) * data_cfg.train_samples_multiplier))
        if data_cfg.max_train_samples is not None:
            num_samples = min(num_samples, data_cfg.max_train_samples)
        num_samples = max(1, num_samples)
        sampler = WeightedRandomSampler(train_ds.sample_weights, num_samples, replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=model_cfg.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=model_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=model_cfg.batch_size,
        collate_fn=collate_fn,
    )

    vocab_size = train_ds.vocab_size
    logger.info("Online vocab size: %d", vocab_size)

    module = OnlineLightningModule(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        vocab_size=vocab_size,
        max_steps=max_steps,
        lr_schedule=lr_schedule,
        save_dir=save_dir or Path("checkpoints/online"),
    )

    total_params = count_parameters(module.model)
    logger.info("Online model parameters: %d", total_params)

    return module, train_loader, valid_loader

