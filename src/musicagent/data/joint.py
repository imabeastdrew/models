"""Weighted joint datasets for multi-source sampling."""

from __future__ import annotations

import dataclasses
import math
import random
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

from torch.utils.data import Dataset

from musicagent.config import DataConfig
from musicagent.data.offline import OfflineDataset
from musicagent.data.online import OnlineDataset


def _normalize_weights(weights: Sequence[float], n: int) -> list[float]:
    if not weights:
        return [1.0 / n for _ in range(n)]
    if len(weights) != n:
        # Broadcast a single value or truncate/extend to match.
        if len(weights) == 1:
            weights = [float(weights[0])] * n
        else:
            weights = list(weights[:n]) + [1.0] * max(0, n - len(weights))
    total = sum(weights)
    if total <= 0:
        return [1.0 / n for _ in range(n)]
    return [w / total for w in weights]


class _WeightedJointDataset(Dataset):
    """Base wrapper that multiplexes multiple sub-datasets.

    Each sub-dataset is sampled proportionally to a target weight. Per-item
    weights are computed as ``weight / len(subdataset)`` so smaller datasets
    receive a higher per-item probability.
    """

    def __init__(
        self,
        cfg: DataConfig,
        sources: Iterable[tuple[str, Path, float]],
        dataset_cls: type[Dataset],
        *,
        eval_mode: bool = False,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg
        self.sources = list(sources)
        if not self.sources:
            raise ValueError("WeightedJointDataset requires at least one source.")

        weights_raw = [w for _, _, w in self.sources]
        self.normalized_weights = _normalize_weights(weights_raw, len(self.sources))

        self.subdatasets: list[Dataset] = []
        self.boundaries: list[int] = []
        total = 0
        for (_, path, _), weight in zip(self.sources, self.normalized_weights):
            # Clone the config with a dataset-specific processed path.
            ds_cfg = dataclasses.replace(cfg, data_processed=path)
            ds = dataset_cls(ds_cfg, split=getattr(self, "split", "train"))  # type: ignore[arg-type]
            self.subdatasets.append(ds)
            total += len(ds)
            self.boundaries.append(total)
        self.total_len = total

        # Precompute per-item sampling weights for WeightedRandomSampler.
        self.sample_weights: list[float] = []
        for ds_weight, ds in zip(self.normalized_weights, self.subdatasets):
            if len(ds) == 0:
                continue
            per_item = ds_weight / len(ds)
            self.sample_weights.extend([per_item] * len(ds))

        # Expose vocab metadata from the first dataset (assumed shared).
        self.vocab_size = getattr(self.subdatasets[0], "vocab_size", 0)
        self.id_to_token = getattr(self.subdatasets[0], "id_to_token", {})

        self.eval_indices: list[tuple[int, int]] | None = None
        if eval_mode:
            self.eval_indices = self._build_eval_indices(seed)

    # ------------------------------------------------------------------ #
    # Deterministic weighted round-robin for evaluation
    # ------------------------------------------------------------------ #
    def _build_eval_indices(self, seed: int) -> list[tuple[int, int]]:
        """Create a deterministic weighted round-robin ordering.

        - Shuffle indices within each sub-dataset with the provided seed.
        - Interleave according to normalized weights using a weighted fair
          queue (smaller "next time" is selected first).
        - Each item appears exactly once; ordering is deterministic.
        """
        rng = random.Random(seed)
        queues: list[deque[int]] = []
        for ds in self.subdatasets:
            idxs = list(range(len(ds)))
            rng.shuffle(idxs)
            queues.append(deque(idxs))

        times = [0.0 for _ in queues]
        steps = [1.0 / w if w > 0 else math.inf for w in self.normalized_weights]
        result: list[tuple[int, int]] = []

        active = {i for i, q in enumerate(queues) if q}
        while active:
            # Choose dataset with smallest "time"; ties broken by index for determinism.
            ds_idx = min(active, key=lambda i: (times[i], i))
            local_idx = queues[ds_idx].popleft()
            result.append((ds_idx, local_idx))
            times[ds_idx] += steps[ds_idx]
            if not queues[ds_idx]:
                active.remove(ds_idx)
                times[ds_idx] = math.inf
        return result

    # ------------------------------------------------------------------ #
    # Dataset interface
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:  # type: ignore[override]
        if self.eval_indices is not None:
            return len(self.eval_indices)
        return self.total_len

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.total_len:
            raise IndexError(idx)
        # Boundaries are cumulative lengths.
        for ds_idx, bound in enumerate(self.boundaries):
            if idx < bound:
                local_idx = idx if ds_idx == 0 else idx - self.boundaries[ds_idx - 1]
                return ds_idx, local_idx
        raise IndexError(idx)

    def __getitem__(self, idx: int):  # type: ignore[override]
        if self.eval_indices is not None:
            ds_idx, local_idx = self.eval_indices[idx]
        else:
            ds_idx, local_idx = self._locate(idx)
        return self.subdatasets[ds_idx][local_idx]


class WeightedJointOfflineDataset(_WeightedJointDataset):
    """Weighted wrapper around OfflineDataset instances."""

    def __init__(
        self,
        cfg: DataConfig,
        sources: Iterable[tuple[str, Path, float]],
        *,
        split: str = "train",
        eval_mode: bool = False,
        seed: int = 42,
    ) -> None:
        self.split = split
        super().__init__(
            cfg,
            sources,
            dataset_cls=OfflineDataset,
            eval_mode=eval_mode,
            seed=seed,
        )


class WeightedJointOnlineDataset(_WeightedJointDataset):
    """Weighted wrapper around OnlineDataset instances."""

    def __init__(
        self,
        cfg: DataConfig,
        sources: Iterable[tuple[str, Path, float]],
        *,
        split: str = "train",
        eval_mode: bool = False,
        seed: int = 42,
    ) -> None:
        self.split = split
        super().__init__(
            cfg,
            sources,
            dataset_cls=OnlineDataset,
            eval_mode=eval_mode,
            seed=seed,
        )
