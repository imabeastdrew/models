"""Test set evaluation for online model."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from musicagent.config import DataConfig, OnlineConfig
from musicagent.data import OnlineDataset, make_online_collate_fn
from musicagent.models import OnlineTransformer
from musicagent.utils import safe_load_state_dict, setup_logging

from .metrics import (
    chord_length_entropy,
    chord_lengths,
    decode_tokens,
    note_in_chord_ratio,
    onset_interval_emd,
    onset_intervals,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class OnlineEvalResult:
    """Result container for online model evaluation."""

    # Loss metrics
    test_loss: float = 0.0
    test_perplexity: float = 0.0

    # Generation metrics
    nic_ratio: float = 0.0
    nic_std: float = 0.0
    onset_interval_emd: float = 0.0
    pred_chord_length_entropy: float = 0.0
    ref_chord_length_entropy: float = 0.0

    # Raw data for further analysis
    per_song_nic: list[float] = field(default_factory=list)
    pred_intervals: list[int] = field(default_factory=list)
    ref_intervals: list[int] = field(default_factory=list)
    pred_lengths: list[int] = field(default_factory=list)
    ref_lengths: list[int] = field(default_factory=list)

    # Cached predictions: {idx: (mel_tokens, pred_tokens, ref_tokens)}
    cached_predictions: dict[int, tuple[list[str], list[str], list[str]]] = field(
        default_factory=dict
    )

    # Metadata
    num_sequences: int = 0


def extract_melody_and_chords(
    interleaved: torch.Tensor,
    melody_vocab_size: int,
    pad_id: int,
    uses_unified_ids_on_disk: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract melody frames and reference chord frames from interleaved sequence.

    OnlineDataset format: [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]
    - Position 0: SOS (chord token in unified space)
    - Odd positions (1, 3, 5, ...): chord tokens (unified space)
    - Even positions (2, 4, 6, ...): melody tokens (unified space)

    Args:
        interleaved: Interleaved sequence tensor.
        melody_vocab_size: Size of melody vocabulary (for legacy unified->chord conversion).
        pad_id: Padding token ID.
        uses_unified_ids_on_disk: Whether the on-disk sequences already use unified IDs.

    Returns:
        melody_frames: Tensor of melody token IDs.
        ref_chord_frames: Tensor of chord token IDs in *chord vocab* space for
            legacy datasets, or unified space for fully unified datasets.
    """
    # Skip SOS at position 0
    seq = interleaved[1:]  # Now: [y₁, x₁, y₂, x₂, ...]

    # Extract chord and melody by alternating positions
    chord_unified = seq[0::2]  # positions 0, 2, 4, ... -> y₁, y₂, y₃, ...
    melody = seq[1::2]         # positions 1, 3, 5, ... -> x₁, x₂, x₃, ...

    # Find valid length (non-PAD in melody)
    pad_mask = melody == pad_id
    first_pad = pad_mask.nonzero(as_tuple=True)[0]
    valid_len = int(first_pad[0].item()) if len(first_pad) > 0 else len(melody)

    melody_frames = melody[:valid_len]
    chord_unified_frames = chord_unified[:valid_len]

    if uses_unified_ids_on_disk:
        # New pipeline: sequences on disk already use unified IDs, so no
        # further conversion is required.
        ref_chord_frames = chord_unified_frames
    else:
        # Legacy pipeline: chords were offset into unified space during
        # OnlineDataset interleaving; convert back to chord-vocab IDs so that
        # id_to_chord mappings built from the original chord vocabularies
        # remain valid.
        ref_chord_frames = torch.where(
            chord_unified_frames == pad_id,
            chord_unified_frames,
            chord_unified_frames - melody_vocab_size,
        )

    return melody_frames, ref_chord_frames


def evaluate_online(
    model: OnlineTransformer,
    test_loader: DataLoader,
    d_cfg: DataConfig,
    device: torch.device,
    melody_vocab_size: int,
    id_to_melody: dict[int, str] | None = None,
    id_to_chord: dict[int, str] | None = None,
    temperature: float = 1.0,
    sample: bool = False,
    log_progress: bool = True,
) -> OnlineEvalResult:
    """Evaluate online model and return metrics with cached predictions.

    Args:
        model: Trained OnlineTransformer model (already on device, in eval mode)
        test_loader: DataLoader for test set (using OnlineDataset)
        d_cfg: Data configuration
        id_to_melody: Mapping from melody token IDs to strings
        id_to_chord: Mapping from chord token IDs to strings
        melody_vocab_size: Size of melody vocabulary (kept for API compatibility)
        device: Device to run evaluation on
        temperature: Sampling temperature (only used with sample=True)
        sample: If True, sample from distribution; otherwise greedy decode
        log_progress: If True, log progress to logger

    Returns:
        OnlineEvalResult containing all metrics and cached predictions
    """
    import numpy as np

    result = OnlineEvalResult()

    dataset = getattr(test_loader, "dataset", None)

    # Determine whether sequences on disk are already in unified ID space.
    # For safety and backward compatibility, we default to *legacy* behavior
    # (False) when the dataset is missing or does not expose the flag, rather
    # than silently assuming unified IDs and risking incorrect decoding.
    if dataset is not None and hasattr(dataset, "uses_unified_ids_on_disk"):
        uses_unified_ids_on_disk = bool(dataset.uses_unified_ids_on_disk)  # type: ignore[attr-defined]
    else:
        uses_unified_ids_on_disk = False

    # If explicit mappings are not provided, fall back to dataset-provided ones.
    # For unified-ID pipelines we use the unified mapping; for legacy pipelines
    # we decode via the original per-track vocabularies.
    if id_to_melody is None or id_to_chord is None:
        if dataset is None:
            raise ValueError(
                "id_to_melody/id_to_chord not provided and test_loader has no dataset."
            )

        if uses_unified_ids_on_disk and hasattr(dataset, "unified_id_to_token"):
            # New pipeline: unified IDs on disk, decode via unified vocab.
            unified_map: dict[int, str] = dataset.unified_id_to_token  # type: ignore[assignment]
            id_to_melody = unified_map
            id_to_chord = unified_map
        elif hasattr(dataset, "id_to_melody") and hasattr(dataset, "id_to_chord"):
            # Legacy pipeline: decode via separate melody / chord vocabularies.
            id_to_melody = dataset.id_to_melody  # type: ignore[assignment]
            id_to_chord = dataset.id_to_chord  # type: ignore[assignment]
        else:
            raise ValueError(
                "Dataset does not expose unified_id_to_token or id_to_melody/id_to_chord "
                "for decoding."
            )

    # --- 1. Test loss / perplexity (teacher-forced, chord positions only) ---
    criterion = nn.CrossEntropyLoss(ignore_index=d_cfg.pad_id, reduction="sum")
    total_loss = 0.0
    total_chord_tokens = 0

    if log_progress:
        logger.info("Computing teacher-forced test loss (chord positions only)...")

    with torch.no_grad():
        for interleaved_batch in test_loader:
            interleaved_batch = interleaved_batch.to(device)

            input_ids = interleaved_batch[:, :-1]
            target_ids = interleaved_batch[:, 1:]

            logits = model(input_ids)

            batch_size, seq_len = target_ids.shape
            chord_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            chord_mask[0::2] = True

            chord_logits = logits[:, chord_mask, :]
            chord_targets = target_ids[:, chord_mask]

            loss = criterion(
                chord_logits.reshape(-1, chord_logits.size(-1)),
                chord_targets.reshape(-1),
            )

            non_pad_mask = chord_targets != d_cfg.pad_id
            num_chord_tokens = non_pad_mask.sum().item()

            total_loss += loss.item()
            total_chord_tokens += num_chord_tokens

    result.test_loss = total_loss / total_chord_tokens if total_chord_tokens > 0 else 0.0
    result.test_perplexity = math.exp(min(result.test_loss, 100))

    if log_progress:
        logger.info(
            f"Test Loss: {result.test_loss:.4f} | Test Perplexity: {result.test_perplexity:.2f}"
        )

    # --- 2. Generate chord sequences & compute metrics ---
    if log_progress:
        logger.info("Generating chord sequences for metric evaluation...")

    num_batches = len(test_loader)
    global_idx = 0

    with torch.no_grad():
        for batch_idx, interleaved_batch in enumerate(test_loader):
            interleaved_batch = interleaved_batch.to(device)
            batch_size = interleaved_batch.size(0)

            for i in range(batch_size):
                interleaved = interleaved_batch[i]

                melody_frames, ref_chord_frames = extract_melody_and_chords(
                    interleaved,
                    melody_vocab_size,
                    d_cfg.pad_id,
                    uses_unified_ids_on_disk,
                )

                if len(melody_frames) == 0:
                    global_idx += 1
                    continue

                pred_chords = model.generate(
                    melody_frames.unsqueeze(0),
                    sos_id=d_cfg.sos_id,
                    eos_id=d_cfg.eos_id,
                    temperature=temperature,
                    sample=sample,
                )[0]

                mel_tokens = decode_tokens(melody_frames.cpu().tolist(), id_to_melody)
                pred_tokens = decode_tokens(pred_chords.cpu().tolist(), id_to_chord)
                ref_tokens = decode_tokens(ref_chord_frames.cpu().tolist(), id_to_chord)

                # Cache predictions
                result.cached_predictions[global_idx] = (mel_tokens, pred_tokens, ref_tokens)

                # Note-in-chord ratio
                nic = note_in_chord_ratio(mel_tokens, pred_tokens)
                result.per_song_nic.append(nic)

                # Onset intervals
                result.pred_intervals.extend(onset_intervals(mel_tokens, pred_tokens))
                result.ref_intervals.extend(onset_intervals(mel_tokens, ref_tokens))

                # Chord lengths
                result.pred_lengths.extend(chord_lengths(pred_tokens))
                result.ref_lengths.extend(chord_lengths(ref_tokens))

                global_idx += 1

            if log_progress and ((batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1):
                logger.info(f"  Processed {batch_idx + 1}/{num_batches} batches")

    # Compute aggregate metrics
    result.nic_ratio = float(np.mean(result.per_song_nic)) if result.per_song_nic else 0.0
    result.nic_std = float(np.std(result.per_song_nic)) if result.per_song_nic else 0.0
    result.onset_interval_emd = onset_interval_emd(result.pred_intervals, result.ref_intervals)
    result.pred_chord_length_entropy = chord_length_entropy(result.pred_lengths)
    result.ref_chord_length_entropy = chord_length_entropy(result.ref_lengths)
    result.num_sequences = len(result.per_song_nic)

    return result


def main():
    """Evaluate online model on test set."""
    from musicagent.utils import load_configs_from_dir

    parser = argparse.ArgumentParser(description="Evaluate Online Model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/online/best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample from distribution instead of greedy decoding",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used with --sample)",
    )
    args = parser.parse_args()

    setup_logging()

    # Load configs from checkpoint directory
    cfg_dir = args.checkpoint.parent
    data_cfg_path = cfg_dir / "data_config.json"
    model_cfg_path = cfg_dir / "model_config.json"

    if data_cfg_path.exists() and model_cfg_path.exists():
        d_cfg, m_cfg = load_configs_from_dir(cfg_dir, OnlineConfig)
        logger.info(f"Loaded configs from {cfg_dir}")
    else:
        d_cfg = DataConfig()
        m_cfg = OnlineConfig()

    if args.device:
        m_cfg.device = args.device
    device = torch.device(m_cfg.device)

    # Load test dataset
    test_ds = OnlineDataset(d_cfg, split="test")
    collate = make_online_collate_fn(pad_id=d_cfg.pad_id)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    melody_vocab_size = test_ds.melody_vocab_size
    chord_vocab_size = test_ds.chord_vocab_size

    # Load model in unified ID space
    vocab_size = test_ds.unified_vocab_size
    chord_token_ids = sorted(test_ds.vocab_chord.values())
    model = OnlineTransformer(
        m_cfg,
        d_cfg,
        vocab_size=vocab_size,
        chord_token_ids=chord_token_ids,
    ).to(device)
    state_dict = safe_load_state_dict(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Run evaluation
    result = evaluate_online(
        model=model,
        test_loader=test_loader,
        d_cfg=d_cfg,
        melody_vocab_size=melody_vocab_size,
        device=device,
        temperature=args.temperature,
        sample=args.sample,
    )

    # Log results
    logger.info("=" * 60)
    logger.info("Metrics (Online Eval)")
    logger.info("=" * 60)
    logger.info(f"Test Loss:                 {result.test_loss:.4f}")
    logger.info(f"Test Perplexity:           {result.test_perplexity:.2f}")
    logger.info("-" * 60)
    logger.info(f"NiC Ratio:                 {result.nic_ratio * 100:.2f}%")
    logger.info(f"Onset Interval EMD:        {result.onset_interval_emd * 1e3:.2f} ×10⁻³")
    logger.info(
        f"Chord Length Entropy:      {result.pred_chord_length_entropy:.2f}  "
        f"(ref: {result.ref_chord_length_entropy:.2f})"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
