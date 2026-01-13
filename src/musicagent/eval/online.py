"""Test set evaluation for online model."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from musicagent.cli import build_eval_online_parser
from musicagent.config import DataConfig, OnlineConfig
from musicagent.data import (
    OnlineDataset,
    WeightedJointOnlineDataset,
    make_online_collate_fn,
)
from musicagent.models import OnlineTransformer
from musicagent.utils.core import safe_load_state_dict, setup_logging

from .metrics import (
    chord_length_entropy,
    chord_lengths,
    chord_silence_counts,
    decode_tokens,
    note_in_chord_ratio,
    onset_interval_emd,
    onset_intervals,
)

if TYPE_CHECKING:
    pass

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
    chord_silence_ratio: float = 0.0
    long_chord_ratio: float = 0.0
    early_stop_ratio: float = 0.0

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
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract melody frames and reference chord frames from interleaved sequence.

    OnlineDataset format: [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]
    - Position 0: SOS (chord token in unified space)
    - Odd positions (1, 3, 5, ...): chord tokens (unified space)
    - Even positions (2, 4, 6, ...): melody tokens (unified space)

    Args:
        interleaved: Interleaved sequence tensor.
        pad_id: Padding token ID.

    Returns:
        melody_frames: Tensor of melody token IDs.
        ref_chord_frames: Tensor of chord token IDs in unified ID space.
    """
    # Skip SOS at position 0
    seq = interleaved[1:]  # Now: [y₁, x₁, y₂, x₂, ...]

    # Extract chord and melody by alternating positions
    chord_unified = seq[0::2]  # positions 0, 2, 4, ... -> y₁, y₂, y₃, ...
    melody = seq[1::2]  # positions 1, 3, 5, ... -> x₁, x₂, x₃, ...

    # Find valid length (non-PAD in melody)
    pad_mask = melody == pad_id
    first_pad = pad_mask.nonzero(as_tuple=True)[0]
    valid_len = int(first_pad[0].item()) if len(first_pad) > 0 else len(melody)

    melody_frames = melody[:valid_len]
    chord_unified_frames = chord_unified[:valid_len]

    # In the unified pipeline, sequences on disk already use the unified ID
    # space produced by preprocessing, so we can treat chord frames directly as
    # chord token IDs in that space.
    return melody_frames, chord_unified_frames


def extract_melody_and_chords_from_mask(
    input_ids: torch.Tensor,
    is_melody: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract melody frames and reference chord frames using is_melody mask.

    OnlineDataset format: [SOS, y₁, x₁, y₂, x₂, ..., y_T, x_T]
    - Position 0: SOS (chord, is_melody=False)
    - Odd positions (1, 3, 5, ...): chord tokens (is_melody=False)
    - Even positions (2, 4, 6, ...): melody tokens (is_melody=True)

    Args:
        input_ids: Interleaved sequence tensor with native vocab IDs.
        is_melody: Boolean mask, True for melody positions.
        pad_id: Padding token ID.

    Returns:
        melody_frames: Tensor of melody token IDs.
        ref_chord_frames: Tensor of chord token IDs.
    """
    # Skip SOS at position 0
    seq = input_ids[1:]  # Now: [y₁, x₁, y₂, x₂, ...]
    mask = is_melody[1:]

    # Extract chord and melody using mask
    # Chord positions: is_melody=False after SOS (positions 0, 2, 4, ...)
    # Melody positions: is_melody=True (positions 1, 3, 5, ...)
    chord_mask = ~mask
    melody_mask = mask

    # Get indices
    chord_indices = torch.where(chord_mask)[0]
    melody_indices = torch.where(melody_mask)[0]

    chord_frames = seq[chord_indices]
    melody_frames = seq[melody_indices]

    # Find valid length (non-PAD in melody)
    pad_mask = melody_frames == pad_id
    first_pad = pad_mask.nonzero(as_tuple=True)[0]
    valid_len = int(first_pad[0].item()) if len(first_pad) > 0 else len(melody_frames)

    # Truncate both to valid length (should be same length)
    melody_frames = melody_frames[:valid_len]
    chord_frames = chord_frames[:valid_len]

    return melody_frames, chord_frames


def evaluate_online(
    model: OnlineTransformer,
    test_loader: DataLoader,
    d_cfg: DataConfig,
    device: torch.device,
    id_to_token: dict[int, str] | None = None,
    temperature: float = 1.0,
    sample: bool = False,
    log_progress: bool = True,
) -> OnlineEvalResult:
    """Evaluate online model and return metrics with cached predictions.

    Args:
        model: Trained OnlineTransformer model (already on device, in eval mode)
        test_loader: DataLoader for test set (using OnlineDataset)
        d_cfg: Data configuration
        id_to_token: Mapping from token IDs to strings
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

    # Resolve ID → token mapping from the dataset if not provided.
    if id_to_token is None:
        if dataset is None:
            raise ValueError(
                "id_to_token not provided and test_loader has no dataset."
            )

        if hasattr(dataset, "id_to_token"):
            id_to_token = dataset.id_to_token  # type: ignore[assignment]
        else:
            raise ValueError("OnlineDataset must expose id_to_token for decoding.")

    # --- 1. Test loss / perplexity (teacher-forced, chord positions only) ---
    criterion = nn.CrossEntropyLoss(ignore_index=d_cfg.pad_id, reduction="sum")
    total_loss = 0.0
    total_chord_tokens = 0

    if log_progress:
        logger.info("Computing teacher-forced test loss (chord positions only)...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids_full = batch["input_ids"].to(device)
            is_melody_full = batch["is_melody"].to(device)

            input_ids = input_ids_full[:, :-1]
            is_melody_input = is_melody_full[:, :-1]
            target_ids = input_ids_full[:, 1:]

            logits = model(input_ids, is_melody_input)

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

    # Accumulators for additional metrics
    total_silent_frames = 0
    total_melody_frames = 0
    total_long_chords = 0
    total_chords = 0
    early_stop_sequences = 0

    def _ended_early(mel_ids: list[int], pred_ids: list[int]) -> bool:
        """Return True if the chord sequence ends (EOS) before the melody."""
        try:
            eos_idx = pred_ids.index(d_cfg.eos_id)
        except ValueError:
            return False

        # Last non-PAD melody frame
        last_mel_idx = len(mel_ids) - 1
        while last_mel_idx >= 0 and mel_ids[last_mel_idx] == d_cfg.pad_id:
            last_mel_idx -= 1

        if last_mel_idx <= 0:
            return False
        return eos_idx < last_mel_idx

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids_batch = batch["input_ids"].to(device)
            is_melody_batch = batch["is_melody"].to(device)
            batch_size = input_ids_batch.size(0)

            for i in range(batch_size):
                input_ids = input_ids_batch[i]
                is_melody = is_melody_batch[i]

                melody_frames, ref_chord_frames = extract_melody_and_chords_from_mask(
                    input_ids,
                    is_melody,
                    d_cfg.pad_id,
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

                mel_tokens = decode_tokens(melody_frames.cpu().tolist(), id_to_token)
                pred_tokens = decode_tokens(pred_chords.cpu().tolist(), id_to_token)
                ref_tokens = decode_tokens(ref_chord_frames.cpu().tolist(), id_to_token)

                # Cache predictions
                result.cached_predictions[global_idx] = (mel_tokens, pred_tokens, ref_tokens)

                # Note-in-chord ratio
                nic = note_in_chord_ratio(mel_tokens, pred_tokens)
                result.per_song_nic.append(nic)

                # Onset intervals
                result.pred_intervals.extend(onset_intervals(mel_tokens, pred_tokens))
                result.ref_intervals.extend(onset_intervals(mel_tokens, ref_tokens))

                # Chord lengths
                lengths_pred = chord_lengths(pred_tokens)
                lengths_ref = chord_lengths(ref_tokens)
                result.pred_lengths.extend(lengths_pred)
                result.ref_lengths.extend(lengths_ref)

                # Chord silence ratio (predicted chords only)
                silent, total = chord_silence_counts(mel_tokens, pred_tokens)
                total_silent_frames += silent
                total_melody_frames += total

                # Long chords ratio (>32 frames)
                long_threshold = 32
                total_long_chords += sum(1 for length in lengths_pred if length > long_threshold)
                total_chords += len(lengths_pred)

                # Early stop ratio (EOS before melody end)
                if _ended_early(
                    melody_frames.cpu().tolist(),
                    pred_chords.cpu().tolist(),
                ):
                    early_stop_sequences += 1

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

    # Aggregate additional diagnostics
    result.chord_silence_ratio = (
        (total_silent_frames / total_melody_frames) * 100.0 if total_melody_frames > 0 else 0.0
    )
    result.long_chord_ratio = (
        (total_long_chords / total_chords) * 100.0 if total_chords > 0 else 0.0
    )
    result.early_stop_ratio = (
        (early_stop_sequences / result.num_sequences) * 100.0 if result.num_sequences > 0 else 0.0
    )

    return result


def main():
    """Evaluate online model on test set."""
    from musicagent.utils import load_configs_from_dir

    parser = build_eval_online_parser()
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of predicted sequences to print with ground truth.",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Print all metrics including diagnostics.",
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

    # Load test dataset (supports weighted multi-source)
    sources = _resolve_sources(d_cfg)
    if len(sources) > 1:
        test_ds = WeightedJointOnlineDataset(
            d_cfg,
            sources,
            split="test",
            eval_mode=True,
            seed=42,
        )
    else:
        test_ds = OnlineDataset(d_cfg, split="test")
    collate = make_online_collate_fn(pad_id=d_cfg.pad_id)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    # Load model with unified vocab size
    vocab_size = test_ds.vocab_size
    model = OnlineTransformer(
        m_cfg,
        d_cfg,
        vocab_size=vocab_size,
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
    logger.info(
        f"NiC Ratio:                 {result.nic_ratio * 100:.2f}% (±{result.nic_std * 100:.2f}%)"
    )
    logger.info(f"Onset Interval EMD:        {result.onset_interval_emd * 1e3:.2f} ×10⁻³")
    logger.info(
        f"Chord Length Entropy:      {result.pred_chord_length_entropy:.2f}  "
        f"(ref: {result.ref_chord_length_entropy:.2f})"
    )
    if args.all_metrics:
        logger.info("-" * 60)
        logger.info(f"Chord Silence Ratio:       {result.chord_silence_ratio:.2f}%")
        logger.info(f"Long Chord Ratio (>32f):   {result.long_chord_ratio:.2f}%")
        logger.info(f"Early Stop Ratio:          {result.early_stop_ratio:.2f}%")
    logger.info("=" * 60)

    if args.num_samples > 0:
        logger.info("\n" + "=" * 60)
        logger.info(f"Sample Predictions ({args.num_samples} sequences)")
        logger.info("=" * 60)

        num_to_show = min(args.num_samples, len(result.cached_predictions))
        for idx in range(num_to_show):
            if idx in result.cached_predictions:
                mel_tokens, pred_tokens, ref_tokens = result.cached_predictions[idx]
                logger.info(f"\n--- Sequence {idx + 1} ---")
                logger.info(f"Melody:   {' '.join(mel_tokens)}")
                logger.info(f"Predicted: {' '.join(pred_tokens)}")
                logger.info(f"Ground Truth: {' '.join(ref_tokens)}")


if __name__ == "__main__":
    main()
