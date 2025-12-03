"""Test set evaluation for offline model."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from musicagent.cli import build_eval_offline_parser
from musicagent.config import DataConfig, OfflineConfig
from musicagent.data import OfflineDataset, make_offline_collate_fn
from musicagent.models import OfflineTransformer
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


@dataclass
class OfflineEvalResult:
    """Result container for offline model evaluation."""

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


def evaluate_offline(
    model: OfflineTransformer,
    test_loader: DataLoader,
    d_cfg: DataConfig,
    device: torch.device,
    id_to_melody: dict[int, str] | None = None,
    id_to_chord: dict[int, str] | None = None,
    temperature: float = 1.0,
    sample: bool = False,
    log_progress: bool = True,
) -> OfflineEvalResult:
    """Evaluate offline model and return metrics with cached predictions.

    Args:
        model: Trained OfflineTransformer model (already on device, in eval mode)
        test_loader: DataLoader for test set (using OfflineDataset)
        d_cfg: Data configuration
        id_to_melody: Mapping from melody token IDs to strings
        id_to_chord: Mapping from chord token IDs to strings
        device: Device to run evaluation on
        temperature: Sampling temperature (only used with sample=True)
        sample: If True, sample from distribution; otherwise greedy decode
        log_progress: If True, log progress to logger

    Returns:
        OfflineEvalResult containing all metrics and cached predictions
    """
    import numpy as np

    result = OfflineEvalResult()

    dataset = getattr(test_loader, "dataset", None)

    # With separate vocabularies, melody IDs remain in unified/melody space
    # (same IDs), but chord IDs are now in chord vocab space. We must use
    # the appropriate mapping for each.
    if id_to_melody is None or id_to_chord is None:
        if dataset is None:
            raise ValueError(
                "id_to_melody/id_to_chord not provided and test_loader has no dataset."
            )

        if hasattr(dataset, "melody_id_to_token") and hasattr(dataset, "chord_id_to_token"):
            id_to_melody = dataset.melody_id_to_token  # type: ignore[assignment]
            id_to_chord = dataset.chord_id_to_token  # type: ignore[assignment]
        elif hasattr(dataset, "unified_id_to_token"):
            # Fallback for legacy datasets without separate vocab files
            unified_map: dict[int, str] = dataset.unified_id_to_token  # type: ignore[assignment]
            id_to_melody = unified_map
            id_to_chord = unified_map
        else:
            raise ValueError(
                "Dataset does not expose melody_id_to_token/chord_id_to_token for decoding."
            )

    # --- 1. Test loss / perplexity (teacher-forced) ---
    criterion = nn.CrossEntropyLoss(ignore_index=d_cfg.pad_id)
    total_loss = 0.0
    total_samples = 0

    if log_progress:
        logger.info("Computing teacher-forced test loss...")

    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_y = tgt[:, 1:]
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_y.reshape(-1))
            total_loss += loss.item() * src.size(0)
            total_samples += src.size(0)

    result.test_loss = total_loss / total_samples if total_samples > 0 else 0.0
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
            # Model never emitted EOS – treat as "not early".
            return False

        # Find last non-PAD frame in melody.
        last_mel_idx = len(mel_ids) - 1
        while last_mel_idx >= 0 and mel_ids[last_mel_idx] == d_cfg.pad_id:
            last_mel_idx -= 1

        # If melody is degenerate or EOS is at/after melody end, not early.
        if last_mel_idx <= 0:
            return False
        return eos_idx < last_mel_idx

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(test_loader):
            src = src.to(device)

            pred = model.generate(
                src,
                max_len=d_cfg.max_len,
                sos_id=d_cfg.sos_id,
                eos_id=d_cfg.eos_id,
                temperature=temperature,
                sample=sample,
            )

            for i in range(src.size(0)):
                mel_ids = src[i].cpu().tolist()
                pred_ids = pred[i].cpu().tolist()
                ref_ids = tgt[i].cpu().tolist()

                mel_tokens = decode_tokens(mel_ids, id_to_melody)
                pred_tokens = decode_tokens(pred_ids, id_to_chord)
                ref_tokens = decode_tokens(ref_ids, id_to_chord)

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

                # Long chords ratio (predicted chords only, >32 frames)
                long_threshold = 32
                total_long_chords += sum(1 for length in lengths_pred if length > long_threshold)
                total_chords += len(lengths_pred)

                # Early stop ratio (EOS before melody end)
                if _ended_early(mel_ids, pred_ids):
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
    """Evaluate offline model on test set."""
    from musicagent.utils import load_configs_from_dir

    parser = build_eval_offline_parser()
    args = parser.parse_args()

    setup_logging()

    # Load configs from checkpoint directory
    cfg_dir = args.checkpoint.parent
    data_cfg_path = cfg_dir / "data_config.json"
    model_cfg_path = cfg_dir / "model_config.json"

    if data_cfg_path.exists() and model_cfg_path.exists():
        d_cfg, m_cfg = load_configs_from_dir(cfg_dir, OfflineConfig)
        logger.info(f"Loaded configs from {cfg_dir}")
    else:
        d_cfg = DataConfig()
        m_cfg = OfflineConfig()

    if args.device:
        m_cfg.device = args.device
    device = torch.device(m_cfg.device)

    # Load test dataset
    test_ds = OfflineDataset(d_cfg, split="test")
    collate = make_offline_collate_fn(pad_id=d_cfg.pad_id)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    # Load model with separate vocab sizes
    melody_vocab_size = test_ds.melody_vocab_size
    chord_vocab_size = test_ds.chord_vocab_size
    model = OfflineTransformer(
        m_cfg,
        d_cfg,
        melody_vocab_size=melody_vocab_size,
        chord_vocab_size=chord_vocab_size,
    ).to(device)
    state_dict = safe_load_state_dict(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Run evaluation
    result = evaluate_offline(
        model=model,
        test_loader=test_loader,
        d_cfg=d_cfg,
        device=device,
        temperature=args.temperature,
        sample=args.sample,
    )

    # Log results
    logger.info("=" * 60)
    logger.info("Metrics (Offline Eval)")
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
