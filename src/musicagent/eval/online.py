"""Test set evaluation for offline model."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from musicagent.config import DataConfig, OfflineConfig
from musicagent.data import OfflineDataset, make_offline_collate_fn
from musicagent.models import OfflineTransformer
from musicagent.utils import setup_logging

from .metrics import (
    chord_length_entropy,
    chord_lengths,
    decode_tokens,
    note_in_chord_counts,
    onset_interval_emd,
    onset_intervals,
)

logger = logging.getLogger(__name__)


def main():
    """Evaluate offline model on test set."""
    parser = argparse.ArgumentParser(description="Evaluate Offline Model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/offline/best_model.pt"),
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

    # Attempt to load saved configs from the checkpoint directory to avoid
    # silent drift between training and evaluation settings. If no configs
    # are found, fall back to default constructors.
    cfg_dir = args.checkpoint.parent
    data_cfg_path = cfg_dir / "data_config.json"
    model_cfg_path = cfg_dir / "model_config.json"

    if data_cfg_path.exists():
        with data_cfg_path.open() as f:
            data_dict = json.load(f)
        # Coerce path-like fields back to Path objects.
        if "data_raw" in data_dict:
            data_dict["data_raw"] = Path(data_dict["data_raw"])
        if "data_processed" in data_dict:
            data_dict["data_processed"] = Path(data_dict["data_processed"])
        d_cfg = DataConfig(**data_dict)
        logger.info(f"Loaded data config from {data_cfg_path}")
    else:
        d_cfg = DataConfig()

    if model_cfg_path.exists():
        with model_cfg_path.open() as f:
            model_dict = json.load(f)
        m_cfg = OfflineConfig(**model_dict)
        logger.info(f"Loaded model config from {model_cfg_path}")
    else:
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

    # Build reverse vocab mappings
    id_to_melody = {v: k for k, v in test_ds.vocab_melody.items()}
    id_to_chord = {v: k for k, v in test_ds.vocab_chord.items()}

    # Load model
    vocab_src = len(test_ds.vocab_melody)
    vocab_tgt = len(test_ds.vocab_chord)
    model = OfflineTransformer(m_cfg, d_cfg, vocab_src, vocab_tgt).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # --- 1. Test loss / perplexity (teacher-forced) ---
    criterion = nn.CrossEntropyLoss(ignore_index=d_cfg.pad_id)
    total_loss = 0.0
    total_samples = 0

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

    test_loss = total_loss / total_samples
    test_ppl = math.exp(min(test_loss, 100))
    logger.info(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")

    # --- 2. Generate chord sequences & compute metrics ---
    logger.info("Generating chord sequences for metric evaluation...")
    # Frame-weighted NiC: accumulate (matches, total) across all sequences
    total_nic_matches = 0
    total_nic_frames = 0
    all_pred_intervals: list[int] = []
    all_ref_intervals: list[int] = []
    all_pred_lengths: list[int] = []
    all_ref_lengths: list[int] = []

    num_batches = len(test_loader)
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(test_loader):
            src = src.to(device)
            # Generate predictions
            pred = model.generate(
                src,
                max_len=d_cfg.max_len,
                sos_id=d_cfg.sos_id,
                eos_id=d_cfg.eos_id,
                temperature=args.temperature,
                sample=args.sample,
            )

            for i in range(src.size(0)):
                mel_ids = src[i].cpu().tolist()
                pred_ids = pred[i].cpu().tolist()
                ref_ids = tgt[i].cpu().tolist()

                mel_tokens = decode_tokens(mel_ids, id_to_melody)
                pred_tokens = decode_tokens(pred_ids, id_to_chord)
                ref_tokens = decode_tokens(ref_ids, id_to_chord)

                # Note-in-chord (frame-weighted)
                matches, total = note_in_chord_counts(mel_tokens, pred_tokens)
                total_nic_matches += matches
                total_nic_frames += total

                # Onset intervals
                all_pred_intervals.extend(onset_intervals(mel_tokens, pred_tokens))
                all_ref_intervals.extend(onset_intervals(mel_tokens, ref_tokens))

                # Chord lengths
                all_pred_lengths.extend(chord_lengths(pred_tokens))
                all_ref_lengths.extend(chord_lengths(ref_tokens))

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                logger.info(f"  Processed {batch_idx + 1}/{num_batches} batches")

    avg_nic = total_nic_matches / total_nic_frames if total_nic_frames > 0 else 0.0
    emd = onset_interval_emd(all_pred_intervals, all_ref_intervals)
    pred_entropy = chord_length_entropy(all_pred_lengths)
    ref_entropy = chord_length_entropy(all_ref_lengths)

    logger.info("=" * 60)
    logger.info("Metrics (Offline Eval)")
    logger.info("=" * 60)
    logger.info(f"Test Loss:                 {test_loss:.4f}")
    logger.info(f"Test Perplexity:           {test_ppl:.2f}")
    logger.info("-" * 60)
    logger.info(f"NiC Ratio:                 {avg_nic * 100:.2f}%")
    logger.info(f"Onset Interval EMD:        {emd * 1e3:.2f} ×10⁻³")
    logger.info(f"Chord Length Entropy:      {pred_entropy:.2f}  (ref: {ref_entropy:.2f})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
