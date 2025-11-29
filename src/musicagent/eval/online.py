"""Evaluation for online model including adaptation dynamics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from musicagent.config import DataConfig, OnlineConfig
from musicagent.data import OfflineDataset, collate_fn
from musicagent.models import OnlineTransformer
from musicagent.utils import setup_logging

from .metrics import (
    chord_length_entropy,
    chord_lengths,
    note_in_chord_ratio,
    onset_interval_emd,
    onset_intervals,
)

logger = logging.getLogger(__name__)


def decode_tokens(ids: list[int], id_to_token: dict[int, str]) -> list[str]:
    """Convert token IDs back to string tokens."""
    return [id_to_token.get(i, "<unk>") for i in ids]


def note_in_chord_at_beat(
    melody_tokens: list[str],
    chord_tokens: list[str],
    frame_rate: int = 4,
) -> dict[int, float]:
    """Compute note-in-chord ratio at each beat position.

    Used for analyzing adaptation dynamics over time.

    Args:
        melody_tokens: List of melody token strings
        chord_tokens: List of chord token strings
        frame_rate: Frames per beat (default 4 for 16th notes)

    Returns:
        Dictionary mapping beat index to NiC ratio at that beat
    """
    from .metrics import chord_pitch_classes, parse_chord_token, parse_melody_token

    num_frames = min(len(melody_tokens), len(chord_tokens))
    num_beats = num_frames // frame_rate

    beat_nic = {}

    for beat in range(num_beats):
        start_frame = beat * frame_rate
        end_frame = start_frame + frame_rate

        matches = 0
        total = 0

        for t in range(start_frame, min(end_frame, num_frames)):
            mel_tok = melody_tokens[t]
            chd_tok = chord_tokens[t]

            midi_pitch = parse_melody_token(mel_tok)
            root, quality = parse_chord_token(chd_tok)

            if midi_pitch is None or root is None or quality is None:
                continue

            melody_pc = midi_pitch % 12
            chord_pcs = chord_pitch_classes(root, quality)

            if melody_pc in chord_pcs:
                matches += 1
            total += 1

        beat_nic[beat] = matches / total if total > 0 else 0.0

    return beat_nic


def perturb_melody(melody: torch.Tensor, semitones: int, start_frame: int) -> torch.Tensor:
    """Transpose a melody by semitones starting from a given frame.

    Used for perturbation tests (e.g., tritone shift mid-song).

    Args:
        melody: Melody token IDs (batch, seq_len)
        semitones: Number of semitones to transpose
        start_frame: Frame index to start transposition

    Returns:
        Perturbed melody tensor
    """
    # This is a simplified version - actual implementation would need
    # access to vocabulary to properly transpose tokens
    return melody.clone()


def main():
    """Evaluate online model on test set."""
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

    # Attempt to load saved configs from the checkpoint directory to avoid
    # silent drift between training and evaluation settings. If no configs
    # are found, fall back to default constructors.
    cfg_dir = args.checkpoint.parent
    data_cfg_path = cfg_dir / "data_config.json"
    model_cfg_path = cfg_dir / "model_config.json"

    if data_cfg_path.exists():
        with data_cfg_path.open() as f:
            data_dict = json.load(f)
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
        m_cfg = OnlineConfig(**model_dict)
        logger.info(f"Loaded model config from {model_cfg_path}")
    else:
        m_cfg = OnlineConfig()

    if args.device:
        m_cfg.device = args.device
    device = torch.device(m_cfg.device)

    # Load test dataset (use OfflineDataset for evaluation since we need
    # separate melody/chord for metric computation)
    test_ds = OfflineDataset(d_cfg, split="test")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Build reverse vocab mappings
    id_to_melody = {v: k for k, v in test_ds.vocab_melody.items()}
    id_to_chord = {v: k for k, v in test_ds.vocab_chord.items()}

    # Load model
    melody_vocab_size = len(test_ds.vocab_melody)
    chord_vocab_size = len(test_ds.vocab_chord)

    model = OnlineTransformer(
        m_cfg, d_cfg, melody_vocab_size, chord_vocab_size
    ).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # --- Generate chord sequences & compute metrics ---
    logger.info("Generating chord sequences for metric evaluation...")
    all_nic: list[float] = []
    all_pred_intervals: list[int] = []
    all_ref_intervals: list[int] = []
    all_pred_lengths: list[int] = []
    all_ref_lengths: list[int] = []

    num_batches = len(test_loader)
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(test_loader):
            src = src.to(device)

            # Generate predictions per-sample using *frame* sequences.
            # OfflineDataset returns sequences with:
            #   index 0      : SOS
            #   indices 1..k : frame-aligned tokens
            #   index k+1    : EOS
            #   >k+1         : PAD
            # We strip SOS / EOS / PAD so the online model sees only frames,
            # matching the representation used during its own training.
            batch_size = src.size(0)

            for i in range(batch_size):
                mel_seq = src[i]

                # Find EOS; if missing, treat full sequence (excluding SOS)
                eos_positions = (mel_seq == d_cfg.eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    eos_idx = int(eos_positions[0].item())
                else:
                    eos_idx = mel_seq.size(0)

                # Frames live in [1, eos_idx)
                melody_frames = mel_seq[1:eos_idx].unsqueeze(0)

                pred_i = model.generate(
                    melody_frames,
                    sos_id=d_cfg.sos_id,
                    eos_id=d_cfg.eos_id,
                    temperature=args.temperature,
                    sample=args.sample,
                )[0]

                mel_ids = mel_seq.cpu().tolist()
                pred_ids = pred_i.cpu().tolist()
                ref_ids = tgt[i].cpu().tolist()

                mel_tokens = decode_tokens(mel_ids, id_to_melody)
                pred_tokens = decode_tokens(pred_ids, id_to_chord)
                ref_tokens = decode_tokens(ref_ids, id_to_chord)

                # Note-in-chord
                nic = note_in_chord_ratio(mel_tokens, pred_tokens)
                all_nic.append(nic)

                # Onset intervals
                all_pred_intervals.extend(onset_intervals(mel_tokens, pred_tokens))
                all_ref_intervals.extend(onset_intervals(mel_tokens, ref_tokens))

                # Chord lengths
                all_pred_lengths.extend(chord_lengths(pred_tokens))
                all_ref_lengths.extend(chord_lengths(ref_tokens))

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                logger.info(f"  Processed {batch_idx + 1}/{num_batches} batches")

    avg_nic = sum(all_nic) / len(all_nic) if all_nic else 0.0
    emd = onset_interval_emd(all_pred_intervals, all_ref_intervals)
    pred_entropy = chord_length_entropy(all_pred_lengths)
    ref_entropy = chord_length_entropy(all_ref_lengths)

    logger.info("=" * 60)
    logger.info("Metrics (Online Eval)")
    logger.info("=" * 60)
    logger.info(f"NiC Ratio:                 {avg_nic * 100:.2f}%")
    logger.info(f"Onset Interval EMD:        {emd * 1e3:.2f} ×10⁻³")
    logger.info(f"Chord Length Entropy:      {pred_entropy:.2f}  (ref: {ref_entropy:.2f})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

