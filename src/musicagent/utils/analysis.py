"""Analysis utilities for model evaluation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptationDynamicsResult:
    """Result of per-beat NiC analysis."""

    valid_beats: list[int]
    """Beat indices that had enough samples."""
    beat_means: list[float]
    """Mean NiC ratio per valid beat."""
    beat_stds: list[float]
    """Standard deviation of NiC ratio per valid beat."""
    samples_per_beat: dict[int, int]
    """Number of samples for each beat (for all beats, not just valid ones)."""


def compute_adaptation_dynamics(
    cached_predictions: dict[int, tuple[list[str], list[str], list[str]]],
    *,
    max_beats: int = 64,
    frame_rate: int = 4,
    min_samples: int = 10,
) -> AdaptationDynamicsResult:
    """Compute per-beat NiC statistics for adaptation dynamics analysis.

    Per paper Section K: excludes beats that are entirely silent.

    Args:
        cached_predictions: Dict mapping index -> (melody_tokens, pred_tokens, ref_tokens)
        max_beats: Maximum number of beats to analyze
        frame_rate: Frames per beat (default 4 for 16th notes)
        min_samples: Minimum samples required per beat to include in results

    Returns:
        AdaptationDynamicsResult with per-beat statistics
    """
    # Import here to avoid circular dependency (eval imports utils)
    from musicagent.eval import note_in_chord_at_beat

    beat_nic_all: dict[int, list[float]] = {b: [] for b in range(max_beats)}

    for mel_tokens, pred_tokens, _ in cached_predictions.values():
        # note_in_chord_at_beat returns None for silent beats
        beat_nic = note_in_chord_at_beat(mel_tokens, pred_tokens, frame_rate=frame_rate)

        for beat, nic in beat_nic.items():
            # Only include non-silent beats (nic is None for silent beats per paper)
            if beat < max_beats and nic is not None:
                beat_nic_all[beat].append(nic)

    # Compute mean and std per beat
    valid_beats: list[int] = []
    beat_means: list[float] = []
    beat_stds: list[float] = []

    for beat in range(max_beats):
        if len(beat_nic_all[beat]) >= min_samples:
            valid_beats.append(beat)
            beat_means.append(float(np.mean(beat_nic_all[beat])))
            beat_stds.append(float(np.std(beat_nic_all[beat])))

    samples_per_beat = {beat: len(beat_nic_all[beat]) for beat in range(max_beats)}

    return AdaptationDynamicsResult(
        valid_beats=valid_beats,
        beat_means=beat_means,
        beat_stds=beat_stds,
        samples_per_beat=samples_per_beat,
    )
