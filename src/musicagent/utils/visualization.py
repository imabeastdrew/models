"""Visualization utils for eval."""

from musicagent.eval import note_in_chord_ratio


def format_tokens_for_display(tokens: list[str], max_n: int = 24) -> str:
    """Format tokens for display, filtering special tokens and compressing holds.

    Args:
        tokens: List of token strings
        max_n: Maximum number of tokens to display

    Returns:
        Formatted string for display
    """
    filtered = [t for t in tokens[: max_n * 2] if not t.startswith("<")]
    if not filtered:
        return "(all rest/special tokens)"

    # Simplify display: remove _on/_hold suffixes, compress holds
    display: list[str] = []
    for t in filtered:
        if t.endswith("_hold") and display and display[-1] == "...":
            continue
        elif t.endswith("_hold"):
            display.append("...")
        else:
            display.append(t.replace("_on", ""))

    return " ".join(display[:max_n])


def show_example(
    cached_predictions: dict[int, tuple[list[str], list[str], list[str]]],
    idx: int,
    max_frames: int = 48,
) -> None:
    """Display a single example with melody, predicted, and reference chords.

    Args:
        cached_predictions: Dict mapping index -> (melody_tokens, pred_tokens, ref_tokens)
        idx: Index of the example to display
        max_frames: Maximum number of frames to show
    """
    if idx not in cached_predictions:
        print(f"⚠ Example {idx} not in cache. Skipping.")
        return

    mel_tokens, pred_tokens, ref_tokens = cached_predictions[idx]
    nic = note_in_chord_ratio(mel_tokens, pred_tokens)

    print(f"\n{'=' * 70}")
    print(f"Example {idx} | NiC: {nic:.1%} | Frames: {len(mel_tokens)}")
    print(f"{'=' * 70}")
    print("\nMelody:")
    print(format_tokens_for_display(mel_tokens, max_frames))
    print("\nPredicted Chords:")
    print(format_tokens_for_display(pred_tokens, max_frames))
    print("\nReference Chords:")
    print(format_tokens_for_display(ref_tokens, max_frames))


def get_examples_by_nic_quality(
    cached_predictions: dict[int, tuple[list[str], list[str], list[str]]],
) -> tuple[int, int, int]:
    """Get example indices sorted by NiC quality.

    Args:
        cached_predictions: Dict mapping index -> (melody_tokens, pred_tokens, ref_tokens)

    Returns:
        Tuple of (best_idx, median_idx, worst_idx) by NiC ratio
    """
    if not cached_predictions:
        raise ValueError("No cached predictions available")

    sorted_indices = sorted(
        cached_predictions.keys(),
        key=lambda idx: note_in_chord_ratio(cached_predictions[idx][0], cached_predictions[idx][1]),
        reverse=True,
    )

    best_idx = sorted_indices[0]
    median_idx = sorted_indices[len(sorted_indices) // 2]
    worst_idx = sorted_indices[-1]

    return best_idx, median_idx, worst_idx


def show_examples_by_quality(
    cached_predictions: dict[int, tuple[list[str], list[str], list[str]]],
    max_frames: int = 48,
) -> None:
    """Show best, median, and worst examples by NiC quality.

    Args:
        cached_predictions: Dict mapping index -> (melody_tokens, pred_tokens, ref_tokens)
        max_frames: Maximum number of frames to show per example
    """
    if not cached_predictions:
        print("⚠ No cached predictions available.")
        return

    best_idx, median_idx, worst_idx = get_examples_by_nic_quality(cached_predictions)

    print("Showing examples by NiC quality (best, median, worst):")
    show_example(cached_predictions, best_idx, max_frames)
    show_example(cached_predictions, median_idx, max_frames)
    show_example(cached_predictions, worst_idx, max_frames)
