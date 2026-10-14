"""Pure segment-list logic for shot -> keyframe timestamp selection.

Deliberately has no cv2/scenedetect import, so it's testable without those
installed. The actual detection (which does need them) lives in
pipeline/02_detect_shots_and_extract_keyframes.py.
"""
import math


def expand_long_segments(shots: list[tuple[float, float]], target_interval: float) -> list[tuple[float, float]]:
    """Subdivide any segment longer than `target_interval` into equal
    sub-segments, so a long continuous shot (or the whole-video fallback
    when no cuts were detected at all) still gets keyframes spread across
    its full duration instead of just one at its midpoint. Nature/travel
    footage is often one continuous shot with slow pans or drone movement
    and no hard cuts -- even AdaptiveDetector correctly finds zero
    boundaries there, since nothing abruptly changes."""
    if target_interval <= 0:
        return shots
    expanded = []
    for start, end in shots:
        duration = end - start
        if duration <= target_interval:
            expanded.append((start, end))
            continue
        n_parts = max(math.ceil(duration / target_interval), 1)
        step = duration / n_parts
        expanded.extend((start + i * step, start + (i + 1) * step) for i in range(n_parts))
    return expanded


def subsample_shots(shots: list[tuple[float, float]], max_keyframes: int) -> list[tuple[float, float]]:
    if len(shots) <= max_keyframes:
        return shots
    step = len(shots) / max_keyframes
    idxs = [int(i * step) for i in range(max_keyframes)]
    return [shots[i] for i in idxs]


def build_tagged_segments(raw_shots: list[tuple[float, float]], target_interval: float) -> list[tuple[int, float, float]]:
    """Applies expand_long_segments per original detected shot (rather than
    on the flat list) so every resulting sub-segment can be tagged with
    which shot it came from: (shot_index, start, end). A shot longer than
    target_interval yields several sub-segments sharing one shot_index --
    their keyframes can later be bundled into a single multi-image caption
    to capture motion within that shot, instead of treated as independent
    stills. A shot short enough to never subdivide just yields one tagged
    segment, same as before this existed."""
    segments = []
    for shot_index, (start, end) in enumerate(raw_shots):
        for sub_start, sub_end in expand_long_segments([(start, end)], target_interval):
            segments.append((shot_index, sub_start, sub_end))
    return segments


def group_by_shot(records: list[dict]) -> list[list[dict]]:
    """Groups frame records (each with a "shot_index" and "keyframe_ts"
    field, e.g. from shots.json) by shot_index, preserving first-seen shot
    order and within-shot chronological order. Used by stage 4 to bundle a
    shot's keyframes into one multi-image captioning call, frames presented
    in the sequence they actually occurred."""
    by_shot: dict[int, list[dict]] = {}
    order: list[int] = []
    for rec in records:
        shot_index = rec["shot_index"]
        if shot_index not in by_shot:
            by_shot[shot_index] = []
            order.append(shot_index)
        by_shot[shot_index].append(rec)
    return [sorted(by_shot[i], key=lambda r: r["keyframe_ts"]) for i in order]
