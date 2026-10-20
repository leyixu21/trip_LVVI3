"""Turn (timestamped transcript segments, timestamped frame captions) into
fused multimodal chunks.

Kept independent of any specific embedding backend: callers pass an
`embed_fn: list[str] -> np.ndarray[N, D]` (L2-normalized rows) so this module
is unit-testable without loading CLIP.
"""
from dataclasses import dataclass, field

import numpy as np

# Named here (not just inlined in fused_text below) so
# pipeline/06_privacy_scrub_chunks.py can strip these structural labels
# before running NER over the text -- an NER model can otherwise
# misclassify the label text itself as a named entity.
TRANSCRIPT_PREFIX = "Transcript: "
CAPTIONS_PREFIX = "Visual observations: "


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class FrameCaption:
    """One VLM caption produced from a bundle of a shot's keyframes (see
    pipeline/04_frame_captioning.py), covering that shot's full (start, end)
    range rather than a single instant."""
    start: float
    end: float
    shot_index: int
    frame_names: list[str]
    caption: str


@dataclass
class FusedChunk:
    video_id: str
    chunk_index: int
    start: float
    end: float
    transcript_text: str
    captions: list[str] = field(default_factory=list)

    @property
    def fused_text(self) -> str:
        parts = []
        if self.transcript_text.strip():
            parts.append(f"{TRANSCRIPT_PREFIX}{self.transcript_text.strip()}")
        if self.captions:
            parts.append(CAPTIONS_PREFIX + " | ".join(self.captions))
        return "\n".join(parts)

    def is_empty(self) -> bool:
        return not self.transcript_text.strip() and not self.captions


def _fixed_grid_windows(segments: list[TranscriptSegment], video_id: str, window_seconds: float) -> list[FusedChunk]:
    """Fallback for videos with no frame captions at all (e.g. a missing
    Allas keyframe tar) -- a uniform arithmetic grid sized off the
    transcript's own duration, with no shots to align chunk boundaries to."""
    if not segments:
        return []
    video_end = max(s.end for s in segments)
    n_windows = max(1, int(np.ceil(video_end / window_seconds)))
    return [
        FusedChunk(
            video_id=video_id,
            chunk_index=i,
            start=i * window_seconds,
            end=min((i + 1) * window_seconds, video_end),
            transcript_text="",
        )
        for i in range(n_windows)
    ]


def _shot_aware_windows(frame_captions: list[FrameCaption], video_id: str, window_seconds: float) -> list[FusedChunk]:
    """Builds chunks by walking shots in chronological order and cutting as
    soon as the accumulated duration reaches window_seconds, so a chunk
    boundary always falls exactly on a shot boundary. Unlike a fixed time
    grid, a shot's caption is never split across two chunks -- and since a
    shot's own duration is already capped (see
    config.TARGET_KEYFRAME_INTERVAL_SECONDS), no single shot can overshoot
    the target by more than that cap."""
    if not frame_captions:
        return []
    shots = sorted(frame_captions, key=lambda fc: fc.start)

    windows: list[FusedChunk] = []
    group: list[FrameCaption] = []
    chunk_start = shots[0].start

    def flush(chunk_end: float):
        if not group:
            return
        windows.append(FusedChunk(
            video_id=video_id,
            chunk_index=len(windows),
            start=chunk_start,
            end=chunk_end,
            transcript_text="",
            captions=[fc.caption for fc in group],
        ))

    for fc in shots:
        group.append(fc)
        if fc.end - chunk_start >= window_seconds:
            flush(fc.end)
            group = []
            chunk_start = fc.end

    flush(shots[-1].end)  # trailing shots that never reached window_seconds
    return windows


def _assign_transcript(windows: list[FusedChunk], segments: list[TranscriptSegment]) -> None:
    """Assigns each transcript segment (by its midpoint) into whichever
    window contains it, mutating transcript_text in place. Windows are
    contiguous and sorted by construction (both _fixed_grid_windows and
    _shot_aware_windows produce them that way), so the first window whose
    end exceeds the midpoint is the right one; a segment whose midpoint
    falls before the first window's start (e.g. speech before the first
    detected shot) is folded into that first window rather than dropped."""
    if not windows:
        return
    for seg in segments:
        mid = (seg.start + seg.end) / 2
        idx = next((i for i, w in enumerate(windows) if mid < w.end), len(windows) - 1)
        windows[idx].transcript_text = (windows[idx].transcript_text + " " + seg.text).strip()


def dedup_captions(captions: list[str], embed_fn, threshold: float) -> list[str]:
    """Drop near-duplicate captions (e.g. static/repeated scenery) within a
    single chunk using embedding cosine similarity, keeping first occurrence."""
    if len(captions) <= 1:
        return list(captions)

    embeddings = embed_fn(captions)
    keep_idx: list[int] = []
    for i in range(len(captions)):
        is_dup = False
        for j in keep_idx:
            sim = float(embeddings[i] @ embeddings[j])
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep_idx.append(i)
    return [captions[i] for i in keep_idx]


def build_fused_chunks(
    segments: list[TranscriptSegment],
    frame_captions: list[FrameCaption],
    video_id: str,
    window_seconds: float,
    embed_fn,
    dedup_threshold: float,
) -> list[FusedChunk]:
    if frame_captions:
        chunks = _shot_aware_windows(frame_captions, video_id, window_seconds)
        for chunk in chunks:
            chunk.captions = dedup_captions(chunk.captions, embed_fn, dedup_threshold)
    else:
        chunks = _fixed_grid_windows(segments, video_id, window_seconds)

    _assign_transcript(chunks, segments)
    return [c for c in chunks if not c.is_empty()]
