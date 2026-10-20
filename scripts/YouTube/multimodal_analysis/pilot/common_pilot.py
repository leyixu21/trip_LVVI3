"""Shared helpers for the B-vs-D 50-video pilot (Cloudflare/clef vs
Qwen3-VL-32B-Instruct). Everything here is additive to the existing
common/ modules -- pilot-only code lives under pilot/, nothing in
pipeline/ or common/ is touched.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parent
MM_DIR = PILOT_DIR.parent  # scripts/YouTube/multimodal_analysis
sys.path.insert(0, str(MM_DIR))

from common import config, keyframe_io  # noqa: E402

from allas_s3 import get_object_bytes
from label_mapping import SUB_ACTIVITY_TO_ACTIVITY  # noqa: E402

PILOT_DATA_DIR = config.DATA_DIR / "YouTube" / "pilot"
SELECTION_PATH = PILOT_DATA_DIR / "video_selection.json"
GROUND_TRUTH_PATH = PILOT_DATA_DIR / "ground_truth.json"
MAPPING_REPORT_PATH = PILOT_DATA_DIR / "mapping_report.json"

PILOT_B_OUTPUT = PILOT_DATA_DIR / "pilot_b_clef.jsonl"
PILOT_D_OUTPUT = PILOT_DATA_DIR / "pilot_d_qwen3vl.jsonl"

LABELED_TASKS_PATH = MM_DIR / "labeled_tasks_LVVI3.json"

# Fairness: both pilots see the same number of frames.
N_SAMPLE_FRAMES = 14
# Cap on transcript characters in prompts/records (both pilots).
TRANSCRIPT_MAX_CHARS = 6000
VIDEO_DESCRIPTION_MAX_CHARS = 600

TAXONOMY_ACTIVITIES: list[str] = []  # filled by load_taxonomy()


def load_taxonomy() -> list[str]:
    """The top-level LVVI3 activity names from the activity column, sorted."""
    global TAXONOMY_ACTIVITIES
    if not TAXONOMY_ACTIVITIES:
        import csv
        with open(config.TAXONOMY_ACTIVITIES_PATH, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        TAXONOMY_ACTIVITIES = sorted({r["activity"].strip() for r in rows if r.get("activity")})
    return TAXONOMY_ACTIVITIES


def load_transcripts_index() -> dict[str, dict]:
    """video_id -> transcript record (video_id, language, segments, audio_events)."""
    idx: dict[str, dict] = {}
    with open(config.TRANSCRIPTS_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            idx[rec["video_id"]] = rec
    return idx


def transcript_text(record: dict, max_chars: int = TRANSCRIPT_MAX_CHARS) -> str:
    """Segments joined as [mm:ss] text lines, capped to max_chars."""
    parts = []
    total = 0
    for seg in record.get("segments", []):
        t = seg.get("text", "").strip()
        if not t:
            continue
        start = int(seg.get("start", 0))
        line = f"[{start // 60:02d}:{start % 60:02d}] {t}"
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line) + 1
    return "\n".join(parts)


def load_ground_truth() -> dict[str, dict]:
    """video_id -> {raw_choices, mapped_activities (top-level), unmapped}."""
    if GROUND_TRUTH_PATH.exists():
        return {r["video_id"]: r for r in json.loads(GROUND_TRUTH_PATH.read_text())}
    tasks = json.loads(LABELED_TASKS_PATH.read_text())
    gt: dict[str, dict] = {}
    for task in tasks:
        raw: set[str] = set()
        for ann in task.get("annotations", []):
            for r in ann.get("result", []):
                if r.get("from_name") == "activity":
                    raw.update(r["value"]["choices"])
        if not raw:
            continue
        mapped, unmapped = set(), set()
        for c in raw:
            if c in SUB_ACTIVITY_TO_ACTIVITY:
                mapped.add(SUB_ACTIVITY_TO_ACTIVITY[c])
            else:
                unmapped.add(c)
        gt[task["data"]["video_id"]] = {
            "video_id": task["data"]["video_id"],
            "title": task["data"].get("title", ""),
            "description": task["data"].get("description", "") or "",
            "raw_choices": sorted(raw),
            "mapped_activities": sorted(mapped),
            "unmapped": sorted(unmapped),
        }
    GROUND_TRUTH_PATH.write_text(json.dumps(sorted(gt.values(), key=lambda r: r["video_id"]), indent=1))
    return gt


_swift_conn = None
_swift_ok: bool | None = None


def _get_swift_conn():
    global _swift_conn, _swift_ok
    if _swift_ok is None:
        from common import allas_utils
        try:
            conn = allas_utils.connect_to_allas()
            conn.get_account()
            _swift_conn, _swift_ok = conn, True
        except Exception:
            _swift_ok = False
    return _swift_conn if _swift_ok else None


def download_shard_bytes(video_id: str) -> bytes | None:
    """{video_id}.tar.gz from Allas, in memory. Tries the project's
    swiftclient path first (works once the .env credentials are valid),
    falls back to the S3 rclone credentials (allas_s3)."""
    conn = _get_swift_conn()
    if conn is not None:
        shard = keyframe_io.download_shard(conn, video_id)
        if shard is not None:
            return shard
    return get_object_bytes(f"{video_id}.tar.gz")


def sample_frames_for_video(video_id: str, n_frames: int = N_SAMPLE_FRAMES):
    """Downloads the video's keyframe shard from Allas and returns
    (shot_records, {frame_name: PIL.Image}) with up to n_frames evenly
    spread across the video's frames. Returns (None, None) if the shard is
    missing (caller skips the video). Never touches disk."""
    shard = download_shard_bytes(video_id)
    if shard is None:
        return None, None
    shot_records, frame_bytes = keyframe_io.read_shard(shard, video_id)
    if not frame_bytes:
        return shot_records, {}
    names = sorted(frame_bytes)
    if len(names) <= n_frames:
        picked = names
    else:
        step = (len(names) - 1) / (n_frames - 1)
        picked = [names[round(i * step)] for i in range(n_frames)]
    images = keyframe_io.decode_images(frame_bytes, picked)
    return shot_records, images


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
