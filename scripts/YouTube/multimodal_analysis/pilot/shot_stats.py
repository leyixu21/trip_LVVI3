"""Shot-count statistics for the 58 pilot videos to inform the Clef
frame-budget decision for the full run.

Downloads each pilot video's keyframe shard from Allas via the pilot's S3
fallback (pilot/allas_s3.py -- the .env OpenStack credentials are expired),
reads ONLY {video_id}/shots.json (plus the frame_*.jpg member names) from
the in-memory tar, and joins with video duration from youtube_videos.csv.

Writes data/YouTube/pilot/shot_stats.json. No model runs, nothing to disk
besides the JSON output.
"""
from __future__ import annotations

import io
import json
import statistics
import tarfile
from pathlib import Path

from allas_s3 import get_object_bytes
from common_pilot import PILOT_DATA_DIR, SELECTION_PATH


def read_shots_and_frame_count(shard: bytes, video_id: str) -> tuple[int, int] | None:
    """Returns (n_shots, n_keyframes) reading only shots.json + member names."""
    shots = None
    n_frames = 0
    with tarfile.open(fileobj=io.BytesIO(shard), mode="r:gz") as tar:
        for member in tar:
            name = member.name
            if name.endswith("shots.json"):
                shots = json.loads(tar.extractfile(member).read())
            elif name.endswith(".jpg"):
                n_frames += 1
    if shots is None:
        return None
    return len(shots), n_frames


def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, round((len(sorted_vals) - 1) * q)))
    return sorted_vals[i]


def dist(vals: list[float]) -> dict:
    s = sorted(vals)
    return {
        "n": len(vals),
        "min": round(s[0], 2),
        "p25": round(pct(s, 0.25), 2),
        "median": round(pct(s, 0.5), 2),
        "mean": round(statistics.mean(s), 2),
        "p75": round(pct(s, 0.75), 2),
        "p90": round(pct(s, 0.90), 2),
        "max": round(s[-1], 2),
    }


def main():
    selection = json.loads(Path(SELECTION_PATH).read_text())
    videos = selection["selected"] + selection.get("spares", [])
    vids = [v["video_id"] for v in videos]
    print(f"{len(vids)} pilot videos")

    # durations from the master video table (30,615 rows, dedupe video_id)
    import csv
    durations: dict[str, float] = {}
    with open("/scratch/project_2006782/leyixu/trip_LVVI3/data/YouTube/youtube_videos.csv", newline="") as f:
        for row in csv.DictReader(f):
            vid = row.get("video_id")
            if not vid or vid in durations:
                continue
            try:
                durations[vid] = float(row["duration_seconds"])
            except (TypeError, ValueError, KeyError):
                pass

    rows = []
    missing = []
    for vid in vids:
        shard = get_object_bytes(f"{vid}.tar.gz")
        if shard is None:
            missing.append(vid)
            continue
        got = read_shots_and_frame_count(shard, vid)
        if got is None:
            missing.append(vid)
            continue
        n_shots, n_frames = got
        dur = durations.get(vid)
        rows.append({
            "video_id": vid,
            "n_shots": n_shots,
            "n_keyframes": n_frames,
            "duration_s": dur,
            "shots_per_min": round(n_shots / (dur / 60), 2) if dur and dur > 0 else None,
        })

    n = len(rows)
    shots = [r["n_shots"] for r in rows]
    frames = [r["n_keyframes"] for r in rows]
    spm = [r["shots_per_min"] for r in rows if r["shots_per_min"] is not None]

    over = {">16": sum(1 for s in shots if s > 16),
            ">24": sum(1 for s in shots if s > 24),
            ">32": sum(1 for s in shots if s > 32),
            ">64": sum(1 for s in shots if s > 64)}

    # shots-per-minute vs duration: binned table
    bins = [(0, 60), (60, 180), (180, 600), (600, 1200), (1200, 1e9)]
    bin_rows = []
    for lo, hi in bins:
        sub = [r for r in rows if r["duration_s"] and lo <= r["duration_s"] < hi]
        sub_spm = [r["shots_per_min"] for r in sub if r["shots_per_min"] is not None]
        bin_rows.append({
            "duration_bin_s": [lo, hi if hi < 1e9 else None],
            "n_videos": len(sub),
            "mean_shots": round(statistics.mean([r["n_shots"] for r in sub]), 1) if sub else None,
            "median_shots": statistics.median([r["n_shots"] for r in sub]) if sub else None,
            "mean_shots_per_min": round(statistics.mean(sub_spm), 2) if sub_spm else None,
            "median_shots_per_min": round(statistics.median(sub_spm), 2) if sub_spm else None,
        })

    out = {
        "n_pilot_videos": len(vids),
        "n_with_shard": n,
        "missing_shards": missing,
        "n_with_duration": sum(1 for r in rows if r["duration_s"]),
        "shots_per_video": dist(shots),
        "keyframes_per_video": dist(frames),
        "shots_per_minute": dist(spm),
        "videos_with_shots_over": {k: {"n": v, "pct": round(v / n, 3)} for k, v in over.items()},
        "duration_vs_shots_bins": bin_rows,
        "per_video": rows,
    }
    (PILOT_DATA_DIR / "shot_stats.json").write_text(json.dumps(out, indent=1))

    print(json.dumps({k: out[k] for k in ("shots_per_video", "keyframes_per_video",
                                          "shots_per_minute", "videos_with_shots_over")}, indent=1))
    for b in bin_rows:
        print(b)
    print("missing:", missing)


if __name__ == "__main__":
    main()
