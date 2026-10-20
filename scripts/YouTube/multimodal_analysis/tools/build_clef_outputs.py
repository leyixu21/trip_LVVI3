"""Build the final YouTube Clef activity-label tables.

Inputs : data/YouTube/clef_labels/clef_full.shard_*.jsonl (per-video Clef output)
         data/YouTube/youtube_videos.csv (metadata; one row per video x area)
Outputs: data/YouTube/clef_labels/youtube_clef_activities_long.csv
             one row per (video_id, activity): probability, predicted flag,
             plus per-video other_activity_p / primary activity columns
         data/YouTube/clef_labels/youtube_clef_videos_wide.csv
             one row per unique video: labels_at_0.5, other_activity_p,
             primary activity, top activity/probability + joined metadata
         data/activities.csv
             master activity taxonomy: the 35 LVVI3 survey activities with
             source='survey'; 'social media' rows are appended after the
             new-activity naming pass.

Usage: python build_clef_outputs.py
"""
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SHARDS = ROOT / "data" / "YouTube" / "clef_labels"
TAXONOMY = ROOT / "data" / "LVVI3" / "LVVI3_activities.csv"
VIDEOS_CSV = ROOT / "data" / "YouTube" / "youtube_videos.csv"
THRESHOLD = 0.5


def load_records():
    recs = []
    for f in sorted(glob.glob(str(SHARDS / "clef_full.shard_*.jsonl"))):
        for line in open(f):
            recs.append(json.loads(line))
    by_id = {r["video_id"]: r for r in recs}
    return by_id


def main():
    by_id = load_records()
    print(f"{len(by_id)} videos loaded from shards")

    taxonomy = sorted(pd.read_csv(TAXONOMY)["activity"].dropna().unique())

    # ---- long table ----
    rows = []
    for vid, r in by_id.items():
        primary = max(r["primary_activity_probs"].items(), key=lambda x: x[1])
        for act in taxonomy:
            p = r["probabilities"].get(act, 0.0)
            rows.append({
                "video_id": vid,
                "activity": act,
                "probability": p,
                "predicted": p >= THRESHOLD,
                "other_activity_p": r["other_activity_p"],
                "primary_activity": primary[0],
                "primary_other_p": r["primary_activity_probs"].get("Other / not listed", 0.0),
            })
    long = pd.DataFrame(rows)
    long.to_csv(SHARDS / "youtube_clef_activities_long.csv", index=False)
    print(f"long table: {len(long)} rows -> youtube_clef_activities_long.csv")

    # ---- wide table with metadata (unique video grain) ----
    meta = pd.read_csv(VIDEOS_CSV).drop_duplicates("video_id")
    wide_rows = []
    for vid, r in by_id.items():
        primary = max(r["primary_activity_probs"].items(), key=lambda x: x[1])
        probs = r["probabilities"]
        top = max(probs.items(), key=lambda x: x[1])
        wide_rows.append({
            "video_id": vid,
            "labels_at_0.5": "; ".join(r["labels_at_0.5"]),
            "n_labels": len(r["labels_at_0.5"]),
            "other_activity_p": r["other_activity_p"],
            "primary_activity": primary[0],
            "primary_other_p": r["primary_activity_probs"].get("Other / not listed", 0.0),
            "top_activity": top[0],
            "top_probability": top[1],
            "n_frames": r["n_frames"],
        })
    wide = pd.DataFrame(wide_rows).merge(meta, on="video_id", how="left")
    wide.to_csv(SHARDS / "youtube_clef_videos_wide.csv", index=False)
    print(f"wide table: {len(wide)} videos -> youtube_clef_videos_wide.csv")

    # ---- master taxonomy ----
    acts = pd.DataFrame({"activity": taxonomy})
    acts["source"] = "survey"
    acts["status"] = "active"
    acts["notes"] = ""
    out = ROOT / "data" / "activities.csv"
    acts.to_csv(out, index=False)
    print(f"master taxonomy: {len(acts)} activities (all source='survey') -> data/activities.csv")


if __name__ == "__main__":
    main()
