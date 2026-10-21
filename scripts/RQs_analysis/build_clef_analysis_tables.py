"""Build RQ-analysis activity tables from the clef model labels.

Joins the clef per-video / per-photo activity labels with:
- structural_activity (7 categories) via `data/activities_structural.csv`
  (survey-derived + inferred mappings — review the `source` column there)
- accompany_type from the archived old Qwen3-VL outputs
  (clef labels activities only; accompany was never re-run)

Outputs (long format, one row per unit x activity):
- `data/YouTube/clef_labels/youtube_clef_analysis.csv`
- `data/Flickr/clef_labels/flickr_clef_analysis.csv`

Run from anywhere: /projappl/project_2006782/leyixu/cpu-env/bin/python build_clef_analysis_tables.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
YT_WIDE = ROOT / "data/YouTube/clef_labels/youtube_clef_videos_wide.csv"
FL_WIDE = ROOT / "data/Flickr/clef_labels/flickr_clef_photos_wide.csv"
ARCHIVE_YT = ROOT / "data/_archive/old_llm_subactivity/youtube/YouTube_inferenced.csv"
ARCHIVE_FL = ROOT / "data/_archive/old_llm_subactivity/flickr/Flickr_inferenced.csv"
STRUCT_MAP = ROOT / "data/activities_structural.csv"
OUT_YT = ROOT / "data/YouTube/clef_labels/youtube_clef_analysis.csv"
OUT_FL = ROOT / "data/Flickr/clef_labels/flickr_clef_analysis.csv"

struct_map = pd.read_csv(STRUCT_MAP).set_index("activity")["structural_activity"].to_dict()


def explode_labels(df, id_col):
    df = df[[id_col, "labels_at_0.5"]].dropna(subset=["labels_at_0.5"]).copy()
    df["activity"] = df["labels_at_0.5"].str.split("; ")
    df = df.explode("activity")
    df["structural_activity"] = df["activity"].map(struct_map)
    return df[[id_col, "activity", "structural_activity"]]


def main():
    missing = [str(p) for p in (YT_WIDE, FL_WIDE, ARCHIVE_YT, ARCHIVE_FL, STRUCT_MAP) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing inputs: {missing}")

    # YouTube: video_id x activity + structural + accompany (per video)
    yt = explode_labels(pd.read_csv(YT_WIDE, usecols=["video_id", "labels_at_0.5"]), "video_id")
    yt_acc = pd.read_csv(ARCHIVE_YT, usecols=["video_id", "accompany_type"]).drop_duplicates("video_id")
    yt = yt.merge(yt_acc, on="video_id", how="left")
    unmapped = sorted(set(yt["activity"]) - set(struct_map))
    yt.to_csv(OUT_YT, index=False)
    print(f"YouTube: {yt.shape} -> {OUT_YT}; unmapped activities: {unmapped}")

    # Flickr: image_id x activity + structural + accompany + human_presence (per photo)
    fl = explode_labels(pd.read_csv(FL_WIDE, usecols=["id", "labels_at_0.5"]), "id").rename(columns={"id": "image_id"})
    fl_acc = pd.read_csv(ARCHIVE_FL, usecols=["image_id", "accompany_type"]).drop_duplicates("image_id")
    fl = fl.merge(fl_acc, on="image_id", how="left")
    unmapped = sorted(set(fl["activity"]) - set(struct_map))
    fl.to_csv(OUT_FL, index=False)
    print(f"Flickr: {fl.shape} -> {OUT_FL}; unmapped activities: {unmapped}")


if __name__ == "__main__":
    main()
