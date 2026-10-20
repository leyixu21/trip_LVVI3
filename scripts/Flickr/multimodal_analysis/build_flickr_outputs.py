"""Merge Flickr Clef shard outputs into final label tables.

Dedupes by photo id (the shard-1/shard-3 slices overlap by 175 ids from the
resubmit cycle; identical records, first wins).

Outputs: data/Flickr/clef_labels/flickr_clef_photos_long.csv
             one row per (id, activity): probability, predicted, ...
         data/Flickr/clef_labels/flickr_clef_photos_wide.csv
             per photo: labels, other_activity_p, primary + joined metadata
"""
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SHARDS = ROOT / "data" / "Flickr" / "clef_labels"
FLICKR_CSV = ROOT / "data" / "Flickr" / "flickr_finland_state_lands.csv"
TAXONOMY = ROOT / "data" / "LVVI3" / "LVVI3_activities.csv"
THRESHOLD = 0.5

by_id = {}
dupes = 0
for f in sorted(glob.glob(str(SHARDS / "flickr_clef.shard_*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r["id"] in by_id:
            dupes += 1
            continue
        by_id[r["id"]] = r
print(f"{len(by_id)} unique photos ({dupes} duplicates dropped)")

taxonomy = sorted(pd.read_csv(TAXONOMY)["activity"].dropna().unique())

rows = []
for pid, r in by_id.items():
    primary = max(r["primary_activity_probs"].items(), key=lambda x: x[1])
    for act in taxonomy:
        p = r["probabilities"].get(act, 0.0)
        rows.append({
            "id": int(pid),
            "activity": act,
            "probability": p,
            "predicted": p >= THRESHOLD,
            "other_activity_p": r["other_activity_p"],
            "primary_activity": primary[0],
            "primary_other_p": r["primary_activity_probs"].get("Other / not listed", 0.0),
        })
long = pd.DataFrame(rows)
long.to_csv(SHARDS / "flickr_clef_photos_long.csv", index=False)
print(f"long: {len(long)} rows")

meta = pd.read_csv(FLICKR_CSV).drop_duplicates("id")
wide_rows = []
for pid, r in by_id.items():
    primary = max(r["primary_activity_probs"].items(), key=lambda x: x[1])
    probs = r["probabilities"]
    top = max(probs.items(), key=lambda x: x[1])
    wide_rows.append({
        "id": int(pid),
        "labels_at_0.5": "; ".join(r["labels_at_0.5"]),
        "n_labels": len(r["labels_at_0.5"]),
        "other_activity_p": r["other_activity_p"],
        "primary_activity": primary[0],
        "primary_other_p": r["primary_activity_probs"].get("Other / not listed", 0.0),
        "top_activity": top[0],
        "top_probability": top[1],
    })
wide = pd.DataFrame(wide_rows).merge(meta, on="id", how="left")
wide.to_csv(SHARDS / "flickr_clef_photos_wide.csv", index=False)
print(f"wide: {len(wide)} photos (metadata matched: {wide['title'].notna().sum()})")
