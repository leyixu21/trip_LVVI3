"""Selects the pilot's 50 videos (+ spares) from the 227 manually annotated
tasks and verifies each candidate's keyframe shard exists in Allas.

Selection pool: tasks with an `activity` annotation whose video_id also has
a transcript in data/YouTube/transcripts.jsonl. Selection is stratified by
mapped top-level activity: we round-robin over activities (sorted by how
rare they are) so the 50 picked videos cover as many of the 22 activities
as possible, then top up with spares.

Outputs (under data/YouTube/pilot/):
  video_selection.json  -- {selected: [...], spares: [...], skipped: {...}}
  mapping_report.json   -- annotation name -> mapping decision + closest CSV
                           sub_activity by edit distance (documentation for
                           REPORT.md)
"""
from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher

from common_pilot import (
    GROUND_TRUTH_PATH,
    MAPPING_REPORT_PATH,
    SELECTION_PATH,
    load_ground_truth,
    load_taxonomy,
    load_transcripts_index,
)
from label_mapping import EXPECTED_UNMAPPED, SUB_ACTIVITY_TO_ACTIVITY

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import allas_utils, config  # noqa: E402


def closest_csv_subactivity(name: str) -> str:
    import csv
    rows = list(csv.DictReader(open(config.TAXONOMY_ACTIVITIES_PATH, newline="")))
    subs = sorted({r["sub_activity"] for r in rows})
    best = max(subs, key=lambda s: SequenceMatcher(None, name.lower(), s.lower()).ratio())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_selected", type=int, default=50)
    ap.add_argument("--n_spares", type=int, default=8)
    args = ap.parse_args()

    gt = load_ground_truth()
    transcripts = load_transcripts_index()
    taxonomy = load_taxonomy()

    # ---- sanity: every raw annotation name is accounted for ----
    all_raw = sorted({c for r in gt.values() for c in r["raw_choices"]})
    unaccounted = [c for c in all_raw if c not in SUB_ACTIVITY_TO_ACTIVITY and c not in EXPECTED_UNMAPPED]
    assert not unaccounted, f"annotation names missing from mapping: {unaccounted}"

    # ---- mapping report (for REPORT.md) ----
    report = {
        "n_annotation_names": len(all_raw),
        "mapped": {c: {"top_level_activity": SUB_ACTIVITY_TO_ACTIVITY[c],
                       "closest_csv_sub_activity": closest_csv_subactivity(c)}
                   for c in all_raw if c in SUB_ACTIVITY_TO_ACTIVITY},
        "unmapped": {c: {"occurrences": sum(1 for r in gt.values() if c in r["raw_choices"]),
                         "closest_csv_sub_activity": closest_csv_subactivity(c)}
                     for c in all_raw if c in EXPECTED_UNMAPPED},
        "taxonomy_activities": taxonomy,
    }
    MAPPING_REPORT_PATH.write_text(json.dumps(report, indent=1))

    # ---- pool ----
    pool = [r for r in gt.values()
            if r["video_id"] in transcripts and r["mapped_activities"]]
    print(f"annotated tasks: {len(gt)}, with transcript + >=1 mapped label: {len(pool)}")

    # stratified round-robin: rare activities first
    by_activity: dict[str, list[dict]] = {}
    for r in pool:
        for act in r["mapped_activities"]:
            by_activity.setdefault(act, []).append(r)
    order = sorted(by_activity, key=lambda a: len(by_activity[a]))
    print("activity coverage in pool:")
    for a in order:
        print(f"  {a}: {len(by_activity[a])}")

    # ---- check Allas keyframe shards for all pool candidates ----
    import allas_s3
    from common_pilot import download_shard_bytes
    have_shard: dict[str, bool] = {}
    for r in pool:
        vid = r["video_id"]
        # cheap HEAD first; fall back to a real download if HEAD is denied
        have_shard[vid] = allas_s3.object_exists(f"{vid}.tar.gz") or download_shard_bytes(vid) is not None
    missing = [r["video_id"] for r in pool if not have_shard[r["video_id"]]]
    print(f"pool videos with keyframe shard: {sum(have_shard.values())}/{len(pool)}; missing: {missing}")

    eligible = [r for r in pool if have_shard[r["video_id"]]]
    primary = {}
    for r in eligible:
        primary[r["video_id"]] = min(r["mapped_activities"],
                                     key=lambda a: (len(by_activity[a]), a))
    eligible.sort(key=lambda r: (len(by_activity[primary[r["video_id"]]]), primary[r["video_id"]], r["video_id"]))

    def pick(n, exclude=()):
        picked, covered, excluded = [], set(), set(exclude)
        # pass 1: maximize activity coverage
        for r in eligible:
            if len(picked) >= n:
                break
            if r["video_id"] in excluded:
                continue
            new = set(r["mapped_activities"]) - covered
            if new:
                picked.append(r)
                covered.update(r["mapped_activities"])
        # pass 2: fill up respecting rarity order
        for r in eligible:
            if len(picked) >= n:
                break
            if r["video_id"] in excluded or r in picked:
                continue
            picked.append(r)
        return picked

    selected = pick(args.n_selected)
    spare_pool = pick(args.n_selected + args.n_spares)
    spares = [r for r in spare_pool if r not in selected][:args.n_spares]

    def slim(r):
        tr = transcripts[r["video_id"]]
        return {
            "video_id": r["video_id"],
            "title": r["title"],
            "description": r["description"][:300],
            "language": tr.get("language"),
            "mapped_activities": r["mapped_activities"],
            "raw_choices": r["raw_choices"],
            "n_transcript_segments": len(tr.get("segments", [])),
        }

    out = {
        "n_selected": len(selected),
        "n_spares": len(spares),
        "selected": [slim(r) for r in selected],
        "spares": [slim(r) for r in spares],
        "skipped_missing_shard": missing,
        "skipped_no_transcript": sorted(set(gt) - set(transcripts)),
        "skipped_unmapped_only": sorted(v for v, r in gt.items()
                                        if r["video_id"] in transcripts and not r["mapped_activities"]),
    }
    SELECTION_PATH.write_text(json.dumps(out, indent=1))
    sel_acts = sorted({a for r in selected for a in r["mapped_activities"]})
    print(f"\nselected {len(selected)} videos covering {len(sel_acts)}/{len(taxonomy)} activities")
    print(f"spares: {len(spares)}; selection -> {SELECTION_PATH}")
    print(f"ground truth -> {GROUND_TRUTH_PATH}")
    print(f"mapping report -> {MAPPING_REPORT_PATH}")


if __name__ == "__main__":
    main()
