"""Export pilot predictions to a manual-check markdown file.

Usage:
  python export_manual_check.py --preds data/YouTube/pilot/pilot_b_clef_v3_k48.jsonl \
      --out data/YouTube/pilot/v3_manual_check.md
"""

import argparse
import json
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parent
GT_PATH = PILOT_DIR.parents[3] / "data" / "YouTube" / "pilot" / "ground_truth.json"
SELECTION_PATH = PILOT_DIR.parents[3] / "data" / "YouTube" / "pilot" / "video_selection.json"
V2_PATH = PILOT_DIR.parents[3] / "data" / "YouTube" / "pilot" / "pilot_b_clef_v2.jsonl"

THRESHOLD = 0.5
TOP_N = 6


def fmt_acts(acts):
    return ", ".join(sorted(acts)) if acts else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Manual check")
    args = ap.parse_args()

    gt = {g["video_id"]: g for g in json.loads(GT_PATH.read_text())}
    sel = {v["video_id"]: v for v in json.loads(SELECTION_PATH.read_text())["selected"]}
    v2 = {}
    if V2_PATH.exists():
        v2 = {json.loads(l)["video_id"]: json.loads(l) for l in V2_PATH.read_text().splitlines() if l.strip()}
    rows = [json.loads(l) for l in Path(args.preds).read_text().splitlines() if l.strip()]

    L = [f"# {args.title}", ""]
    L.append(f"Threshold ≥{THRESHOLD} for labels. GT = mapped manual annotations. "
             f"✓ = predicted label is in GT, ✗ = GT label missed, ! = predicted but not in GT.")
    L.append("")

    # ---- summary table ----
    L.append("## Summary (all videos)")
    L.append("")
    L.append("| # | Video | Title | GT | Predicted (≥%.1f) | Match |" % THRESHOLD)
    L.append("|---|-------|-------|----|--------------------|-------|")
    for i, r in enumerate(rows, 1):
        vid = r["video_id"]
        g = gt.get(vid, {})
        gt_acts = set(g.get("mapped_activities") or [])
        pred = set(r["labels_at_0.5"])
        tp = gt_acts & pred
        status = "✓ exact" if pred == gt_acts and gt_acts else f"{len(tp)}/{len(gt_acts)}"
        title = (sel.get(vid, {}).get("title") or "?").replace("|", "/")[:48]
        L.append(f"| {i} | `{vid}` | {title} | {fmt_acts(gt_acts)} | {fmt_acts(pred)} | {status} |")
    L.append("")

    # ---- per-video cards ----
    L.append("## Details")
    L.append("")
    for i, r in enumerate(rows, 1):
        vid = r["video_id"]
        g = gt.get(vid, {})
        gt_acts = set(g.get("mapped_activities") or [])
        unmapped = g.get("unmapped") or []
        pred = set(r["labels_at_0.5"])
        title = (sel.get(vid, {}).get("title") or "?").replace("|", "/")
        top = sorted(r["probabilities"].items(), key=lambda x: -x[1])[:TOP_N]

        L.append(f"### {i}. {title}")
        L.append(f"`{vid}` — frames={r['n_frames']}, tokens={r['n_state_tokens']}, "
                 f"{r['forward_s']}s | **other_p={r['other_activity_p']}**, "
                 f"primary_other_p={r['primary_activity_probs'].get('Other / not listed', 0)}")
        L.append("")
        L.append(f"**GT:** {fmt_acts(gt_acts)}" + (f" · *unmapped:* {', '.join(unmapped)}" if unmapped else ""))
        L.append("")
        L.append("| Activity | P | vs GT |")
        L.append("|----------|---|-------|")
        shown = set()
        for act, p in top:
            mark = "✓" if act in gt_acts else "!"
            L.append(f"| {act} | {p:.2f} | {mark} |")
            shown.add(act)
        for act in sorted(gt_acts - shown):
            L.append(f"| {act} | {r['probabilities'].get(act, 0):.2f} | ✗ missed |")
        L.append("")
        if vid in v2:
            L.append(f"*v2 (32 frames, creator prompt):* {fmt_acts(set(v2[vid]['labels_at_0.5']))}")
            L.append("")

    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out} ({len(rows)} videos)")


if __name__ == "__main__":
    main()
