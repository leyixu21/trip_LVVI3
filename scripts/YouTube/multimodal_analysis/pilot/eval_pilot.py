"""Evaluate pilot activity predictions against manual ground truth.

Supports two prediction formats (per video_id):
  - Pilot B (Clef): {"probabilities": {activity: p, ...}, "forward_s": ...}
  - Pilot D (VLM):  {"labels": [activity, ...]}  (or "predicted_labels")

Usage:
  python eval_pilot.py --preds pilot_b_clef.jsonl --out eval_b.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

GT_PATH = Path(__file__).resolve().parents[4] / "data" / "YouTube" / "pilot" / "ground_truth.json"
FULL_SCOPE_VIDEOS = 3763


def load_gt():
    gt = {}
    for g in json.loads(GT_PATH.read_text()):
        acts = set(g.get("mapped_activities") or [])
        if acts:
            gt[g["video_id"]] = acts
    return gt


def load_preds(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        vid = r["video_id"]
        if "probabilities" in r:
            out[vid] = {"probs": r["probabilities"], "meta": r}
        else:
            labels = r.get("labels") or r.get("predicted_labels") or []
            out[vid] = {"labels": set(labels), "meta": r}
    return out


def metrics(gt_sets, pred_sets, taxonomy):
    tp = fp = fn = 0
    per_act = {a: {"tp": 0, "fp": 0, "fn": 0} for a in taxonomy}
    exact = 0
    for vid, gt in gt_sets.items():
        pred = pred_sets.get(vid, set())
        if pred == gt:
            exact += 1
        for a in taxonomy:
            g, p = a in gt, a in pred
            if g and p:
                tp += 1; per_act[a]["tp"] += 1
            elif p and not g:
                fp += 1; per_act[a]["fp"] += 1
            elif g and not p:
                fn += 1; per_act[a]["fn"] += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f1s = []
    for a in taxonomy:
        d = per_act[a]
        p = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
        r = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
        if p + r:
            f1s.append((a, 2 * p * r / (p + r), d))
    macro = sum(f for _, f, _ in f1s) / len(f1s) if f1s else 0.0
    return {
        "n_videos_evaluated": len(gt_sets),
        "micro_precision": round(prec, 4),
        "micro_recall": round(rec, 4),
        "micro_f1": round(f1, 4),
        "macro_f1": round(macro, 4),
        "exact_subset_accuracy": round(exact / len(gt_sets), 4) if gt_sets else 0.0,
        "per_activity_f1": {a: round(f, 3) for a, f, _ in sorted(f1s)},
        "per_activity_support": {a: d["tp"] + d["fn"] for a, _, d in sorted(f1s)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = load_gt()
    preds = load_preds(args.preds)
    gt = {vid: acts for vid, acts in gt.items() if vid in preds}
    taxonomy = sorted({a for acts in gt.values() for a in acts} |
                      {a for p in preds.values() for a in (p.get("probs") or p.get("labels") or set())})

    report = {"preds_file": args.preds, "n_gt_videos": len(gt), "n_pred_videos": len(preds)}
    metas = [p["meta"] for p in preds.values()]
    for key in ("forward_s", "download_s", "n_frames", "n_state_tokens"):
        vals = [m[key] for m in metas if key in m]
        if vals:
            report[f"mean_{key}"] = round(sum(vals) / len(vals), 3)
    if "mean_forward_s" in report:
        per_gpu = report["mean_forward_s"] * FULL_SCOPE_VIDEOS / 3600
        report["full_scope_hours_1gpu"] = round(per_gpu, 1)
        report["full_scope_hours_4gpu"] = round(per_gpu / 4, 1)

    if "probs" in next(iter(preds.values()), {}):
        sweep = {}
        best = None
        for t in [x / 100 for x in range(5, 96, 5)]:
            pred_sets = {vid: {a for a, p in p["probs"].items() if p >= t} for vid, p in preds.items()}
            m = metrics(gt, pred_sets, taxonomy)
            sweep[str(t)] = m
            if best is None or m["micro_f1"] > sweep[best]["micro_f1"]:
                best = str(t)
        report["threshold_sweep"] = sweep
        report["best_threshold"] = best
        report["metrics_at_best"] = sweep[best]
    else:
        pred_sets = {vid: p["labels"] for vid, p in preds.items()}
        report["metrics"] = metrics(gt, pred_sets, taxonomy)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "threshold_sweep"}, indent=2))


if __name__ == "__main__":
    main()
