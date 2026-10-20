"""Evaluate pilot B (clef probabilities) and pilot D (qwen3-vl label lists)
against the mapped ground truth, and extrapolate runtimes to the full
3,763-video scope.

Metrics:
  - label-level micro/macro precision, recall, F1 (multi-label, per video)
  - per-video exact-subset accuracy (predicted set == ground-truth set)
    * B: probability-threshold sweep (0.05..0.95) + at 0.5
    * D: fixed (temperature-0 list)
  - B top-k accuracy: fraction of videos where every GT label is in the
    top-k probabilities
  - B calibration: mean P(true) on GT-positives vs GT-negatives, and the
    rank separation (Mann-Whitney-style AUC)
  - runtime: mean per-video forward_s per pilot; extrapolation to 3,763
    videos on 1/2/4 GPUs (assuming linear scaling, batches saturate 1 GPU)

Writes data/YouTube/pilot/metrics.json and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from common_pilot import (
    GROUND_TRUTH_PATH,
    PILOT_B_OUTPUT,
    PILOT_D_OUTPUT,
    SELECTION_PATH,
    load_ground_truth,
    load_taxonomy,
    read_jsonl,
)

N_FULL_SCOPE = 3763


def prf1(gold_sets: list[set], pred_sets: list[set]) -> dict:
    """Micro and macro precision/recall/F1 over label instances."""
    tp = fp = fn = 0
    per_label: dict[str, dict[str, int]] = {}
    for gold, pred in zip(gold_sets, pred_sets):
        for a in gold & pred:
            tp += 1
            per_label.setdefault(a, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        for a in pred - gold:
            fp += 1
            per_label.setdefault(a, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
        for a in gold - pred:
            fn += 1
            per_label.setdefault(a, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    f1s, ps, rs = [], [], []
    for a, c in per_label.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        ps.append(p); rs.append(r); f1s.append(f)
    return {
        "micro_precision": round(micro_p, 4), "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(statistics.mean(ps), 4) if ps else 0.0,
        "macro_recall": round(statistics.mean(rs), 4) if rs else 0.0,
        "macro_f1": round(statistics.mean(f1s), 4) if f1s else 0.0,
    }


def exact_subset_accuracy(gold_sets: list[set], pred_sets: list[set]) -> float:
    return round(sum(1 for g, p in zip(gold_sets, pred_sets) if g == p) / len(gold_sets), 4) if gold_sets else 0.0


def topk_full_coverage(gold_sets: list[set], prob_lists: list[list[str]], k: int) -> float:
    """Fraction of videos whose entire GT set is contained in the model's
    top-k predicted activities."""
    n = 0
    for gold, ranked in zip(gold_sets, prob_lists):
        if gold and set(ranked[:k]) >= gold:
            n += 1
    return round(n / len(gold_sets), 4) if gold_sets else 0.0


def rank_auc(positives: list[float], negatives: list[float]) -> float:
    """P(pos > neg) over all pairs (probabilistic separation / AUC)."""
    if not positives or not negatives:
        return float("nan")
    wins = total = 0
    for p in positives:
        for n in negatives:
            total += 1
            wins += (p > n) + 0.5 * (p == n)
    return round(wins / total, 4)


def runtime_stats(records: list[dict], key: str = "forward_s") -> dict:
    vals = [r[key] for r in records if key in r]
    if not vals:
        return {}
    mean = statistics.mean(vals)
    out = {
        "n": len(vals),
        "mean_s": round(mean, 3),
        "median_s": round(statistics.median(vals), 3),
        "total_min": round(sum(vals) / 60, 1),
        "full_scope_gpu_hours": round(mean * N_FULL_SCOPE / 3600, 1),
    }
    for gpus in (1, 2, 4):
        out[f"full_scope_wall_hours_{gpus}gpu"] = round(mean * N_FULL_SCOPE / 3600 / gpus, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b_output", default=str(PILOT_B_OUTPUT))
    ap.add_argument("--d_output", default=str(PILOT_D_OUTPUT))
    ap.add_argument("--out", default=str(PILOT_B_OUTPUT.parent / "metrics.json"))
    args = ap.parse_args()

    selection = json.loads(Path(SELECTION_PATH).read_text())
    selected_ids = [v["video_id"] for v in selection["selected"]]
    gt_all = load_ground_truth()
    taxonomy = load_taxonomy()

    b_records = {r["video_id"]: r for r in read_jsonl(Path(args.b_output))}
    d_records = {r["video_id"]: r for r in read_jsonl(Path(args.d_output))}

    gold = {vid: set(gt_all[vid]["mapped_activities"]) for vid in selected_ids if vid in gt_all}

    metrics: dict = {"n_selected": len(selected_ids), "n_evaluated": len(gold), "b": {}, "d": {}}

    # ---------------- Pilot B ----------------
    b_gold, b_probs = [], []
    b_meta = []
    for vid in selected_ids:
        if vid in gold and vid in b_records:
            b_gold.append(gold[vid])
            b_probs.append(b_records[vid]["probabilities"])
            b_meta.append(b_records[vid])
    if b_gold:
        sweep = []
        best = None
        for t10 in range(5, 96, 5):
            t = t10 / 100
            preds = [{a for a, p in probs.items() if p >= t} for probs in b_probs]
            m = prf1(b_gold, preds)
            acc = exact_subset_accuracy(b_gold, preds)
            row = {"threshold": t, **m, "exact_subset_accuracy": acc}
            sweep.append(row)
            if best is None or m["micro_f1"] > best["micro_f1"]:
                best = row
        ranked = [sorted(probs, key=probs.get, reverse=True) for probs in b_probs]
        pos = [p[a] for probs, g in zip(b_probs, b_gold) for a in g for p in [probs]]
        neg = [p[a] for probs, g in zip(b_probs, b_gold) for a in taxonomy if a not in g for p in [probs]]
        metrics["b"] = {
            "n_videos": len(b_gold),
            "threshold_sweep": sweep,
            "best_f1_threshold": best,
            "at_0.5": next(r for r in sweep if r["threshold"] == 0.5),
            "topk_full_coverage": {f"top{k}": topk_full_coverage(b_gold, ranked, k) for k in (1, 2, 3)},
            "calibration": {
                "mean_p_on_gt_positives": round(statistics.mean(pos), 4) if pos else None,
                "mean_p_on_gt_negatives": round(statistics.mean(neg), 4) if neg else None,
                "pairwise_rank_auc": rank_auc(pos, neg),
                "n_pos_instances": len(pos),
                "n_neg_instances": len(neg),
            },
            "runtime": runtime_stats(b_meta),
            "download_time": runtime_stats(b_meta, "download_s"),
            "n_frames": [r.get("n_frames") for r in b_meta],
        }

    # ---------------- Pilot D ----------------
    d_gold, d_preds, d_meta = [], [], []
    unmatched_total = 0
    for vid in selected_ids:
        if vid in gold and vid in d_records:
            d_gold.append(gold[vid])
            d_preds.append(set(d_records[vid]["labels"]))
            d_meta.append(d_records[vid])
            unmatched_total += len(d_records[vid].get("unmatched_names", []))
    if d_gold:
        metrics["d"] = {
            "n_videos": len(d_gold),
            "prf1": prf1(d_gold, d_preds),
            "exact_subset_accuracy": exact_subset_accuracy(d_gold, d_preds),
            "runtime": runtime_stats(d_meta),
            "download_time": runtime_stats(d_meta, "download_s"),
            "n_unmatched_names_total": unmatched_total,
            "n_frames": [r.get("n_frames") for r in d_meta],
        }

    Path(args.out).write_text(json.dumps(metrics, indent=1))

    # ---------------- summary print ----------------
    def fmt(d, keys):
        return "  ".join(f"{k}={d.get(k)}" for k in keys)

    print(f"evaluated {len(gold)} videos")
    if metrics["b"]:
        b = metrics["b"]
        print("\n[Pilot B: clef]")
        print("  @0.5      ", fmt(b["at_0.5"], ["micro_precision", "micro_recall", "micro_f1", "macro_f1", "exact_subset_accuracy"]))
        print("  @best F1  ", fmt(b["best_f1_threshold"], ["threshold", "micro_precision", "micro_recall", "micro_f1", "macro_f1", "exact_subset_accuracy"]))
        print("  top-k full coverage:", b["topk_full_coverage"])
        print("  calibration:", json.dumps(b["calibration"]))
        print("  runtime:", json.dumps(b["runtime"]))
    if metrics["d"]:
        d = metrics["d"]
        print("\n[Pilot D: qwen3-vl-32b]")
        print("  ", fmt(d["prf1"], ["micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1"]))
        print("  exact_subset_accuracy=", d["exact_subset_accuracy"], " unmatched_names=", d["n_unmatched_names_total"])
        print("  runtime:", json.dumps(d["runtime"]))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
