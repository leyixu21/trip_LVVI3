"""Cluster the raw new-activity names from the naming pass.

Embeds distinct names (multilingual MiniLM, same model as the project's RAG
index), then agglomerative clustering on cosine distance. Output is a review
table: one row per cluster with the most frequent member as the candidate
label, member variants, video counts, and example videos.

Input : data/YouTube/clef_labels/new_activity_names.jsonl
Output: data/YouTube/clef_labels/new_activity_clusters.csv

Usage: python cluster_new_activities.py [--distance 0.25]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
IN = ROOT / "data" / "YouTube" / "clef_labels" / "new_activity_names.jsonl"
OUT = ROOT / "data" / "YouTube" / "clef_labels" / "new_activity_clusters.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=float, default=0.25,
                    help="agglomerative clustering distance threshold (cosine)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in IN.read_text().splitlines() if l.strip()]
    named = [r for r in rows if not r["error"] and r["new_activity"].strip()
             and r["new_activity"].strip().lower() != "none"]
    print(f"{len(named)} named rows")

    # canonical surface form: lowercase, strip trailing punctuation/periods
    def canon(s: str) -> str:
        return s.strip().lower().rstrip(".").strip()

    freq = Counter()
    examples = defaultdict(list)
    for r in named:
        c = canon(r["new_activity"])
        freq[c] += 1
        if len(examples[c]) < 3:
            examples[c].append(r["video_id"])

    names = sorted(freq, key=lambda n: -freq[n])
    print(f"{len(names)} distinct names")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb = model.encode(names, normalize_embeddings=True)
    sim = emb @ emb.T
    dist = 1.0 - sim

    from sklearn.cluster import AgglomerativeClustering
    cl = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                 linkage="average", distance_threshold=args.distance)
    labels = cl.fit_predict(dist)

    clusters = defaultdict(list)
    for name, lab in zip(names, labels):
        clusters[lab].append(name)

    out_rows = []
    for lab, members in sorted(clusters.items(), key=lambda kv: -sum(freq[m] for m in kv[1])):
        rep = max(members, key=lambda m: freq[m])
        total = sum(freq[m] for m in members)
        vids = []
        for m in sorted(members, key=lambda m: -freq[m]):
            vids.extend(examples[m])
        out_rows.append({
            "cluster_id": lab,
            "candidate_label": rep,
            "n_videos": total,
            "n_variants": len(members),
            "variants": "; ".join(f"{m} ({freq[m]})" for m in sorted(members, key=lambda m: -freq[m])),
            "example_videos": "; ".join(vids[:5]),
        })
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} clusters -> {OUT}")
    print()
    print(df[["candidate_label", "n_videos", "n_variants"]].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
