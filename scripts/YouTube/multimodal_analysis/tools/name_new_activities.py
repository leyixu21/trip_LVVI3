"""Name the new (out-of-taxonomy) activities detected by Clef.

Stage 1 of new-activity discovery: for every video with other_activity_p
>= threshold, ask Poro-2-70B (via CSC Aitta, text-only) to name the
uncovered recreational activity from title/description/transcript, with an
evidence snippet and confidence. Vision escalation for low-confidence rows
is a separate later step.

Input : data/YouTube/clef_labels/youtube_clef_videos_wide.csv
        data/YouTube/transcripts.jsonl
Output: data/YouTube/clef_labels/new_activity_names.jsonl
        {video_id, other_activity_p, response, new_activity, evidence, confidence, error}

Runs from the trip_LVVI3 root (aitta token comes from .env via llm_chat).
Usage: python name_new_activities.py [--threshold 0.5] [--limit 0]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts" / "YouTube" / "multimodal_analysis"))

# Aitta token: trip_LVVI3/.env does not carry one; fall back to the sibling
# project's .env (same user, same CSC project).
from dotenv import load_dotenv  # noqa: E402
import os  # noqa: E402

load_dotenv(ROOT / ".env")
if not os.environ.get("AITTA_API_TOKEN"):
    load_dotenv("/scratch/project_2006782/leyixu/influencer_YouTube/.env")

from common import config, llm_chat  # noqa: E402

WIDE = ROOT / "data" / "YouTube" / "clef_labels" / "youtube_clef_videos_wide.csv"
TRANSCRIPTS = ROOT / "data" / "YouTube" / "transcripts.jsonl"
OUT = ROOT / "data" / "YouTube" / "clef_labels" / "new_activity_names.jsonl"

TRANSCRIPT_MAX_CHARS = 3000
DESCRIPTION_MAX_CHARS = 500

SYSTEM_PROMPT = """You are annotating outdoor-recreation videos. A fixed activity list was offered:
{activities}

The video contains a recreational activity that is NOT adequately covered by any of those categories. Based on the video title, description, and transcript, name that uncovered activity: a concise English activity phrase of 1-5 words (e.g. "sauna", "husky safari", "northern lights watching", "geocaching"). If several uncovered activities occur, name the most prominent one. If there is genuinely nothing, answer "none".

Respond with ONLY a JSON object: {{"new_activity": "...", "evidence": "short quote or concrete paraphrase from the text", "confidence": "high|medium|low"}}"""


def transcript_text(record: dict) -> str:
    parts = []
    total = 0
    for seg in record.get("segments", []):
        t = (seg.get("text") or "").strip()
        if not t:
            continue
        line = f"[{int(seg.get('start', 0))}] {t}"
        if total + len(line) > TRANSCRIPT_MAX_CHARS:
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)


def parse_json_object(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    wide = pd.read_csv(WIDE)
    candidates = wide[wide["other_activity_p"] >= args.threshold]
    if args.limit:
        candidates = candidates.head(args.limit)
    print(f"{len(candidates)} candidate videos (other_activity_p >= {args.threshold})", flush=True)

    tr_idx = {}
    with open(TRANSCRIPTS) as f:
        for line in f:
            rec = json.loads(line)
            tr_idx[rec["video_id"]] = rec

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["video_id"])
    out_f = OUT.open("a")
    todo = [v for v in candidates["video_id"] if v not in done]
    print(f"{len(todo)} to name ({len(done)} already done)", flush=True)

    taxonomy = sorted(pd.read_csv(ROOT / "data" / "LVVI3" / "LVVI3_activities.csv")["activity"].dropna().unique())
    model = next(m for m in config.ENSEMBLE_MODELS if m["key"] == "poro2_70b")
    system = SYSTEM_PROMPT.format(activities="; ".join(taxonomy))

    BATCH = 32
    for start in range(0, len(todo), BATCH):
        batch_vids = todo[start: start + BATCH]
        messages_list = []
        for vid in batch_vids:
            row = candidates[candidates["video_id"] == vid].iloc[0]
            tr = transcript_text(tr_idx.get(vid, {"segments": []}))
            user = (
                f"Video title: {row['title']}\n"
                f"Video description: {str(row['description'])[:DESCRIPTION_MAX_CHARS]}\n"
                f"Detected in-taxonomy activities: {row['labels_at_0.5']}\n\n"
                f"Transcript (may be Finnish/Swedish/English):\n{tr or '(no speech)'}\n\n"
                "Name the uncovered recreational activity."
            )
            messages_list.append([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
        try:
            responses = llm_chat.chat_batch(messages_list, model, max_tokens=200)
        except Exception as e:
            print(f"[batch {start}] chat_batch failed: {e}", flush=True)
            responses = [""] * len(batch_vids)
        for vid, resp in zip(batch_vids, responses):
            obj = parse_json_object(resp or "")
            rec = {
                "video_id": vid,
                "other_activity_p": float(candidates[candidates["video_id"] == vid]["other_activity_p"].iloc[0]),
                "response": (resp or "")[:2000],
                "new_activity": obj.get("new_activity", ""),
                "evidence": obj.get("evidence", ""),
                "confidence": obj.get("confidence", ""),
                "error": "" if obj else "unparsed",
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
        print(f"[done] {min(start + BATCH, len(todo))}/{len(todo)} named", flush=True)

    out_f.close()
    print("naming finished", flush=True)


if __name__ == "__main__":
    main()
