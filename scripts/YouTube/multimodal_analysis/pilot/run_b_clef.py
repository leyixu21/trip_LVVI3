"""Pilot B: Cloudflare/clef (27B multimodal decision model, self-hosted) over
the pilot's selected videos.

One record per video: state = {title, description, transcript}, images =
evenly-sampled keyframes (same budget as Pilot D), questions = one `noul`
question per top-level LVVI3 activity (22). One forward pass decides all 22
activities at once and returns per-activity P(true). Runs batched through
collate_records for throughput.

Environment: runs INSIDE the aida vllm wrapper container (0.29.0: torch
2.13 + transformers 5.17 + PIL). The model snapshot is pre-downloaded to
the shared HF cache; joint_schema_model.py is imported from the snapshot
directory. Allas shards come in memory via pilot/allas_s3.py (pure-python
boto3 on PYTHONPATH) because the project's swiftclient credentials expired.

Output JSONL (data/YouTube/pilot/pilot_b_clef.jsonl), one line per video:
  video_id, probabilities {activity: P(true)}, labels_at_0.5, n_frames,
  n_state_tokens, download_s, forward_s (this video's share of its batch)
Resumable: videos already present in the output are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

PILOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PILOT_DIR))

from common_pilot import (  # noqa: E402
    PILOT_B_OUTPUT,
    SELECTION_PATH,
    TRANSCRIPT_MAX_CHARS,
    VIDEO_DESCRIPTION_MAX_CHARS,
    append_jsonl,
    load_taxonomy,
    load_transcripts_index,
    read_jsonl,
    sample_frames_for_video,
)

CREATOR_RULE = (
    "Is the activity \"{act}\" performed in this video? Answer true only if it "
    "is clearly depicted in the frames or clearly described in the narration; "
    "answer false if it is merely mentioned in passing or absent. Any person "
    "in the video counts, not only the creator."
)

OTHER_QUESTION_ID = "other_activity"
OTHER_RULE = (
    "Is an outdoor or recreational activity performed in this video that is "
    "NOT adequately covered by any of the other categories asked about? "
    "Answer true only if someone is clearly doing something recreational that "
    "fits none of the listed categories."
)
PRIMARY_QUESTION_ID = "primary_activity"
PRIMARY_OTHER_OPTION = "Other / not listed"

# Bound image token cost; at K=48 records are ~21.5k tokens, so the runner
# raises encode_record's max_length accordingly.
MAX_PIXELS = 1280 * 720


def resize_frame(img, max_pixels: int = MAX_PIXELS):
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))))


def frame_timestamp_s(frame_name: str) -> float | None:
    m = re.match(r"frame_\d+_(\d+\.\d+)\.jpg$", frame_name)
    return float(m.group(1)) if m else None


def build_record(video: dict, taxonomy: list[str], transcript_idx: dict,
                 frame_names: list[str]) -> dict:
    tr = transcript_idx[video["video_id"]]
    from common_pilot import transcript_text
    frame_lines = []
    for i, name in enumerate(frame_names):
        ts = frame_timestamp_s(name)
        label = f"{int(ts) // 60:02d}:{int(ts) % 60:02d}" if ts is not None else "unknown time"
        frame_lines.append(f"image {i + 1} was captured at {label} in the video")
    state = {
        "video_title": video["title"],
        "video_description": video["description"][:VIDEO_DESCRIPTION_MAX_CHARS],
        "transcript_with_timestamps": transcript_text(tr),
        "frames": frame_lines,
        "note": ("Transcript may be in any language; titles/descriptions too. "
                 "Timestamps in the transcript ([mm:ss]) and the frame capture "
                 "times refer to the same video clock."),
    }
    questions = {
        act: {"type": "noul", "instructions": CREATOR_RULE.format(act=act)}
        for act in taxonomy
    }
    questions[OTHER_QUESTION_ID] = {"type": "noul", "instructions": OTHER_RULE}
    questions[PRIMARY_QUESTION_ID] = {
        "type": "choice",
        "instructions": ("Which single category best describes the "
                         "PRIMARY (most prominent) activity performed in this "
                         "video? If none fits adequately, choose Other."),
        "criteria": {**{act: act for act in taxonomy},
                     PRIMARY_OTHER_OPTION: "The main activity is not covered."},
    }
    return {
        "id": video["video_id"],
        "state": state,
        "questions": questions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="Local snapshot dir of Cloudflare/clef")
    ap.add_argument("--selection", default=str(SELECTION_PATH))
    ap.add_argument("--output", default=str(PILOT_B_OUTPUT.parent / "pilot_b_clef_v3_k48.jsonl"))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N videos")
    ap.add_argument("--n_frames", type=int, default=48,
                    help="evenly-sampled keyframes per video")
    ap.add_argument("--max_length", type=int, default=24576,
                    help="encode_record token budget (must cover frames + state)")
    ap.add_argument("--shard_index", type=int, default=0,
                    help="this job handles videos[shard_index::num_shards]")
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    selection = json.loads(Path(args.selection).read_text())
    videos = selection["selected"] + selection.get("spares", [])
    if args.num_shards > 1:
        videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard_index]
    if args.limit:
        videos = videos[: args.limit]

    done = {r["video_id"] for r in read_jsonl(Path(args.output))}
    videos = [v for v in videos if v["video_id"] not in done]
    print(f"pilot B: {len(videos)} videos to run ({len(done)} already done)", flush=True)

    taxonomy = load_taxonomy()
    transcript_idx = load_transcripts_index()

    sys.path.insert(0, args.model_path)
    from joint_schema_model import collate_records, encode_record, load_release_model

    t0 = time.time()
    model, processor = load_release_model(args.model_path, device="cuda")
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    skipped_no_shard = []
    for start in range(0, len(videos), args.batch_size):
        batch_videos = videos[start: start + args.batch_size]
        encoded, metas = [], []
        for video in batch_videos:
            vid = video["video_id"]
            t_dl = time.time()
            shots, images = sample_frames_for_video(vid, n_frames=args.n_frames)
            download_s = time.time() - t_dl
            if not images:
                print(f"[skip] {vid}: no keyframes shard", flush=True)
                skipped_no_shard.append(vid)
                continue
            frames = [resize_frame(img) for img in images.values()]
            record = build_record(video, taxonomy, transcript_idx, list(images))
            record["images"] = frames
            record["images"] = frames
            record["media_kwargs"] = {"max_pixels": MAX_PIXELS}
            try:
                enc = encode_record(processor.tokenizer, record, processor=processor,
                                    max_length=args.max_length)
            except ValueError as e:
                print(f"[skip] {vid}: encode failed: {e}", flush=True)
                continue
            encoded.append(enc)
            metas.append({"video": video, "download_s": download_s, "n_frames": len(frames)})

        if not encoded:
            continue
        pad_id = processor.tokenizer.pad_token_id
        batch = collate_records(encoded, pad_id, torch.device("cuda"))
        t_fwd = time.time()
        with torch.inference_mode():
            all_logits = model(batch)
        forward_s = time.time() - t_fwd
        per_video = forward_s / len(encoded)

        for meta, enc, logits in zip(metas, encoded, all_logits):
            probs = {}
            other_p = None
            primary_probs = {}
            for question, q_logits in zip(enc.questions, logits):
                p = q_logits.float().softmax(-1)
                if "true" in question.option_ids:
                    val = round(p[question.option_ids.index("true")].item(), 4)
                    if question.question_id == OTHER_QUESTION_ID:
                        other_p = val
                    else:
                        probs[question.question_id] = val
                else:
                    primary_probs = {
                        opt: round(p[i].item(), 4)
                        for i, opt in enumerate(question.option_ids)
                    }
            rec = {
                "video_id": meta["video"]["video_id"],
                "probabilities": probs,
                "labels_at_0.5": sorted([a for a, p in probs.items() if p >= 0.5]),
                "other_activity_p": other_p,
                "primary_activity_probs": primary_probs,
                "n_frames": meta["n_frames"],
                "n_state_tokens": len(enc.input_ids),
                "download_s": round(meta["download_s"], 3),
                "forward_s": round(per_video, 3),
            }
            append_jsonl(Path(args.output), rec)
            top = max(probs, key=probs.get)
            print(f"[done] {rec['video_id']} fwd={per_video:.2f}s "
                  f"top={top}:{probs[top]:.2f} other={other_p:.2f} "
                  f"ntok={rec['n_state_tokens']}", flush=True)

    print(f"skipped (no shard): {skipped_no_shard}", flush=True)
    print("pilot B finished", flush=True)


if __name__ == "__main__":
    main()
