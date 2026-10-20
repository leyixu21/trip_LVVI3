"""Flickr activity labeling with Cloudflare/clef (self-hosted, 35 LVVI3 activities).

One record per photo: state = {title, description, tags}, images = [photo],
questions = one `noul` question per top-level LVVI3 activity (35, from the
`activity` column of data/LVVI3/LVVI3_activities.csv) plus the open-set
`other_activity` noul question and the `primary_activity` choice question
(with "Other / not listed"). One forward pass decides all questions at once.

Photos come from data/Flickr/images_z/<id>.jpg (see download_flickr_images.py);
the selection file lists photos with a local image.

Output JSONL (one line per photo):
  id, probabilities {activity: P(true)}, labels_at_0.5, other_activity_p,
  primary_activity_probs, n_state_tokens, forward_s
Resumable: ids already present in the output are skipped.

Runs inside the aida vllm wrapper 0.29.0 container (see sbatch/run_clef_flickr.sh).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PILOT_DIR = Path(__file__).resolve().parents[1] / "YouTube" / "multimodal_analysis" / "pilot"
sys.path.insert(0, str(PILOT_DIR))

from common_pilot import append_jsonl, load_taxonomy, read_jsonl  # noqa: E402

CREATOR_RULE = (
    "Is the activity \"{act}\" performed in this image? Answer true only if it "
    "is clearly depicted; answer false if it is merely referenced by a sign or "
    "caption, or absent. Any person in the image counts."
)
OTHER_QUESTION_ID = "other_activity"
OTHER_RULE = (
    "Is an outdoor or recreational activity performed in this image that is "
    "NOT adequately covered by any of the other categories asked about? "
    "Answer true only if someone is clearly doing something recreational that "
    "fits none of the listed categories."
)
PRIMARY_QUESTION_ID = "primary_activity"
PRIMARY_OTHER_OPTION = "Other / not listed"

TEXT_MAX_CHARS = 1000
MAX_PIXELS = 1280 * 720


def load_photo(photo: dict, images_dir: str | None) -> Image.Image | None:
    """Local file if already downloaded, otherwise stream from photo_url."""
    import io
    import urllib.request
    if images_dir:
        img_path = Path(images_dir) / f"{photo['id']}.jpg"
        if img_path.exists():
            try:
                return Image.open(img_path).convert("RGB")
            except Exception:
                pass
    url = photo.get("photo_url")
    if not url:
        return None
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            time.sleep(1 + attempt)
    return None


def resize_frame(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w * h <= MAX_PIXELS:
        return img
    scale = (MAX_PIXELS / (w * h)) ** 0.5
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))))


def build_record(photo: dict, taxonomy: list[str]) -> dict:
    state = {
        "photo_title": photo.get("title") or "",
        "photo_description": (photo.get("description") or "")[:TEXT_MAX_CHARS],
        "photo_tags": photo.get("tags") or "",
        "note": "Text may be in any language.",
    }
    questions = {
        act: {"type": "noul", "instructions": CREATOR_RULE.format(act=act)}
        for act in taxonomy
    }
    questions[OTHER_QUESTION_ID] = {"type": "noul", "instructions": OTHER_RULE}
    questions[PRIMARY_QUESTION_ID] = {
        "type": "choice",
        "instructions": ("Which single category best describes the "
                         "PRIMARY (most prominent) activity in this image? "
                         "If none fits adequately, choose Other."),
        "criteria": {**{act: act for act in taxonomy},
                     PRIMARY_OTHER_OPTION: "The main activity is not covered."},
    }
    return {"id": str(photo["id"]), "state": state, "questions": questions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Local snapshot dir of Cloudflare/clef")
    ap.add_argument("--selection", required=True, help="JSON list of {id, title, description, tags}")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=8192)
    args = ap.parse_args()

    photos = json.loads(Path(args.selection).read_text())
    if args.num_shards > 1:
        photos = [p for i, p in enumerate(photos) if i % args.num_shards == args.shard_index]
    if args.limit:
        photos = photos[: args.limit]

    done = {r["id"] for r in read_jsonl(Path(args.output))}
    photos = [p for p in photos if str(p["id"]) not in done]
    print(f"flickr clef: {len(photos)} photos to run ({len(done)} already done)", flush=True)

    taxonomy = load_taxonomy()

    sys.path.insert(0, args.model_path)
    from joint_schema_model import collate_records, encode_record, load_release_model

    t0 = time.time()
    model, processor = load_release_model(args.model_path, device="cuda")
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    skipped_no_image = []
    for start in range(0, len(photos), args.batch_size):
        batch_photos = photos[start: start + args.batch_size]
        encoded, metas = [], []
        for photo in batch_photos:
            img = load_photo(photo, args.images_dir or None)
            if img is None:
                skipped_no_image.append(str(photo["id"]))
                continue
            try:
                img = resize_frame(img)
            except Exception as e:
                print(f"[skip] {photo['id']}: unreadable image: {e}", flush=True)
                continue
            record = build_record(photo, taxonomy)
            record["images"] = [img]
            try:
                enc = encode_record(processor.tokenizer, record, processor=processor,
                                    max_length=args.max_length)
            except ValueError as e:
                print(f"[skip] {photo['id']}: encode failed: {e}", flush=True)
                continue
            encoded.append(enc)
            metas.append({"photo": photo})

        if not encoded:
            continue
        pad_id = processor.tokenizer.pad_token_id
        batch = collate_records(encoded, pad_id, torch.device("cuda"))
        t_fwd = time.time()
        with torch.inference_mode():
            all_logits = model(batch)
        forward_s = time.time() - t_fwd
        per_photo = forward_s / len(encoded)

        for meta, enc, logits in zip(metas, encoded, all_logits):
            probs, other_p, primary_probs = {}, None, {}
            for question, q_logits in zip(enc.questions, logits):
                p = q_logits.float().softmax(-1)
                if "true" in question.option_ids:
                    val = round(p[question.option_ids.index("true")].item(), 4)
                    if question.question_id == OTHER_QUESTION_ID:
                        other_p = val
                    else:
                        probs[question.question_id] = val
                else:
                    primary_probs = {opt: round(p[i].item(), 4)
                                     for i, opt in enumerate(question.option_ids)}
            rec = {
                "id": str(meta["photo"]["id"]),
                "probabilities": probs,
                "labels_at_0.5": sorted([a for a, p in probs.items() if p >= 0.5]),
                "other_activity_p": other_p,
                "primary_activity_probs": primary_probs,
                "n_state_tokens": len(enc.input_ids),
                "forward_s": round(per_photo, 3),
            }
            append_jsonl(Path(args.output), rec)
            top = max(probs, key=probs.get)
            print(f"[done] {rec['id']} fwd={per_photo:.2f}s "
                  f"top={top}:{probs[top]:.2f} other={other_p:.2f}", flush=True)

    print(f"skipped (no image): {skipped_no_image}", flush=True)
    print("flickr clef finished", flush=True)


if __name__ == "__main__":
    main()
