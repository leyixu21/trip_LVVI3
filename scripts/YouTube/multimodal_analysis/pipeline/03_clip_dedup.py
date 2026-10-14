"""
Stage 3: CLIP dedup

Shot detection still over-produces near-identical keyframes for long static
shots (e.g. a locked drone shot, or scene-detection false positives). Embed
every keyframe with CLIP's image encoder and greedily drop frames that are
near-duplicates (cosine sim above threshold) of a frame already kept, within
each video.

Survivors replace the video's keyframe shard in Allas in place: the shard is
downloaded, decoded, and rebuilt entirely in memory (common/keyframe_io.py)
-- a new `{video_id}.tar.gz` holding just the kept frames + a shots.json of
just the kept records is uploaded straight over the object
02_detect_shots_and_extract_keyframes.py wrote. No frame or shard ever
touches local disk. Each record's shot_index (which original shot its frame
belongs to) survives untouched -- records are filtered here, never
reconstructed.

Usage:
  python 03_clip_dedup.py
  python 03_clip_dedup.py --video_ids_file=<path/to/video_ids.txt>
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import allas_utils, config, keyframe_io


def load_clip(device: str):
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME, cache_dir=config.CACHE_DIR)
    model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME, cache_dir=config.CACHE_DIR).to(device).eval()
    return processor, model


@torch.no_grad()
def embed_images(images: list[Image.Image], processor, model, device: str) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt").to(device)
    output = model.get_image_features(**inputs)
    # transformers >=4.56 or so returns a BaseModelOutputWithPooling (the
    # projected image embedding lives in .pooler_output) instead of the
    # plain tensor older versions returned directly.
    features = output.pooler_output if hasattr(output, "pooler_output") else output
    return torch.nn.functional.normalize(features, dim=-1)


def dedup_video(video_id: str, conn, processor, model, device: str) -> int:
    shard_bytes = keyframe_io.download_shard(conn, video_id)
    if shard_bytes is None:
        return 0
    shot_records, frame_bytes = keyframe_io.read_shard(shard_bytes, video_id)
    if not shot_records:
        return 0

    decoded = keyframe_io.decode_images(frame_bytes, [rec["frame_name"] for rec in shot_records])
    kept_records = [rec for rec in shot_records if rec["frame_name"] in decoded]
    images = [decoded[rec["frame_name"]] for rec in kept_records]

    if not images:
        return 0

    embeddings = embed_images(images, processor, model, device)

    keep_idx = []
    for i in range(len(kept_records)):
        is_dup = False
        for j in keep_idx:
            sim = float(embeddings[i] @ embeddings[j])
            if sim >= config.CLIP_DEDUP_SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            keep_idx.append(i)

    filtered = [kept_records[i] for i in keep_idx]

    # Rebuild the shard from just the survivors and upload it over the
    # existing Allas object -- see module docstring for why. If the upload
    # fails, the original (un-deduped) shard is left untouched in Allas, so
    # this video is simply picked back up as-is on the next run.
    new_shard_bytes = keyframe_io.build_shard_bytes(video_id, filtered, frame_bytes)
    if not keyframe_io.upload_shard(conn, video_id, new_shard_bytes):
        print(f"[allas] failed to upload deduplicated shard for {video_id}")

    return len(filtered)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_ids_file", type=Path, default=None,
                         help="One video_id per line; defaults to every video_id in config.VIDEOS_CSV")
    return parser.parse_args()


def main():
    import pandas as pd

    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available -- refusing to silently fall back to CPU "
            "for CLIP. Check the GPU allocation (gres=gpu:...) and that the "
            "container was run with `singularity exec --nv`."
        )
    device = "cuda"
    print(f"Loading CLIP on {device}...")
    processor, model = load_clip(device)

    if args.video_ids_file is not None:
        video_ids = [line.strip() for line in args.video_ids_file.read_text().splitlines() if line.strip()]
    else:
        df = pd.read_csv(config.VIDEOS_CSV)
        video_ids = df["video_id"].tolist()

    print(f"Deduping keyframes for up to {len(video_ids)} videos "
          f"(videos without a keyframe tar in Allas yet are skipped)")

    conn = allas_utils.connect_to_allas()
    start = time.time()
    n_skipped = 0
    with tqdm(total=len(video_ids), desc="Videos", unit="video") as pbar:
        for vid in video_ids:
            n_kept = dedup_video(vid, conn, processor, model, device)
            if n_kept:
                tqdm.write(f"{vid}: kept {n_kept} keyframes")
            else:
                n_skipped += 1
            pbar.set_postfix(skipped=n_skipped)
            pbar.update(1)

    print(f"Done in {time.time() - start:.1f}s -- {n_skipped} videos skipped (no tar / no keyframes)")


if __name__ == "__main__":
    main()
