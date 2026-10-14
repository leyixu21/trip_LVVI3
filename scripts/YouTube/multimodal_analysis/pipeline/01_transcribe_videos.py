"""
Whisper transcription of the audio stream.

For each video: download from Allas, decode the audio once, transcribe with
Whisper (large-v2, auto language detection), filter, summarize. There is
exactly one GPU and exactly one Whisper model instance per job -- download +
ffmpeg decode (I/O-bound) run concurrently across a thread pool so the next
video's audio is ready by the time the GPU is free, but the GPU transcribe
step itself always runs one video at a time in the main thread. (An earlier
version ran everything, including the GPU step, across 4 forked
ProcessPoolExecutor workers -- that loaded 4 redundant copies of the model
and, when CUDA failed to initialize for those workers, silently fell back to
CPU with no error, burning a full GPU allocation for hours at ~1% of the
achievable throughput. get_model() below now fails loudly instead.)

Supports Slurm-array sharding via --shard_index/--num_shards: each task
processes video_ids[shard_index::num_shards] and writes its own output file
(default: TRANSCRIPTS_JSONL with a .{shard_index} suffix), so concurrent
array tasks never race on one file. Concatenate the shard files into
TRANSCRIPTS_JSONL once all tasks finish.

Output: one shard-or-plain JSONL, one JSON object per line ->
  {
    "video_id": ...,
    "language": <ISO 639-1 code Whisper detected>,
    "segments": [
      {"start", "end", "text", "language", "no_speech_prob", "avg_logprob", "compression_ratio"},
      ...
    ],
    "n_segments_total": ...,
    "n_segments_dropped_hallucinated": ...,
    "audio_events": {"speech": .., "music_or_ambient": .., "silence": .., "duration_sec": ..},
  }

Videos that fail (download or transcription error) are skipped, same as
before -- no line is written for them.

Usage:
  python 01_transcribe_videos.py
  python 01_transcribe_videos.py --video_ids_file=<path/to/video_ids.txt>
  python 01_transcribe_videos.py --shard_index=$SLURM_ARRAY_TASK_ID --num_shards=$SLURM_ARRAY_TASK_COUNT
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
import whisper
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import allas_utils, config
from common.whisper_utils import SAMPLE_RATE, audio_event_summary, filter_hallucinated_segments, load_audio

_MODEL = None


def get_model():
    """Loads Whisper onto the GPU exactly once per job. Raises immediately
    if CUDA isn't available instead of letting whisper.load_model() fall
    back to CPU on its own -- large-v2 on CPU is ~50-100x slower, so a
    silent fallback means the job crawls for hours (or hits its walltime)
    while the GPU it was allocated sits idle, instead of failing fast."""
    global _MODEL
    if _MODEL is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available -- refusing to silently fall back to "
                "CPU for Whisper large-v2. Check the GPU allocation "
                "(gres=gpu:...) and that the container was run with `singularity "
                "exec --nv`."
            )
        _MODEL = whisper.load_model(config.WHISPER_MODEL_NAME, device="cuda", download_root=config.CACHE_DIR)
    return _MODEL


def download_and_decode(video_id: str) -> dict:
    """I/O-bound phase: download from Allas + decode audio via ffmpeg.
    Network I/O and the ffmpeg subprocess both release the GIL, so several
    of these can run concurrently in threads while the GPU transcribes a
    different video in the main thread.

    Returns {"video_id", "audio": np.ndarray, "error": None} on success, or
    {"video_id", "error": <message>} on failure."""
    conn = allas_utils.connect_to_allas()
    config.VIDEOS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    video_path = config.VIDEOS_TMP_DIR / f"{video_id}.mp4"

    try:
        if not allas_utils.download_object(conn, config.ALLAS_VIDEO_CONTAINER, f"{video_id}.mp4", str(video_path)):
            return {"video_id": video_id, "error": "download failed"}
        audio = load_audio(str(video_path))
        return {"video_id": video_id, "audio": audio, "error": None}
    except Exception as e:
        return {"video_id": video_id, "error": str(e)}
    finally:
        if video_path.exists():
            video_path.unlink()


def transcribe_decoded(video_id: str, audio) -> dict:
    """GPU-bound phase: run the shared Whisper model on already-decoded
    audio. Must only ever be called from one thread/process at a time --
    there's exactly one GPU and one model instance for the whole job."""
    try:
        model = get_model()
        result = model.transcribe(audio, condition_on_previous_text=False)
        language = result["language"]

        raw_segments = [
            {
                "start": s["start"],
                "end": s["end"],
                "text": s["text"].strip(),
                "language": language,
                "no_speech_prob": s["no_speech_prob"],
                "avg_logprob": s["avg_logprob"],
                "compression_ratio": s["compression_ratio"],
            }
            for s in result["segments"]
        ]

        kept_segments, n_dropped = filter_hallucinated_segments(
            raw_segments,
            no_speech_prob_threshold=config.WHISPER_NO_SPEECH_PROB_THRESHOLD,
            avg_logprob_threshold=config.WHISPER_AVG_LOGPROB_THRESHOLD,
            compression_ratio_threshold=config.WHISPER_COMPRESSION_RATIO_THRESHOLD,
        )

        events = audio_event_summary(
            audio, SAMPLE_RATE, kept_segments,
            frame_seconds=config.AUDIO_EVENT_FRAME_SECONDS,
            silence_rms_threshold=config.AUDIO_EVENT_SILENCE_RMS_THRESHOLD,
        )

        return {
            "video_id": video_id,
            "language": language,
            "segments": kept_segments,
            "n_segments_total": len(raw_segments),
            "n_segments_dropped_hallucinated": n_dropped,
            "audio_events": events,
            "error": None,
        }
    except Exception as e:
        return {"video_id": video_id, "error": str(e)}


def load_video_ids(video_ids_file: Path | None) -> list[str]:
    """Same source as pipeline/02_detect_shots_and_extract_keyframes.py's
    --list_video_ids mode: an explicit one-id-per-line file if given,
    otherwise every video_id in config.VIDEOS_CSV."""
    if video_ids_file is not None:
        return [line.strip() for line in video_ids_file.read_text().splitlines() if line.strip()]
    df = pd.read_csv(config.VIDEOS_CSV)
    return df["video_id"].tolist()


def already_transcribed_ids() -> set[str]:
    """video_ids that already have a line in any TRANSCRIPTS_JSONL* file --
    the plain file and/or any shard part left by a Slurm array run -- so a
    rerun over the same list (e.g. after a partial failure, or resuming one
    shard) only pays for what's genuinely missing."""
    ids = set()
    for path in sorted(config.TRANSCRIPTS_JSONL.parent.glob(config.TRANSCRIPTS_JSONL.name + "*")):
        with open(path) as f:
            ids.update(json.loads(line)["video_id"] for line in f if line.strip())
    return ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_ids_file", type=Path, default=None,
                         help="One video_id per line; defaults to every video_id in config.VIDEOS_CSV")
    parser.add_argument("--shard_index", type=int, default=0,
                         help="0-based index of this task's shard (pairs with --num_shards for a Slurm array job)")
    parser.add_argument("--num_shards", type=int, default=1,
                         help="Total shards; this task processes video_ids[shard_index::num_shards]")
    parser.add_argument("--output", type=Path, default=None,
                         help="Override the output JSONL path. Defaults to config.TRANSCRIPTS_JSONL, or that "
                              "path with a .{shard_index} suffix appended when --num_shards > 1")
    parser.add_argument("--download_workers", type=int, default=4,
                         help="Threads downloading+decoding audio concurrently while the GPU transcribes")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"--shard_index ({args.shard_index}) must be in [0, --num_shards) ({args.num_shards})")

    video_ids = load_video_ids(args.video_ids_file)
    video_ids = video_ids[args.shard_index::args.num_shards]

    output_path = args.output
    if output_path is None:
        output_path = config.TRANSCRIPTS_JSONL
        if args.num_shards > 1:
            output_path = output_path.with_name(output_path.name + f".{args.shard_index}")

    done = already_transcribed_ids()
    n_before = len(video_ids)
    video_ids = [v for v in video_ids if v not in done]
    if len(video_ids) < n_before:
        print(f"Skipping {n_before - len(video_ids)} already-transcribed videos")
    print(f"Transcribing {len(video_ids)} videos with timestamps -> {output_path}")

    start = time.time()
    n_segments_kept_total = 0
    n_failed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor, open(output_path, "a") as out:
        futures = {executor.submit(download_and_decode, vid): vid for vid in video_ids}
        with tqdm(total=len(futures), desc="Videos", unit="video") as pbar:
            for future in as_completed(futures):
                vid = futures[future]
                try:
                    dl_result = future.result()
                except Exception as e:
                    dl_result = {"video_id": vid, "error": str(e)}

                if dl_result["error"] is not None:
                    n_failed += 1
                    tqdm.write(f"{vid}: FAILED - {dl_result['error']}")
                    pbar.set_postfix(segments=n_segments_kept_total, failed=n_failed)
                    pbar.update(1)
                    continue

                # GPU step: always runs here, in the main thread, one video
                # at a time -- only the download/decode phase above is
                # parallelized across threads.
                result = transcribe_decoded(vid, dl_result["audio"])

                if result["error"] is not None:
                    n_failed += 1
                    tqdm.write(f"{result['video_id']}: FAILED - {result['error']}")
                else:
                    n_kept = len(result["segments"])
                    n_segments_kept_total += n_kept
                    tqdm.write(
                        f"{result['video_id']}: {n_kept}/{result['n_segments_total']} segments kept "
                        f"(lang={result['language']}, speech={result['audio_events']['speech']:.0%})"
                    )
                    del result["error"]
                    out.write(json.dumps(result) + "\n")
                    out.flush()

                pbar.set_postfix(segments=n_segments_kept_total, failed=n_failed)
                pbar.update(1)

    print(f"Done in {time.time() - start:.1f}s -- {n_segments_kept_total} segments kept total, {n_failed} videos failed")


if __name__ == "__main__":
    main()
