"""
Shot detection + keyframe extraction.

Single-video mode (--video_id/--video_path) plus a second, unrelated
--list_video_ids mode that writes the video-id list
sbatch/run_02_detect_shots_keyframes.sh's array job indexes into (same
source as transcribe_videos.py: config.VIDEOS_CSV). Both modes live here so
this script is self-contained in exactly two files; the shell script owns
downloading the source video, while this script owns uploading the
resulting keyframe shard -- it's built and uploaded straight from memory
(common/keyframe_io.py), so no tar file (nor any individual frame file)
ever touches local disk. That matters on CSC: project scratch has a limited
file-count/storage quota, while Allas is the purpose-built bulk object
store for exactly this data.

Uses AdaptiveDetector (copes with camera motion better than ContentDetector,
which under-detects on continuous nature/travel footage) plus
expand_long_segments (common/shot_detection.py) to backfill keyframe density
for any segment -- real shot or the whole-video fallback -- longer than
config.TARGET_KEYFRAME_INTERVAL_SECONDS, since genuinely continuous footage
can still produce zero detected cuts. Every sub-segment produced this way is
tagged with shot_index, the original detected shot it came from, so
downstream stages can bundle a shot's keyframes together instead of treating
each still independently.

Usage:
  python 02_detect_shots_and_extract_keyframes.py \\
      --video_id=<id> --video_path=<path/to/video.mp4>
  python 02_detect_shots_and_extract_keyframes.py --list_video_ids video_ids_missing.txt

Note: pass --video_id (and any other flag whose value comes from a video id)
with `=`, not a space -- some YouTube ids start with "-", and argparse reads
a bare `--video_id -abc123` as the (unknown) option "-abc123" rather than as
--video_id's value.

Output: uploads a single WebDataset-style tar shard to Allas as
`{video_id}.tar.gz`, containing
  frame_{idx:05d}_{timestamp_sec:.2f}.jpg
  shots.json -> [{"shot_index": .., "start": .., "end": .., "keyframe_ts": .., "frame_name": ..}, ...]
under a `{video_id}/` prefix (see common/keyframe_io.py). Frames are encoded
straight to jpg bytes in memory (never written to individual files).

Exit code is non-zero on zero keyframes or a failed Allas upload, so the
calling shell script's `set -e` treats either as a failure.
"""
import argparse
import sys
from pathlib import Path

import cv2
from scenedetect import AdaptiveDetector, SceneManager, open_video

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import allas_utils, config, keyframe_io
from common.shot_detection import build_tagged_segments, subsample_shots


def write_video_id_list(out_path: Path) -> int:
    """Writes one video id per line to out_path; returns the count. Same
    source as 01_transcribe_videos.py: every video_id in config.VIDEOS_CSV,
    filtered to what's actually present in the Allas source-video bucket,
    minus whatever already has a keyframe tar in ALLAS_KEYFRAME_CONTAINER.
    That container is shared with the influencer_YouTube project (same
    bucket name), so this also skips videos that project already processed
    -- no separate cross-project bookkeeping needed. Writes to a temp file
    and renames atomically so concurrent array tasks racing to generate
    this (see the shell script's lock) can't observe a torn file."""
    import pandas as pd

    df = pd.read_csv(config.VIDEOS_CSV)
    video_ids = df["video_id"].tolist()

    conn = allas_utils.connect_to_allas()
    available = allas_utils.list_container_video_ids(conn, config.ALLAS_VIDEO_CONTAINER)
    done = allas_utils.list_container_video_ids(conn, config.ALLAS_KEYFRAME_CONTAINER)
    video_ids = [v for v in video_ids if v in available and v not in done]

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text("\n".join(video_ids) + "\n")
    tmp_path.replace(out_path)

    print(f"Wrote {len(video_ids)} video ids to {out_path}")
    return len(video_ids)


def detect_shots(video_path: str) -> tuple[list[tuple[float, float]], float]:
    """Returns (shots, duration_seconds). Opens the video once via
    PySceneDetect's own open_video() and reads duration off that same
    VideoStream, instead of a second cv2.VideoCapture just for metadata.

    Uses the pyav backend rather than the default opencv one: opencv's
    ffmpeg backend has a stall-detection timeout that occasionally fires on
    local files with an unusual container layout (~146s hang -> VideoOpenFailure
    on an otherwise-valid download), which pyav doesn't hit."""
    video = open_video(video_path, backend="pyav")
    duration = video.duration.get_seconds() if video.duration is not None else 0.0

    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=config.SHOT_DETECTOR_ADAPTIVE_THRESHOLD))
    scene_manager.detect_scenes(video=video)

    shots = [(s.get_seconds(), e.get_seconds()) for s, e in scene_manager.get_scene_list()]
    shots = [(s, e) for s, e in shots if (e - s) >= config.MIN_SHOT_LEN_SECONDS]
    if not shots:
        shots = [(0.0, duration)]
    return shots, duration


class KeyframeExtractor:
    """Wraps a single pyav-backed VideoStream (same backend as detect_shots),
    reused across every extraction call in a run instead of opening the file
    once per keyframe. seek()-then-read() mirrors PySceneDetect's own
    save_images() usage of this backend (scenedetect/output/image.py)."""

    def __init__(self, video_path: str):
        self.video = open_video(video_path, backend="pyav")

    def extract(self, timestamp_sec: float) -> bytes | None:
        """Returns encoded jpg bytes for the frame at timestamp_sec, or None
        if the seek/read failed. Encoding straight to bytes (rather than
        cv2.imwrite to a path) is what lets run() below skip ever writing an
        individual frame file to disk."""
        self.video.seek(timestamp_sec)
        frame = self.video.read()
        if frame is False:
            return None
        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok else None

    def close(self):
        self.video = None


def run(video_id: str, video_path: str, conn) -> int | None:
    """Returns the number of keyframes uploaded, or None if the Allas
    upload itself failed (distinct from 0, which means shot detection ran
    but found nothing to extract)."""
    raw_shots, duration = detect_shots(video_path)

    # Sub-segments of the same original shot share a shot_index (see
    # build_tagged_segments), so downstream stages can bundle their
    # keyframes into one motion-aware unit instead of treating each as an
    # independent still.
    segments = build_tagged_segments(raw_shots, config.TARGET_KEYFRAME_INTERVAL_SECONDS)
    segments = subsample_shots(segments, config.MAX_KEYFRAMES_PER_VIDEO)

    shot_records = []
    frame_bytes = {}
    extractor = KeyframeExtractor(video_path)
    try:
        for i, (shot_index, start, end) in enumerate(segments):
            mid = (start + end) / 2.0
            frame_name = f"frame_{i:05d}_{mid:.2f}.jpg"
            data = extractor.extract(mid)
            if data is not None:
                frame_bytes[frame_name] = data
                shot_records.append({
                    "shot_index": shot_index, "start": start, "end": end,
                    "keyframe_ts": mid, "frame_name": frame_name,
                })
    finally:
        extractor.close()

    if shot_records:
        shard_bytes = keyframe_io.build_shard_bytes(video_id, shot_records, frame_bytes)
        if not keyframe_io.upload_shard(conn, video_id, shard_bytes):
            return None

    print(f"{video_id}: {len(shot_records)} keyframes (duration={duration:.1f}s, {len(segments)} segments, {len(raw_shots)} original shots)")
    return len(shot_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id")
    parser.add_argument("--video_path")
    parser.add_argument(
        "--list_video_ids", type=Path,
        help="Instead of processing one video, write the array job's video-id list to this path and exit.",
    )
    args = parser.parse_args()

    if args.list_video_ids:
        n = write_video_id_list(args.list_video_ids)
        if n == 0:
            sys.exit(1)
        return

    if not (args.video_id and args.video_path):
        parser.error("--video_id and --video_path are required unless --list_video_ids is given")

    conn = allas_utils.connect_to_allas()
    n_keyframes = run(args.video_id, args.video_path, conn)
    if n_keyframes is None:
        print(f"{args.video_id}: failed to upload keyframe shard to Allas", file=sys.stderr)
        sys.exit(1)
    if n_keyframes == 0:
        print(f"{args.video_id}: 0 keyframes extracted -- treating as failure", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
