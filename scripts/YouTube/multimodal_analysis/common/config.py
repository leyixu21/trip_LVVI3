"""
Shared configuration for the transcription + shot/keyframe extraction stages
of the LVVI3 multimodal analysis pipeline (ported from
influencer_YouTube/scripts_revision_01/multimodal_analysis).

Paths default to the CSC (Puhti/Mahti) scratch layout used by the rest of
this project, but every path can be overridden with an environment variable
so the same code can be smoke-tested off-cluster.
"""
import os
from pathlib import Path

# -------------------
# Paths
# -------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get(
    "MMA_DATA_DIR",
    "/scratch/project_2006782/leyixu/LVVI3_YouTube/data",
))
CACHE_DIR = os.environ.get(
    "MMA_CACHE_DIR",
    "/scratch/project_2006782/leyixu/transformers_cache",
)

# Master video table: same file multimodal_json_input.py and the rest of
# this folder already read from -- must have a "video_id" column.
VIDEOS_CSV = DATA_DIR / "YouTube" / "YouTube_videos.csv"

# Intermediate artifact directories/files (created on demand by each stage)
# Transcripts are consolidated into a single JSONL (one line per video)
# rather than one file per video, to avoid a directory of thousands of tiny
# files and the bookkeeping that comes with it.
TRANSCRIPTS_JSONL = DATA_DIR / "YouTube" / "transcripts.jsonl"
VIDEOS_TMP_DIR = DATA_DIR / "YouTube" / "videos_tmp"

# -------------------
# Allas object storage containers
# -------------------
ALLAS_VIDEO_CONTAINER = "2006782_youtube_videos"
# Deliberately a new container, not the old "2006782_youtube_videos_shots"
# (CSV + loose-jpg format written by the retired run_video_shot_detection.sh)
# -- this pipeline writes a different artifact shape (shots.json + tar), so
# it gets its own container rather than overwriting/mixing with the old one.
# Each video's keyframes live in exactly one object here,
# ALLAS_KEYFRAME_CONTAINER/{video_id}.tar.gz (shots.json + its frame jpgs,
# see common/keyframe_io.py) -- no keyframe or shard is ever written to
# local disk.
ALLAS_KEYFRAME_CONTAINER = "2006782_youtube_videos_keyframes"

# -------------------
# Stage 1: Whisper transcription (pipeline/01_transcribe_videos.py)
# -------------------
# large-v2 (not "medium") to match the model size already budgeted for on
# the v100 GPU allocation this project's transcription jobs have used.
WHISPER_MODEL_NAME = "large-v2"

# Hallucination filtering: nature/travel videos have long music/ambient
# stretches with no speech, and Whisper sometimes fills these with invented
# text instead of staying silent. A segment is dropped if Whisper itself
# flagged it as likely non-speech AND the decoder had low confidence in its
# own output (the same two signals Whisper's internal temperature-fallback
# decoding uses), or if its text is degenerate/repetitive (high compression
# ratio is the classic signature of hallucinated gibberish).
WHISPER_NO_SPEECH_PROB_THRESHOLD = 0.6
WHISPER_AVG_LOGPROB_THRESHOLD = -1.0
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4

# Lightweight speech / music-or-ambient / silence proportions, derived from
# the same decoded audio Whisper already read -- no extra model needed.
# Non-speech frames are split on RMS energy: near-silent frames are
# "silence", louder non-speech frames are "music_or_ambient".
AUDIO_EVENT_FRAME_SECONDS = 1.0
AUDIO_EVENT_SILENCE_RMS_THRESHOLD = 0.01

# -------------------
# Stage 2: shot detection + keyframe extraction
# (pipeline/02_detect_shots_and_extract_keyframes.py)
# -------------------
# AdaptiveDetector (rolling-average content score) copes with camera motion
# -- pans, drone movement -- better than plain ContentDetector, which is
# tuned for hard cuts and tends to under-detect on continuous nature/travel
# footage. adaptive_threshold=3.0 matches the `scenedetect detect-adaptive`
# CLI default.
SHOT_DETECTOR_ADAPTIVE_THRESHOLD = 3.0
MIN_SHOT_LEN_SECONDS = 1.0
MAX_KEYFRAMES_PER_VIDEO = 120  # safety cap for very long / static videos
# Upper bound on the gap between keyframes, applied on top of detected shots.
# Even AdaptiveDetector finds zero boundaries in genuinely continuous
# footage (nothing "cuts" in a slow drone pan), which without this would
# collapse a long, content-rich video down to a single keyframe. Any segment
# longer than this (including the whole-video fallback when zero cuts are
# found) is subdivided so no gap between keyframes exceeds it.
TARGET_KEYFRAME_INTERVAL_SECONDS = 15.0

# -------------------
# Stage 3: CLIP dedup (pipeline/03_clip_dedup.py)
# -------------------
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_DEDUP_SIMILARITY_THRESHOLD = 0.9  # cosine sim above this -> treat as duplicate
