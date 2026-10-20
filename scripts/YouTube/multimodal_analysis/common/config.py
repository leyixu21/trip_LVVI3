"""
Shared configuration for the LVVI3 multimodal analysis pipeline (ported
from influencer_YouTube/scripts/multimodal_analysis).

Paths default to the CSC (Roihu) scratch layout used by the rest of
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
    "/scratch/project_2006782/leyixu/trip_LVVI3/data",
))
CACHE_DIR = os.environ.get(
    "MMA_CACHE_DIR",
    "/scratch/project_2006782/leyixu/transformers_cache",
)

# Master video table. NOTE: 30,615 rows but only 17,650 UNIQUE video_id
# (one row per video x protected-area intersection) -- every consumer must
# drop_duplicates('video_id') before counting or iterating.
VIDEOS_CSV = DATA_DIR / "YouTube" / "youtube_videos.csv"

# Intermediate artifact directories/files (stages 4-9). CAPTIONS_DIR is
# only a staging area for in-progress stage-4 shards; the merged output is
# DATA_DIR / f"frame_captions_{model_key}.jsonl", a sibling of
# TRANSCRIPTS_JSONL (same one-line-per-video convention).
CAPTIONS_DIR = DATA_DIR / "YouTube" / "frame_captions"
TRANSCRIPTS_JSONL = DATA_DIR / "YouTube" / "transcripts.jsonl"
RAG_INDEX_DIR = DATA_DIR / "YouTube" / "rag_index"
CHUNK_LABELS_DIR = DATA_DIR / "YouTube" / "chunk_labels"
# Stage 9 output: the video-level ensemble vote, one pickle per vote
# threshold ({taxonomy}_vote{1..N}of{N}.pkl).
VIDEO_LABELS_DIR = DATA_DIR / "YouTube" / "video_labels"

VIDEOS_TMP_DIR = DATA_DIR / "YouTube" / "videos_tmp"

# -------------------
# Taxonomy (LVVI3, activities-only)
# -------------------
# The LVVI3 survey's activity taxonomy: 86 sub-activity rows under 22 unique
# top-level activities. Per the project decision we label at ACTIVITY level,
# so labels are the 22 unique `activity` values (io_utils.Taxonomy.labels
# dedupes by label_col). The CSV has no definition/example columns, so
# retrieval/display text is just the label itself (definition_col/example_col
# None); label_col doubles as group_col.
TAXONOMY_ACTIVITIES_PATH = DATA_DIR / "LVVI3" / "LVVI3_activities.csv"

TAXONOMIES = {
    "activities": {
        "path": TAXONOMY_ACTIVITIES_PATH,
        "label_col": "activity",
        "group_col": "activity",
        "definition_col": None,
        "example_col": None,
    },
}

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

# -------------------
# Stage 4: neutral-instructed captioning VLM (shot-bundled, multi-image)
# (pipeline/04_frame_captioning.py)
# -------------------
CAPTION_VLM_MODELS = [
    {"key": "llava_next", "hf_name": "llava-hf/llava-v1.6-mistral-7b-hf", "family": "llava_next"},
    {"key": "qwen3_vl",   "hf_name": "Qwen/Qwen3-VL-8B-Instruct",         "family": "qwen3_vl"},
]
CAPTION_MAX_NEW_TOKENS = 160
# Deliberately neutral / objective -- must not leak taxonomy vocabulary
# (activity names), otherwise downstream labeling would be biased by the
# captioner's guesses rather than the LLM's own independent reasoning.
# Specific about equipment/posture/action on purpose: that's what lets the
# downstream text-only labeling infer an activity (e.g. "skis, poles, snow"
# -> skiing) without the caption ever naming it.
CAPTION_SYSTEM_PROMPT = """
You are a neutral visual observer. You will be shown several frames, in
chronological order, from a single continuous shot of a video. Describe
ONLY what is literally visible across the frames:
- Setting/environment: terrain, landscape, indoor/outdoor, weather, time of day
- People: precise physical posture and action (e.g. "crouching to inspect a
  plant", "paddling", "walking with trekking poles"), not generic phrasing
  like "a person outside"
- Equipment/gear/objects: name them specifically (e.g. "skis", "kayak",
  "tent", "climbing rope", "camera on a tripod") rather than generically
- Any visible motion or change from the first frame to the last
Do not interpret intent, mood, or activity names.
Do not speculate about anything outside the frame.
Respond with 1-3 factual, specific sentences describing the shot as a whole,
no more than 60 words.
""".strip()
CAPTION_USER_PROMPT = "Describe this shot."

# -------------------
# Stage 5: alignment + chunk fusion (pipeline/05_align_fuse_chunks.py)
# -------------------
CHUNK_WINDOW_SECONDS = 20.0
CAPTION_DEDUP_WITHIN_CHUNK_THRESHOLD = 0.95  # cosine sim, CLIP text embeddings

# -------------------
# Stage 6: privacy NER scrub (in-place on fused_chunks_{model_key}.jsonl)
# (pipeline/06_privacy_scrub_chunks.py)
# -------------------
# Multilingual (covers the Finnish/English mix in these transcripts) NER
# model; entity_group tags follow CoNLL-style PER/LOC/ORG/MISC.
PRIVACY_NER_MODEL_NAME = "Babelscape/wikineural-multilingual-ner"
PRIVACY_NER_ENTITY_TYPES = ["PER", "LOC", "ORG", "MISC"]
PRIVACY_NER_SCORE_THRESHOLD = 0.8
# The model's max position embeddings is 512 tokens, and fused_text can run
# much longer, so redaction runs over whitespace-aligned windows of at most
# this many characters rather than the raw text directly. ~800 chars stays
# comfortably under 512 tokens even for less token-efficient languages.
PRIVACY_NER_WINDOW_MAX_CHARS = 800
PRIVACY_NER_BATCH_SIZE = 64

# Regex pass for structured PII patterns NER doesn't reliably catch --
# emails, URLs, HTML tags, emojis, and phone-number-like digit strings.
# Runs before NER (see privacy_ner.scrub_structured_pii / 06's docstring).
PRIVACY_REGEX_MIN_PHONE_DIGITS = 6  # digit chars (ignoring separators) to flag as phone-like, e.g. "040 123 4567"
PRIVACY_REGEX_MIN_DIGIT_RUN = 4     # bare digit run length to flag, e.g. zip/PIN codes -- short numbers like "15" or "3.5" stay untouched

# -------------------
# Stage 7: taxonomy RAG index (pipeline/07_build_taxonomy_rag_index.py)
# -------------------
RAG_EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# The activities taxonomy has only 22 unique labels, and top_k must offer
# ALL of them in every stage-8 prompt (retrieval over 22 candidates is
# near-meaningless anyway), so keep top_k above the label count.
RAG_TOP_K = 25

# -------------------
# Stage 8: chunk labeling (pipeline/08_ensemble_label_chunks.py)
# -------------------
# Study design here: a SINGLE ensemble model, poro2_70b (LumiOpen/
# Llama-Poro-2-70B-Instruct) via CSC LUMI's Aitta hosted-inference API (no
# GPU needed, but per-user rate-limited -- hence a single non-array job).
# The other backend configs are kept so --model_key can switch if needed:
#   "vllm_local": loaded once into a local vLLM engine on the job's own GPU.
#   "aitta": a network call to CSC LUMI's Aitta hosted-inference API
#     (AITTA_BASE_URL below).
#   "alibaba": same shape as "aitta" but against Alibaba Cloud Model
#     Studio's DashScope endpoint (ALIBABA_BASE_URL below).
ENSEMBLE_MODELS = [
    {"key": "poro2_70b",    "hf_name": "LumiOpen/Llama-Poro-2-70B-Instruct",    "family": "lumiopen", "size": "big", "backend": "aitta"},
    {"key": "poro2_8b",     "hf_name": "LumiOpen/Llama-Poro-2-8B-Instruct",     "family": "lumiopen", "size": "small", "backend": "vllm_local"},
    {"key": "qwen3_8b",     "hf_name": "Qwen/Qwen3-8B",                         "family": "qwen",     "size": "small", "backend": "vllm_local"},
    {"key": "qwen36_27b",   "hf_name": "qwen3.6-27b",                           "family": "qwen",     "size": "big",   "backend": "alibaba"},
    {"key": "llama3_1_8b",  "hf_name": "meta-llama/Llama-3.1-8B-Instruct",      "family": "meta",     "size": "small", "backend": "vllm_local"},
    {"key": "llama3_3_70b", "hf_name": "meta-llama/Llama-3.3-70B-Instruct",     "family": "meta",     "size": "big",   "backend": "aitta"},
]
ENSEMBLE_MAX_NEW_TOKENS = 512

# Defensive cap on a single chunk's fused_text within the stage-8 prompt.
# Chunks are normally short, but an unbounded outlier can exceed a
# vllm_local model's context window and crash the whole batch.
CHUNK_TEXT_MAX_CHARS = 4000

# Video title/description are included as context in every chunk's stage-8
# prompt. Descriptions repeat unchanged across every chunk of the same
# video, so they're truncated to bound per-prompt token cost.
VIDEO_DESCRIPTION_MAX_CHARS = 400

# CSC LUMI's hosted-inference API (OpenAI-compatible), used for the "aitta"
# backend model above. Needs AITTA_API_TOKEN in the environment/.env (see
# docs.csc.fi -> Aitta -> "Get your access token"); rate-limited per-minute
# per user, so common/llm_chat.py retries a 429 with a fixed 60s backoff.
AITTA_BASE_URL = "https://aitta-api.csc.fi/openai/v1"
AITTA_RATE_LIMIT_BACKOFF_SECONDS = 60
AITTA_MAX_RETRIES = 5
# Requests within one chat_batch() call run concurrently (worker pool in
# _chat_batch_aitta) to overlap network round-trip latency instead of
# serializing it. Lower this first if the per-minute rate limit trips often.
AITTA_MAX_CONCURRENT_REQUESTS = 8

# Alibaba Cloud Model Studio's DashScope international OpenAI-compatible
# endpoint, used by the "alibaba" backend model (qwen36_27b). Needs
# QWEN_API_KEY in the environment/.env. Same concurrent-worker-pool +
# 429-backoff shape as Aitta above; values are untuned starting guesses.
ALIBABA_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
ALIBABA_RATE_LIMIT_BACKOFF_SECONDS = 60
ALIBABA_MAX_RETRIES = 5
ALIBABA_MAX_CONCURRENT_REQUESTS = 8

# -------------------
# Stage 9: video-level ensemble vote (pipeline/09_aggregate_video_labels.py)
# -------------------
# A model joins the vote only if it processed at least
# ENSEMBLE_MODEL_MIN_COVERAGE of the videos that have fused chunks. With the
# single-model design the vote degenerates to that model's any-chunk union,
# and ENSEMBLE_VOTE_MIN_SELECTED=1 marks vote1of1 as the threshold the
# downstream analysis consumes.
ENSEMBLE_MODEL_MIN_COVERAGE = 0.5
ENSEMBLE_VOTE_MIN_SELECTED = 1
