# LVVI3 YouTube multimodal analysis pipeline

Ported from `influencer_YouTube/scripts/multimodal_analysis`, adapted to the
LVVI3 taxonomy (activities-only) and this project's paths/data. Labels the
3,763 transcribed YouTube videos with LVVI3 outdoor-activity labels, using
Whisper transcripts + VLM frame captions, and a single LLM
(poro2_70b via CSC LUMI's Aitta API) for chunk-level labeling.

## Layout

- `pipeline/01..09_*.py` — the stages (run from the `sbatch/` wrappers).
- `common/` — shared config and utils (`config.py` holds every path and
  parameter; nothing is hardcoded elsewhere). All paths can be overridden
  with `MMA_DATA_DIR` / `MMA_CACHE_DIR` for off-cluster smoke tests.
- `sbatch/` — SLURM wrappers (GPU stages use the gh200 partition +
  gpu-env singularity container; CPU stages use the `small` partition +
  cpu-env venv).
- `tests/test_pipeline_logic.py` — GPU-free sanity checks for the
  pure-Python logic; run with the cpu-env python.

## Data locations (all under `/scratch/project_2006782/leyixu/trip_LVVI3/data`)

- `YouTube/youtube_videos.csv` — master video table. **30,615 rows but only
  17,650 unique `video_id`** (one row per video × protected-area
  intersection) — every stage `drop_duplicates("video_id")` before use.
- `YouTube/transcripts.jsonl` — Whisper transcripts (3,763 videos), written
  by stage 01.
- Keyframes and source videos live **only in Allas**:
  `2006782_youtube_videos` (mp4s) and `2006782_youtube_videos_keyframes`
  (`{video_id}.tar.gz` per video: `shots.json` + frame jpgs, handled
  in-memory by `common/keyframe_io.py`).
- `LVVI3/LVVI3_activities.csv` — the taxonomy: 22 unique top-level
  `activity` labels over 86 sub-activity rows. We label at activity level;
  the CSV has no definition columns, so the retrieval/display text is just
  the label.
- Stage outputs (all in `data/YouTube/`): `frame_captions_{vlm}.jsonl` (04),
  `fused_chunks_{vlm}.jsonl` (05, scrubbed in place by 06),
  `rag_index/activities.pkl` (07), `chunk_labels/activities/{model}.json` (08),
  `video_labels/activities_vote{1..N}of{N}.pkl` (09).

## Stages

1. **01 transcribe** — Whisper (large-v2) per-video transcription from
   Allas, with hallucination filtering; appends to `transcripts.jsonl.*`
   shards per array task. `sbatch/run_01_transcribe_videos.sh`, then
   `cat transcripts.jsonl.* > transcripts.jsonl`.
2. **02 shots+keyframes** — PySceneDetect adaptive shot detection + keyframe
   extraction; uploads `{video_id}.tar.gz` shards to Allas (nothing local).
   `sbatch/run_02_detect_shots_keyframes.sh` (array over `video_ids_filtered.txt`).
3. **03 CLIP dedup** — drops near-duplicate keyframes within each video's
   shard (re-uploads pruned shard). `sbatch/run_03_clip_dedup.sh`.
4. **04 frame captioning** — shot-bundled multi-image VLM captioning
   (qwen3_vl; llava_next kept as an alternative `--model_key`), strictly
   descriptive prompts so no taxonomy vocabulary leaks. Videos missing an
   Allas shard are skipped + counted. Job array over `--chunk_index`; merge
   with `04_frame_captioning.py --model_key qwen3_vl --merge_only` (the
   single-job case merges automatically).
   `sbatch/run_04_frame_captioning_qwen3_vl.sh`.
5. **05 align+fuse** — shot-aware 20 s chunks fusing transcript segments and
   captions (`fused_text = "Transcript: ...\nVisual observations: ..."`),
   with per-chunk CLIP alignment scores. Needs a GPU for CLIP.
   `sbatch/run_05_align_fuse_chunks.sh qwen3_vl`.
6. **06 privacy scrub** — regex + multilingual NER PII redaction, in place on
   `fused_chunks_{vlm}.jsonl` (mandatory before prompts go to hosted APIs).
   `sbatch/run_06_privacy_scrub_chunks.sh qwen3_vl`.
7. **07 RAG index** — sentence-transformers embeddings over the 22 activity
   labels → `rag_index/activities.pkl`. `RAG_TOP_K=25` > 22, so every
   stage-8 prompt effectively offers all candidates.
   `sbatch/run_07_build_rag_index.sh`.
8. **08 chunk labeling** — per chunk: RAG candidates → prompt →
   `common/llm_chat.chat_batch`. Single model by design: **poro2_70b via
   Aitta** (`AITTA_API_TOKEN` in `.env`), one non-array CPU job (per-user
   rate limits make arrays counterproductive). Other backends
   (vllm_local/alibaba models) stay in `config.ENSEMBLE_MODELS` so
   `--model_key` can switch. `sbatch/run_08_ensemble_poro2_70b.sh`;
   `sbatch/run_08_merge_labels.sh poro2_70b` only if a sharded run was used.
9. **09 video vote** — any-chunk union per model, cross-model vote at every
   threshold `vote1ofN..voteNofN`. With the single-model design this
   degenerates to `activities_vote1of1.pkl`
   (`ENSEMBLE_VOTE_MIN_SELECTED=1`). `sbatch/run_09_aggregate_video_labels.sh`.

## Scope decisions (vs the reference pipeline)

- **Activities-only**: `config.TAXONOMIES` has just the LVVI3 activities
  entry; the stage-8 prompt's experiences wording and codebook-criteria
  bullets are removed.
- **Single ensemble model** (poro2_70b/aitta) instead of a 6-model ensemble;
  stage-9 machinery kept so extra models can be folded in later.
- Dedup of `video_id` everywhere the videos CSV is read; transcripts input
  is `data/YouTube/transcripts.jsonl` (not `transcripts_whisper.jsonl`).
- Stage 04 tolerates missing Allas keyframe shards (skip + count).

## Testing

```bash
/projappl/project_2006782/leyixu/cpu-env/bin/python3 tests/test_pipeline_logic.py
```

106 checks, no GPU/Allas/network needed (heavy model imports are only
exercised through fakes; the stage-06 module import does need torch +
transformers, which is why cpu-env's python is used).
