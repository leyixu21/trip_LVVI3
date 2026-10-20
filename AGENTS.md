# Token-saving rules for AI agents working in this project

This project holds ~30k-row CSVs, thousands of per-video JSON files, and large
binaries. Reading them into the conversation context is the main token cost.
Follow these rules to work cheaply.

## 0. First, read this file — do not explore blindly

Most answers need only 1–3 targeted reads, not a directory walk.

## 1. Project layout (know this instead of listing)

- `data/YouTube/` — main YouTube tables and pipeline outputs:
  - `youtube_videos.csv` (renamed from `YouTube_videos.csv`) — **30,615 rows
    but only 17,650 unique `video_id`** (one row per video × protected-area
    intersection). Always `drop_duplicates('video_id')` before counting.
  - `transcripts.jsonl` — Whisper transcripts, 3,763 videos (one JSON line
    each: `video_id, language, segments[{start,end,text,...}], audio_events`).
  - `preprocessing_data/` — intermediate **CSVs** (all CSV since 2026-10;
    `ft_language_predict.csv` has stringified-list cells → `ast.literal_eval`).
  - `videos_tmp/` — stage-01 temp dir (usually empty).
- `data/YouTube/pilot/` — 50-video pilot results (Clef decision model vs
  Qwen3-VL comparison; see `REPORT.md`). Small files, safe to read.
- `data/YouTube/clef_labels/` — **final Clef activity labels** for the full
  YouTube run (3,745 videos): `clef_full.shard_*.jsonl` (raw per-video
  probabilities), `youtube_clef_activities_long.csv` (video × activity tidy
  table), `youtube_clef_videos_wide.csv` (per-video labels + metadata),
  `full_selection.json`. Rebuild with
  `scripts/YouTube/multimodal_analysis/tools/build_clef_outputs.py`.
- `data/Flickr/clef_labels/` — Flickr Clef labels (16,773 photos),
  same record format.
- `data/activities.csv` — master activity taxonomy (survey + future
  social-media activities; `source` column). The inference prompt source
  remains `data/LVVI3/LVVI3_activities.csv` — do NOT edit that one for
  analysis purposes.
- `data/LVVI3/` — LVVI3 survey data, **all CSV since 2026-10**: `lvvi3_survey_en.csv`
  is the main table (451×128, read by every RQ notebook), plus `lvvi3_or.csv`,
  `lvvi3_nbt.csv`, `lvvi3_or_en.csv`, `lvvi3_nbt_en.csv`. `LVVI3_activities.csv`
  = activity taxonomy used for inference (35 activities, `activity` column
  only; small, safe to read). `*.xlsx` — binary, never `Read`.
- `data/Flickr/` — `flickr_finland_state_lands*.gpkg/csv`, `flickr_trips*.gpkg`,
  `images_z/` (downloaded photos), `clef_labels/`. `*.gpkg` — never `Read`.
- `data/boundaries/` — geodata.
- `data/_archive/old_llm_subactivity/{youtube,flickr}/` — **old Qwen3-VL
  sub_activity run** (thousands of files: `inference_json_output/`,
  `accompany_inference_json_output/`, `YouTube_inferenced.csv`,
  `Flickr_inferenced.csv`, `multimodal_json_input/`). Count with
  `ls | wc -l`; read at most one sample. Delete after clef labels fully
  replace it. See its README.
- **Facial recognition was removed entirely (2026-10):** no
  `gender_age_output/`, `person_detection_output/`, `*_faces.csv`, or
  `demographic_clustering.ipynb` anywhere (recoverable from git history only).
- `scripts/YouTube/data_preprocessing/0N_*` — numbered preprocessing steps:
  01 exploration → 02 NBT identification (Qwen3-14B) → 03 NBT spatial
  filtering → 04 activity labeling (**LEGACY**, reads the archive; superseded
  by the clef pipeline). `scripts/LVVI3/data_preprocessing/0N_*` — 01 survey
  decode/translate → 02 sub_activity→activity mapping → `lvvi3_survey_en.csv`.
  `scripts/Flickr/data_preprocessing/0N_*` — 01 download → 02 activity
  labeling (**LEGACY**). Full table: `scripts/README.md`.
- `scripts/RQs_analysis/` — research-question analyses, incl.
  `trip_identification.ipynb` (Flickr trip building, moved here from
  preprocessing 2026-10) and generated maps in `figures/`.
- `scripts/YouTube/multimodal_analysis/` — **current** multimodal pipeline
  (ported from `influencer_YouTube`): `pipeline/01…09_*.py`, `common/`,
  `sbatch/`, `tests/`, `README.md`. Stages 01–03 = transcribe / shots+keyframes
  / CLIP dedup; 04 = frame captioning (Qwen3-VL); 05 = align & fuse 20 s
  chunks; 06 = privacy scrub; 07 = taxonomy RAG index; 08 = LLM chunk labeling
  (activities only, LVVI3 22-activity taxonomy); 09 = video-level aggregation.
- **Keyframes exist only in Allas** (container per `common/config.py`), never
  on local disk. Videos likewise. `.env` holds Allas/API credentials — never
  read it; use env var names from `common/llm_chat.py` / `config.py`.

## 2. Never read these into context

- `*.pkl` / `*.xlsx` / `*.gpkg` — binary (no pkl should remain in `data/` —
  preprocessing outputs are CSV since 2026-10; if you find one, convert it).
  Inspect via a short `python -c` printing shape/columns/head(3) only.
- `*.jsonl`, big `*.csv` — never read whole. `head -n 2` for schema or
  `python -c "...df.head(3).to_string()"`.
- `*.ipynb` — full outputs can be 10k+ tokens. Use
  `/projappl/project_2006782/leyixu/cpu-env/bin/jupyter nbconvert --to script --stdout <nb> | head -n 80`
  (the system `jupyter` lacks nbconvert).
- `slurm-*.out` logs — `tail -n 30` at most.
- `data/_archive/old_llm_subactivity/` — one sample max, never a listing
  with contents.
- `.env` — never.

## 3. Prefer targeted commands over reads

- Find code with `Grep` on a specific pattern, not Glob + read everything.
- Columns/shape: `python -c "import pandas as pd; df=pd.read_csv('...'); print(df.shape); print(list(df.columns))"`.
- Compare with sibling project: `diff influencer_YouTube/scripts/multimodal_analysis/<f> trip_LVVI3/scripts/YouTube/multimodal_analysis/<f>` and read only differing hunks.

## 4. Delegate heavy exploration

If a task requires scanning many files, spawn an explore subagent and ask for a
summary. Do not page large outputs through this conversation.

## 5. Keep the loop cheap

- State assumptions and act when a wrong guess is low-cost.
- Batch related questions in one message.
- Summarize command output; point to file:line instead of pasting.

## 6. Notes

- Manual ground truth (Label Studio, 588 tasks) is in git history:
  `scripts/YouTube/multimodal_analysis/labeled_tasks*.json` (deleted from the
  working tree; restore with `git checkout -- <path>` if needed).
- HPC: project account `project_2006782`, Puhti/Mahti-style sbatch under
  `sbatch/`; GPU stages use the gh200 partition + singularity wrapper and
  venvs under `/projappl/project_2006782/leyixu/`.
