# Scripts overview — trip_LVVI3

Numbered `0N_` prefixes show the run order within each source. All paths inside notebooks are
relative with **kernel CWD = the notebook's own folder** (`../../../data/...` from a
`data_preprocessing` folder, `../../data/...` from `RQs_analysis/`).

Preprocessing prepares data for the RQ analyses; anything that answers a research question
lives in `RQs_analysis/`.

## YouTube — `scripts/YouTube/`

| Script | Purpose | Main outputs |
|---|---|---|
| `data_preprocessing/01_youtube_data_exploration.ipynb` | EDA of the Finland video pool (language, channels, space-time) | none (exploration) |
| `data_preprocessing/02_youtube_nbt_identification.py` | Qwen3-14B Yes/No nature-based-tourism relevance from title+description | `data/YouTube/preprocessing_data/nbt_identification_qwen3_14b_{batch}.csv` → consolidated `nbt_identification_qwen3_14b.csv` |
| `data_preprocessing/03_youtube_nbt_spatial_filtering.ipynb` | NBT filter + spatial intersection with state lands / protected areas | `data/YouTube/nbt_videos_in_state_owned_lands.gpkg/.csv` |
| `data_preprocessing/04_youtube_activity_labeling.ipynb` | **LEGACY** old Qwen3-VL per-shot labels → LVVI3 taxonomy (sub_activity kept at preprocessing level only) | `data/_archive/old_llm_subactivity/youtube/YouTube_inferenced.csv` |
| `data_preprocessing/video_shot_detection.py` | Standalone Google Video Intelligence shot-change helper | none |
| `multimodal_analysis/` | **CURRENT** pipeline: clef decision model activity labeling (`pilot/`, `sbatch/run_clef_full.sh` → `data/YouTube/clef_labels/*.jsonl`) | clef label shards |

## Flickr — `scripts/Flickr/`

| Script | Purpose | Main outputs |
|---|---|---|
| `data_preprocessing/01_flickr_data_download.ipynb` | Reconstruct global Flickr parquet from Allas, bbox + precise intersect with state lands | `data/Flickr/flickr_finland_state_lands_all.gpkg` |
| `data_preprocessing/02_flickr_activity_labeling.ipynb` | **LEGACY** old Qwen3-VL per-image labels → LVVI3 taxonomy | `data/_archive/old_llm_subactivity/flickr/Flickr_inferenced.csv` |
| `multimodal_analysis/` | **CURRENT**: `download_flickr_images.py`, `run_clef_flickr.py` + `sbatch/` → `data/Flickr/clef_labels/flickr_clef.shard_*.jsonl` | clef label shards |

## LVVI3 survey — `scripts/LVVI3/data_preprocessing/`

| Script | Purpose | Main outputs |
|---|---|---|
| `01_lvvi3_data_exploration.ipynb` | Decode survey xlsx (OR + NBT sheets), map coded columns to activity names, translate to English | `data/LVVI3/lvvi3_or.csv`, `lvvi3_nbt.csv`, `lvvi3_or_en.csv`, `lvvi3_nbt_en.csv` |
| `02_lvvi3_data_preprocessing.ipynb` | Harmonize sub_activity variants, map sub_activity → activity/structural_activity | `data/LVVI3/lvvi3_survey_en.csv` (main survey table for all RQ notebooks) |

## Research questions — `scripts/RQs_analysis/`

| Notebook | Purpose |
|---|---|
| `trip_identification.ipynb` | Flickr photo-trip identification (24 h gap rule) → `data/Flickr/flickr_trips.gpkg` / `flickr_trips_all.gpkg`; activity-level trip EDA; figures → `figures/` |
| `data_exploration.ipynb` | Cross-source EDA (LVVI3 + YouTube + Flickr) |
| `GRD_conference.ipynb` | Conference paper analyses (activity comparisons across sources) |
| `RQ1_analysis.ipynb` | RQ1: latent-class (StepMix) analysis of trip × structural_activity |
| `RQ2_analysis.ipynb` | RQ2: visit/trip-type clustering (MCA + agglomerative) |

## Archives / removed

- `data/_archive/old_llm_subactivity/` — old Qwen3-VL sub_activity outputs (see its README; delete after clef aggregation).
- `scripts/_archive/old_llm_subactivity/` — legacy Flickr Qwen3-VL inference scripts.
- Facial recognition (DeepFace/Qwen3-VL gender-age + person detection) was **removed entirely** in 2026-10 reorganization: data deleted, sections stripped from RQ notebooks, `demographic_clustering.ipynb` deleted (recoverable from git history).
- **All preprocessing outputs are CSV** (pkl was converted 2026-10-20); do not reintroduce pkl.

## Known gaps (need reconstruction before reruns)

- `LVVI3_activities_inference_postprocessing.csv` (sub_activity → structural_activity/type mapping) is missing — it lived in the deleted `LVVI3_YouTube` project. Referenced by `04_youtube_activity_labeling.ipynb`, `02_flickr_activity_labeling.ipynb`.
- `data/YouTube/preprocessing_data/videos_info_finland_all.csv` (input of 01/03) is large but present; `ft_language_predict.csv` language cells are stringified lists (`ast.literal_eval` on read).
