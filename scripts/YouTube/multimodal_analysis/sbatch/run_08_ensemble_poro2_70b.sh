#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=small

# No GPU -- served by CSC LUMI's Aitta hosted-inference API (a network
# call, see common/llm_chat.py / config.AITTA_BASE_URL). A single job, not
# an array: Aitta's rate limit is per-user, so concurrent array tasks would
# just contend for the same quota rather than add throughput.
#
# --mem=32G (was 8G in the reference project's first runs): the full-dataset
# run holds every prompt/response in memory until the job finishes (nothing
# is sharded/flushed mid-run), which OOM-killed at 8G partway through a real
# run there.
#
# Needs AITTA_API_TOKEN in .env (docs.csc.fi -> Aitta -> "Get your access
# token") and sentence-transformers installed in cpu-env for the RAG
# retrieval step:
#   /projappl/project_2006782/leyixu/cpu-env/bin/pip install sentence-transformers

set -ex

echo "Starting ensemble labeling (poro2_70b) at $(date)"

source /projappl/project_2006782/leyixu/cpu-env/bin/activate

cd /scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis

echo "Running 08_ensemble_label_chunks.py --model_key poro2_70b --vlm_model_key qwen3_vl..."
python3 pipeline/08_ensemble_label_chunks.py --model_key poro2_70b --vlm_model_key qwen3_vl  # change the VLM key if llava_next was used instead

echo "Ensemble labeling (poro2_70b) finished at $(date)"
