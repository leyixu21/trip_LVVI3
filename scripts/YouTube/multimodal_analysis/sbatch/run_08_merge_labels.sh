#!/bin/bash

# Concatenates every chunk's shard files (written by a run_08_ensemble_*
# job -- an array for vllm_local models, a single job for aitta models)
# into the single CHUNK_LABELS_DIR/{taxonomy_name}/{model_key}.json stage 9
# reads -- see pipeline/08_ensemble_label_chunks.py's --merge_only. With the
# single aitta job (run_08_ensemble_poro2_70b.sh) this is only needed if the
# job's own end-of-run merge (num_chunks == 1) didn't happen; keep it for
# sharded/re-run cases. No GPU and no architecture-specific container:
# SentenceTransformer/vllm/openai are all lazily imported inside
# 08_ensemble_label_chunks.py, so --merge_only only ever touches
# pandas/numpy/dotenv, already present in the plain x86_64 cpu-env.
#
#   sbatch run_08_ensemble_poro2_70b.sh                  # single job, note the job id
#   sbatch --dependency=afterok:<that job id> run_08_merge_labels.sh poro2_70b

#SBATCH --account=project_2006782
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=small

MODEL_KEY="${1:?Usage: sbatch run_08_merge_labels.sh <model_key>}"

set -ex

echo "Starting ensemble-label shard merge (model_key=$MODEL_KEY) at $(date)"

source /projappl/project_2006782/leyixu/cpu-env/bin/activate

cd /scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis

python3 pipeline/08_ensemble_label_chunks.py --model_key "$MODEL_KEY" --merge_only

echo "Merge finished at $(date)"
