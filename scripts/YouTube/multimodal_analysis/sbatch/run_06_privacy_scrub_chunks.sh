#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=16G
#SBATCH --partition=gpumedium

set -e

echo "Starting privacy NER scrub job at $(date)"

# Same rationale as run_05_align_fuse_chunks.sh: the NER model is a plain
# transformers AutoModelForTokenClassification, and gpu-env's container
# already ships torch/transformers matched to CUDA/aarch64, so this runs
# through it rather than duplicating those packages in cpu-env.
VLLM_WRAPPER_BASE=/appl/soft/manual/aida/aarch64/vllm/wrappers/0.19.1
export VIRTUAL_ENV=/projappl/project_2006782/leyixu/gpu-env
export _O_SOURCE=/dev/null
set +e
source "$VLLM_WRAPPER_BASE/common.sh"
set -e

cd /scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis

# Must match whichever VLM stage 5 fused (see run_05_align_fuse_chunks.sh) --
# scrubs data/YouTube/fused_chunks_{MODEL_KEY}.jsonl in place.
MODEL_KEY="${1:?Usage: sbatch run_06_privacy_scrub_chunks.sh <llava_next|qwen3_vl>}"

echo "Running 06_privacy_scrub_chunks.py --model_key $MODEL_KEY..."
singularity --silent exec --nv "$VLLM_WRAPPER_BASE/$CONTAINER_IMAGE" python3 pipeline/06_privacy_scrub_chunks.py --model_key "$MODEL_KEY"

echo "Privacy NER scrub job finished at $(date)"
