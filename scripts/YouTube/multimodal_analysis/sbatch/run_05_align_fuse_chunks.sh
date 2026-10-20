#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=16G
#SBATCH --partition=gpumedium

set -e

echo "Starting CLIP alignment + chunk fusion job at $(date)"

# CLIP's text tower is tiny, but cpu-env (x86_64) has no torch/transformers
# and we deliberately don't want to duplicate those there -- the vllm
# module's container already ships them (matched to CUDA 12.9/aarch64), so
# this runs through the same gpu-env container as stages 1/4 instead. See
# run_01_transcribe_videos.sh for the wrapper-sourcing rationale.
VLLM_WRAPPER_BASE=/appl/soft/manual/aida/aarch64/vllm/wrappers/0.19.1
export VIRTUAL_ENV=/projappl/project_2006782/leyixu/gpu-env
export _O_SOURCE=/dev/null  # silences a harmless `dirname: missing operand` from common.sh
set +e
source "$VLLM_WRAPPER_BASE/common.sh"
set -e

cd /scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis

# Which stage-4 VLM's captions to fuse (llava_next or qwen3_vl) -- pick
# whichever you settled on after comparing them:
#   sbatch run_05_align_fuse_chunks.sh qwen3_vl
MODEL_KEY="${1:?Usage: sbatch run_05_align_fuse_chunks.sh <llava_next|qwen3_vl>}"

echo "Running 05_align_fuse_chunks.py --model_key $MODEL_KEY..."
singularity --silent exec --nv "$VLLM_WRAPPER_BASE/$CONTAINER_IMAGE" python3 pipeline/05_align_fuse_chunks.py --model_key "$MODEL_KEY"

echo "Alignment + fusion job finished at $(date)"
