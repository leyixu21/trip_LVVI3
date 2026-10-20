#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=16G
#SBATCH --partition=gpumedium


set -e

echo "Starting CLIP keyframe dedup job at $(date)"

# cv_env (this job's original venv) doesn't exist on Roihu -- this now uses
# gpu-env, the same aarch64/GPU-container venv run_01_transcribe_videos.sh
# uses (confirmed to already have torch+CUDA, transformers, and PIL). See
# that script: gpu-env's python wrapper only forwards --nv when invoked
# from inside an already-nv container, so it must be passed explicitly via
# `singularity exec --nv` here or the container never sees the GPU.
VLLM_WRAPPER_BASE=/appl/soft/manual/aida/aarch64/vllm/wrappers/0.19.1
export VIRTUAL_ENV=/projappl/project_2006782/leyixu/gpu-env
export _O_SOURCE=/dev/null  # silences a harmless `dirname: missing operand` from common.sh
# common.sh's own internal `test -e` checks can return non-zero by design;
# under `set -e` that would kill this script before it does anything, so
# errexit is suspended just for the sourcing.
set +e
source "$VLLM_WRAPPER_BASE/common.sh"
set -e

cd /scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis

echo "Running 03_clip_dedup.py..."
singularity --silent exec --nv "$VLLM_WRAPPER_BASE/$CONTAINER_IMAGE" python3 pipeline/03_clip_dedup.py

echo "CLIP dedup job finished at $(date)"
