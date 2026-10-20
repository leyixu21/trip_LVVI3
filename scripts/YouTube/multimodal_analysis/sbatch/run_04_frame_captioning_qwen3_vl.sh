#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=48G
#SBATCH --partition=gpumedium
#SBATCH --array=0-9

NUM_CHUNKS=10

set -ex

echo "Starting VLM frame captioning job (qwen3_vl) chunk $SLURM_ARRAY_TASK_ID/$NUM_CHUNKS at $(date)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

VLLM_WRAPPER_BASE=/appl/soft/manual/aida/aarch64/vllm/wrappers/0.19.1
export VIRTUAL_ENV=/projappl/project_2006782/leyixu/gpu-env
export _O_SOURCE=/dev/null  # silences a harmless `dirname: missing operand` from common.sh
# common.sh does its own internal (non-fatal-by-design) `test -e` checks
# that can return non-zero; under `set -e` that would kill this script
# before it ever gets to the actual job, so errexit is suspended just for
# the sourcing.
set +e
source "$VLLM_WRAPPER_BASE/common.sh"
set -e

cd /scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis

echo "Running 04_frame_captioning.py --model_key qwen3_vl --chunk_index $SLURM_ARRAY_TASK_ID --num_chunks $NUM_CHUNKS..."
singularity --silent exec --nv "$VLLM_WRAPPER_BASE/$CONTAINER_IMAGE" python3 pipeline/04_frame_captioning.py \
    --model_key qwen3_vl --chunk_index "$SLURM_ARRAY_TASK_ID" --num_chunks "$NUM_CHUNKS"

echo "Frame captioning job (qwen3_vl) chunk $SLURM_ARRAY_TASK_ID/$NUM_CHUNKS finished at $(date)"
