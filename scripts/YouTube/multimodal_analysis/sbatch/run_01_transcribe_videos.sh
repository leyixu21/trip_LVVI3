#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=32G
#SBATCH --partition=gpumedium
#SBATCH --array=0-15

# Usage:
#   sbatch run_01_transcribe_videos.sh [video_ids_file]
# video_ids_file is optional -- one video id per line. Defaults to every
# video_id in common/config.py's VIDEOS_CSV (see 01_transcribe_videos.py).
#
# Runs as a Slurm array: each of the SLURM_ARRAY_TASK_COUNT tasks above gets
# its own GPU and processes video_ids[SLURM_ARRAY_TASK_ID::SLURM_ARRAY_TASK_COUNT]
# (interleaved, so long/short videos are spread evenly across tasks), writing
# to its own transcripts.jsonl.$SLURM_ARRAY_TASK_ID so tasks never race on
# one file. Re-running (whole array or a single failed --array=N task) skips
# whatever's already in any transcripts.jsonl* file. Once every task has
# finished, merge with:
#   cat data/YouTube/transcripts.jsonl.* > data/YouTube/transcripts.jsonl
#
# --time and --array's width are both starting points, not measured: adjust
# --array=0-N to trade GPU-parallelism for queue wait, and --time once you've
# seen one task's actual videos/hour on this node. Each task is independently
# resumable, so a task that times out can just be resubmitted (`sbatch
# --array=<id> run_01_transcribe_videos.sh`) rather than losing its progress.

set -e

echo "Starting transcribe-videos job at $(date) (array task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT)"
echo "Running on host: $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"

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

cd /scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis/pipeline

VIDEO_IDS_FILE_ARG=()
if [ -n "$1" ]; then
    VIDEO_IDS_FILE_ARG=(--video_ids_file="$1")
fi

echo "Running 01_transcribe_videos.py..."
singularity --silent exec --nv "$VLLM_WRAPPER_BASE/$CONTAINER_IMAGE" python3 01_transcribe_videos.py \
    --shard_index="$SLURM_ARRAY_TASK_ID" \
    --num_shards="$SLURM_ARRAY_TASK_COUNT" \
    "${VIDEO_IDS_FILE_ARG[@]}"

echo "Finished at $(date)"
