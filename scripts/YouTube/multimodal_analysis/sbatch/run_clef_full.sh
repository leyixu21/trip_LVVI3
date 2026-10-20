#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=64G
#SBATCH --partition=gpumedium
#SBATCH --job-name=clef_full
#SBATCH --array=0-3

# Full run: Cloudflare/clef activity labeling over every transcribed video
# with a keyframe shard (3,745 videos, selection built in
# data/YouTube/clef_labels/full_selection.json). Same setup as
# run_pilot_b_clef.sh (aida vllm wrapper 0.29.0 container, local HF snapshot,
# Allas shards in memory via the S3 fallback). Each array task takes one
# interleaved slice (videos[i::4]) and writes its own
# data/YouTube/clef_labels/clef_full.shard_<i>.jsonl, so tasks never race on
# one file. The runner skips video_ids already present in ITS shard file, so
# a failed task can be resubmitted with sbatch --array=<i> ... without
# redoing finished videos. The activity taxonomy (35 top-level labels) is
# read from data/LVVI3/LVVI3_activities.csv at runtime.

set -e

echo "Starting clef full run (task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT) at $(date) on $(hostname)"

VLLM_WRAPPER_BASE=/appl/soft/manual/aida/aarch64/vllm/wrappers/0.29.0
export _O_SOURCE=/dev/null
set +e
source "$VLLM_WRAPPER_BASE/common.sh"
set -e

export HF_HOME=/scratch/project_2006782/leyixu/transformers_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PILOT_DIR=/scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis/pilot
PYLIBS=/scratch/project_2006782/leyixu/pilot_pylibs/extracted
SNAPSHOT=$(ls -d "$HF_HOME"/hub/models--Cloudflare--clef/snapshots/*/)
LABELS_DIR=/scratch/project_2006782/leyixu/trip_LVVI3/data/YouTube/clef_labels

export PYTHONPATH="$PILOT_DIR:$PYLIBS${PYTHONPATH:+:$PYTHONPATH}"

echo "preflight checks..."
singularity --silent exec --nv "$VLLM_WRAPPER_BASE/$CONTAINER_IMAGE" python3 - <<'EOF'
import socket
for mod in ("torch", "transformers", "PIL", "boto3", "dotenv", "swiftclient"):
    __import__(mod)
print("imports ok")
try:
    socket.create_connection(("a3s.fi", 443), timeout=10).close()
    print("allas S3 endpoint reachable")
except OSError as e:
    raise SystemExit(f"ALLAS UNREACHABLE from compute node: {e}")
EOF

OUT="$LABELS_DIR/clef_full.shard_${SLURM_ARRAY_TASK_ID}.jsonl"

echo "Running run_b_clef.py -> $OUT (snapshot: $SNAPSHOT)..."
singularity --silent exec --nv "$VLLM_WRAPPER_BASE/$CONTAINER_IMAGE" python3 \
    "$PILOT_DIR/run_b_clef.py" \
    --model_path="$SNAPSHOT" \
    --selection="$LABELS_DIR/full_selection.json" \
    --output="$OUT" \
    --shard_index="$SLURM_ARRAY_TASK_ID" \
    --num_shards="$SLURM_ARRAY_TASK_COUNT" \
    --batch_size=2

echo "Finished at $(date)"
