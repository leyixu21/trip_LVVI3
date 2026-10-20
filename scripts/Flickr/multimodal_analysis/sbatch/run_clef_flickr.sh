#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=64G
#SBATCH --partition=gpumedium
#SBATCH --job-name=flickr_clef
#SBATCH --array=0-3

# Flickr activity labeling with Cloudflare/clef over the photos downloaded to
# data/Flickr/images_z/ (see download_flickr_images.py). Same container/setup
# as run_clef_full.sh; each array task takes one interleaved slice
# (photos[i::4]) and writes its own
# data/Flickr/clef_labels/flickr_clef.shard_<i>.jsonl. Resumable per shard.

set -e

echo "Starting flickr clef (task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT) at $(date) on $(hostname)"

VLLM_WRAPPER_BASE=/appl/soft/manual/aida/aarch64/vllm/wrappers/0.29.0
export _O_SOURCE=/dev/null
set +e
source "$VLLM_WRAPPER_BASE/common.sh"
set -e

export HF_HOME=/scratch/project_2006782/leyixu/transformers_cache
FLICKR_SCRIPT=/scratch/project_2006782/leyixu/trip_LVVI3/scripts/Flickr/multimodal_analysis/run_clef_flickr.py
PILOT_DIR=/scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis/pilot
PYLIBS=/scratch/project_2006782/leyixu/pilot_pylibs/extracted
SNAPSHOT=$(ls -d "$HF_HOME"/hub/models--Cloudflare--clef/snapshots/*/)
FLICKR_DIR=/scratch/project_2006782/leyixu/trip_LVVI3/data/Flickr

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PILOT_DIR:$PYLIBS${PYTHONPATH:+:$PYTHONPATH}"

echo "preflight checks..."
singularity --silent exec --nv "$VLLM_WRAPPER_BASE/$CONTAINER_IMAGE" python3 - <<'EOF'
for mod in ("torch", "transformers", "PIL"):
    __import__(mod)
print("imports ok")
EOF

OUT="$FLICKR_DIR/clef_labels/flickr_clef.shard_${SLURM_ARRAY_TASK_ID}.jsonl"

echo "Running run_clef_flickr.py -> $OUT (snapshot: $SNAPSHOT)..."
singularity --silent exec --nv "$VLLM_WRAPPER_BASE/$CONTAINER_IMAGE" python3 \
    "$FLICKR_SCRIPT" \
    --model_path="$SNAPSHOT" \
    --selection="$FLICKR_DIR/clef_labels/flickr_selection.json" \
    --images_dir="$FLICKR_DIR/images_z" \
    --output="$OUT" \
    --shard_index="$SLURM_ARRAY_TASK_ID" \
    --num_shards=4 \
    --batch_size=4

echo "Finished at $(date)"
