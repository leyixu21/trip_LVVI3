#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=small

# Stage 9: video-level ensemble vote over the chunk labels. With this
# project's single-model design the vote degenerates to poro2_70b's
# any-chunk union (vote1of1); the multi-model threshold files are still
# written if more models are added later. Pure CPU (pandas only) -- runs in
# seconds via the cpu-env venv, no GPU, no container. Needs the merged
# chunk labels (chunk_labels/{taxonomy}/{model_key}.json) to exist, i.e.
# stage 8 (+ run_08_merge_labels.sh if sharded) done first.
#
# Usage:
#   sbatch sbatch/run_09_aggregate_video_labels.sh            # qwen3_vl chunks
#   sbatch sbatch/run_09_aggregate_video_labels.sh llava_next # if that VLM was used

set -e

echo "Starting video-level ensemble vote job at $(date)"

module load python-data
source /projappl/project_2006782/leyixu/cpu-env/bin/activate

cd /scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis

VLM_MODEL_KEY="${1:-qwen3_vl}"
echo "Running 09_aggregate_video_labels.py --vlm_model_key $VLM_MODEL_KEY..."
python pipeline/09_aggregate_video_labels.py --vlm_model_key "$VLM_MODEL_KEY"

echo "Ensemble vote job finished at $(date)"
