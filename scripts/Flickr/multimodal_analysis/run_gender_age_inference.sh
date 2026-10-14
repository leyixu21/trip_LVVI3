#!/bin/bash
#SBATCH --account=project_2006782
#SBATCH --time=08:00:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module load python-data

source /projappl/project_2006782/leyixu/cv_env/bin/activate

# CPU-only execution
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=2

SCRATCH_DIR=/scratch/project_2006782/leyixu/LVVI3_YouTube/scripts/Flickr/multimodal_analysis
cd "$SCRATCH_DIR"

python gender_age_inference.py