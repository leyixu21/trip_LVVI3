#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G


set -euo pipefail

# ------------------------ Load modules ------------------------ #
module purge
module load python-data

# ------------------------ Activate virtual environment ------------------------ #
source /projappl/project_2006782/leyixu/cv_env/bin/activate

# ------------------------ Directories ------------------------ #
cd /scratch/project_2006782/leyixu/LVVI3_YouTube/scripts/Flickr/multimodal_analysis

# ------------------------ Run Python script ------------------------ #
python multimodal_llm_inference.py
