#!/bin/bash
#SBATCH --account=project_2006782
#SBATCH --time=00:10:00
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G

set -euo pipefail

# ------------------------ Load modules ------------------------ #
module purge
module load python-data

# ------------------------ Directories ------------------------ #
cd /scratch/project_2006782/leyixu/LVVI3_YouTube/scripts/Flickr/multimodal_analysis/

# ------------------------ Run Python script ------------------------ #
python multimodal_json_input.py
