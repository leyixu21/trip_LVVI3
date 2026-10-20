#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=small
#SBATCH --job-name=name_new_act

# Stage-1 naming pass for Clef-detected new activities (text-only, Poro-2-70B
# via CSC Aitta). Runs from the project root so .env (AITTA_API_TOKEN) is
# found by dotenv. Resumable: videos already in new_activity_names.jsonl are
# skipped. Usage: sbatch run_name_new_activities.sh [threshold]

set -e
cd /scratch/project_2006782/leyixu/trip_LVVI3
source /projappl/project_2006782/leyixu/cpu-env/bin/activate

THRESHOLD="${1:-0.5}"
python3 scripts/YouTube/multimodal_analysis/tools/name_new_activities.py --threshold "$THRESHOLD"
