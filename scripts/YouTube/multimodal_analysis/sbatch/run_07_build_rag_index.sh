#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=small

set -e

echo "Starting taxonomy RAG index build job at $(date)"

# This job only ever runs on Roihu's x86_64 CPU partitions ("small"), but
# sbatch by default exports the submitting shell's full environment into
# the job -- including MODULEPATH. If that shell's session started on an
# aarch64 node (e.g. from working on the GPU stages), the inherited
# MODULEPATH points at aarch64 module trees, and `module load` here
# silently resolves aarch64 binaries that then fail with "Exec format
# error" on this (x86_64) node. Rebuild MODULEPATH from Roihu's own
# x86_64 init file first so module resolution never depends on the
# submitting shell's architecture. (Same gotcha as run_02_detect_shots_keyframes.sh.)
unset MODULEPATH
while read -r entry; do
    for expanded in $entry; do
        [ -d "$expanded" ] && MODULEPATH="${MODULEPATH:+$MODULEPATH:}$expanded"
    done
done < /appl/modulefiles/.modulepath_init_x86_64
export MODULEPATH

module purge
module load python-data
source /projappl/project_2006782/leyixu/cpu-env/bin/activate

cd /scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis

echo "Running 07_build_taxonomy_rag_index.py..."
python pipeline/07_build_taxonomy_rag_index.py

echo "RAG index build job finished at $(date)"
