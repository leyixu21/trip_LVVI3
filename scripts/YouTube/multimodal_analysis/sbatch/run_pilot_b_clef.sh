#!/bin/bash

#SBATCH --account=project_2006782
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=64G
#SBATCH --partition=gpumedium
#SBATCH --job-name=pilot_b_clef

# Pilot B: Cloudflare/clef (27B decision model) on the 50-video pilot.
# Runs inside the aida vllm wrapper 0.29.0 container (torch 2.13,
# transformers 5.17, PIL -- the stock env can host Clef; the model card's
# torch 2.11 + transformers 5.10.2 combo is not installable here because the
# login node is x86_64 while GH200 nodes are aarch64 and compute nodes have
# no internet). Clef's joint_schema_model.py is imported from the local HF
# snapshot. boto3/dotenv/swiftclient come from a pure-python PYTHONPATH dir
# prepared on the login node; Allas shards are fetched in memory via the
# pilot's S3 fallback (the .env OpenStack credentials are expired).

set -e

echo "Starting pilot B (clef) at $(date) on $(hostname)"

VLLM_WRAPPER_BASE=/appl/soft/manual/aida/aarch64/vllm/wrappers/0.29.0
export _O_SOURCE=/dev/null
set +e
source "$VLLM_WRAPPER_BASE/common.sh"
set -e

export HF_HOME=/scratch/project_2006782/leyixu/transformers_cache
PILOT_DIR=/scratch/project_2006782/leyixu/trip_LVVI3/scripts/YouTube/multimodal_analysis/pilot
PYLIBS=/scratch/project_2006782/leyixu/pilot_pylibs/extracted
SNAPSHOT=$(ls -d "$HF_HOME"/hub/models--Cloudflare--clef/snapshots/*/)

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

echo "Running run_b_clef.py (snapshot: $SNAPSHOT)..."
EXTRA_ARGS=()
if [ -n "$1" ]; then EXTRA_ARGS=(--limit="$1"); fi
singularity --silent exec --nv "$VLLM_WRAPPER_BASE/$CONTAINER_IMAGE" python3 \
    "$PILOT_DIR/run_b_clef.py" \
    --model_path="$SNAPSHOT" \
    "${EXTRA_ARGS[@]}"

echo "Finished at $(date)"
