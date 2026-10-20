#!/bin/bash
#SBATCH --account=project_2006782
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=16G
#SBATCH --partition=gpumedium
#SBATCH --job-name=pilot_probe

# Environment probe for the B-vs-D pilot: checks what a GH200 node offers for
# running Cloudflare/clef outside the vLLM wrapper (bare venv) vs inside it.

set -x
uname -m
nvidia-smi | head -12

echo "=== internet on compute node? ==="
curl -sI --max-time 10 https://pypi.org 2>&1 | head -3
curl -sI --max-time 10 https://huggingface.co 2>&1 | head -3

echo "=== bare node pythons ==="
ls /usr/bin/python3* 2>/dev/null
which python3.12 python3.13 2>/dev/null
python3 --version 2>&1

echo "=== wrapper 0.19.1 container versions ==="
VW=/appl/soft/manual/aida/aarch64/vllm/wrappers/0.19.1
export _O_SOURCE=/dev/null
set +e
source $VW/common.sh
set -e
singularity --silent exec --nv $VW/$CONTAINER_IMAGE python3 -c "
import sys; print('python', sys.version)
import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())
import transformers; print('transformers', transformers.__version__)
try:
    import PIL; print('PIL', PIL.__version__)
except Exception as e: print('PIL MISSING', e)
import vllm; print('vllm', vllm.__version__)
"

echo "=== wrapper 0.29.0 container versions ==="
VW2=/appl/soft/manual/aida/aarch64/vllm/wrappers/0.29.0
if [ -d "$VW2" ]; then
  export _O_SOURCE=/dev/null
  set +e
  source $VW2/common.sh
  set -e
  singularity --silent exec --nv $VW2/$CONTAINER_IMAGE python3 -c "
import sys; print('python', sys.version)
import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())
import transformers; print('transformers', transformers.__version__)
try:
    import PIL; print('PIL', PIL.__version__)
except Exception as e: print('PIL MISSING', e)
import vllm; print('vllm', vllm.__version__)
"
fi

echo "=== bare-node CUDA libs ==="
ls /usr/local/cuda 2>/dev/null | head -3
ldconfig -p 2>/dev/null | grep -c libcudart
echo "probe done"
