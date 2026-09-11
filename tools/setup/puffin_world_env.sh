#!/usr/bin/env bash
# Build the `puffin-world` environment (Python 3.10, torch 2.7.0 + cu126).
#
# micromamba, not conda: flash-attn needs a real CUDA toolkit in the env, and
# the order below matters -- flash-attn compiles against the torch installed in
# step 2, so it cannot be a line in requirements.txt.
#
#   bash tools/setup/puffin_world_env.sh
#   micromamba activate puffin-world
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UP="$ROOT/third_party/clones/puffin/Puffin-World"
ENV_NAME="${ENV_NAME:-puffin-world}"

[ -f "$UP/requirements.txt" ] || { echo "missing $UP; run tools/setup/clone_upstream.sh" >&2; exit 1; }

# 1. env + CUDA toolkit (the host has a driver but no nvcc; flash-attn needs one)
micromamba create -y -n "$ENV_NAME" -c conda-forge -c nvidia \
    python=3.10 cuda-toolkit=12.6 ninja

eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ENV_NAME"

# 2. torch stack first
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

# 3. upstream's pinned deps (transformers/deepspeed/xtuner/diffusers/...)
pip install -r "$UP/requirements.txt"

# 4. flash-attn, compiled against the torch above
pip install flash-attn==2.8.3 --no-build-isolation

# 5. this repo's library, so runners can `import gs_playground`
pip install -e "$ROOT"

# sanity check (upstream's own)
cd "$UP" && PYTHONPATH=./:${PYTHONPATH:-} python -c \
    "import torch, transformers, deepspeed, xtuner, flash_attn, trimesh; print('ok', torch.__version__, torch.cuda.is_available())"
