#!/usr/bin/env bash
# Build the puffin-world environment (Python 3.10, torch 2.7.0 + cu126).
#
# micromamba with a project-local prefix, matching the house convention
# (<project>/envs/<name>-<commit>-py<ver>-cu<ver>): the env is pinned to the
# upstream commit it was built for, so a moved pin cannot silently reuse an env
# built against different requirements.
#
# Order matters: flash-attn compiles against the torch installed in step 2, so
# it cannot be a line in requirements.txt. The host has a driver but no nvcc,
# so the env brings its own CUDA toolkit.
#
#   bash tools/setup/puffin_world_env.sh
#   micromamba activate ./envs/puffin-world-<sha>-py310-cu126
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UP="$ROOT/third_party/clones/puffin/Puffin-World"

[ -f "$UP/requirements.txt" ] || { echo "missing $UP; run tools/setup/clone_upstream.sh" >&2; exit 1; }

# Ignore the broken NVIDIA extra-index in the user-level pip config; see pip.conf.
export PIP_CONFIG_FILE="$ROOT/tools/setup/pip.conf"

SHA="$(git -C "$ROOT/third_party/clones/puffin" rev-parse --short HEAD)"
PREFIX="${ENV_PREFIX:-$ROOT/envs/puffin-world-$SHA-py310-cu126}"
echo "== env prefix: $PREFIX"

# 1. env + CUDA toolkit
if [ ! -x "$PREFIX/bin/python" ]; then
    micromamba create -y -p "$PREFIX" -c conda-forge -c nvidia \
        python=3.10 cuda-toolkit=12.6 ninja git-lfs
else
    echo "   (env prefix exists, reusing)"
fi

# `micromamba shell hook` emits code that reads MAMBA_ROOT_PREFIX unguarded,
# which trips `set -u`; give it a value and relax nounset across the eval only.
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
set +u
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$PREFIX"
set -u
export CUDA_HOME="$PREFIX"

# 2. torch stack first
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

# 3. upstream's pinned deps (transformers/deepspeed/xtuner/diffusers/...)
#
# requirements.txt is a pip freeze taken on the authors' internal cluster and
# pins petrel-oss-sdk, their object-store SDK, which exists on no public index.
# Drop it here rather than patching upstream; src/aoss_client/ supplies a
# filesystem-backed stand-in for the import that needs it.
REQ="$PREFIX/requirements.local.txt"
grep -vE '^(petrel-oss-sdk|aoss-client)\b' "$UP/requirements.txt" > "$REQ"
echo "== dropped from requirements.txt:"
grep -nE '^(petrel-oss-sdk|aoss-client)\b' "$UP/requirements.txt" || echo "   (none)"
pip install -r "$REQ"

# 4. flash-attn, compiled against the torch above (slow: tens of minutes)
MAX_JOBS="${MAX_JOBS:-16}" pip install flash-attn==2.8.3 --no-build-isolation

# 5. mmengine's lazy-import config parser calls pkg_resources, removed in
# setuptools>=81; upstream's own train.py cannot load a config without it.
pip install "setuptools<81"

# 6. this repo's library, so runners can `import gs_playground`
pip install -e "$ROOT"

# gsplat's example trainer (DL3DV reconstruction) needs these beyond the lib
pip install gsplat plyfile spconv-cu126 tyro nerfview splines tensorboard
# gsplat examples need rmbrualla's pycolmap (SceneManager API), NOT the
# official COLMAP bindings that share the name on PyPI
pip install "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"
# fused-ssim's setup.py imports torch, so build isolation must be off
pip install --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"
# compat fixes to site-packages (upstream's pins are mutually incompatible)
bash "$ROOT/tools/setup/fix_env_compat.sh"

# upstream's own sanity check
cd "$UP" && PYTHONPATH=./:${PYTHONPATH:-} python -c \
    "import torch, transformers, deepspeed, xtuner, flash_attn, trimesh; \
     print('ok', torch.__version__, 'cuda', torch.cuda.is_available())"
