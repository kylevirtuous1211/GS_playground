#!/usr/bin/env bash
# Build the SuRFLo environment (Python 3.10, torch 2.4.1 + cu124).
#
#   bash tools/setup/surflo_env.sh
#
# Upstream's "Option A" with their cu124 recipe, the one they test most: a
# self-contained micromamba prefix bringing its own nvcc 12.4 and gxx 13, so
# nothing on the host has to match (the host has a driver but no nvcc). The
# prefix is named after the pinned upstream commit, as the house convention
# does, so a moved pin cannot silently reuse an env built for another one.
#
# Everything the clone builds lands untracked inside it (egg-info, the
# rasterizer build dir); its tracked tree stays at the pin.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLONE="$ROOT/third_party/clones/surflo"
[ -f "$CLONE/install/environment-cu124.yml" ] || {
    echo "missing $CLONE; run tools/setup/clone_upstream.sh" >&2; exit 1; }

# Ignore the broken NVIDIA extra-index in the user-level pip config; see pip.conf.
export PIP_CONFIG_FILE="$ROOT/tools/setup/pip.conf"

SHA="$(git -C "$CLONE" rev-parse --short HEAD)"
PREFIX="$ROOT/envs/surflo-$SHA-py310-cu124"
echo "== env prefix: $PREFIX"

echo "== 1. submodules (nvdiffrast, Depth-Anything-3), at the commits the pin records"
git -C "$CLONE" submodule update --init

echo "== 2. conda env from upstream's yml"
if [ ! -x "$PREFIX/bin/python" ]; then
    # From install/, so the yml's `-r requirements-cu124.txt` resolves.
    ( cd "$CLONE/install" && micromamba create -y -p "$PREFIX" -f environment-cu124.yml )
else
    echo "   (env prefix exists, reusing)"
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
set +u
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$PREFIX"
set -u
python -c "import torch; assert torch.__version__.startswith('2.4.1'), torch.__version__; print('   torch', torch.__version__)"

echo "== 3. surflo itself, editable"
python -m pip install -e "$CLONE[texture]"

echo "== 4. CUDA hook (CUDA_HOME, CC, CXX from the prefix), then reactivate"
( cd "$CLONE" && CONDA_PREFIX="$PREFIX" bash install/activate_cuda.sh )
set +u
micromamba deactivate
micromamba activate "$PREFIX"
set -u
echo "   CUDA_HOME=$CUDA_HOME  nvcc: $(nvcc --version | tail -1)"

echo "== 5. CUDA extensions, sm_89 only (RTX 6000 Ada)"
( cd "$CLONE" && TORCH_CUDA_ARCH_LIST="8.9" bash install/build_extensions.sh --all )

echo "== 6. addict: Depth-Anything-3 imports it and nothing installs it"
# Found by running verify_install.py: without it the monodepth expert, which
# every guided preset but `no_expert` uses, reports "addict not importable".
python -m pip install addict

echo "== 7. verify"
( cd "$CLONE" && python install/verify_install.py --check-isolation )
