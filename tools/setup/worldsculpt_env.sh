#!/usr/bin/env bash
# Build the WorldSculpt overlay env.
#
#   bash tools/setup/worldsculpt_env.sh
#
# Layered on the puffin-world base like envs/trellis-overlay, so the base env
# is never mutated. The base supplies torch 2.7+cu126, flash-attn, the CUDA
# toolkit, and the stock spconv/cumm wheels, which is what TRELLIS.2 wants.
#
# Deliberately a separate overlay from envs/worldgrow-overlay: WorldGrow
# installs forked int32 spconv/cumm that TRELLIS.2 was not built against, and
# one shared overlay would force one of the two methods onto the other's fork.
#
# natten is compiled for this machine's compute capability only (8.9, RTX 6000
# Ada). It is the slow step; everything else is wheels.
#
# transformers is pinned to 4.57.1 per WorldSculpt's README. The base env
# carries transformers 5.3.0 for Puffin, and the overlay shadows it here
# without touching that.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OVERLAY="$ROOT/envs/worldsculpt-overlay"

# shellcheck source=/dev/null
source "$ROOT/tools/setup/activate_puffin_world.sh"
BASE_PY="$(command -v python)"
echo "== base python: $BASE_PY"

if [ ! -x "$OVERLAY/bin/python" ]; then
    # --seed so the venv carries its own pip. Everything below installs with
    # pip rather than `uv pip` on purpose: uv resolves against the venv alone
    # and does not count the base env's torch as installed, so any package
    # that depends on torch drags in a second torch (and a full CUDA 13 wheel
    # set) that shadows the base one. pip honours system-site-packages and
    # sees torch 2.7+cu126 already there.
    uv venv --system-site-packages --seed --python "$BASE_PY" "$OVERLAY"
fi
# shellcheck source=/dev/null
source "$OVERLAY/bin/activate"
echo "== overlay python: $(command -v python)"

ARCH="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
echo "== compute capability: $ARCH"

echo "== wheels"
python -m pip install --quiet --upgrade pip
python -m pip install \
    "transformers==4.57.1" peft diffusers accelerate \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    trimesh zstandard kornia timm gradio plyfile matplotlib \
    scikit-image scikit-learn fpsample iopath pycocotools ftfy \
    "setuptools<81" cmake ninja

# The wheel WorldSculpt's README pins; the PyPI utils3d is a different package
# at a different API.
python -m pip install --no-deps \
    "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"

# natten's build needs cmake on PATH, which the wheel above provides.
echo "== natten 0.21.0 (compiles, slow)"
NATTEN_CUDA_ARCH="$ARCH" NATTEN_N_WORKERS="${NATTEN_N_WORKERS:-8}" \
    python -m pip install "natten==0.21.0" --no-build-isolation

# WorldSculpt's README lists none of these, because it says "follow the
# TRELLIS.2 installation guide" and they live there. The vendored `pixal3d`
# package imports all of them: cumesh and o_voxel at module load, nvdiffrast
# for rendering. Missing cumesh is what the first run here died on, three
# stages into the pipeline.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$ARCH}"
export MAX_JOBS="${MAX_JOBS:-16}"

EXT="${EXT_DIR:-/tmp/gs_playground_ext}"
mkdir -p "$EXT"
echo "== cumesh (compiles CUDA)"
[ -d "$EXT/CuMesh/.git" ] || git clone --recursive \
    https://github.com/JeffreyXiang/CuMesh.git "$EXT/CuMesh"
python -m pip install --no-build-isolation "$EXT/CuMesh"

# o_voxel's C++ sources include <Eigen/Dense>, and neither the base env nor
# the build requirements provide it. Eigen is header-only, so fetching the
# release and putting it on CPATH is enough; installing it into the base env
# would mutate an env other studies depend on.
echo "== Eigen headers for o_voxel"
[ -d "$EXT/eigen-3.4.0" ] || {
    curl -sSL -o "$EXT/eigen.tar.gz" \
        "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
    tar xzf "$EXT/eigen.tar.gz" -C "$EXT"
}
export CPATH="$EXT/eigen-3.4.0:${CPATH:-}"

echo "== o_voxel, from the pinned TRELLIS.2 clone (pulls flex_gemm too)"
python -m pip install --no-build-isolation "$ROOT/third_party/clones/trellis2/o-voxel"

echo "== nvdiffrast"
python -m pip install --no-build-isolation \
    "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git"

python - <<'PY'
import torch, natten, transformers, utils3d
print("torch", torch.__version__, "| natten", natten.__version__,
      "| transformers", transformers.__version__)
PY

echo
echo "built: $OVERLAY"
echo "use it with:"
echo "  source tools/setup/activate_puffin_world.sh && source envs/worldsculpt-overlay/bin/activate"
