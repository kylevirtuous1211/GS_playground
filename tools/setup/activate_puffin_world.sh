#!/usr/bin/env bash
# Source this to get a working Puffin-World runtime; runners source it too.
#
#   source tools/setup/activate_puffin_world.sh
#
# Sets, beyond micromamba activation:
#   CUDA_HOME          deepspeed refuses to import without it (its op builder
#                      probes nvcc even when nothing needs compiling); the env
#                      carries its own CUDA toolkit at the prefix.
#   PYTHONPATH         the clone root, for upstream's `src.*` and `configs.*`.
#   PIP_CONFIG_FILE    keeps any pip use inside the env off the broken NGC
#                      index in the user-level pip.conf.
#   GS_PLAYGROUND_DATA taken from the untracked .env.local if unset there;
#                      the tracked tree carries no machine-specific path.
_GSP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_GSP_SHA="$(git -C "$_GSP_ROOT/third_party/clones/puffin" rev-parse --short HEAD)"
_GSP_PREFIX="$_GSP_ROOT/envs/puffin-world-$_GSP_SHA-py310-cu126"

[ -x "$_GSP_PREFIX/bin/python" ] || {
    echo "no env at $_GSP_PREFIX -- bash tools/setup/puffin_world_env.sh" >&2
    return 1 2>/dev/null || exit 1
}

# Both the micromamba shell hook and cuda-toolkit's activate.d scripts read
# variables unguarded (MAMBA_ROOT_PREFIX, NVCC_PREPEND_FLAGS), so activation
# dies under a caller's `set -u`. Suspend nounset for the activation only and
# restore the caller's setting after.
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
case $- in *u*) _GSP_HAD_NOUNSET=1;; *) _GSP_HAD_NOUNSET=0;; esac
set +u
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$_GSP_PREFIX"
[ "$_GSP_HAD_NOUNSET" = 1 ] && set -u

export CUDA_HOME="$_GSP_PREFIX"
# conda's cuda-toolkit keeps headers/libs under targets/; nvcc knows, but the
# host compiler steps of torch JIT extensions (gsplat) do not
export CPATH="$_GSP_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$_GSP_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PIP_CONFIG_FILE="$_GSP_ROOT/tools/setup/pip.conf"
export PYTHONPATH="$_GSP_ROOT/third_party/clones/puffin/Puffin-World:$_GSP_ROOT/third_party/clones/trellis2${PYTHONPATH:+:$PYTHONPATH}"
# TRELLIS.2 sparse backends: spconv wheel is installed; flex_gemm is not built
export SPARSE_CONV_BACKEND="${SPARSE_CONV_BACKEND:-spconv}"
# machine-local settings (untracked); currently just GS_PLAYGROUND_DATA
[ -f "$_GSP_ROOT/.env.local" ] && . "$_GSP_ROOT/.env.local"
