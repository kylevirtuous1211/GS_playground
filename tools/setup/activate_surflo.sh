#!/usr/bin/env bash
# Source this to get the SuRFLo runtime; runners source it too.
#
#   source tools/setup/activate_surflo.sh
#
# Its own micromamba prefix, not an overlay: SuRFLo wants torch 2.4.1+cu124
# and nvcc 12.4, the puffin env has 2.7+cu126. It deliberately does not source
# activate_puffin_world.sh, whose CUDA_HOME and SPARSE_CONV_BACKEND belong to
# that env. CUDA_HOME, CC and CXX come from the activate.d hook SuRFLo's
# install/activate_cuda.sh wrote into the prefix.
_GSP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_GSP_SHA="$(git -C "$_GSP_ROOT/third_party/clones/surflo" rev-parse --short HEAD)"
_GSP_PREFIX="$_GSP_ROOT/envs/surflo-$_GSP_SHA-py310-cu124"

[ -x "$_GSP_PREFIX/bin/python" ] || {
    echo "no env at $_GSP_PREFIX -- bash tools/setup/surflo_env.sh" >&2
    return 1 2>/dev/null || exit 1
}

# The micromamba hook and the activate.d scripts read variables unguarded;
# suspend nounset for the activation only (as activate_puffin_world.sh does).
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
case $- in *u*) _GSP_HAD_NOUNSET=1;; *) _GSP_HAD_NOUNSET=0;; esac
set +u
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$_GSP_PREFIX"
[ "$_GSP_HAD_NOUNSET" = 1 ] && set -u

export PIP_CONFIG_FILE="$_GSP_ROOT/tools/setup/pip.conf"
export PYTHONPATH="$_GSP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$_GSP_ROOT/.env.local" ] && . "$_GSP_ROOT/.env.local"
