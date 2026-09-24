#!/usr/bin/env bash
# E08c: WorldSculpt's released weights, run here.
#
#   bash experiments/worldsculpt/run_E08c_worldsculpt.sh [scene ...]
#
# A reproduction arm on the authors' own data. WorldSculpt needs posed frames
# **plus per-instance masks and 3D boxes**, which their released scenes carry
# and our DL3DV scenes do not, so running it on our corpus is a separate
# grounding project rather than a flag on this runner.
#
# Defaults to one Marble scene: it is the smallest, and it is the case the
# paper leads with, turning a generated 3DGS world into object-level meshes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CLONE="$ROOT/third_party/clones/worldsculpt"
DATASET="${DATASET:-Marble}"
INPUT_ROOT="${INPUT_ROOT:-$ROOT/data/worldsculpt_input/$DATASET}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/worldsculpt/$DATASET}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/data/models/worldsculpt}"
SCENES=("${@:-marble_serene_living_room_countryside_view}")

[ -d "$CKPT_ROOT/ss_ft64_mv_lora_ibr_texverse" ] || {
    echo "no LoRA weights at $CKPT_ROOT" >&2
    echo "  hf download AlayaLab/WorldSculpt --local-dir $CKPT_ROOT" >&2
    exit 1
}

# inference.py resolves the Pixal3D base weights from `<clone>/pretrained/Pixal3D`
# if that path exists and falls back to the HuggingFace id otherwise. There is
# no env override, so the local copy is linked in. The clone stays byte-identical
# to its pin: this is an untracked symlink beside the source, not an edit.
[ -e "$CLONE/pretrained" ] || ln -s "$CKPT_ROOT" "$CLONE/pretrained"

# shellcheck source=/dev/null
source tools/setup/activate_puffin_world.sh
# shellcheck source=/dev/null
source envs/worldsculpt-overlay/bin/activate

# The clone vendors its own `pixal3d` package; it must win over the trellis2
# clone the base activation puts on PYTHONPATH.
export PYTHONPATH="$CLONE:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export SPCONV_ALGO=native

# Preflight. The vendored pixal3d package imports these at module load, and
# the crop stage runs for minutes before the first one is touched, so a missing
# extension otherwise surfaces as a traceback well into the run.
python - <<'PY' || exit 1
import importlib.util, sys
missing = [m for m in ("cumesh", "o_voxel", "nvdiffrast", "utils3d", "natten")
           if importlib.util.find_spec(m) is None]
if missing:
    print(f"[preflight] missing: {', '.join(missing)}", file=sys.stderr)
    print("  bash tools/setup/worldsculpt_env.sh", file=sys.stderr)
    sys.exit(1)
print("[preflight] extensions present")
PY

mkdir -p "$OUTPUT_ROOT"
echo "== scenes: ${SCENES[*]}"
echo "== in  $INPUT_ROOT"
echo "== out $OUTPUT_ROOT"

INPUT_ROOT="$INPUT_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" CKPT_ROOT="$CKPT_ROOT" \
SAMPLER=official SS_STEP=15000 SHAPE_STEP=15000 \
RENDER=1 FACE_BUDGET=1000000 GPU="${GPU:-0}" \
    "$CLONE/inference.sh" "${SCENES[@]}" 2>&1 | tee "$OUTPUT_ROOT/run.log"

echo
echo "assets in $OUTPUT_ROOT"
