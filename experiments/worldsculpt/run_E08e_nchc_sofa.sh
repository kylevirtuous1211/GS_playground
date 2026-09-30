#!/usr/bin/env bash
# E08e: WorldSculpt on our own capture, the NCHC sofa scene.
#
#   bash experiments/worldsculpt/run_E08e_nchc_sofa.sh
#
# WorldSculpt needs per-object masks and a world box per object and leaves
# finding them out of scope. gs_playground.worldsculpt.ground supplies them
# from the scene's COLMAP poses, its trained 3DGS and HQ-SAM label maps an
# earlier project associated across views. The object list is the reviewed,
# tracked e08e_objects.json; `ground propose` made its first draft.
#
# Exploratory: no ground truth exists for this scene, so nothing is scored.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source tools/setup/activate_puffin_world.sh
: "${NCHC_SOFA_RUN:?set NCHC_SOFA_RUN in .env.local}"

SCENE=nchc_sofa_20260727_143647
DATA_DIR="$NCHC_SOFA_RUN/data/editreadygs_video/$SCENE"
MODEL_DIR="$NCHC_SOFA_RUN/output/editreadygs_video/$SCENE/3dgs_output"
OBJECTS="$ROOT/experiments/worldsculpt/e08e_objects.json"
SCENE_DIR="$ROOT/data/worldsculpt_input/NCHC/$SCENE"
OUT="$ROOT/outputs/worldsculpt/NCHC"
CASE_ROOT="$OUT/$SCENE"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Upstream resumes from whatever is on disk: an existing mesh.pt or scene.glb
# is reused even if the selection changed underneath it. Refuse to mix.
want="$(sha256sum "$OBJECTS" | cut -d' ' -f1)"
if [ -d "$SCENE_DIR" ]; then
    have="$(cat "$SCENE_DIR/selection.sha256")"
    [ "$have" = "$want" ] || {
        echo "selection changed since $SCENE_DIR was built; delete it and $CASE_ROOT" >&2
        exit 1
    }
else
    python -m gs_playground.worldsculpt.ground build \
        --data-dir "$DATA_DIR" --model-dir "$MODEL_DIR" \
        --objects "$OBJECTS" --out "$SCENE_DIR"
fi

mkdir -p "$OUT"
source tools/stamp_provenance.sh
stamp_provenance "$OUT" worldsculpt

DATASET=NCHC INPUT_ROOT="$ROOT/data/worldsculpt_input/NCHC" OUTPUT_ROOT="$OUT" \
    bash experiments/worldsculpt/run_E08c_worldsculpt.sh "$SCENE"

python -m gs_playground.worldsculpt.ground check \
    --scene-dir "$SCENE_DIR" --case-root "$CASE_ROOT" --log "$OUT/run.log" \
    --out "$ROOT/results/worldsculpt/e08e_checks.json"

echo
echo "assets in $CASE_ROOT"
