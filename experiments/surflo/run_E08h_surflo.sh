#!/usr/bin/env bash
# E08h: SuRFLo (arXiv 2606.13644), pre-registered in PREREG_E08h.md.
#
#   bash experiments/surflo/run_E08h_surflo.sh A   # garden sample: reproduction attempt
#   bash experiments/surflo/run_E08h_surflo.sh B   # NCHC sofa: exploratory; only after A passed
#
# Every arm runs upstream's own scripts/infer.py from the clone, patched only to
# also save the fitted Gaussians and to record VRAM
# (third_party/patches/surflo_save-guided-state.patch). A run directory with a
# `done` marker is skipped, so an interrupted stage resumes where it stopped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAGE="${1:?usage: run_E08h_surflo.sh A|B}"
CLONE="$ROOT/third_party/clones/surflo"
CKPT="${CKPT:-$ROOT/data/models/surflo/surflo_v0.pt}"

[ -f "$CKPT" ] || {
    echo "no weights at $CKPT" >&2
    echo "  hf download AntoineGuedon/Surflo-v0 surflo_v0.pt --local-dir $(dirname "$CKPT")" >&2
    exit 1
}
# The pre-registration must be committed before any run: that commit order is
# the evidence that the criteria were set before the numbers were seen.
if [ -z "$(git log --format=%h -- experiments/surflo/PREREG_E08h.md)" ]; then
    echo "experiments/surflo/PREREG_E08h.md is not committed; commit it before any run" >&2
    exit 1
fi

# infer <out> <mode> <guided preset or -> <image folder> <n images> <seed>
infer() {
    local out="$1" mode="$2" preset="$3" folder="$4" n="$5" seed="$6"
    if [ -f "$out/done" ]; then echo "   $out: done, skipped"; return; fi
    rm -rf "$out" && mkdir -p "$out"
    local args=(mode="$mode" ckpt="$CKPT" source.image_folder="$folder"
                source.n_images="$n" num_query_points=100000 seed="$seed"
                output_dir="$out" hydra.run.dir="$out/hydra" save_guided_state=true)
    if [ "$mode" = guided ]; then args+=(guided="$preset"); fi
    echo "   $out"
    ( source tools/setup/activate_surflo.sh && cd "$CLONE" && python scripts/infer.py "${args[@]}" ) \
        > "$out/infer.log" 2>&1 || { echo "   FAILED: see $out/infer.log" >&2; exit 1; }
    touch "$out/done"
}

# puffin <python args...>: our own tools, which need plyfile and gsplat
puffin() {
    ( source tools/setup/activate_puffin_world.sh &&
      export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" && python "$@" )
}

# export_all <out> <result json>: every guided run's Gaussians -> 3DGS asset, checked
export_all() {
    local out="$1" result="$2" scenes=()
    for state in $(find "$out" -name guided_state.pt | sort); do
        local scene; scene="$(dirname "$state")"
        scenes+=("$scene")
        [ -f "$scene/reference_renders.npz" ] ||
            ( source tools/setup/activate_surflo.sh &&
              python -m gs_playground.surflo.export reference --run "$scene" )
        [ -f "$scene/point_cloud.ply" ] || puffin -m gs_playground.surflo.export export --run "$scene"
    done
    puffin -m gs_playground.surflo.export check --runs "${scenes[@]}" --out "$result"
}

case "$STAGE" in
A)
    OUT="$ROOT/outputs/surflo/garden_sample"
    FOLDER="$CLONE/media/sample"
    mkdir -p "$OUT"
    source tools/stamp_provenance.sh
    stamp_provenance "$OUT" surflo
    echo "== Stage A: the README's commands on the authors' 16-view garden sample, n = 3 each"
    for k in 1 2 3; do infer "$OUT/guided_default/run$k" guided default "$FOLDER" 16 42; done
    for k in 1 2 3; do infer "$OUT/plain/run$k" plain - "$FOLDER" 16 42; done
    echo "== Stage A readings"
    puffin -m gs_playground.surflo.evaluate stage-a --root "$OUT" \
        --out "$ROOT/results/surflo/e08h_stage_a.json"
    echo "== export check on Stage A's guided runs"
    export_all "$OUT" "$ROOT/results/surflo/e08h_export_check.json"
    ;;
B)
    # Pre-registered: Stage B runs only if Stage A passed all four criteria.
    python3 -c "import json,sys; sys.exit(0 if json.load(open('results/surflo/e08h_stage_a.json')).get('passes') else 1)" || {
        echo "Stage A has not passed (results/surflo/e08h_stage_a.json); Stage B does not run" >&2; exit 1; }
    [ -f "$ROOT/.env.local" ] && . "$ROOT/.env.local"
    : "${NCHC_SOFA_RUN:?set NCHC_SOFA_RUN in .env.local}"
    SCENE=nchc_sofa_20260727_143647
    FOLDER="$NCHC_SOFA_RUN/data/editreadygs_video/$SCENE/images"
    MODEL="$NCHC_SOFA_RUN/output/editreadygs_video/$SCENE/3dgs_output"
    DEPTH="$ROOT/outputs/worldsculpt/NCHC/ground/depth"   # E08e's cache of that 3DGS's depth
    [ "$(ls "$DEPTH" | wc -l)" = 364 ] || { echo "missing E08e's depth cache at $DEPTH" >&2; exit 1; }
    OUT="$ROOT/outputs/surflo/nchc_sofa"
    mkdir -p "$OUT"
    source tools/stamp_provenance.sh
    stamp_provenance "$OUT" surflo
    echo "== Stage B: NCHC sofa, three arms x seeds {42, 0, 1} (exploratory)"
    for seed in 42 0 1; do
        infer "$OUT/guided_default_16/seed$seed" guided default "$FOLDER" 16 "$seed"
        infer "$OUT/plain_16/seed$seed" plain - "$FOLDER" 16 "$seed"
        infer "$OUT/guided_long_32/seed$seed" guided long "$FOLDER" 32 "$seed"
    done
    echo "== export check on Stage B's guided runs"
    export_all "$OUT" "$ROOT/results/surflo/e08h_export_check.json"
    echo "== Stage B scores (authors' metric code, surflo env)"
    ( source tools/setup/activate_surflo.sh &&
      python -m gs_playground.surflo.evaluate stage-b --root "$OUT" --model-dir "$MODEL" \
          --depth-dir "$DEPTH" --out "$ROOT/results/surflo/e08h_stage_b.json" )
    ;;
*)
    echo "unknown stage $STAGE (A or B)" >&2; exit 1 ;;
esac
