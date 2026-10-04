#!/usr/bin/env bash
# E08n (exploratory): instance and class labels on SuRFLo's Gaussians from one encoding.
# SuRFLo runs with IGGT's backbone and keeps its tokens at layers 4/11/17/23; IGGT's
# instance head reads those tokens (no second backbone pass); the per-pixel instance
# features are lifted onto SuRFLo's Gaussians through its refined cameras, grouped,
# and each group is named by CLIP against ScanNet++'s top-100 classes.
#
#   bash experiments/iggt/run_E08n_semantics_on_surflo.sh
#
# Each stage skips what it already wrote.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[ -f .env.local ] && . ./.env.local   # machine-local data roots (NCHC_SOFA_RUN for our capture)
CLONE="$ROOT/third_party/clones/surflo"
CKPT="$ROOT/data/models/surflo/surflo_v0.pt"
IGGT="$ROOT/data/models/iggt/iggt_checkpoint.pth"
OUT="$ROOT/outputs/iggt/semantics_on_surflo"
declare -A FOLDER=(
    [nchc_sofa]="$NCHC_SOFA_RUN/data/editreadygs_video/nchc_sofa_20260727_143647/images"
    [6115eddb86]="$HOME/datasets/scannetpp/derived/surflo_inputs/6115eddb86"
    [garden]="$HOME/datasets/mipnerf360/derived/garden_train161_images_4")
SCENES=(nchc_sofa 6115eddb86 garden)
grep -q save_vggt_tokens "$CLONE/configs/infer.yaml" || { echo "apply third_party/patches/surflo_*.patch" >&2; exit 1; }
grep -q pad_to_window third_party/clones/iggt/iggt/heads/window_sa.py || { echo "apply third_party/patches/iggt_*.patch" >&2; exit 1; }
mkdir -p "$OUT"
source tools/stamp_provenance.sh
stamp_provenance "$OUT" surflo
stamp_provenance "$OUT/iggt_code" iggt

surflo_py() { ( source tools/setup/activate_surflo.sh &&
                export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" && python "$@" ); }
puffin_py() { ( source tools/setup/activate_puffin_world.sh &&
                export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" && python "$@" ); }

for scene in "${SCENES[@]}"; do
    run="$OUT/$scene/surflo"
    echo "== $scene"
    if [ ! -f "$run/done" ]; then
        [ -d "$run" ] && mv "$run" "$run.incomplete.$(date +%s)"   # keep a failed attempt, never delete
        mkdir -p "$run"
        ( source tools/setup/activate_surflo.sh && cd "$CLONE" && python scripts/infer.py mode=guided guided=default \
              ckpt="$CKPT" vggt_weights="$IGGT" source.image_folder="${FOLDER[$scene]}" source.n_images=16 \
              num_query_points=100000 seed=42 output_dir="$run" hydra.run.dir="$run/hydra" \
              save_guided_state=true save_vggt_tokens=true mesh.enabled=false ) > "$run/infer.log" 2>&1 ||
            { echo "   FAILED: $run/infer.log" >&2; exit 1; }
        state="$(ls "$run"/*/guided_state.pt)"
        puffin_py -m gs_playground.surflo.export export --run "$(dirname "$state")" > "$run/export.log" 2>&1
        touch "$run/done"
    fi
    scene_dir="$(dirname "$(ls "$run"/*/guided_state.pt)")"
    [ -f "$OUT/$scene/features.npz" ] ||
        surflo_py -m gs_playground.iggt.on_surflo features --scene-dir "$scene_dir" --out "$OUT/$scene/features.npz"
    [ -f "$OUT/$scene/lifted.npz" ] ||
        puffin_py -m gs_playground.iggt.on_surflo lift --scene-dir "$scene_dir" --features "$OUT/$scene/features.npz" \
            --out "$OUT/$scene/lifted.npz"
    puffin_py -m gs_playground.iggt.on_surflo group --lifted "$OUT/$scene/lifted.npz" --out "$OUT/$scene/groups.npz"
    puffin_py -m gs_playground.iggt.on_surflo label --scene-dir "$scene_dir" --lifted "$OUT/$scene/lifted.npz" \
        --groups "$OUT/$scene/groups.npz" --out "$OUT/$scene/labels.json"
done
echo "== viewer"
puffin_py -m gs_playground.iggt.on_surflo demo --root "$OUT" --scenes "${SCENES[@]}" --out "$OUT/viewer"
