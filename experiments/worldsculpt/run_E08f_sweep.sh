#!/usr/bin/env bash
# E08f Stage B: WorldSculpt parameters vs mesh completeness, on a fixed subset.
#
#   bash experiments/worldsculpt/run_E08f_sweep.sh
#
# Arms and reading rules: PREREG_E08f.md; arms themselves: e08f_configs.json.
# Runs upstream's own crop and reconstruction scripts with inference.sh's
# flags, changing one thing per config. Needs E08e's run (its crops are the
# base case) and Stage A's observed points (sweep stage-a) first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source tools/setup/activate_puffin_world.sh
: "${NCHC_SOFA_RUN:?set NCHC_SOFA_RUN in .env.local}"

SCENE=nchc_sofa_20260727_143647
DATA_DIR="$NCHC_SOFA_RUN/data/editreadygs_video/$SCENE"
MODEL_DIR="$NCHC_SOFA_RUN/output/editreadygs_video/$SCENE/3dgs_output"
CONFIGS="$ROOT/experiments/worldsculpt/e08f_configs.json"
OBJECTS="$ROOT/experiments/worldsculpt/e08e_objects.json"
BASE_SCENE="$ROOT/data/worldsculpt_input/NCHC/$SCENE"
BASE_CASE="$ROOT/outputs/worldsculpt/NCHC/$SCENE"
SWEEP="$ROOT/outputs/worldsculpt/NCHC/sweep"
OBSERVED="$SWEEP/observed.npz"
CLONE="$ROOT/third_party/clones/worldsculpt"
CKPT_ROOT="$ROOT/data/models/worldsculpt"
SS_DIR="$CKPT_ROOT/ss_ft64_mv_lora_ibr_texverse"
SHAPE_DIR="$CKPT_ROOT/shape_ft1024_mv_lora_ibr_texverse_fixedmem05"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

[ -f "$BASE_CASE/_crops/anchor_views.json" ] || { echo "run E08e first: no crops in $BASE_CASE" >&2; exit 1; }
[ -f "$OBSERVED" ] || { echo "run sweep stage-a first: no $OBSERVED" >&2; exit 1; }
mkdir -p "$SWEEP"
source tools/stamp_provenance.sh
stamp_provenance "$SWEEP" worldsculpt

SUBSET="$(python -c "import json; print(','.join(f'obj{k:02d}' for k in json.load(open('$CONFIGS'))['objects']))")"
python - "$CONFIGS" "$OBJECTS" "$SWEEP/subset_selection.json" <<'PY'
import json, sys
subset = set(json.load(open(sys.argv[1]))["objects"])
sel = json.load(open(sys.argv[2]))
for obj in sel["objects"]:
    obj["keep"] = obj["id"] in subset
json.dump(sel, open(sys.argv[3], "w"), indent=1)
PY

# upstream's crop stage with inference.sh's flags; $1 scene, $2 case root, rest extra flags
crops() {
    local scene="$1" case="$2"; shift 2
    [ -f "$case/_crops/anchor_views.json" ] && return 0
    ( cd "$CLONE" && python prepare_crops_scene.py --scene_dir "$scene" --case_root "$case" \
        --crop_resolution 1024 --save_alignments --alpha_erode_kernel 0 --alpha_erode_iters 0 \
        --min_mask_ratio 0.001 --max_crop_ratio 3.0 "$@" ) > "$case.crops.log" 2>&1
}

for spec in erode4:-4 dilate4:4; do
    name="${spec%%:*}"; morph="${spec##*:}"
    scene="$ROOT/data/worldsculpt_input/NCHC_sweep/$name"
    [ -d "$scene" ] || python -m gs_playground.worldsculpt.ground build \
        --data-dir "$DATA_DIR" --model-dir "$MODEL_DIR" --objects "$SWEEP/subset_selection.json" \
        --out "$scene" --mask-morph "$morph" --images-from "$BASE_SCENE"
    mkdir -p "$SWEEP/$name"
    crops "$scene" "$SWEEP/$name" --mask_fit_scale
done
mkdir -p "$SWEEP/nofit"
crops "$BASE_SCENE" "$SWEEP/nofit"

# reconstruction runs in the WorldSculpt overlay, as in the E08c runner
# shellcheck source=/dev/null
source envs/worldsculpt-overlay/bin/activate
export PYTHONPATH="$CLONE:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export SPCONV_ALGO=native

python - "$CONFIGS" <<'PY' > "$SWEEP/plan.tsv"
import json, sys
c = json.load(open(sys.argv[1]))
for cfg in c["configs"]:
    print("\t".join([cfg["name"], c["cases"][cfg["case"]], str(cfg.get("seed", 42)),
                     str(cfg.get("max_views", 20)), cfg.get("view_select", "area")]))
PY
while IFS=$'\t' read -r name case seed views select; do
    echo "== $name: case $case seed $seed max_views $views view_select $select"
    # upstream resolves its base weights as pretrained/Pixal3D relative to the
    # clone, which is where inference.sh runs it from
    ( cd "$CLONE" && CUDA_VISIBLE_DEVICES="${GPU:-0}" python reconstruct_batch.py \
        --case_root "$ROOT/$case" --instances "$SUBSET" --recon_subdir "_sweep_$name" \
        --views all --no-ema --vis_ss --sampler official --no_tex --no_glb \
        --seed "$seed" --max_views "$views" --view_select "$select" \
        --ss_config "$SS_DIR/config.json" --ss_ckpt_dir "$SS_DIR/ckpts" --ss_step 15000 \
        --shape_config "$SHAPE_DIR/config.json" --shape_ckpt_dir "$SHAPE_DIR/ckpts" --shape_step 15000 \
        >> "$SWEEP/$name.log" 2>&1 ) || { echo "   FAILED: see $SWEEP/$name.log"; exit 1; }
    echo "   $(ls "$ROOT/$case/_sweep_$name"/*/mesh.pt 2>/dev/null | wc -l) meshes"
done < "$SWEEP/plan.tsv"

deactivate
export PYTHONPATH="$ROOT/src"   # the clone's vendored modules must not shadow ours
python -m gs_playground.worldsculpt.sweep score-b \
    --data-dir "$DATA_DIR" --model-dir "$MODEL_DIR" --configs "$CONFIGS" --objects "$OBJECTS" \
    --observed "$OBSERVED" --e08e-recon "$BASE_CASE/_recon" \
    --e08e-log "$ROOT/outputs/worldsculpt/NCHC/run.log" --sweep-log "$SWEEP/default_s42.log" \
    --out "$ROOT/results/worldsculpt/e08f_stage_b.json"
