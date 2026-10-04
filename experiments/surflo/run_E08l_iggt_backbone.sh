#!/usr/bin/env bash
# E08l: SuRFLo with its VGGT-1B against IGGT's fine-tuned backbone, on garden
# and three ScanNet++ v2 val scenes: token drift, geometry against the GT mesh,
# agreement between the two, novel views on garden. Pre-registered in
# PREREG_E08l.md.
#
#   bash tools/fetch/fetch_mipnerf360.sh                  # once: garden
#   bash tools/fetch/fetch_scannetpp_surflo_inputs.sh     # once: ScanNet++ inputs
#   hf download lifuguan/IGGT_official iggt_checkpoint.pth --revision 12aa6fa --local-dir data/models/iggt
#   bash experiments/surflo/run_E08l_iggt_backbone.sh
#
# A run directory with a `done` marker is skipped, so a stopped run resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [ -z "$(git log --format=%h -- experiments/surflo/PREREG_E08l.md)" ]; then
    echo "experiments/surflo/PREREG_E08l.md is not committed; commit it before any run" >&2
    exit 1
fi

CLONE="$ROOT/third_party/clones/surflo"
CKPT="$ROOT/data/models/surflo/surflo_v0.pt"
IGGT="$ROOT/data/models/iggt/iggt_checkpoint.pth"
[ -f .env.local ] && . ./.env.local   # machine-local data roots
SCANNETPP="${GS_PLAYGROUND_SCANNETPP:?set GS_PLAYGROUND_SCANNETPP (e.g. in .env.local) to the ScanNet++ v2 root}/data"
GARDEN="$ROOT/data/mipnerf360/garden"
GT="$ROOT/outputs/surflo/views_sweep/gt.npz"   # E08j: garden's frames as SuRFLo preprocesses them
OUT="$ROOT/outputs/surflo/iggt_backbone"
declare -A FOLDER=([garden]="$HOME/datasets/mipnerf360/derived/garden_train161_images_4")
for s in 825d228aec 6115eddb86 13c3e046d7; do FOLDER[$s]="$HOME/datasets/scannetpp/derived/surflo_inputs/$s"; done
SCENES=(garden 825d228aec 6115eddb86 13c3e046d7)

for f in "$CKPT" "$IGGT" "$GT"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
for s in "${SCENES[@]}"; do [ -d "${FOLDER[$s]}" ] || { echo "missing ${FOLDER[$s]}" >&2; exit 1; }; done
grep -q vggt_weights "$CLONE/configs/infer.yaml" || { echo "apply third_party/patches/surflo_vggt-weights.patch" >&2; exit 1; }
mkdir -p "$OUT"
source tools/stamp_provenance.sh
stamp_provenance "$OUT" surflo

surflo_py() { ( source tools/setup/activate_surflo.sh &&
                export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" && python "$@" ); }
puffin_py() { ( source tools/setup/activate_puffin_world.sh &&
                export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" && python "$@" ); }

echo "== validity check 1: the swap is real and complete"
surflo_py -m gs_playground.surflo.backbone check --surflo-ckpt "$CKPT" --iggt "$IGGT" \
    --out "$ROOT/results/surflo/e08l_weights_check.json"

# infer <out> <scene> <backbone> <mode> <seed>
infer() {
    local out="$1" scene="$2" backbone="$3" mode="$4" seed="$5"
    if [ -f "$out/done" ]; then echo "   $out: done, skipped"; return; fi
    rm -rf "$out" && mkdir -p "$out"
    local args=(mode="$mode" ckpt="$CKPT" source.image_folder="${FOLDER[$scene]}"
                source.n_images=16 num_query_points=100000 seed="$seed"
                output_dir="$out" hydra.run.dir="$out/hydra" save_guided_state=true
                mesh.enabled=false)   # Amendment 1: no metric uses mesh.ply; one run hung in it
    if [ "$mode" = guided ]; then args+=(guided=default); fi
    if [ "$backbone" = iggt ]; then args+=(vggt_weights="$IGGT"); fi
    echo "   $out"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$out/gpu_before.csv" || true
    local t0; t0=$(date +%s)
    ( source tools/setup/activate_surflo.sh && cd "$CLONE" && python scripts/infer.py "${args[@]}" ) \
        > "$out/infer.log" 2>&1 || { echo "   FAILED: see $out/infer.log" >&2; exit 1; }
    echo $(( $(date +%s) - t0 )) > "$out/wall_s"
    if [ "$mode" = guided ]; then
        local state; state="$(ls "$out"/*/guided_state.pt)"
        puffin_py -m gs_playground.surflo.export export --run "$(dirname "$state")" > "$out/export.log" 2>&1 ||
            { echo "   export FAILED: $out/export.log" >&2; exit 1; }
    fi
    touch "$out/done"
}

for scene in "${SCENES[@]}"; do
    echo "== $scene"
    [ -f "$OUT/$scene/drift.json" ] ||
        surflo_py -m gs_playground.surflo.backbone drift --surflo-ckpt "$CKPT" --iggt "$IGGT" \
            --folder "${FOLDER[$scene]}" --n-images 16 --out "$OUT/$scene/drift.json"
    for backbone in vggt iggt; do
        infer "$OUT/$scene/$backbone/plain/seed42" "$scene" "$backbone" plain 42
        for seed in 42 0 1; do
            infer "$OUT/$scene/$backbone/n16/seed$seed" "$scene" "$backbone" guided "$seed"
        done
    done
    # validity check 2: plain mode, repeated, must be bitwise identical
    infer "$OUT/$scene/vggt/plain/seed42_repeat" "$scene" vggt plain 42
done

echo "== garden novel views (E08j's protocol; sanity 2 gates each backbone's test numbers)"
for backbone in vggt iggt; do
    [ -f "$OUT/garden/$backbone/nvs.json" ] ||
        puffin_py -m gs_playground.surflo.nvs eval --gt "$GT" --data-dir "$GARDEN" \
            --sweep-root "$OUT/garden/$backbone" --out "$OUT/garden/$backbone/nvs.json"
done

echo "== scoring"
surflo_py -m gs_playground.surflo.swap score --root "$OUT" --scannetpp "$SCANNETPP" \
    --out "$ROOT/results/surflo/e08l_iggt_backbone.json"
