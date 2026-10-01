#!/usr/bin/env bash
# E08j: SuRFLo with 16, 32, 64 and 161 views of garden's training split - time,
# VRAM and the novel-view PSNR of the 3DGS it fits, beside E08i's 3DGS.
# Pre-registered in PREREG_E08j.md (with Amendment 1).
#
#   bash tools/fetch/fetch_mipnerf360.sh                # once: the dataset
#   bash experiments/surflo/run_E08j_views_sweep.sh
#
# A run directory with a `done` marker is skipped, so a stopped sweep resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [ -z "$(git log --format=%h -- experiments/surflo/PREREG_E08j.md)" ]; then
    echo "experiments/surflo/PREREG_E08j.md is not committed; commit it before any run" >&2
    exit 1
fi

DS="$HOME/datasets/mipnerf360"
TRAIN="$DS/derived/garden_train161_images_4"
GARDEN="$ROOT/data/mipnerf360/garden"   # data/mipnerf360 links the whole dataset
[ "$(ls "$TRAIN" 2>/dev/null | wc -l)" = 161 ] || { echo "run tools/fetch/fetch_mipnerf360.sh" >&2; exit 1; }
[ -d "$GARDEN/sparse/0" ] || { echo "data/mipnerf360 must link $DS/360_v2" >&2; exit 1; }
CLONE="$ROOT/third_party/clones/surflo"
CKPT="$ROOT/data/models/surflo/surflo_v0.pt"
BASELINE="$ROOT/outputs/surflo/perscene_garden/3dgs/ply/point_cloud_29999.ply"   # E08i
OUT="$ROOT/outputs/surflo/views_sweep"
mkdir -p "$OUT"
source tools/stamp_provenance.sh
stamp_provenance "$OUT" surflo

surflo_py() { ( source tools/setup/activate_surflo.sh && python "$@" ); }
puffin_py() { ( source tools/setup/activate_puffin_world.sh &&
                export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" && python "$@" ); }

[ -f "$OUT/gt.npz" ] || surflo_py -m gs_playground.surflo.nvs gt --images-dir "$GARDEN/images_4" --out "$OUT/gt.npz"

# infer <out> <n views> <seed>; returns non-zero instead of exiting, so the
# pre-registered cap on N = 161 can record a failure as a result
infer() {
    local out="$1" n="$2" seed="$3"
    if [ -f "$out/done" ]; then echo "   $out: done, skipped"; return 0; fi
    rm -rf "$out" && mkdir -p "$out"
    echo "   $out"
    # the GPU is shared: who else was on it, for any out-of-memory reading
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$out/gpu_before.csv" || true
    local t0; t0=$(date +%s)
    ( source tools/setup/activate_surflo.sh && cd "$CLONE" && python scripts/infer.py \
          mode=guided guided=default ckpt="$CKPT" source.image_folder="$TRAIN" \
          source.n_images="$n" num_query_points=100000 seed="$seed" \
          output_dir="$out" hydra.run.dir="$out/hydra" save_guided_state=true ) \
        > "$out/infer.log" 2>&1 || { echo $(( $(date +%s) - t0 )) > "$out/wall_s"; return 1; }
    echo $(( $(date +%s) - t0 )) > "$out/wall_s"
    # set -e does not apply inside a function called from `if !`: check each step
    local state; state="$(ls "$out"/*/guided_state.pt)" || return 1
    local scene; scene="$(dirname "$state")"
    puffin_py -m gs_playground.surflo.export export --run "$scene" > "$out/export.log" 2>&1 || {
        echo "   export FAILED: $out/export.log" >&2; exit 1; }
    touch "$out/done"
}

echo "== SuRFLo, guided default, N x seeds {42, 0, 1}"
OOM='OutOfMemoryError|CUDA out of memory'
for n in 16 32 64 161; do
    for seed in 42 0 1; do
        run="$OUT/n$n/seed$seed"
        # the cap (Amendment 1): seed 42 at N = 161 out of memory or over 20 min skips seeds 0 and 1
        if [ "$n" = 161 ] && [ "$seed" != 42 ] && [ -f "$OUT/n161_capped" ]; then
            mkdir -p "$run"
            echo "skipped by the pre-registered cap: $(cat "$OUT/n161_capped")" > "$run/status"
            echo "   n161/seed$seed: $(cat "$run/status")"
            continue
        fi
        if [ "$n" = 161 ] && [ "$seed" = 42 ] && [ -f "$run/status" ]; then
            echo "   n161/seed42: $(cat "$run/status")"; continue
        fi
        if ! infer "$run" "$n" "$seed"; then
            if [ "$n" = 161 ] && [ "$seed" = 42 ] && grep -q -E "$OOM" "$run/infer.log"; then
                echo "seed 42 ran out of GPU memory after $(cat "$run/wall_s") s: $(grep -E "$OOM" "$run/infer.log" | tail -1 | cut -c1-200)" > "$OUT/n161_capped"
                echo "failed: $(cat "$OUT/n161_capped")" > "$run/status"
                echo "   n161/seed42 $(cat "$run/status")"
                continue
            fi
            echo "   FAILED: $run/infer.log: $(grep -E 'Error' "$run/infer.log" | tail -1 | cut -c1-200)" >&2
            exit 1
        fi
        if [ "$n" = 161 ] && [ "$seed" = 42 ] && [ "$(cat "$run/wall_s")" -gt 1200 ]; then
            echo "seed 42 took $(cat "$run/wall_s") s > 1200 s" > "$OUT/n161_capped"
        fi
    done
done

echo "== novel views (sanity 2 first, then the test views, the E08i 3DGS beside them)"
puffin_py -m gs_playground.surflo.nvs eval --gt "$OUT/gt.npz" --data-dir "$GARDEN" \
    --sweep-root "$OUT" --baseline-ply "$BASELINE" --out "$ROOT/results/surflo/e08j_views_sweep.json"
