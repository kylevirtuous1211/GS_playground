#!/usr/bin/env bash
# E08i: what 3DGS and 2DGS cost on Mip-NeRF 360 garden on this GPU, beside
# SuRFLo (E08h). Pre-registered in PREREG_E08i.md.
#
#   bash experiments/surflo/run_E08i_perscene_garden.sh
#
# gsplat's own trainers, the standard protocol (data factor 4, every 8th view
# held out, 30k steps). Each trainer's per-process GPU memory is sampled once a
# second by nvidia-smi beside it. An arm with a `done` marker is skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [ -z "$(git log --format=%h -- experiments/surflo/PREREG_E08i.md)" ]; then
    echo "experiments/surflo/PREREG_E08i.md is not committed; commit it before any run" >&2
    exit 1
fi

# shellcheck source=/dev/null
source tools/setup/activate_puffin_world.sh
: "${MIPNERF360_GARDEN:?set MIPNERF360_GARDEN in .env.local}"

# Stage through symlinks: gsplat's parser writes images_4_png beside the data,
# and the share stays read-only.
DATA="$ROOT/data/mipnerf360/garden"
mkdir -p "$DATA"
for d in images images_4 sparse; do
    [ -e "$DATA/$d" ] || ln -s "$MIPNERF360_GARDEN/$d" "$DATA/$d"
done

OUT="$ROOT/outputs/surflo/perscene_garden"
mkdir -p "$OUT"
source tools/stamp_provenance.sh
stamp_provenance "$OUT" gsplat
EXAMPLES="$ROOT/third_party/clones/gsplat/examples"

# train <arm> <trainer.py and its args...>
train() {
    local arm="$1"; shift
    local out="$OUT/$arm"
    if [ -f "$out/done" ]; then echo "   $arm: done, skipped"; return; fi
    rm -rf "$out" && mkdir -p "$out"
    echo "== $arm"
    ( cd "$EXAMPLES" && exec python "$@" --data-dir "$DATA" --data-factor 4 --result-dir "$out" \
          --max-steps 30000 --eval-steps 30000 --save-steps 30000 --disable-viewer ) \
        > "$out/train.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$out/pid"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -lms 1000 \
        > "$out/vram.csv" 2>/dev/null &
    local smi=$!
    local t0; t0=$(date +%s)
    local status=0
    wait "$pid" || status=$?
    echo $(( $(date +%s) - t0 )) > "$out/wall_s"
    kill "$smi" 2>/dev/null || true
    [ "$status" = 0 ] || { echo "   FAILED ($status): see $out/train.log" >&2; exit 1; }
    touch "$out/done"
}

train 3dgs simple_trainer.py default --save-ply
train 2dgs simple_trainer_2dgs.py --normal-loss --dist-loss

python -m gs_playground.surflo.perscene --root "$OUT" \
    --surflo "$ROOT/results/surflo/e08h_stage_a.json" \
    --out "$ROOT/results/surflo/e08i_perscene_garden.json"
