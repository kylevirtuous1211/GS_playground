#!/usr/bin/env bash
# E08m: do VGGT's cameras (Mip-NeRF 360, up to 128 frames) and IGGT's instance
# features (Mip-NeRF 360 stability, ScanNet++ T-mIoU, up to 32 frames) degrade
# with more input frames? Fixed anchors, growing context. Pre-registered in
# PREREG_E08m.md.
#
#   bash tools/fetch/fetch_mipnerf360.sh                  # once
#   bash tools/fetch/fetch_scannetpp_surflo_inputs.sh     # once (three of the five scenes, locally)
#   bash experiments/iggt/run_E08m_view_count.sh
#
# Each stage skips what it already wrote, so a stopped run resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [ -z "$(git log --format=%h -- experiments/iggt/PREREG_E08m.md)" ]; then
    echo "experiments/iggt/PREREG_E08m.md is not committed; commit it before any run" >&2
    exit 1
fi
grep -q 'frames_chunk_size=None' third_party/clones/iggt/iggt/models/vggt.py ||
    { echo "apply third_party/patches/iggt_*.patch (tools/setup/clone_upstream.sh)" >&2; exit 1; }
[ -f data/models/iggt/iggt_checkpoint.pth ] || { echo "missing the IGGT checkpoint" >&2; exit 1; }
[ -d "$HOME/datasets/mipnerf360/360_v2/garden" ] || { echo "run tools/fetch/fetch_mipnerf360.sh" >&2; exit 1; }

OUT="$ROOT/outputs/iggt/view_count"
RES="$ROOT/results/iggt"
mkdir -p "$OUT" "$RES"
source tools/stamp_provenance.sh
stamp_provenance "$OUT" iggt
stamp_provenance "$OUT/vggt_code" surflo   # VGGT-1B runs through SuRFLo's vendored copy

surflo_py() { ( source tools/setup/activate_surflo.sh &&
                export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" && python "$@" ); }
puffin_py() { ( source tools/setup/activate_puffin_world.sh &&
                export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" && python "$@" ); }
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/gpu_before.csv" || true

echo "== validity check 1: the IGGT many-frames fix"
surflo_py -m gs_playground.iggt.view_count check --out "$RES/e08m_validity.json"

echo "== part 1: VGGT-1B cameras against COLMAP (Mip-NeRF 360)"
surflo_py -m gs_playground.iggt.view_count vggt --out "$RES/e08m_vggt_cameras.json"

echo "== part 2b ground truth: ScanNet++ instance masks from the mesh"
surflo_py -m gs_playground.iggt.view_count gt --out-dir "$OUT"
# validity check 3: the mesh covers the anchor frames (the overlays are looked at before T-mIoU is read)
puffin_py - "$OUT" <<'EOF'
import sys
from pathlib import Path
import numpy as np
from gs_playground.iggt.view_count import HIT_FRACTION_MIN, SPP_SCENES
low = {s: float(np.load(Path(sys.argv[1]) / s / "gt.npz")["hit_fraction"].min()) for s in SPP_SCENES}
print("validity check 3, lowest hit fraction per scene:", {s: round(v, 3) for s, v in low.items()})
if min(low.values()) < HIT_FRACTION_MIN:
    raise SystemExit("validity check 3 failed: the mesh misses too much of an anchor frame")
print("validity check 3 passes")
EOF

echo "== part 2: IGGT instance features (Mip-NeRF 360 and ScanNet++)"
surflo_py -m gs_playground.iggt.view_count iggt --out-dir "$OUT"

echo "== part 2: stability and T-mIoU"
puffin_py -m gs_playground.iggt.view_count analyse --out-dir "$OUT" --out "$RES/e08m_iggt_instances.json"
puffin_py - "$RES" <<'EOF'
import json, sys
from pathlib import Path
res = Path(sys.argv[1])
print("verdict part 1 (VGGT):", json.loads((res / "e08m_vggt_cameras.json").read_text())["reading"]["verdict"])
print("verdict part 2b (IGGT):", json.loads((res / "e08m_iggt_instances.json").read_text())["reading"]["verdict"])
EOF
