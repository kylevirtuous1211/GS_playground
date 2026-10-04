#!/usr/bin/env bash
# Copy the DSLR training frames (official train list minus frames flagged
# is_bad) of the ScanNet++ v2 scenes E08l feeds SuRFLo into a local derived
# directory, one folder per scene, since SuRFLo takes a folder and samples it.
# The ScanNet++ release stays where it is (GS_PLAYGROUND_SCANNETPP); this is a
# small derived copy, staged under .partial, verified, published by rename,
# with a manifest.
#
#   bash tools/fetch/fetch_scannetpp_surflo_inputs.sh [scene ...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$REPO/.env.local" ] && . "$REPO/.env.local"   # machine-local data roots
SRC="${GS_PLAYGROUND_SCANNETPP:?set GS_PLAYGROUND_SCANNETPP (e.g. in .env.local) to the ScanNet++ v2 root}/data"
ROOT="$HOME/datasets/scannetpp"
DEST="$ROOT/derived/surflo_inputs"
SCENES=("$@")
[ ${#SCENES[@]} -gt 0 ] || SCENES=(825d228aec 6115eddb86 13c3e046d7)   # E08l
mkdir -p "$DEST"

for s in "${SCENES[@]}"; do
    if [ -d "$DEST/$s" ]; then echo "$s: exists"; continue; fi
    stage="$DEST/.$s.partial"
    rm -rf "$stage" && mkdir -p "$stage"
    python3 - "$SRC/$s" "$stage" "$DEST/$s.SHA256SUMS" <<'EOF'
import hashlib, json, shutil, sys
from pathlib import Path
src, stage, sums = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
t = json.loads((src / "dslr/nerfstudio/transforms_undistorted.json").read_text())
names = sorted(f["file_path"] for f in t["frames"] if not f.get("is_bad", False))
lines = []
for name in names:
    original = src / "dslr/resized_undistorted_images" / name
    shutil.copyfile(original, stage / name)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if hashlib.sha256((stage / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"copy of {name} differs from the source")
    lines.append(f"{digest}  {name}")
sums.write_text("\n".join(lines) + "\n")
print(f"{src.name}: {len(names)} of {len(t['frames'])} training frames (is_bad dropped), verified")
EOF
    mv "$stage" "$DEST/$s"
done

cat > "$ROOT/README.md" <<EOF
# scannetpp (local derived copies)

The ScanNet++ v2 release lives at $SRC (licensed; not copied here in bulk).
This root holds only small derived selections, owned by GS_playground.

- derived/surflo_inputs/<scene>/: the DSLR resized_undistorted_images of the scene's official training
  frames (dslr/nerfstudio/transforms_undistorted.json "frames"), minus frames flagged is_bad; byte copies,
  SHA-256 per file in derived/surflo_inputs/<scene>.SHA256SUMS, each checked against the source.
  Built by GS_playground's tools/fetch/fetch_scannetpp_surflo_inputs.sh for E08l (SuRFLo takes a folder).
- Licence: ScanNet++ terms of use (research only). Retention: working copy, rebuildable from that release.
EOF
echo "published $DEST: $(ls "$DEST" | grep -v SHA256SUMS | tr '\n' ' ')"
