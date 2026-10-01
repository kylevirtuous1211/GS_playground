#!/usr/bin/env bash
# Fetch the official Mip-NeRF 360 scenes into the machine-local dataset root,
# following the fleet storage policy (~/datasets/AGENTS.md): stage under a
# .partial directory, verify, write a manifest, publish by one rename.
#
#   bash tools/fetch/fetch_mipnerf360.sh
#
# Then the repository links the whole dataset directory, relatively:
#   data/mipnerf360 -> ../../../datasets/mipnerf360/360_v2
#
# Also builds one derived view of garden for E08j: its standard training split
# (every image but every 8th, sorted by name) at images_4, as hardlinks, since
# SuRFLo takes a folder and the policy forbids symlinking individual files.
set -euo pipefail

URL="http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
BYTES=12535427936
MD5="cef2ef3aeaf0c062dbe65130bc249870"   # the object's GCS ETag, which is its MD5
ROOT="$HOME/datasets/mipnerf360"
STAGE="$HOME/datasets/.mipnerf360.partial"

if [ -d "$ROOT" ]; then echo "$ROOT exists; nothing to do"; exit 0; fi
mkdir -p "$STAGE"
cd "$STAGE"

if [ ! -f 360_v2.zip ] || [ "$(stat -c %s 360_v2.zip)" != "$BYTES" ]; then
    curl -fSL --retry 3 -C - -o 360_v2.zip "$URL"
fi
[ "$(stat -c %s 360_v2.zip)" = "$BYTES" ] || { echo "size mismatch" >&2; exit 1; }
echo "$MD5  360_v2.zip" | md5sum -c -

rm -rf 360_v2 && mkdir 360_v2
unzip -q 360_v2.zip -d 360_v2
find 360_v2 -type f -printf '%P\n' | sort > files.txt
( cd 360_v2 && xargs -a ../files.txt -d '\n' sha256sum ) > SHA256SUMS

# derived: garden's standard training split, for SuRFLo's folder input
TRAIN="derived/garden_train161_images_4"
mkdir -p "$TRAIN"
i=0
while IFS= read -r name; do
    [ $((i % 8)) -ne 0 ] && ln "360_v2/garden/images_4/$name" "$TRAIN/$name"
    i=$((i + 1))
done < <(ls 360_v2/garden/images_4 | sort)

cat > README.md <<EOF
# mipnerf360

Official Mip-NeRF 360 scenes (Barron et al., CVPR 2022), from $URL.
Fetched $(date -u +%Y-%m-%dT%H:%M:%SZ) by GS_playground's tools/fetch/fetch_mipnerf360.sh.

- Source zip: $BYTES bytes, MD5 $MD5 (verified), kept as 360_v2.zip.
- Payload: 360_v2/, $(wc -l < files.txt) files, $(du -sb 360_v2 | cut -f1) bytes; per-file SHA-256 in SHA256SUMS.
- derived/garden_train161_images_4: hardlinks to garden's images_4 minus every 8th image by sorted name
  ($(ls "$TRAIN" | wc -l) files), the standard training split; used by GS_playground E08j.
- gsplat writes images_<factor>_png beside a scene's images when it trains or parses at a factor; those are derived too.
- Owner: GS_playground. Licence: as released by the authors for research. Retention: working copy, re-fetchable.
EOF

cd "$HOME/datasets"
mv "$STAGE" "$ROOT"
echo "published $ROOT"
