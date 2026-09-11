#!/usr/bin/env bash
# Rebuild third_party/clones/ from third_party/PINS.tsv, at the pinned commits.
# Idempotent: an existing clone is fetched and checked out, never deleted.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PINS="$ROOT/third_party/PINS.tsv"
CLONES="$ROOT/third_party/clones"

mkdir -p "$CLONES"
tail -n +2 "$PINS" | while IFS=$'\t' read -r name url commit subdir notes; do
    [ -n "${name:-}" ] || continue
    dest="$CLONES/$name"
    if [ ! -d "$dest/.git" ]; then
        echo "== cloning $name"
        git clone --quiet "$url" "$dest"
    else
        echo "== fetching $name"
        git -C "$dest" fetch --quiet --all --tags
    fi
    if ! git -C "$dest" diff --quiet || ! git -C "$dest" diff --cached --quiet; then
        echo "!! $name has uncommitted changes; refusing to checkout $commit" >&2
        echo "   capture them first: git -C $dest diff > $ROOT/third_party/patches/${name}_<slug>.patch" >&2
        continue
    fi
    git -C "$dest" checkout --quiet --detach "$commit"
    echo "   $name @ $(git -C "$dest" rev-parse --short HEAD)"
done

shopt -s nullglob
for patch in "$ROOT"/third_party/patches/*.patch; do
    name="$(basename "$patch")"; name="${name%%_*}"
    echo "== applying $(basename "$patch") to $name"
    git -C "$CLONES/$name" apply --check "$patch" \
        && git -C "$CLONES/$name" apply "$patch" \
        || echo "!! $(basename "$patch") does not apply; the pin moved" >&2
done
