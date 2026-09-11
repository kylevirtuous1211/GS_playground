#!/usr/bin/env bash
# Stamp a run directory with the exact code that produced it.
#
#   source "$ROOT/tools/stamp_provenance.sh"
#   stamp_provenance "$run" puffin
#
# The working diff matters as much as the commit: research code is edited
# between runs faster than it is committed, so the sha alone answers the wrong
# question.
stamp_provenance() {
    local run="$1" upstream="${2:-}"
    local root; root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    mkdir -p "$run"
    git -C "$root" rev-parse HEAD 2>/dev/null > "$run/repo.sha" || echo unborn > "$run/repo.sha"
    git -C "$root" diff                      > "$run/repo.patch"
    git -C "$root" status --porcelain        > "$run/repo.status"
    if [ -n "$upstream" ]; then
        local up="$root/third_party/clones/$upstream"
        git -C "$up" rev-parse HEAD 2>/dev/null > "$run/upstream.sha" || echo unknown > "$run/upstream.sha"
        git -C "$up" diff                    > "$run/upstream.patch"
    fi
    {
        echo "host      $(hostname)"
        echo "date_utc  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "python    $(command -v python3) $(python3 -V 2>&1)"
        echo "cuda      $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | paste -sd'; ')"
        echo "cmd       $0 $*"
    } > "$run/env.txt"
}
