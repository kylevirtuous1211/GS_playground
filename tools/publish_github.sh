#!/usr/bin/env bash
# Publish local `main` to GitHub as a filtered history: the archived studies,
# the private lab log, the handoff notes and the agent instructions never
# leave this machine. The local repository has no remote on purpose; this
# script is the only way anything reaches GitHub.
#
#   bash tools/publish_github.sh            # dry run: filter, check, report
#   bash tools/publish_github.sh --push     # the same, then push to GitHub main
#   PUBLISH_SOURCE=docs/x bash tools/publish_github.sh   # dry run another branch
#
# It publishes the committed ref, never the working tree.
#
# git filter-branch keeps authors, dates and messages, so the same local
# history always filters to the same commits and a later publish is a
# fast-forward; the push is never forced.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="https://github.com/kylevirtuous1211/GS_playground.git"
SOURCE="${PUBLISH_SOURCE:-main}"
PUSH=0
[ "${1:-}" = "--push" ] && PUSH=1

# Every path that must never be published, at its current place and at every
# place it lived before (LOG E00c and E00d moved the archived studies).
DROP=(
    ':(glob)**/archive/**'
    LOG.md HANDOFF.md CLAUDE.md
    src/aoss_client
    experiments/sota-gs experiments/puffin-gsvoxel experiments/puffin-world-repro experiments/worldgrow
    src/gs_playground/sota src/gs_playground/gsvoxel src/gs_playground/puffin_world.py
    src/gs_playground/eval src/gs_playground/report src/gs_playground/train
    src/gs_playground/worldgrow src/gs_playground/datasets
    results/worldgrow results/tables results/collect.py
    probes/probe_e02_key_mismatch.py
    tools/setup/worldgrow_env.sh
    tools/fetch/fetch_dl3dv_subset.sh tools/fetch/normalize_dl3dv.sh tools/fetch/reconstruct_dl3dv.sh
    tools/fetch/archive_dl3dv_from_nano4.sh tools/fetch/write_dl3dv_cameras.sh
    tools/fetch/fetch_puffin_world_weights.sh
    tests/test_dl3dv_adapter.py tests/test_e05_split.py
)
# Path names that would mean the list above missed something. The base env
# keeps the name of the fork it is built from (third_party/README.md).
BAD_PATH='archive|worldgrow|puffin|gsvoxel|dl3dv|trellis|sota|aoss'
ALLOWED_PATH='^tools/setup/(puffin_world_env|activate_puffin_world)\.sh$'
# Contents that must not appear in any published version of any file (logins,
# hosts): one extended regex per line in an untracked file, so this script
# does not publish them itself.
BLOCKLIST="$ROOT/.git/info/publish_blocklist"
[ -s "$BLOCKLIST" ] || { echo "missing $BLOCKLIST (one regex per line of what must never be published)" >&2; exit 1; }

if [ "$PUSH" = 1 ] && [ "$SOURCE" != main ]; then
    echo "only main is pushed; PUBLISH_SOURCE is for dry runs" >&2
    exit 1
fi
echo "== source: $SOURCE at $(git -C "$ROOT" log -1 --format='%h %s' "$SOURCE")"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/gs_playground_publish.XXXXXX")"
git clone -q --no-local --single-branch --branch "$SOURCE" "$ROOT" "$WORK/repo"
cd "$WORK/repo"
git remote remove origin
before=$(git rev-list --count HEAD)

printf -v drop_args '%q ' "${DROP[@]}"
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --prune-empty \
    --index-filter "git rm -r -q --cached --ignore-unmatch -- $drop_args" -- "$SOURCE" > /dev/null
git for-each-ref --format='%(refname)' refs/original | xargs -r -n1 git update-ref -d
git reflog expire --expire=now --all
git gc -q --prune=now

echo "== filtered $SOURCE: $before commits -> $(git rev-list --count HEAD), $(git ls-files | wc -l) files at the tip"
du -sh .git | sed 's/^/   .git /'

fail=0
leaked_paths=$(git log --name-only --format= HEAD | sort -u | grep -iE "$BAD_PATH" | grep -vE "$ALLOWED_PATH" || true)
if [ -n "$leaked_paths" ]; then
    echo "!! paths in the filtered history that should have been dropped:"; echo "$leaked_paths" | sed 's/^/   /'
    fail=1
fi
leaked_content=$(git grep -I -l -E -f "$BLOCKLIST" $(git rev-list HEAD) -- . 2>/dev/null | sort -u || true)
if [ -n "$leaked_content" ]; then
    echo "!! files in the filtered history matching $BLOCKLIST:"; echo "$leaked_content" | sed 's/^/   /'
    fail=1
fi

echo "== relative links in the tip's Markdown that point at nothing"
git ls-files '*.md' | while read -r md; do
    dir=$(dirname "$md")
    { grep -oE '\]\([^)#[:space:]]+\)|src="[^"#]+"' "$md" || true; } |
        sed -E 's/^\]\(//; s/\)$//; s/^src="//; s/"$//' |
        { grep -vE '^(https?:|mailto:)' || true; } | while read -r link; do
            target=$(realpath -m --relative-to=. "$dir/$link")
            [ -e "$target" ] || echo "   $md -> $link"
        done
done
echo "== tip files that still name an archived study (reported, not blocked)"
git grep -I -c -i -E 'worldgrow|puffin|trellis|dl3dv|gs-?voxel' -- . | sed 's/^/   /' || true

if [ "$fail" = 1 ]; then
    echo "not publishing: fix the drop list. The filtered clone is at $WORK/repo" >&2
    exit 1
fi
if [ "$PUSH" = 0 ]; then
    echo "dry run: nothing pushed. The filtered clone is at $WORK/repo"
    exit 0
fi
git push "$REMOTE" "HEAD:refs/heads/main"
echo "pushed $(git rev-parse --short HEAD) to $REMOTE main"
