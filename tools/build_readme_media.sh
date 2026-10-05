#!/usr/bin/env bash
# Rebuild docs/media/ (the README's clips and stills) from runs already on disk,
# all on Mip-NeRF 360 garden: SuRFLo at 16 views (E08j) and IGGT semantics on
# SuRFLo (E08n). Our own captures are not public, so nothing here renders them.
#
#   bash tools/build_readme_media.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

puffin_py() { ( source tools/setup/activate_puffin_world.sh &&
                export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" && python "$@" ); }

puffin_py -m gs_playground.readme_media surflo        # gsplat
puffin_py -m gs_playground.readme_media semantics     # gsplat
du -h docs/media/*
