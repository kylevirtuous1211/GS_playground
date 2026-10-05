#!/usr/bin/env bash
# Rebuild docs/media/ (the README's clips and stills) from runs already on disk:
# SuRFLo at 16 views (E08j) and IGGT semantics on SuRFLo (E08n), both on
# Mip-NeRF 360 garden, and WorldSculpt's objects of the authors' released Marble
# living room (E08g). Our own captures are not public, so nothing here renders
# them. Each part runs in the env that has its renderer.
#
#   bash tools/build_readme_media.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

puffin_py() { ( source tools/setup/activate_puffin_world.sh &&
                export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" && python "$@" ); }
surflo_py() { ( source tools/setup/activate_surflo.sh &&
                export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" && python "$@" ); }

puffin_py -m gs_playground.readme_media surflo        # gsplat
puffin_py -m gs_playground.readme_media semantics     # gsplat
surflo_py -m gs_playground.readme_media worldsculpt   # nvdiffrast
du -h docs/media/*
