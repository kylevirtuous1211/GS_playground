#!/usr/bin/env bash
# Rebuild docs/media/ (the README's clips and stills) from runs already on disk:
# SuRFLo on our sofa (E08h), IGGT semantics on SuRFLo (E08n), WorldSculpt's
# sofa objects (E08g). Each part runs in the env that has its renderer.
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
