# outputs/

Untracked.
One directory per run: `outputs/<study>/<arm>/`.
Semantic slugs only - never a uuid, never a bare timestamp, because the deliverable is a comparison table and a table cannot be built from `output/a3f9c21b04`.

Every run directory carries its own provenance, written by `tools/stamp_provenance.sh` before training starts:

| file | what |
|---|---|
| `repo.sha`, `repo.patch`, `repo.status` | this repository's commit *and* its uncommitted diff |
| `upstream.sha`, `upstream.patch` | the pinned clone's commit and diff |
| `env.txt` | host, UTC date, python, GPU |
| `metrics.json` | every number the run produced. Printed is not saved. |

`results/collect.py` reads the `metrics.json` files from here and builds `results/tables/`.
Redirect with `GS_PLAYGROUND_OUTPUTS=/mnt/...`.
