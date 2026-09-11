# GS_playground

Camera-centric 3D world generation.
Two threads: reproduce **Puffin-World** training from scratch, then ask whether its native-3D-world-state formulation buys anything when the appearance stream is 3D Gaussian Splatting rather than a multi-view diffusion transformer.

## The finding

Nothing yet.
This repository was laid out on 2026-09-11 and has produced no numbers.
When it has one, it goes here, with its scope and the single command that reproduces it.

## Where things are

| | |
|---|---|
| `LOG.md` | numbered, append-only. The record of what was run and what it settled. |
| `experiments/<study>/` | one directory per question; `run_E##_*.sh` is the runner for log entry `E##`. |
| `results/` | tracked. Every number quoted anywhere else in this repo comes from here. |
| `src/gs_playground/` | the installed library. Runners import from it; it never imports from them. |
| `third_party/` | `PINS.tsv` + `patches/`. `clones/` is untracked and rebuildable. |
| `outputs/`, `data/` | untracked. Large, regenerable, and stamped with their provenance. |

## Reproducing anything

```bash
bash tools/setup/clone_upstream.sh          # third_party/clones/ at pinned commits
bash tools/setup/puffin_world_env.sh        # the puffin-world env (python 3.10, torch 2.7 cu126)
micromamba activate puffin-world
bash experiments/puffin-world-repro/run_E01_stage1_alignment.sh
python results/collect.py                   # outputs/ -> results/tables/
```

`python results/collect.py --check` exits non-zero when the tracked tables disagree with `outputs/`.
Run it before quoting a number.

## Hardware

One RTX 6000 Ada, 48 GB.
That constraint is why the 1.5B Puffin-World arm is the primary one; see `experiments/puffin-world-repro/README.md`.

## Upstream and licence

Puffin / Puffin-World is NTU S-Lab License 1.0 (see `third_party/clones/puffin/LICENSE`).
It is pinned, not vendored.
