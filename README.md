# GS_playground

State-of-the-art Gaussian splatting methods, each run here on our own data and each with a demo a person can open.
Breadth is the point; `CLAUDE.md` says what "done" means for a method, and `results/DEMOS.md` lists the demos.

Methods so far: [`worldgrow`](experiments/worldgrow/README.md), [`worldsculpt`](experiments/worldsculpt/README.md).
The earlier Puffin-World reproduction and the Puffin x GS-Voxel latent study are archived in [`experiments/archive/`](experiments/archive/).

## The finding

Nothing yet.
This repository was laid out on 2026-09-11 and has produced no numbers.
When it has one, it goes here, with its scope and the single command that reproduces it.

## Where things are

| | |
|---|---|
| `LOG.md` | numbered, append-only. The record of what was run and what it settled. |
| `experiments/<method>/` | one directory per method, named after its paper; `run_E##_*.sh` is the runner for log entry `E##`. |
| `results/` | tracked. Every number quoted anywhere else in this repo comes from here, per method in `results/<method>/`. |
| `src/gs_playground/` | the installed library: `<method>/` per method plus shared `gs/`, `viewer.py`, `datasets/`. Runners import from it; it never imports from them. |
| `*/archive/` | retired studies, same layout underneath. |
| `third_party/` | `PINS.tsv` + `patches/`. `clones/` is untracked and rebuildable. |
| `outputs/`, `data/` | untracked. Large, regenerable, and stamped with their provenance. |

## Reproducing anything

```bash
bash tools/setup/clone_upstream.sh          # third_party/clones/ at pinned commits
bash tools/setup/puffin_world_env.sh        # the puffin-world env (python 3.10, torch 2.7 cu126)
micromamba activate puffin-world
bash tools/setup/worldgrow_env.sh           # an overlay env on top of it
bash experiments/worldgrow/run_E08b_worldgrow.sh
python results/collect.py                   # outputs/ -> results/tables/
```

`python results/collect.py --check` exits non-zero when the tracked tables disagree with `outputs/`.
Run it before quoting a number.

## Hardware

One RTX 6000 Ada, 48 GB, so a method has to run on one GPU to be tried here first.
Big jobs go to the nano4 Slurm cluster.

## Upstream and licence

Puffin / Puffin-World is NTU S-Lab License 1.0 (see `third_party/clones/puffin/LICENSE`).
It is pinned, not vendored.
