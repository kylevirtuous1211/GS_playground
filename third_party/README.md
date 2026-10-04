# Upstream

`PINS.tsv` is the record: repo, url, commit, and which subdirectory we actually use.
`clones/` is untracked and rebuildable from it; `tools/setup/clone_upstream.sh` does that.

## The two rules that make this work

**Nothing is ever written inside a clone.**
Not checkpoints, not logs, not `work_dirs/`, not downloaded weights.
Upstream's own default is `work_dirs/<config_name>/` relative to the repo root, so every runner passes `--work-dir` explicitly and points it at `outputs/`.
A clone that only ever gets read can be deleted and re-cloned without losing anything, which is what makes `PINS.tsv` a sufficient record.

**Additions go in `src/gs_playground/`, not on the fork.**
Import upstream and override it.
A patch in `patches/` is only for edits to *upstream lines* - a bug we had to fix in their code, a hard-coded path we had to unhard-code.
Each patch is one commit with a one-line reason, and it appears in the table below or it does not exist.

## Patches

| patch | applies to | why |
|---|---|---|
| `gsplat_datasets_pkg.patch` | gsplat v1.5.3 | adds `examples/datasets/__init__.py`: as a namespace package it loses to the installed HF `datasets` package (regular packages beat namespace packages regardless of sys.path order), so `from datasets.colmap import ...` imports the wrong module |
| `surflo_save-guided-state.patch` | SuRFLo | saves guided mode's fitted state, so the 3DGS SuRFLo fits but never writes can be exported (`gs_playground.surflo.export`) |
| `surflo_vggt-weights.patch` | SuRFLo | a `vggt_weights` config key that loads another VGGT checkpoint (IGGT's backbone) into SuRFLo strictly; off by default |
| `surflo_save-vggt-tokens.patch` | SuRFLo | a `save_vggt_tokens` config key that saves the aggregator tokens of layers 4/11/17/23, so IGGT's heads can read SuRFLo's own encoding |
| `iggt_inference-shims.patch` | IGGT | inference without basicsr, detectron2, apex or the transformers import: replaces the four small helpers it used from them |
| `iggt_many-frames.patch` | IGGT | its point and instance heads mis-handled frame chunking above 12 frames; they now run on all frames at once, the path it already took at 12 or fewer |
| `iggt_any-grid.patch` | IGGT | its window attention needed a patch grid divisible by the window; reflect-pads and crops back (HAT's scheme), so SuRFLo's 518-wide grid works; bitwise unchanged at 504x336 |
| `worldsculpt_rembg-optional.patch` | WorldSculpt | its background remover is a gated HF model the scene pipeline never calls (crops carry alpha); a missing model no longer stops the pipeline from loading |

Regenerate a patch after editing a clone:

```bash
git -C third_party/clones/<name> diff > third_party/patches/<name>_<slug>.patch
```

and re-apply after a fresh clone with `git -C third_party/clones/<name> apply <patch>` (`clone_upstream.sh` does this for every patch).
If a patch stops applying, the pin moved: update `PINS.tsv` and the patch in the same commit.

## puffin

The base environment (`tools/setup/puffin_world_env.sh`) is built from the requirements of the pinned Puffin fork, which is why that pin stays and the env carries its name.
