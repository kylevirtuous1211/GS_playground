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

Regenerate a patch after editing a clone:

```bash
git -C third_party/clones/puffin diff > third_party/patches/puffin_<slug>.patch
```

and re-apply after a fresh clone with `git -C third_party/clones/puffin apply <patch>`.
If a patch stops applying, the pin moved: update `PINS.tsv` and the patch in the same commit.

## puffin

Fork of [KangLiao929/Puffin](https://github.com/KangLiao929/Puffin) (NTU S-Lab License 1.0).
Two subprojects; we use **`Puffin-World/`**.

- `Puffin/` - *Thinking with Camera* (ICLR 2026), single-image camera-centric understanding + generation.
- `Puffin-World/` - *Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States* (arXiv 2609.04196).
  Vision encoder (RADIO) + LLM (Qwen2.5) + DiT (SD3.5), trained in four stages; represents a scene as physics (gravity field, latitude), geometry (depth) and appearance (RGB).

Entry points we care about, all relative to `clones/puffin/Puffin-World/`:

| what | path |
|---|---|
| training entry | `scripts/train.py` (accepts `--work-dir`; `scripts/train_ddp.sh` does not forward it, so runners call `torchrun scripts/train.py` directly) |
| stage configs | `configs/pipelines/final_stage_{1,2,3,4}_*.py` |
| model sizes | `configs/models/qwen2_5_1_5b_radiov4H_sd3p5L.py` (small), `qwen2_5_7b_radiov3H_sd3p5M.py` (paper) |
| multi-view data | `configs/datasets/multi_view/gen_*.py` |
| index caches | `scripts/build_dataset_caches.py` |
| eval | `scripts/evaluation/{understanding,generation,generation_multi_view}.py` |
| docs | `documents/{TRAINING,EVALUATION,EVALUATION_World,DATASET_PIPELINE,ANNOTATION_CAMERA,ANNOTATION_DEPTH}.md` |
