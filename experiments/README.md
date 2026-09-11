# Experiments

A **study** is a directory: one question, one dataset family, one set of arms.
A **log entry** is a runner: `run_E##_<slug>.sh` inside a study.
The `E##` numbering is immutable and lives on the runners, so a study can grow without renumbering anything.

Reusable code belongs in `src/gs_playground/`, not here.
A runner is allowed to be a shell script that sets up a run directory, stamps provenance, calls one entry point, and parses the result to disk.
It is not allowed to be the place a model or a dataset is defined.

## Index

| entry | study | question | outcome |
|---|---|---|---|
| E00 | - | repository layout | done, see `LOG.md` |
| E01 | `puffin-world-repro` | does Stage I alignment train at the 1.5B size on one 48GB card? | not yet run |
| E02 | `puffin-world-repro` | do the released Pro weights run end-to-end locally? | **pass** (prereg criterion 1 mis-calibrated; see LOG) |
| E03 | `puffin-gsvoxel` | does the fitting-free voxelizer hold on indoor scenes? | **pass**: 34.51 dB / 0.987 SSIM over 12 scenes |
| E04 | `puffin-gsvoxel` | can the two-branch sparse VAE represent this data at all? | **pass**: overfit IoU 1.0, 31.37 dB vs bars 0.95 / 25 |
| E05a | `puffin-gsvoxel` | does held-out recon improve from N=3 to N=10 scenes? | **yes**: +1.36 dB held-out (bar +0.5) |
| E04b | `puffin-gsvoxel` | chamfer vs ordered direct loss | running |
| E06 | `puffin-gsvoxel` | what do the latents linearly encode? | queued |

## Studies

| study | question |
|---|---|
| [`puffin-world-repro`](puffin-world-repro/) | Can Puffin-World be trained from scratch here, and does the four-stage recipe hold at the 1.5B size? |
| [`puffin-gsvoxel`](puffin-gsvoxel/) | Can Puffin-World's world-state conditioning be carried into a GS-Voxel-style structured 3DGS latent? Design record. Does **not** depend on the reproduce - see E00b. |
