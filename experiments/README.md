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

## Studies

| study | question |
|---|---|
| [`puffin-world-repro`](puffin-world-repro/) | Can Puffin-World be trained from scratch here, and does the four-stage recipe hold at the 1.5B size? |
| [`puffin-gsvoxel`](puffin-gsvoxel/) | Can Puffin-World's world-state conditioning be carried into a GS-Voxel-style structured 3DGS latent? Design record; no runners until the reproduce lands. |
