# Experiments

A **method** is a directory named after its paper, in lower case: `worldgrow/`, `worldsculpt/`.
A **log entry** is a runner: `run_E##_<slug>.sh` inside a method directory.
The `E##` numbering is immutable and lives on the runners and in `LOG.md`, never in directory names.
Retired studies live in `archive/`.

Each method is a **reproduction arm first**: per `.claude/rules/research-discipline.md`, the first run of a borrowed method reports whether it reproduces on the authors' own setting before anything of ours is compared to it.

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
| E04b | `puffin-gsvoxel` | chamfer vs ordered direct loss | **negative**: ordered wins by 0.78 dB, stays default |
| E06 | `puffin-gsvoxel` | what do the latents linearly encode? | **done**: occupancy strongly, colour partly; factorisation clean; gravity + semantics not linearly present |
| E08b | `worldgrow` | do WorldGrow's released weights run here? | **yes**: six worlds, 3 m to 9 m, peak VRAM flat at 9.5-11.6 GB |
| E08c | `worldsculpt` | does WorldSculpt run here on the authors' data? | **yes**: 13/13 objects in 8.5 min; no room shell, by design |
| E08d | `worldgrow` | does WorldGrow follow a text prompt or guidance? | **no**: output changes (IoU 0.62-0.84 vs 0.999 repeat), room type does not (CLIP 0-1/16) |
| E08e | `worldsculpt` | does WorldSculpt run on our own capture (NCHC sofa)? | **yes**: 24/24 objects reach `scene.glb`, placement right by eye; masks predicted, boxes ours |
| E08f | `worldsculpt` | which parameters change object count and mesh completeness? | count: views needed (stride, min_views); completeness: only 1 view and mask erosion/dilation beat seed noise; **shape stage nondeterministic at a fixed seed** (Stage B and that finding retracted by E08g: random conv weights) |
| E08g | `worldsculpt` | why did WorldSculpt's meshes look like dust? | **our bug**: spconv backend left the shape decoder's convs at random init; fixed, E08c/E08e rerun, deterministic |
| E08h | `surflo` | does SuRFLo reproduce on its own garden sample, and how does it do on our sofa? | **reproduces** (runtime/VRAM/outputs; accuracy not attempted, no GT released); 3DGS export exact (59-63 dB); sofa: unguided F1 0.87 vs guided 0.58, cause untested |
| E08i | `surflo` | what does per-scene optimisation cost on garden here, beside SuRFLo? | 3DGS 35.7 min / 10.8 GiB, 2DGS 23.0 min / 6.1 GiB (both valid, PSNR 27.6 / 26.8); SuRFLo 16-25x faster guided, 80-125x plain; memory about the same |

## Methods

| method | paper | what it needs |
|---|---|---|
| [`worldgrow`](worldgrow/) | WorldGrow, AAAI 2026 Oral, arXiv 2510.21682 | nothing but weights: the released pipeline is unconditional |
| [`worldsculpt`](worldsculpt/) | WorldSculpt, arXiv 2609.05416 | posed frames **plus per-instance masks and 3D boxes** |
| [`surflo`](surflo/) | SuRFLo, NeurIPS 2026, arXiv 2606.13644 | 2-32 unposed photos; nothing else |

## Archive

| study | question |
|---|---|
| [`archive/puffin-world-repro`](archive/puffin-world-repro/) | Can Puffin-World be trained from scratch here, and does the four-stage recipe hold at the 1.5B size? |
| [`archive/puffin-gsvoxel`](archive/puffin-gsvoxel/) | Can Puffin-World's world-state conditioning be carried into a GS-Voxel-style structured 3DGS latent? Includes E07's TRELLIS run. |
