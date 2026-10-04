# Experiments

A **method** is a directory named after its paper, in lower case: `surflo/`, `worldsculpt/`.
A **log entry** is a runner: `run_E##_<slug>.sh` inside a method directory.
The `E##` numbering is immutable and lives on the runners and in `LOG.md`, never in directory names.
Retired studies live in `archive/`, which stays local and is not part of the public mirror.

Each method is a **reproduction arm first**: per `.claude/rules/research-discipline.md`, the first run of a borrowed method reports whether it reproduces on the authors' own setting before anything of ours is compared to it.

Reusable code belongs in `src/gs_playground/`, not here.
A runner is allowed to be a shell script that sets up a run directory, stamps provenance, calls one entry point, and parses the result to disk.
It is not allowed to be the place a model or a dataset is defined.

## Index

| entry | method | question | outcome |
|---|---|---|---|
| E08c | `worldsculpt` | does WorldSculpt run here on the authors' data? | **yes**: 13/13 objects in 8.5 min; no room shell, by design |
| E08e | `worldsculpt` | does WorldSculpt run on our own capture (NCHC sofa)? | **yes**: 24/24 objects reach `scene.glb`, placement right by eye; masks predicted, boxes ours |
| E08f | `worldsculpt` | which parameters change object count and mesh completeness? | count: views needed (stride, min_views); completeness: only 1 view and mask erosion/dilation beat seed noise; **shape stage nondeterministic at a fixed seed** (Stage B and that finding retracted by E08g: random conv weights) |
| E08g | `worldsculpt` | why did WorldSculpt's meshes look like dust? | **our bug**: spconv backend left the shape decoder's convs at random init; fixed, E08c/E08e rerun, deterministic |
| E08h | `surflo` | does SuRFLo reproduce on its own garden sample, and how does it do on our sofa? | **reproduces** (runtime/VRAM/outputs; accuracy not attempted, no GT released); 3DGS export exact (59-63 dB); sofa: unguided F1 0.87 vs guided 0.58, cause untested |
| E08i | `surflo` | what does per-scene optimisation cost on garden here, beside SuRFLo? | 3DGS 35.7 min / 10.8 GiB, 2DGS 23.0 min / 6.1 GiB (both valid, PSNR 27.6 / 26.8); SuRFLo 16-25x faster guided, 80-125x plain; memory about the same |
| E08j | `surflo` | how do cost and novel-view PSNR move with the number of views on garden? | 16/32/64 views: 22.98/23.60/24.50 dB refined against 31.84 for per-scene 3DGS; VRAM 12.6/15.4/22.3 GiB |
| E08k | `surflo` | all 161 garden training views, on an emptied card (post-hoc E08j arm) | 24.92 dB refined, 45.6 GiB; still 6.9 dB under 3DGS |
| E08l | `surflo` | does SuRFLo still work on IGGT's fine-tuned VGGT backbone? | **not compatible** by the pre-registered rule: every guided metric moved on every scene (garden -0.23 dB; ScanNet++ geometry better, but those scenes are in IGGT's training data) |
| E08m | `iggt` | do VGGT's cameras (to 128 frames) and IGGT's instances (to 32) degrade past their training range? | **no decline** on either |
| E08n | `iggt` | instance and class labels on SuRFLo's Gaussians from one encoding | exploratory: instances clean on sofa and garden; class names partly right |

## Methods

| method | paper | what it needs |
|---|---|---|
| [`surflo`](surflo/) | SuRFLo, NeurIPS 2026, arXiv 2606.13644 | 2-32 unposed photos; nothing else |
| [`iggt`](iggt/) | IGGT, ICLR 2026, arXiv 2510.22706 | photos; here it runs on SuRFLo's inputs and encoding |
| [`worldsculpt`](worldsculpt/) | WorldSculpt, arXiv 2609.05416 | posed frames **plus per-instance masks and 3D boxes** |

## Archive

Retired studies are indexed in `archive/README.md`, which stays local.
