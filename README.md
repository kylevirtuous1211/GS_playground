<h1 align="center">GS_playground</h1>

<p align="center">
Recent Gaussian-splatting and feed-forward 3D methods, run end to end on our own captures and on public benchmark scenes, each with something to look at.
</p>

<table>
<tr>
<td width="50%"><img src="docs/media/surflo_sofa.webp" alt="SuRFLo's 3D Gaussians of our lounge, flown between two of its input cameras"></td>
<td width="50%"><img src="docs/media/worldsculpt_sofa_turntable.webp" alt="WorldSculpt's 24 object meshes of the same lounge, turning"></td>
</tr>
<tr>
<td align="center"><a href="#surflo-16-unposed-photos-to-3d-gaussians"><b>SuRFLo</b></a>: 16 unposed photos of our lounge in, 3D Gaussians out</td>
<td align="center"><a href="#worldsculpt-a-room-taken-apart-into-object-meshes"><b>WorldSculpt</b></a>: the same capture, 24 objects we picked and boxed, each completed as a mesh</td>
</tr>
</table>

Each method here is pinned to its upstream commit, patched only where it had to be, run on data we captured or chose, and given a demo a person can open.
Numbers on this page come from the tracked `results/` files or method READMEs named next to them.

## SuRFLo: 16 unposed photos to 3D Gaussians

<img src="docs/media/surflo_sofa_inputs.jpg" width="100%" alt="The 16 input photos">

[SuRFLo](https://github.com/Anttwo/Surflo) (Guédon et al., NeurIPS 2026, [arXiv 2606.13644](https://arxiv.org/abs/2606.13644)) builds a mesh, and in guided mode a 3D Gaussian scene, from 2 to 32 unposed photos, on top of a frozen VGGT.
The clip at the top is SuRFLo's 3D Gaussians of our own handheld capture of a lounge: it was given the 16 frames above, picked from the video, with no poses and no depth.
The clip flies between two of those input cameras, so most of what it shows lies between the views SuRFLo was given.
It is rendered at 1.5 times the resolution SuRFLo fitted, with gsplat's usual 0.3-pixel dilation, and the two cameras were picked so the path does not fly through a pillar.

What we added:

- **The 3DGS export.** Guided mode fits Gaussians to the photos but never writes them; [`gs_playground.surflo.export`](src/gs_playground/surflo/export.py) saves them as a standard 3DGS PLY, which matches SuRFLo's own renders at a per-run median of 59 to 63 dB PSNR, though a few single cameras fall to 12 to 21 dB, not yet explained ([`e08h_export_check.json`](results/surflo/e08h_export_check.json)).
- **Cost and novel-view quality against a per-scene 3DGS**, on Mip-NeRF 360 garden, from 16 input views to all 161 training views (table below).
- **A backbone swap**: SuRFLo on IGGT's fine-tuned VGGT, the basis of the semantics below.

| Mip-NeRF 360 garden | input views | time | peak GPU memory allocated | novel-view PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| SuRFLo | 16 | 2.0 min | 11.7 GiB | 22.98 dB | 0.705 |
| SuRFLo | 64 | 2.3 min | 21.9 GiB | 24.50 dB | 0.775 |
| SuRFLo, post hoc | 161 | 3.2 min | 45.3 GiB | 24.92 dB | 0.777 |
| 3DGS (gsplat, 30k steps, COLMAP poses) | 161 | 35.7 min | 10.8 GiB | 31.84 dB | 0.954 |

- An exploratory sweep: one scene, one RTX 6000 Ada, SuRFLo's default guided preset ([`e08j_views_sweep.json`](results/surflo/e08j_views_sweep.json), [`e08i_perscene_garden.json`](results/surflo/e08i_perscene_garden.json)).
- SuRFLo rows are medians over 3 seeds; the 3DGS row is one run.
- The 161-view row is a post-hoc arm run with nothing else on the 48 GB card; the run as planned, with another user's 7.1 GiB on the card, ran out of memory ([`e08j_n161_posthoc.json`](results/surflo/e08j_n161_posthoc.json)).
- Time is SuRFLo's wall clock including model loading, and 3DGS's training time.
- Memory is SuRFLo's ODE stage only, which understates its whole run at large view counts, and all of 3DGS's training.
- PSNR and SSIM are the median over 24 held-out views at SuRFLo's 518x322, after test-time refinement of the test cameras; the 3DGS is rendered at its 1297x840 training resolution and downscaled like the ground truth (at full resolution it scores 27.65 dB).

SuRFLo is 11 to 18 times faster than training a 3DGS here and needs no poses, but its novel views sit 7 to 9 dB below the per-scene 3DGS, and going from 16 to 161 input views closes about 2 dB of that gap.
Two things this does not separate: the default preset keeps SuRFLo at about 0.19M Gaussians whatever the view count (a larger preset was not tried), and SuRFLo leaves out the far background, which a full-image PSNR counts against it.
On our lounge (exploratory, one capture, default preset), guided mode's geometry agrees with a reference built from our own dense 3DGS less than unguided mode's does (F1 0.58 against 0.87, [`e08h_stage_b.json`](results/surflo/e08h_stage_b.json)); the cause is untested.

### Semantics from the same encoding: IGGT on SuRFLo

<img src="docs/media/iggt_semantics_sofa.webp" width="100%" alt="The same flight: SuRFLo's Gaussians, coloured by IGGT instance group, and coloured by CLIP class">

<details>
<summary>Class colours</summary>
<img src="docs/media/iggt_semantics_sofa_legend.png" width="520" alt="Class legend">
</details>

[IGGT](https://github.com/lifuguan/IGGT_official) (Li et al., [arXiv 2510.22706](https://arxiv.org/abs/2510.22706)) is VGGT fine-tuned with an 8-dimensional instance head.
We ran SuRFLo on IGGT's backbone and let IGGT's instance head read the very tokens SuRFLo had computed, so one backbone pass gives both the surface and per-pixel instance features (against IGGT's own forward pass, per-pixel cosine similarity median 1.000000, minimum 0.99966).
The features are carried onto SuRFLo's Gaussians through its cameras and grouped, and each group holding at least 0.2% of the Gaussians is named by CLIP from ScanNet++'s 100 most common classes plus a few outdoor words.

This one is **exploratory**, judged by eye only.
On the lounge, instance groups mostly separate the sofa, cushions, pillar, hanging plants and TV, but neighbouring stools merge into one group and the floor and walls split into several; on a ScanNet++ room the groups are fragmented.
Class names are right for some distinct objects (sofa, floor, plant, TV) and wrong for much of the rest: on the lounge the most common name is "blind rail" (the far window wall), white walls often come out as "whiteboard" and the stools as "table".
On garden the ground is named "table", even with the table it surrounds inpainted out of its crops; the cause is untested.

Two IGGT bugs were fixed on the way ([`third_party/patches/`](third_party/patches/)): its heads broke above 12 input frames, and its window attention could not take SuRFLo's 518-pixel-wide input.
Two questions were measured with pre-registered metrics:

- **More frames:** neither VGGT's cameras (Mip-NeRF 360, AUC@30, median over seven scenes, 0.986 at 24 frames, its training maximum, and 0.987 at 128) nor IGGT's instances (up to 32 frames, ScanNet++ ground truth on scenes from its training data) degrade past their training range ([`results/iggt/`](results/iggt/)).
- **SuRFLo on IGGT's backbone:** no longer the same SuRFLo, since every guided metric moved beyond seed noise on every scene: slightly worse on garden, better on the ScanNet++ scenes IGGT was trained on ([`e08l_iggt_backbone.json`](results/surflo/e08l_iggt_backbone.json)).

## WorldSculpt: a room taken apart into object meshes

<img src="docs/media/worldsculpt_sofa_view.jpg" width="100%" alt="One input frame, the meshes' normals, one colour per object, and the objects over the frame">

[WorldSculpt](https://github.com/AlayaLab/WorldSculpt) ([arXiv 2609.05416](https://arxiv.org/abs/2609.05416)) completes every object in a scene as a whole mesh, from posed frames, a mask per object in each frame, and a 3D box per object.
Our lounge capture already had COLMAP poses, a trained 3DGS and label maps associated across views from an earlier project.
From those we built the per-frame object masks and one box per object ([`gs_playground.worldsculpt.ground`](src/gs_playground/worldsculpt/ground.py)), with the objects to keep picked by an automatic filter and then on a review page.
All 24 objects given to it come out as meshes in one scene ([`e08e_checks.json`](results/worldsculpt/e08e_checks.json)), in about 13 minutes on one GPU from prepared crops ([method README](experiments/worldsculpt/README.md#our-own-captures-e08e)).
The turntable at the top shows one flat colour per object, meshes decimated for display, framed on the sofa group, so three far objects leave the shot for most of the turn.
The still above is WorldSculpt's own render of one input view: the frame, the meshes' normals, one colour per object, and the objects over the frame.

**A silent failure, found and fixed.**
Our base environment selected the spconv sparse-convolution backend (upstream defaults to flex_gemm), and under it 80 of the shape decoder's 292 parameter tensors stayed at random initialisation without a warning.
Every object came out as plausible-looking dust: on the authors' scene, a table decoded into about 15,000 disconnected blobs.
With flex_gemm the checkpoint loads in full, the sofas, stools and pillows look like clean solids, and two runs at one seed give bitwise-identical vertices (on the authors' scene).
The details are in [the method's README](experiments/worldsculpt/README.md#traps).

**Limits.**
Walls, floor and ceiling were not given to it, so the room's shell is absent by construction.
The potted plant is missing: the label maps gave it the walls' id, so it was never handed over as an object.
The hanging planter came out as a large flat slab, and the TV as a deep solid block instead of a thin panel.

## Running it

```bash
bash tools/setup/clone_upstream.sh                    # every pin in third_party/PINS.tsv, patches applied
bash tools/setup/puffin_world_env.sh                  # the base env (python 3.10, torch 2.7, cu126)
bash tools/setup/surflo_env.sh                        # SuRFLo and IGGT
bash tools/setup/worldsculpt_env.sh                   # WorldSculpt, layered on the base env
```

Each method's README in [`experiments/`](experiments/) gives its weights, its runners and the traps we hit.
Runners read machine-local data roots from an untracked `.env.local`, sourced by the activation scripts (`GS_PLAYGROUND_DATA`, `GS_PLAYGROUND_SCANNETPP`, `MIPNERF360_GARDEN`, ...); a runner names any variable it is missing.
Our own lounge capture is not part of this repository; Mip-NeRF 360 is fetched by [`tools/fetch/fetch_mipnerf360.sh`](tools/fetch/fetch_mipnerf360.sh), and ScanNet++ needs its own licence.
Each demo on this page also has a self-contained web viewer (Spark and three.js for Gaussians, three.js for meshes, served with `python -m http.server`); the viewers are built locally from runs into the untracked `outputs/` and are not in the repository, and [`results/DEMOS.md`](results/DEMOS.md) lists each one with its builder.
The clips here are built from those runs by [`tools/build_readme_media.sh`](tools/build_readme_media.sh).

## How results are kept

- A run that settles a question is pre-registered first: its metric and decision rule are committed before it starts (`experiments/<method>/PREREG_E##.md`), later changes are dated amendments that say what had already been seen, and post-hoc arms are labelled post hoc.
- A run that only explores is labelled exploratory: on this page, the view-count table, the lounge geometry comparison, the semantics and WorldSculpt on the lounge.
- Runners write the numbers they report to `results/<method>/` and stamp their output directory with this repository's and the upstream's commit and diff, the host, GPU and command.
- The lab notebook these runs were logged in stays private; the method READMEs, pre-registrations and `results/` carry most of what it settled.

| where | what |
|---|---|
| [`experiments/<method>/`](experiments/) | one directory per method, named after its paper; `run_E##_<slug>.sh` is a runner |
| [`src/gs_playground/`](src/gs_playground/) | the library: one package per method, plus shared Gaussian I/O and rendering, the web viewer, and this page's media builder |
| [`results/`](results/) | the numbers each runner writes, per method |
| [`third_party/`](third_party/) | `PINS.tsv` (upstream commits) and `patches/` (each with its reason); clones are rebuilt, not vendored |
| [`docs/media/`](docs/media/) | the clips and stills on this page |

All of it ran on one RTX 6000 Ada (48 GB).

## Upstream code and licences

| method | upstream | pinned | licence |
|---|---|---|---|
| SuRFLo | [Anttwo/Surflo](https://github.com/Anttwo/Surflo) | `bf14c63` | code under the Gaussian-Splatting licence (Inria, MPII), non-commercial research use, with VGGT-derived files under Meta's VGGT License; weights CC BY-NC 4.0 |
| IGGT | [lifuguan/IGGT_official](https://github.com/lifuguan/IGGT_official) | `bec6604` | code MIT per its README; weights derived from VGGT-1B (CC BY-NC) |
| WorldSculpt | [AlayaLab/WorldSculpt](https://github.com/AlayaLab/WorldSculpt) | `fac6b83` | own code and LoRA weights Apache 2.0; vendored Pixal3D code and base weights MIT; DINOv3 weights under Meta's DINOv3 License (gated) |
| gsplat | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | `937e299` (v1.5.3, its examples; the library comes from the env) | Apache 2.0 |

Nothing upstream is vendored; [`tools/setup/clone_upstream.sh`](tools/setup/clone_upstream.sh) clones every pin (a few belong to retired studies or only feed the env builds) and applies our patches.

This repository's own code is under the [Apache License 2.0](LICENSE); upstream code and weights keep the licences above, including the non-commercial terms on SuRFLo's and IGGT's weights.
