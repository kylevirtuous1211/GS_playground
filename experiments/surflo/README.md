# SuRFLo

Guédon et al., "Surflo: Consistent 3D Surface Flow Model with Global State", NeurIPS 2026, arXiv 2606.13644.
Code [`Anttwo/Surflo`](https://github.com/Anttwo/Surflo), pinned in `third_party/PINS.tsv`; weights `AntoineGuedon/Surflo-v0` (CC BY-NC 4.0, research only).

## Results: our lounge and Mip-NeRF 360 garden

<img src="../../docs/media/surflo_sofa.webp" alt="SuRFLo's 3D Gaussians of our lounge, flown between two of its input cameras">
<img src="../../docs/media/surflo_sofa_inputs.jpg" width="100%" alt="The 16 input photos">

SuRFLo builds a mesh, and in guided mode a 3D Gaussian scene, from 2 to 32 unposed photos, on top of a frozen VGGT.
The clip is SuRFLo's 3D Gaussians of our own handheld capture of a lounge: it was given the 16 frames under it, picked from the video, with no poses and no depth.
The clip flies between two of those input cameras, so most of what it shows lies between the views SuRFLo was given.
It is rendered at 1.5 times the resolution SuRFLo fitted, with gsplat's usual 0.3-pixel dilation, and the two cameras were picked so the path does not fly through a pillar.

What we added:

- **The 3DGS export.** Guided mode fits Gaussians to the photos but never writes them; [`gs_playground.surflo.export`](../../src/gs_playground/surflo/export.py) saves them as a standard 3DGS PLY, which matches SuRFLo's own renders at a per-run median of 59 to 63 dB PSNR, though a few single cameras fall to 12 to 21 dB, not yet explained ([`e08h_export_check.json`](../../results/surflo/e08h_export_check.json)).
- **Cost and novel-view quality against a per-scene 3DGS**, on Mip-NeRF 360 garden, from 16 input views to all 161 training views (table below).
- **A backbone swap**: SuRFLo on IGGT's fine-tuned VGGT, the basis of [semantics on SuRFLo](../iggt/README.md#semantics-on-surflo-e08n).

| Mip-NeRF 360 garden | input views | time | peak GPU memory allocated | novel-view PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| SuRFLo | 16 | 2.0 min | 11.7 GiB | 22.98 dB | 0.705 |
| SuRFLo | 64 | 2.3 min | 21.9 GiB | 24.50 dB | 0.775 |
| SuRFLo, post hoc | 161 | 3.2 min | 45.3 GiB | 24.92 dB | 0.777 |
| 3DGS (gsplat, 30k steps, COLMAP poses) | 161 | 35.7 min | 10.8 GiB | 31.84 dB | 0.954 |

- An exploratory sweep: one scene, one RTX 6000 Ada, SuRFLo's default guided preset ([`e08j_views_sweep.json`](../../results/surflo/e08j_views_sweep.json), [`e08i_perscene_garden.json`](../../results/surflo/e08i_perscene_garden.json)).
- SuRFLo rows are medians over 3 seeds; the 3DGS row is one run.
- The 161-view row is a post-hoc arm run with nothing else on the 48 GB card; the run as planned, with another user's 7.1 GiB on the card, ran out of memory ([`e08j_n161_posthoc.json`](../../results/surflo/e08j_n161_posthoc.json)).
- Time is SuRFLo's wall clock including model loading, and 3DGS's training time.
- Memory is SuRFLo's ODE stage only, which understates its whole run at large view counts, and all of 3DGS's training.
- PSNR and SSIM are the median over 24 held-out views at SuRFLo's 518x322, after test-time refinement of the test cameras; the 3DGS is rendered at its 1297x840 training resolution and downscaled like the ground truth (at full resolution it scores 27.65 dB).

SuRFLo is 11 to 18 times faster than training a 3DGS here and needs no poses, but its novel views sit 7 to 9 dB below the per-scene 3DGS, and going from 16 to 161 input views closes about 2 dB of that gap.
Two things this does not separate: the default preset keeps SuRFLo at about 0.19M Gaussians whatever the view count (a larger preset was not tried), and SuRFLo leaves out the far background, which a full-image PSNR counts against it.
On our lounge (exploratory, one capture, default preset), guided mode's geometry agrees with a reference built from our own dense 3DGS less than unguided mode's does (F1 0.58 against 0.87, [`e08h_stage_b.json`](../../results/surflo/e08h_stage_b.json)); the cause is untested.

## What it needs

A folder of 2-32 photos of one scene (the authors say up to 80).
No poses, no depth, no masks: VGGT-1B predicts the cameras, and guided mode refines them.
Its output is in VGGT's frame (camera 0 at the origin, no metric scale), so comparing with anything else needs an alignment.

## Running it

```bash
# once: the env (CUDA extensions built and verified), then the weights
bash tools/setup/surflo_env.sh
hf download AntoineGuedon/Surflo-v0 surflo_v0.pt --local-dir data/models/surflo
# E08h: the authors' garden sample (A), then our sofa (B, only if A passed)
bash experiments/surflo/run_E08h_surflo.sh A
bash experiments/surflo/run_E08h_surflo.sh B
# E08i, E08j: Mip-NeRF 360; per-scene 3DGS and 2DGS; SuRFLo at 16-161 views
bash tools/fetch/fetch_mipnerf360.sh
bash experiments/surflo/run_E08i_perscene_garden.sh
bash experiments/surflo/run_E08j_views_sweep.sh
```

Each run is upstream's own `scripts/infer.py`, with our patch's `save_guided_state=true`.
Outputs per run: `mesh.ply`, `mesh_textured.ply`, `point_cloud_{normals,rgb}.ply` (upstream's), and `guided_state.pt`.
`gs_playground.surflo.export` turns the latter into `point_cloud.ply` (a standard 3DGS, SH degree 3) plus `cameras.json`.

## Traps

- It needs its own env: torch 2.4.1 + cu124, not the puffin env's 2.7 + cu126.
  Never source `activate_puffin_world.sh` for it.
- Depth-Anything-3, which every guided preset but `no_expert` uses, imports `addict`, and nothing upstream installs it.
- Guided mode is not bitwise reproducible at a fixed seed; plain mode is.
- Its renderer draws Gaussians without the 0.3 px low-pass most viewers add; render the export with gsplat `eps2d=0` to match it exactly.
- Its cameras carry one focal length per input view, as VGGT predicted it (about 2% spread); guided mode keeps these intrinsics fixed and refines only rotation and translation (`surflo/inference/guided.py:833-847`).
  A novel-view evaluation that fixes the median focal and refines only the pose loses up to 1.2 dB on SuRFLo's own input views; refine a focal scale too (E08j).
- Memory grows with the view count (22 GiB reserved in the ODE stage at 64 views), and the Depth-Anything-3 guidance before the ODE peaks higher still: 161 views need 45.6 GiB, so they fit a 48 GB card only when nothing else is on it (E08j, E08k).
- To compare against a 3DGS trained at full resolution, do not render it at SuRFLo's 518 width: zoom-out dilation costs it about 8.6 dB. Render at its training resolution and downscale like the ground truth (E08j, Amendment 1).
