# SuRFLo

Guédon et al., "Surflo: Consistent 3D Surface Flow Model with Global State", NeurIPS 2026, arXiv 2606.13644.
Code `Anttwo/Surflo`, pinned in `third_party/PINS.tsv`; weights `AntoineGuedon/Surflo-v0` (CC BY-NC 4.0, research only).

## What it needs

A folder of 2-32 photos of one scene (the authors say up to 80).
No poses, no depth, no masks: VGGT-1B predicts the cameras, and guided mode refines them.
Its output is in VGGT's frame (camera 0 at the origin, no metric scale), so comparing with anything else needs an alignment.

## Running it

```bash
bash tools/setup/surflo_env.sh                        # once: env, CUDA extensions, verify
hf download AntoineGuedon/Surflo-v0 surflo_v0.pt --local-dir data/models/surflo
bash experiments/surflo/run_E08h_surflo.sh A          # the authors' garden sample
bash experiments/surflo/run_E08h_surflo.sh B          # NCHC sofa; runs only if A passed
bash tools/fetch/fetch_mipnerf360.sh                  # once: official Mip-NeRF 360 into ~/datasets
bash experiments/surflo/run_E08i_perscene_garden.sh   # 3DGS and 2DGS cost on garden, beside it
bash experiments/surflo/run_E08j_views_sweep.sh       # 16-161 views: cost and novel-view PSNR
```

Each run is upstream's own `scripts/infer.py`, with our patch's `save_guided_state=true`.
Outputs per run: `mesh.ply`, `mesh_textured.ply`, `point_cloud_{normals,rgb}.ply` (upstream's), and `guided_state.pt`.
`gs_playground.surflo.export` turns the latter into `point_cloud.ply` (a standard 3DGS, SH degree 3) plus `cameras.json`.

## Traps

- It needs its own env: torch 2.4.1 + cu124, not the puffin env's 2.7 + cu126. Never source `activate_puffin_world.sh` for it.
- Depth-Anything-3, which every guided preset but `no_expert` uses, imports `addict`, and nothing upstream installs it.
- Guided mode is not bitwise reproducible at a fixed seed; plain mode is.
- Its renderer draws Gaussians without the 0.3 px low-pass most viewers add; render the export with gsplat `eps2d=0` to match it exactly.
- Guided mode refines a focal length per input view (about 2% spread). A novel-view evaluation that fixes the median focal and refines only the pose loses up to 1.2 dB on SuRFLo's own input views; refine a focal scale too (E08j).
- Memory grows with the view count (22 GiB reserved in the ODE stage at 64 views), and the Depth-Anything-3 guidance before the ODE peaks higher still: 161 views need 45.6 GiB, so they fit a 48 GB card only when nothing else is on it (E08j, E08k).
- To compare against a 3DGS trained at full resolution, do not render it at SuRFLo's 518 width: zoom-out dilation costs it about 8.6 dB. Render at its training resolution and downscale like the ground truth (E08j, Amendment 1).
