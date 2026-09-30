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
```

Each run is upstream's own `scripts/infer.py`, with our patch's `save_guided_state=true`.
Outputs per run: `mesh.ply`, `mesh_textured.ply`, `point_cloud_{normals,rgb}.ply` (upstream's), and `guided_state.pt`.
`gs_playground.surflo.export` turns the latter into `point_cloud.ply` (a standard 3DGS, SH degree 3) plus `cameras.json`.

## Traps

- It needs its own env: torch 2.4.1 + cu124, not the puffin env's 2.7 + cu126. Never source `activate_puffin_world.sh` for it.
- Depth-Anything-3, which every guided preset but `no_expert` uses, imports `addict`, and nothing upstream installs it.
- Guided mode is not bitwise reproducible at a fixed seed; plain mode is.
- Its renderer draws Gaussians without the 0.3 px low-pass most viewers add; render the export with gsplat `eps2d=0` to match it exactly.
