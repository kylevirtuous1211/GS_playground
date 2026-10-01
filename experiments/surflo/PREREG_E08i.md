# E08i pre-registration: what per-scene optimisation costs on Mip-NeRF 360 garden, on this GPU

Written 2026-10-01, before any run below; the commit that adds this file precedes them, and the runner refuses to start without it.
**A cost measurement, not an accuracy comparison.** It answers how long and how much GPU memory the standard per-scene methods take on garden on our RTX 6000 Ada, to set beside SuRFLo's cost on the same card (E08h).

## Arms

Both with gsplat v1.5.3 (`third_party/clones/gsplat`, pinned), in the puffin env, on garden from the share (`MIPNERF360_GARDEN` in `.env.local`), staged through symlinks into `data/mipnerf360/garden` so nothing is written to the share.
Standard Mip-NeRF 360 protocol: `--data-factor 4` (1297x840), every 8th image held out (161 train, 24 test), 30,000 steps, gsplat's default seed.

| arm | trainer | flags beyond the protocol |
|---|---|---|
| `3dgs` | `simple_trainer.py default` | the same as our DL3DV corpus (`--save-ply --disable-viewer`) |
| `2dgs` | `simple_trainer_2dgs.py` | `--normal-loss --dist-loss`: 2DGS as its paper defines it; gsplat ships both off |

n = 1 per arm, stated as such: these are wall-clock costs whose run-to-run spread on one machine is a few percent, against an expected gap to SuRFLo of an order of magnitude or more.
Nothing else runs on the GPU during the measurement; another user's idle 7.3 GB stays resident and is excluded by measuring per process.

## Metrics

- **Training time**: gsplat's own `ellipse_time` at step 30,000 (from the first training step; excludes data loading and the one-off resize of the images).
- **Wall clock**: the whole trainer command, including that resize and the evaluation at step 30,000.
- **Peak VRAM**: gsplat's `mem` (`torch.cuda.max_memory_allocated`), and the trainer process's peak in `nvidia-smi --query-compute-apps` sampled every second (what a user sees; includes the allocator cache).
- **Validity check, before any cost is quoted**: held-out PSNR at step 30,000 on the 24 test views must be at least 25 dB. A run below that is reported as failed and its cost is not quoted as the method's cost.

## What it is set beside

- **SuRFLo on the same card (E08h, `results/surflo/e08h_stage_a.json`)**: guided, 86.0 s ODE (about 2 min 7 s end to end), 12.8 GiB reserved; plain, 17.1 s ODE (about 45 s end to end), 10.6 GiB.
- **Published measurements on other GPUs**, verified against their sources and reported beside, not compared as nulls: gsplat team, 3DGS on garden, 3013 s and 9.88 GB on a TITAN RTX; 2DGS without regularisation, 1298 s and 4.79 GB on an RTX 4090.

## Named differences, so the comparison is read correctly

- The per-scene arms use the standard protocol (161 views with COLMAP poses at 1297x840); SuRFLo used 16 views at 518x336 with no poses. This compares each method as it is normally run, not equal inputs.
- SuRFLo's paper ran its per-scene baselines on the same 16 views with VGGT poses; that matched-input setting is not run here.
- 2DGS here includes its two regularisers, the gsplat team's garden number does not.
