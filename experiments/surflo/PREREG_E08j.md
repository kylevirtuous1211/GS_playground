# E08j pre-registration: how SuRFLo's cost and novel-view quality change with the number of input views (garden)

Written 2026-10-01, before any run below; the commit that adds this file precedes them, and the runner refuses to start without it.
**Exploratory sweep**: one scene, one preset; it describes curves, it does not publish a number for the method.

## Question

On Mip-NeRF 360 garden, as SuRFLo is given more views (up to the full standard training split), how do its time, its GPU memory and the novel-view PSNR of the 3DGS it fits change, and where does that PSNR sit against a 3DGS trained on the same training split (E08i)?

## Inputs and arms

- Input pool: garden's standard training split, the 161 images that are not every 8th one (sorted by name), at `images_4` (1297x840), staged by symlinks in `data/mipnerf360/garden_train161`. The 24 standard test views are never an input.
- SuRFLo picks its N inputs with its own uniform sampling over that pool (`source.n_images=N`).
- Arms: N in {16, 32, 64, 161}, all `mode=guided guided=default`, 100k query points; seeds {42, 0, 1}, n = 3 per N. The preset is held fixed to isolate N (the README suggests `long` for more views; not swept here).
- **Cap, stated before the run**: if the first N = 161 run fails for memory or takes over 20 minutes of wall clock, its other seeds are skipped and reported as skipped; a failure is a result, not a hole.

## Metrics, per run

- **Cost**: `ode_inference_s` and `peak_vram_ode_gib`/`peak_alloc_ode_gib` from SuRFLo's summary (our patch, as in E08h), plus the whole command's wall clock.
- **Novel-view PSNR (primary)** and SSIM on the 24 test views, of our export of the Gaussians SuRFLo fits (rendered with gsplat, SH 3, `eps2d=0`, which E08h showed matches SuRFLo's renderer at 59-63 dB), against the test images processed exactly as SuRFLo processes its inputs (`load_and_preprocess_images`, `no_stretch`, 518 wide: 518x322 for garden).
- **Test cameras for SuRFLo**, which has none: the test views' cameras from gsplat's own Parser (the frame E08i's 3DGS lives in), mapped into SuRFLo's frame by a Sim(3) Umeyama fit from SuRFLo's refined input-camera centres to the Parser centres of the same frames; intrinsics are the median of SuRFLo's input focals, principal point at the centre.
- **Test-time pose refinement**, standard for pose-free methods: per test view, 200 Adam steps on a 6-DoF pose correction minimising L1 to the test image, Gaussians frozen. Reported with and without it.

## What it is set beside

- **3DGS from E08i** (161 training views, COLMAP poses, 30k steps), rendered at the same 24 test views at the same 518x322 resolution and crop, from gsplat's Parser cameras with intrinsics scaled and cropped like the images; reported both without and with the same test-time pose refinement, so the comparison is symmetric.
- 2DGS is not evaluated for novel views: it is a surface method and needs a different rasteriser; its E08i PSNR stands as it is.

## Sanity checks, before reading any test PSNR

1. **Preprocessing**: our crop and scale for the test images must reproduce `load_and_preprocess_images`' output exactly (max absolute difference 0).
2. **The camera path**: for one run per N, SuRFLo's own input views evaluated through the whole test path (Parser camera, Umeyama mapping, median intrinsics, pose refinement) must score within 1 dB of the same views rendered with SuRFLo's own refined cameras. If not, the test path is wrong and no test PSNR is read.

## Reading

Descriptive: per N, the median and range over the three seeds of time, peak VRAM and test PSNR (with and without pose refinement), set beside 3DGS's test PSNR.
Named differences: SuRFLo is evaluated at 518x322 because that is the only resolution it fits; 3DGS is re-rendered at that resolution for the comparison, though it was trained at 1297x840. The per-image exposure SuRFLo learns has no value for a test view, so test renders use none. SuRFLo drops far background by design, which full-image PSNR counts against it.
