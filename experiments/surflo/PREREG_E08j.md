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

## Amendment 1, 2026-10-01, before any SuRFLo run of this sweep

Made after an adversarial review of the evaluation code (25 agents, every finding put to a refuter) and before any SuRFLo inference for E08j; no SuRFLo test number existed when it was written.

1. **The 3DGS baseline is rendered at its training resolution, then put through the ground truth's own operation.**
   The review measured the pre-registered method on E08i's 3DGS: rendering it natively at 518x322 (intrinsics scaled, `eps2d` 0.3) scores a median 22.89 dB raw on the test views, against 31.45 dB when the same model is rendered at 1297x840 and then downscaled exactly as the ground truth is (PIL bicubic to 518x335, crop to 518x322); per-view gap median 8.6 dB, range 2.8 to 10.5.
   The cause is the known zoom-out dilation of 3DGS (Mip-Splatting): renders come out brighter and thickened (mean intensity 0.415 against 0.379 for the ground truth), and pose refinement cannot undo it.
   The pre-registered method would set SuRFLo beside a baseline handicapped by a renderer artifact, not by its reconstruction.
   Disclosure: these baseline numbers were seen before this amendment, and the change raises the baseline, which works against SuRFLo in the comparison.
   Amended: render at the Parser's resolution with its own intrinsics, quantise to 8 bits, apply the same resize and crop as the ground truth, score; pose refinement for the baseline runs at that training resolution against the full-resolution test image.
   The pre-registered native-518 variant is still computed and reported, labelled as such.
   SuRFLo is unchanged: it was fit at exactly the evaluation resolution, so its native render is already the like-for-like one.
   The principle for both: each model is rendered at the resolution it was fit at, then put through the ground truth's operation.
2. **SSIM is reported with and without pose refinement**, as section "Metrics" implies; the code computed it only with.
3. **Sanity check 2 gates**: it runs on every seed-42 run before any test view is evaluated, and if any gap exceeds 1 dB the evaluation writes only the sanity block and stops with an error.
4. **The cap, as written above, applies only to seed 42 at N = 161**, and the failure branch only to running out of GPU memory (`CUDA out of memory` / `OutOfMemoryError` in its log); any other failure, or a failure of seed 0 or 1, stops the sweep as at every other N.
   The GPU is shared, so the compute processes on it are recorded before each run.
   Skipped and failed runs are written into the results JSON with their reason, and the per-N median and range over seeds are computed by the evaluation, not by hand.
5. **Data**, not a protocol change: the share copy of garden was deleted on 2026-10-02 (Taiwan time), so the inputs, ground truth and Parser cameras come from the official release (`~/datasets/mipnerf360`, zip MD5 verified).
   Its `images`, `images_4` and `sparse/0/*.bin` are byte-identical to the share copy by that copy's own SHA-256 receipt, and the `images_4_png` gsplat derives from it is byte-identical to the one E08i's 3DGS trained on.
   The 161-image input pool is a directory of hardlinks, `~/datasets/mipnerf360/derived/garden_train161_images_4`, not symlinks under `data/` (the storage policy forbids per-file symlinks); same 161 files.

## Amendment 2, 2026-10-01, after sanity check 2 failed and before any test PSNR was computed

Sanity check 2 failed at N = 16 (seed 42): SuRFLo's own cameras scored 28.76 dB on its input views, the test path 27.53 dB, a gap of 1.23 dB; N = 32 and 64 passed (0.45 and 0.23 dB).
As pre-registered, the evaluation stopped there and no test view was rendered (`results/surflo/e08j_views_sweep.json` holds only the sanity block and the sweep status).

A probe on the input views only (`probes/surflo/e08j_sanity2_gap.py`, `results/surflo/e08j_sanity2_gap_probe.json`) located the whole gap in the intrinsics:

| N = 16, input views, median PSNR | dB |
|---|---|
| own cameras, raw | 28.76 |
| test path, median K, 200 steps of pose refinement | 27.53 |
| test path, **each view's own K**, 200 steps | 28.77 |
| test path, median K, 1000 steps | 27.52 |

The Sim(3) alignment is not the cause (centre residual 0.5% of the camera extent, rotation offset median 0.23 deg), nor is the refinement budget.
SuRFLo refines a focal length per input view (spread about 2.3% around the median at every N), which a pose-only refinement cannot absorb; per view the loss ranges from 0 to 4.6 dB.

Amended: the test-time refinement also optimises one focal scale per view (fx and fy multiplied by the same factor, principal point fixed), with the same 200 Adam steps and learning rate on its logarithm.
gsplat gives no gradient for the intrinsics, so that one scalar's gradient is a central finite difference of the same L1 loss (step 1e-3 in log scale); the final render uses the refined focal exactly.
It applies to both arms, SuRFLo and the 3DGS baseline, so the comparison stays symmetric; for 3DGS, whose COLMAP intrinsics are shared and known, it is one extra degree of freedom it does not need.
Sanity check 2 is rerun with the amended refinement and the same 1 dB threshold, and still gates every test number.
The raw (unrefined) numbers are unchanged: median K for SuRFLo, the Parser's K for 3DGS.

Also recorded here, not a change: N = 161, seed 42 ran out of GPU memory after 92 s, in SuRFLo's monocular-depth normal guidance, needing 2.86 GiB more with 38.4 GiB in use, while two other processes held 7.1 GiB of the 48 GB card (`gpu_before.csv`); the cap skipped seeds 0 and 1 as pre-registered.
Whether N = 161 fits on an otherwise empty 48 GB card is therefore not tested by this sweep.
