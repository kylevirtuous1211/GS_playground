# E08h pre-registration: SuRFLo, reproduced on its own garden sample, then run on the NCHC sofa

Written 2026-09-30, after the env was built and before any `infer.py` call; the commit that adds this file precedes every run below.
Nothing in this repository has run SuRFLo before (searched `LOG.md`, `HANDOFF.md`, all branches).

## Source

Guédon, Nakamura, Dufour, Lei, Nishino, Kanazawa, "Surflo: Consistent 3D Surface Flow Model with Global State", NeurIPS 2026, arXiv 2606.13644.
Code `Anttwo/Surflo` at `bf14c6375a92911c45795710cd00bf2af17e9a13`, with one patch of ours (`third_party/patches/surflo_save-guided-state.patch`) that only adds outputs and VRAM bookkeeping.
Checkpoint `AntoineGuedon/Surflo-v0/surflo_v0.pt`; its sha256 is in `data/models/surflo/surflo_v0.pt.sha256` and is quoted in the LOG entry.

## Stage A: reproduction attempt on the authors' own setting

Input: the 16 garden views the repository ships in `media/sample/` (518x336, already downscaled by the authors).
Command: the README's first guided command, `mode=guided guided=default source.n_images=16 num_query_points=100000`, and the same with `mode=plain`.

Borrowed claims, from the README's runtime table (one H100, 100k points, DA3 priors and meshing excluded):
- `plain`: about 8 s and 8.5 GiB peak VRAM.
- `guided=default`: about 45 s and 14 GiB peak VRAM.
- The guided command writes `mesh.ply`, `point_cloud_normals.ply`, `point_cloud_rgb.ply` and a textured mesh.

**Not attempted: the paper's accuracy numbers.**
No Mip-NeRF 360 ground truth and no per-scene number is released (the released evaluation data is Tanks and Temples only), so no Chamfer or F1 on garden can be compared with theirs.
The LOG entry will say "not attempted", never "reproduced".

Metrics:
- ODE time: `ode_inference_s` from their own `_infer_summary.json`, the same span their table times.
- Peak VRAM: `torch.cuda.max_memory_reserved` over the ODE span (`peak_vram_ode_gib`, added by our patch), after releasing the allocator cache the DA3 expert leaves behind; `max_memory_allocated` over the same span is recorded beside it.
  Their measuring method is not stated; this is a named difference.
- n = 3 runs per arm at seed 42; the median and the range are reported.
- Determinism: whether the three runs give identical outputs is recorded, not assumed.

Differences from their setting: an RTX 6000 Ada rather than an H100; our build of their cu124 environment; the VRAM measure above.

**Stage A passes only if all hold:**
1. The outputs listed above exist and are non-empty for every guided run (`final.ply` for plain).
2. Median ODE time is within 5x of their figure for that arm. An Ada is expected to be about 1.5-3x slower than an H100; beyond 5x something is wrong with the build.
3. Median peak VRAM is within ±30% of their figure for that arm.
4. By eye, the guided mesh shows three things named here before looking: the garden table, the pot on it, and the ground plane.

Stage B does not start unless Stage A passes.

## Export check (engineering verification, not a research claim)

Our exporter turns the Gaussians guided mode fits into a standard 3DGS `.ply` (SH degree 3) plus `cameras.json`.
Metric: PSNR between a gsplat render of the exported `.ply` (`eps2d=0`, SH degree 3, black background) and SuRFLo's own `render_surflo` render of the same Gaussians (`kernel_size=0`, black background), at every refined input camera.
**Pass: median >= 40 dB on every guided run.**
Null: the same comparison with the exported higher-order SH zeroed (DC only). It must come out at least 5 dB lower than the full export, or the check cannot detect a wrong SH layout and a pass means nothing.

## Stage B: NCHC sofa, exploratory lane

Labelled exploratory: it learns how SuRFLo behaves on our capture; it does not publish a number.
Question: how close is SuRFLo's geometry to our dense reconstruction of the same room, and what do guidance and more views change?

- Input: `nchc_sofa_20260727_143647`, 364 frames at 1272x715 on the read-only share, sampled uniformly by SuRFLo's own `source.n_images`.
- Reference (pseudo-ground truth, not ground truth): points back-projected from the expected depth of our 30k-iteration 3DGS over all 364 frames (the depth E08e cached at 636x358), every second pixel where alpha > 0.95. It is trained on 364 views against SuRFLo's 16 or 32.
- Reference box: the 0.5-99.5 percentile box of those points, because the 3DGS depth has floaters beyond the room that would otherwise set the scale; reference points outside it are dropped.
- Alignment: a Sim(3) Umeyama fit (their `umeyama_alignment`) from SuRFLo's input-camera centres (refined ones for guided, VGGT's for plain) to the COLMAP centres of the same frames. No ICP, which is stricter than the authors' protocol (a named difference).
- Metrics, the authors' code imported (`surflo/metrics/eval_alignment.py`: `voxel_downsample`, `chamfer_and_fscore`), not their numbers: prediction cropped to the reference box, both sides voxel-downsampled at 0.001 × the box diagonal, Chamfer (mean of the two directions' mean distances) divided by the diagonal, and precision, recall and F1 at τ = 0.01 × the diagonal.
- Scored on point clouds, as their `evaluate.py` does: guided `point_cloud_normals.ply` (opacity >= 0.1, their evaluation cull) and plain `final.ply`. The mesh is not scored.
- Arms: `guided=default` with 16 views (reference arm); `plain` with 16 views (a matched control differing only in guidance); `guided=long` with 32 views (their README recommends `long` for more views).
- Seeds {42, 0, 1}, n = 3 per arm.
- Reading rule, in the spirit of E08f: an arm differs from the reference arm on a metric only if the median over the three seeds of its paired change (arm minus reference arm, same seed) exceeds the reference arm's seed range (max minus min over its three seeds) in magnitude; otherwise it is reported as within seed noise.
- Known limit: the reference includes walls and floor that SuRFLo's cull radius may drop, so recall is bounded below 1 for reasons other than geometric quality.
