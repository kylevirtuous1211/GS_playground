# E08l pre-registration: IGGT's fine-tuned VGGT backbone inside SuRFLo

Written 2026-10-03, before any run below; the commit that adds this file precedes them, and the runner refuses to start without it.
Step 1 of a plan towards semantics carried by the VGGT encoding SuRFLo already runs; it settles a yes/no question, so it is pre-registered, on one preset and four scenes.

## Question

SuRFLo was trained on the tokens of a frozen VGGT-1B (aggregator layers 4, 11, 17, 23 and the last layer's camera token), and also uses VGGT's camera and depth heads to initialise.
IGGT (Instance-Grounded Geometry Transformer, ICLR 2026, `lifuguan/IGGT_official`) is the same architecture fine-tuned for instance grouping (backbone learning rate 1e-6), with an instance head that reads the same four layers.
If SuRFLo's VGGT-1B is replaced by IGGT's backbone, how far do the tokens and the heads' cameras and depths move, and does SuRFLo's output change measurably?

## Arms

- Backbone: **VGGT-1B as SuRFLo ships it** (the `vggt.*` weights in `surflo_v0.pt`), against **IGGT** (`iggt_checkpoint.pth`, HF revision 12aa6fa, 5198776140 bytes, sha256 2e165f7d18b64de1a6e18ce0022f2ada034faaaae007cd3e9a4a15eccf4644d2; `module.` prefix stripped, its 256 `part_head.*` / `part_adaptor.*` keys dropped, the remaining 1797 loaded strictly into `vggt`, after SuRFLo's own checkpoint load). Only the backbone differs; the SuRFLo checkpoint, code and config are the same.
- Mode: `plain` at seed 42 (bitwise deterministic), and `guided=default` at seeds {42, 0, 1}.
- N = 16 views, `num_query_points=100000`, SuRFLo's own uniform sampling (deterministic, so both arms and all seeds see the same 16 views).
- Scenes:
  - Mip-NeRF 360 garden: E08j's pool of 161 training views (`~/datasets/mipnerf360/derived/garden_train161_images_4`); not in IGGT's training data.
  - ScanNet++ v2 `nvs_sem_val` scenes 825d228aec, 6115eddb86 and 13c3e046d7: DSLR `resized_undistorted_images` of the official training frames minus those flagged `is_bad`.
  Named difference: those ScanNet++ scenes are in IGGT's released training archive (and ScanNet++ was in VGGT-1B's training); for a compatibility test that is acceptable, for a semantic evaluation it would not be.

## Metrics

1. **Token drift** (descriptive), on identical preprocessed inputs (each scene's 16 views, SuRFLo's `no_stretch` 518 preprocessing, SuRFLo's autocast dtype): per layer {4, 11, 17, 23} for the patch tokens, and for the last layer's camera token, the mean cosine similarity and the relative L2 (`|a - b| / |a|`) between VGGT-1B and IGGT; camera rotation difference (degrees), focal difference (%), and the median relative depth difference of the two backbones' heads.
   Scale it is read against, stated now: the same quantities between VGGT-1B run under autocast and in fp32, i.e. the numerical noise SuRFLo already lives with.
2. **Geometry against ground truth** (ScanNet++): Chamfer distance and F1 of SuRFLo's points (guided: `point_cloud_normals.ply` at opacity >= 0.1; plain: `final.ply`, as E08h) against the vertices of `scans/mesh_aligned_0.05.ply`, after a Sim(3) Umeyama fit of SuRFLo's camera centres to the DSLR camera centres in the mesh frame (nerfstudio poses mapped by `MESH_FROM_NERFSTUDIO`).
   F1 at tau = 5 cm (primary, metric) and at tau = 0.01 x the box diagonal (SuRFLo's own definition), with SuRFLo's metric code.
   Ground truth is cropped to the vertices seen by at least one of the 16 input views (inside the image, in front of the camera, and not occluded: within 5 cm of the depth ray-cast from the mesh), and predictions to that region's box.
3. **Agreement between arms**: F1 at 5 cm and Chamfer between the IGGT and VGGT outputs of one scene at one seed (frames aligned through their camera centres).
   Its null, stated now: the same agreement between two VGGT-arm seeds of guided mode (42 against 0, 42 against 1), i.e. what SuRFLo's own run-to-run variation already produces.
4. **Novel views on garden** (guided): E08j's protocol unchanged (24 standard test views, sanity check 2 gating, refinement of pose and focal, raw and refined PSNR and SSIM).
5. **Cost**: ODE time, peak VRAM; expected unchanged by construction, recorded.

## Validity checks, before reading any metric

1. The swap is real and complete: every one of the 1797 tensors in the loaded backbone equals the checkpoint's; IGGT's backbone differs from VGGT-1B (maximum absolute difference > 0; about 1e-4 is expected); and the backbone SuRFLo ships equals `facebook/VGGT-1B` (if not, that difference is reported and VGGT-1B means SuRFLo's copy).
2. Determinism: the VGGT-arm plain run, repeated, is bitwise identical.
3. Pipeline regression: the VGGT arm's garden guided refined test PSNR lands in, or within 0.2 dB of, E08j's N = 16 range (22.91 to 22.98 dB).

## Reading

- Per scene and metric, as in E08f and E08h: the IGGT arm changed a guided metric only if the median over its three seeds differs from the VGGT arm's median by more than the VGGT arm's seed range.
- Plain-mode differences are deterministic and reported with the guided seed range beside them as a scale; the verdict rests on the guided metrics.
- Agreement (metric 3): the swap is within SuRFLo's own variation if the cross-backbone agreement is no lower than the lowest cross-seed agreement of the VGGT arm.
- **Compatible**, the answer to the question: no guided metric changed on any of the four scenes, and the cross-backbone agreement is within SuRFLo's own variation on every scene.
- Token drift is descriptive; it is read against the autocast-vs-fp32 scale, without a threshold.
