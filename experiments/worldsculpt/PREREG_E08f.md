# E08f pre-registration: parameters vs how many objects come out, and how complete they are

Written 2026-09-30, after E08e's default run was launched and before any E08f run or any E08e mesh was looked at.
**Exploratory.** It learns relationships; it does not publish a number.
One scene, one capture, about ten objects in Stage B: a large effect would show, a small one would not.

## Question

On our own capture (NCHC sofa, E08e), how does each parameter change
(a) how many objects the grounding stage hands to WorldSculpt, and
(b) how complete each object's mesh is?

One parameter moves at a time (OFAT); everything else stays at E08e's default.

## Stage A: grounding parameters -> count and box completeness (all label ids, CPU)

Recomputed from E08e's cached depth with `gs_playground.worldsculpt.ground.Params` overridden one field at a time.
Only the automatic filter is applied; E08e's manual review is not re-applied, because it was done once by eye.

| parameter | default | values |
|---|---|---|
| `support_frac` | 0.05 | 0.02, 0.05, 0.10, 0.20 |
| `min_views` | 8 | 4, 8, 12, 16 |
| `speck_frac` | 0.02 | 0, 0.02, 0.10, and `largest_only` |
| `stride` | 3 | 1, 3, 6, 12 |
| `clip_gate` | on | on, off |
| `struct_extent` | 0.4 | 0.3, 0.4, 0.5 |

Metrics per setting:
- **count**: ids kept, and ids rejected per reason code.
- **overlap with the default**: Jaccard of the kept id set against the default's, so "more objects" can be told apart from "different objects".
- **box completeness**: for each kept id, the share of all its back-projected points (every frame, before the support filter) that fall inside its box; reported as the median over kept ids, and separately over the ids kept by both this setting and the default.

Stage A is descriptive: no reading rule, the curves are the result.
Whatever it shows, E08e's selection is not revised from it.

## Stage B: WorldSculpt parameters -> mesh completeness (fixed subset, GPU)

**Object subset**, fixed by rule before any mesh was seen: E08e's kept objects split into three tertiles by largest box side, the 3, 3 and 4 with the most views from each.
That gives ids **6, 9, 18, 20, 21, 22, 56, 67, 74, 222** (pillows 21/56/74, stools 6/18/20, TV 22, console item 9, ottoman row 67, and 222).

Each config runs upstream's own `prepare_crops_scene.py` and `reconstruct_batch.py` with `inference.sh`'s flags, changing only the swept one; no composition.

| config | change from default |
|---|---|
| `default_s42`, `default_s0`, `default_s1` | seed 42 (E08e's), 0, 1: **the null** |
| `views12`, `views6`, `views3`, `views1` | `--max_views` |
| `fps` | `--view_select fps` instead of `area` |
| `erode4`, `dilate4` | final masks eroded / dilated by 4 px at full resolution |
| `nofit` | no `--mask_fit_scale` |

Metrics per object, with tau = 2% of the object's box diagonal:
- **completeness (primary)**: share of the object's observed surface points (centres of the supported voxels in its best component, at Stage A defaults) within tau of the mesh surface.
  It asks whether the parts a camera saw were built.
- **held-out silhouette**: on frames WorldSculpt never received (outside the stride-3 set) where the object passes E08e's frame gate, the mesh projected to a silhouette against the HQ-SAM mask: recall and IoU.
  The mask is predicted, so this is agreement with a reference, not accuracy. Null: the projected box hull's IoU with the same mask.
- **fragments**: connected components of the mesh.
- **count**: objects that produced a mesh at all.
- secondary, **accuracy-like**: share of mesh surface samples within tau of observed points. Generated back sides are legitimately far from anything observed, so this is expected to be low and is not read as an error rate.

**Reading rule.** For each object, the seed range r is max minus min of the metric over the three default seeds.
A config changes completeness only if the median over objects of (config minus mean of the three seeds) exceeds the median over objects of r in magnitude.
Otherwise it is reported as within seed noise.

**Known circularity.** The boxes, and so the canonical cube each mesh is placed in, come from the same rendered depth as the observed surface points.
Completeness therefore measures consistency with our own grounding, not absolute accuracy.

## Sanity checks before reading anything

- `default_s42` must reproduce E08e's meshes for the subset (same vertex counts), or the sweep runner has drifted from `inference.sh`.
- Stage A at default parameters must reproduce E08e's automatic kept set exactly.

## Amendment, 2026-09-30, before any Stage B metric was computed

The sanity check "`default_s42` must reproduce E08e's meshes (same vertex counts)" cannot hold, and not because of the sweep runner.
Upstream's shape stage is nondeterministic at a fixed seed: `reconstruct_batch.py` run twice back to back on obj09 with `inference.sh`'s exact flags, seed 42, gave F = 120,370 and then F = 31,136 faces, from the same 2,364 occupied Stage 1 voxels.
`default_s42` and E08e differ by the same kind of factor (3.5-4x fewer faces) on all ten objects.

Replaced with what the runner can be held to: for every subset object, `default_s42` must match E08e exactly on the Stage 1 occupied-voxel count (the seeded, deterministic stage), the selected views and anchor, and `T_canon_to_metric`.
Checked by hand for obj06 before this amendment; `score-b` now checks it for all ten from the two logs.

The seed null is unchanged and now also covers this run-to-run nondeterminism, which is the variation a user of the method actually faces.
Face and vertex counts are not read as a quality measure anywhere.

## Notes written after Stage B results were computed

Neither changes a pre-registered reading; both are recorded here so nothing is silently redefined.

1. **Fragments cannot be measured on the raw output.** `mesh.pt` is an unwelded set of 4,000-17,000 small patches per object; welding vertices at any tolerance up to the 1/1024 grid spacing leaves the largest patch at 0.1% of the faces (obj09).
   Upstream remeshes only when it writes a GLB, which the sweep does not do. The metric is dropped from the reading rather than redefined.
2. **The completeness tolerance is below the reference's own quantisation.** tau = 2% of the box diagonal is 0.016-0.054 units, while the observed points are voxel centres at E/400 = 0.045 (half-diagonal 0.039), so absolute completeness is capped well below 1 even for a perfect surface and favours large objects.
   The pre-registered metric and its verdicts stand as the result. A **post-hoc** variant with tau floored at half a voxel diagonal (`completeness_posthoc_voxel_tau`) was added only to test whether the verdicts depend on this: they do not (the same three configs change, the same five stay within seed noise).

## Retraction, 2026-09-30, after all of the above

Stage B is retracted (`LOG.md` E08g).
Every mesh it scored was decoded with the shape decoder's 40 sparse convs at random initialisation, because the runner inherited `SPARSE_CONV_BACKEND=spconv`.
The amendment above was that fault, not upstream: with `flex_gemm`, two runs at one seed are bitwise identical, so a rerun can restore the original "same vertex counts" sanity check.
Stage A never touches the model and stands.
