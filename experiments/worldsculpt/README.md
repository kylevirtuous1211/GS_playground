# WorldSculpt

WorldSculpt, arXiv 2609.05416: a LoRA over the Pixal3D base that completes every object in a scene as a whole mesh, from posed frames, a mask per object in each frame, and a 3D box per object.
Code [`AlayaLab/WorldSculpt`](https://github.com/AlayaLab/WorldSculpt), pinned in `third_party/PINS.tsv`; weights HF `AlayaLab/WorldSculpt`.
Run first on the authors' Marble scene (E08c, the reproduction arm), then on our own capture (E08e, E08g).

## Results: our lounge, taken apart (E08e, E08g)

<img src="../../docs/media/worldsculpt_sofa_turntable.webp" width="100%" alt="WorldSculpt's 24 object meshes of our lounge, turning">

<img src="../../docs/media/worldsculpt_sofa_view.jpg" width="100%" alt="One input frame, the meshes' normals, one colour per object, and the objects over the frame">

Our lounge capture already had COLMAP poses, a trained 3DGS and label maps associated across views from an earlier project.
From those we built the per-frame object masks and one box per object ([`gs_playground.worldsculpt.ground`](../../src/gs_playground/worldsculpt/ground.py)), with the objects to keep picked by an automatic filter and then on a review page.
All 24 objects given to it come out as meshes in one scene ([`e08e_checks.json`](../../results/worldsculpt/e08e_checks.json)), in about 13 minutes on one RTX 6000 Ada from prepared crops.
The turntable shows one flat colour per object, meshes decimated for display, framed on the sofa group, so three far objects leave the shot for most of the turn.
The still is WorldSculpt's own render of one input view: the frame, the meshes' normals, one colour per object, and the objects over the frame.

**A silent failure, found and fixed:** our base environment's sparse-convolution backend left the shape decoder's convolutions at random weights without a warning, and every mesh came out as plausible-looking dust; see [Traps](#traps).

**Limits.**
Walls, floor and ceiling were not given to it, so the room's shell is absent by construction.
The potted plant is missing: the label maps gave it the walls' id, so it was never handed over as an object.
The hanging planter came out as a large flat slab, and the TV as a deep solid block instead of a thin panel.

## Running it

```bash
bash tools/setup/worldsculpt_env.sh                                    # an overlay on the base env
hf download AlayaLab/WorldSculpt --local-dir data/models/worldsculpt   # its weights
bash experiments/worldsculpt/run_E08c_worldsculpt.sh
```

## What it consumes

A scene directory with `transforms.json` and the frames and masks it names.
Upstream reads, per `prepare_crops_scene.py`: one global intrinsics (`fl_x`, `fl_y`, `cx`, `cy`, `w`, `h`), `frames[]` with `file_path` and an OpenGL camera-to-world `transform_matrix`, and `instances[]` with `pass_index` and `aabb_world`.
Masks are found by convention at `masks/objNN/NNNN.png` (NN the pass index, NNNN the frame's position in `frames`), grayscale, thresholded at >127; depth is never read.
An object with no usable mask frame is dropped without an error.

## Our own captures (E08e)

`gs_playground.worldsculpt.ground` builds that directory for a scene with COLMAP poses, a trained 3DGS and label maps associated across views:

```bash
python -m gs_playground.worldsculpt.ground propose ...   # boxes, filter, review sheet
# review outputs/worldsculpt/NCHC/ground/index.html, commit the selection JSON
bash experiments/worldsculpt/run_E08e_nchc_sofa.sh        # build, run, count every object through each stage
```

The reviewed selection is `e08e_objects.json`; the runner refuses to reuse an input built from a different selection, because upstream resumes from whatever meshes are on disk.

## Traps

**The sparse-conv backend must be `flex_gemm`** (log entry E08g).
Our base activation exported `SPARSE_CONV_BACKEND=spconv` for older TRELLIS work, and the first WorldSculpt runs inherited it; upstream's default is `flex_gemm`.
Under spconv the sparse convs in the vendored `pixal3d` name their parameters `conv.weight` and `conv.bias`, the checkpoint stores `weight` and `bias`, and `pixal3d` loads its decoders with `strict=False`.
So the shape decoder loads with 80 of its 292 parameter tensors (the weights and biases of all 40 sparse convs) left at random initialisation, without a warning.
The output is plausible-looking dust: on the authors' scene, a table decoded into 165,250 faces whose largest connected piece held under 0.1% of them (about 15,000 closed blobs), against 4,444,904 faces and 96% under flex_gemm, from the same crops and seed.
The sparse-structure stage is unaffected, so boxes and placement look right while every mesh is wrong.
Both runners now export `SPARSE_CONV_BACKEND=flex_gemm` after the base activation, and the preflight fails if it is missing.
With it, two runs at one seed give bitwise-identical vertices for all 13 objects of the authors' scene.

**Upstream resumes from whatever meshes are on disk**, so a changed selection or backend needs a fresh output directory; the E08e runner refuses to reuse an input built from a different selection.
