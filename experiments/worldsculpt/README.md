# worldsculpt

WorldSculpt (arXiv 2609.05416), released LoRA over the Pixal3D base, run here on the authors' Marble data.
Reproduction arm, log entry E08c; demo row in `results/DEMOS.md`.

```bash
bash tools/setup/worldsculpt_env.sh            # envs/worldsculpt-overlay, on the puffin-world base
hf download AlayaLab/WorldSculpt --local-dir data/models/worldsculpt   # its weights; the runner links them in as the clone's pretrained/
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
