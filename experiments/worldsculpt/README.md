# worldsculpt

WorldSculpt (arXiv 2609.05416), released LoRA over the Pixal3D base, run here on the authors' Marble data.
Reproduction arm, log entry E08c; demo row in `results/DEMOS.md`.

```bash
bash tools/setup/worldsculpt_env.sh            # envs/worldsculpt-overlay, on the puffin-world base
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
