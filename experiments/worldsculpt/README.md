# worldsculpt

WorldSculpt (arXiv 2609.05416), released LoRA over the Pixal3D base, run here on the authors' Marble data.
Reproduction arm, log entry E08c; demo row in `results/DEMOS.md`.

```bash
bash tools/setup/worldsculpt_env.sh            # envs/worldsculpt-overlay, on the puffin-world base
bash experiments/worldsculpt/run_E08c_worldsculpt.sh
```

## What it consumes

**WorldSculpt** needs a scene directory holding `transforms.json` with camera
intrinsics, `frames` (image, pose, `mask_paths`, optional depth) and
`instances` (per-object `aabb_world`, `center_world`, `extent`). Their released
scenes (UE-MeshyScene, and five Marble worlds) carry all of it. **Our DL3DV
scenes carry frames and poses but no instance masks and no 3D boxes**, so
running WorldSculpt on our own data is a grounding project, not a flag: it
needs per-frame instance masks and a 3D box per object. That is scoped as a
separate step, not smuggled into the reproduction arm.
