# Demos

**Tracked index of every demo this repository can show.**
The viewers themselves live under `outputs/` and are untracked; what is tracked is the row below and the builder script that regenerates the viewer from a run.

A row belongs here when the demo opens and shows the method's output clearly.
Showing the result on its own is enough; a **comparison** is owed only when the demo implies a judgement, and then what it sits beside has to be a real counterpart rather than something borrowed to fill the frame.

| demo | what it shows | open it | built by | log entry |
|---|---|---|---|---|
| E07 scene gap | TRELLIS run on ten real DL3DV rooms, three conditioning modes, each generation beside the photo it came from | `python -m http.server -d outputs/archive/puffin-gsvoxel/e07_scene_gap/viewer` | `gs_playground.archive.gsvoxel.e07_viewer_assets` (assets only; its `index.html` was hand-written and is **not** reproducible) | `LOG.md` E07 |
| E08b WorldGrow | six generated worlds from 3 m to 9 m of floor, orbiting together, beside a real DL3DV interior rendered from its own cameras | `python -m http.server -d outputs/worldgrow/viewer` | `gs_playground.worldgrow.demo` | `LOG.md` E08b |
| E08c WorldSculpt | a Marble living room taken apart into 13 object meshes, each view beside the frame it came from; the room's walls and floor are visibly absent | `python -m http.server -d outputs/worldsculpt/viewer` | `gs_playground.worldsculpt.demo` | `LOG.md` E08c |
| E08d WorldGrow prompts | eight prompt/guidance arms at four fixed seeds, each seed's column comparable down the page, plus one orbitable splat per arm; IoU and CLIP counts against the released setting | `python -m http.server -d outputs/worldgrow/e08d_prompts/viewer` | `gs_playground.worldgrow.prompt_probe` | `LOG.md` E08d |
| E08e WorldSculpt on NCHC sofa | our own lounge capture taken apart into 24 object meshes, each per-view strip beside the frame it came from, plus what each object was given (our mask and box) | `python -m http.server -d outputs/worldsculpt/NCHC/viewer` | `gs_playground.worldsculpt.demo --page nchc` | `LOG.md` E08e |
| E08f WorldSculpt parameter sweep | which knobs change how many objects come out (Stage A, 256 label ids, stands) and how complete their meshes are (Stage B, **retracted** by E08g and marked so on the page: its meshes were decoded with random conv weights) | `python -m http.server -d outputs/worldsculpt/NCHC/sweep/viewer` | `gs_playground.worldsculpt.sweep_demo` | `LOG.md` E08f |
| E08g WorldSculpt scenes in 3D | both WorldSculpt scenes (Marble 13 objects, NCHC sofa 24) as orbitable meshes in one page, like the paper's project page; hover shows an object's input box, click shows the crop it was conditioned on | `python -m http.server -d outputs/worldsculpt/viewer3d` | `gs_playground.worldsculpt.scene_viewer` | `LOG.md` E08g |
| E08h SuRFLo | a mesh and a 3DGS from 16 photos: the authors' garden sample and our NCHC sofa, each beside its input views, our exported 3DGS (SuRFLo never saves it) and, for the sofa, our 3DGS labelled as a reference | `python -m http.server -d outputs/surflo/viewer` | `gs_playground.surflo.demo` (surflo env) | `LOG.md` E08h |
| E08m IGGT instances | IGGT's 8-dim instance features on 12 garden views: each view above its feature PCA colours and the instances IGGT's own demo clustering makes of them (on garden it merges table, pot and ground; see the entry) | open `outputs/iggt/garden/render/contact_sheet.png` (a still) | `gs_playground.iggt.instances infer` (surflo env) then `render` (puffin env) | `LOG.md` E08m |

## Rebuilding a viewer

Each builder takes the run directory it summarises and writes a self-contained directory: the renderer vendored in, assets beside it, and a `manifest.json` the page reads.
Nothing in a viewer directory is hand-edited, so a stale demo is regenerated rather than patched.

```bash
python -m gs_playground.archive.gsvoxel.e07_viewer_assets \
    --generation outputs/archive/puffin-gsvoxel/e07_scene_gap/generation.json \
    --scenes-root "$GS_PLAYGROUND_DATA/dl3dv/scenes" \
    --out outputs/archive/puffin-gsvoxel/e07_scene_gap/viewer
```

The WorldGrow page as published was built with a 120,000-gaussian cap (the builder's default is 200,000); this command reproduces it byte for byte:

```bash
R="$GS_PLAYGROUND_DATA/dl3dv/gs_iters=30000_factor=4/0c6f5d61c7936a255784bc6ffae6f4055e6a00482581b20f031ef02905009e46"
python -m gs_playground.worldgrow.demo \
    --generation outputs/worldgrow/generation.json \
    --real-scene "$R/ply/point_cloud_29999.ply" --real-cameras "$R/cameras.json" \
    --max-gaussians 120000 --out outputs/worldgrow/viewer
```

Anything done to an asset to make it loadable, such as E07's opacity-ranked subsampling, is stated on the page itself.
