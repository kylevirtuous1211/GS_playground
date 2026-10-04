# Demos

**Tracked index of every demo this repository can show.**
The viewers themselves live under `outputs/` and are untracked; what is tracked is the row below and the builder script that regenerates the viewer from a run.

A row belongs here when the demo opens and shows the method's output clearly.
Showing the result on its own is enough; a **comparison** is owed only when the demo implies a judgement, and then what it sits beside has to be a real counterpart rather than something borrowed to fill the frame.

| demo | what it shows | open it | built by | log entry |
|---|---|---|---|---|
| E08c WorldSculpt | a Marble living room taken apart into 13 object meshes, each view beside the frame it came from; the room's walls and floor are visibly absent | `python -m http.server -d outputs/worldsculpt/viewer` | `gs_playground.worldsculpt.demo` | E08c |
| E08e WorldSculpt on NCHC sofa | our own lounge capture taken apart into 24 object meshes, each per-view strip beside the frame it came from, plus what each object was given (our mask and box) | `python -m http.server -d outputs/worldsculpt/NCHC/viewer` | `gs_playground.worldsculpt.demo --page nchc` | E08e |
| E08f WorldSculpt parameter sweep | which knobs change how many objects come out (Stage A, 256 label ids, stands) and how complete their meshes are (Stage B, **retracted** by E08g and marked so on the page: its meshes were decoded with random conv weights) | `python -m http.server -d outputs/worldsculpt/NCHC/sweep/viewer` | `gs_playground.worldsculpt.sweep_demo` | E08f |
| E08g WorldSculpt scenes in 3D | both WorldSculpt scenes (Marble 13 objects, NCHC sofa 24) as orbitable meshes in one page, like the paper's project page; hover shows an object's input box, click shows the crop it was conditioned on | `python -m http.server -d outputs/worldsculpt/viewer3d` | `gs_playground.worldsculpt.scene_viewer` | E08g |
| E08h SuRFLo | a mesh and a 3DGS from 16 photos: the authors' garden sample and our NCHC sofa, each beside its input views, our exported 3DGS (SuRFLo never saves it) and, for the sofa, our 3DGS labelled as a reference | `python -m http.server -d outputs/surflo/viewer` | `gs_playground.surflo.demo` (surflo env) | E08h |
| E08m IGGT instances | IGGT's 8-dim instance features on 12 garden views: each view above its feature PCA colours and the instances IGGT's own demo clustering makes of them (on garden it merges table, pot and ground; see the entry) | open `outputs/iggt/garden/render/contact_sheet.png` (a still) | `gs_playground.iggt.instances infer` (surflo env) then `render` (puffin env) | E08m |
| E08n semantics on SuRFLo | SuRFLo's Gaussians from 16 photos on our NCHC sofa, a ScanNet++ room and garden, each beside the same Gaussians coloured by IGGT instance group and by CLIP class, with a class legend (exploratory; known misses stated on the page) | `python -m http.server -d outputs/iggt/semantics_on_surflo/viewer` | `gs_playground.iggt.on_surflo` via `experiments/iggt/run_E08n_semantics_on_surflo.sh` | E08n |

## Rebuilding a viewer

Each builder takes the run directory it summarises and writes a self-contained directory: the renderer vendored in, assets beside it, and a `manifest.json` the page reads.
Nothing in a viewer directory is hand-edited, so a stale demo is regenerated rather than patched.
The exact command for each is in its builder's docstring or its runner.

Anything done to an asset to make it loadable, such as subsampling a large splat, is stated on the page itself.

## README media

The clips and stills in the top-level `README.md` live in `docs/media/` and are built from the runs above by `bash tools/build_readme_media.sh` (`gs_playground.readme_media`), never by hand.
