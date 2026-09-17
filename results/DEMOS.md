# Demos

**Tracked index of every demo this repository can show.**
The viewers themselves live under `outputs/` and are untracked; what is tracked is the row below and the builder script that regenerates the viewer from a run.

A row belongs here only when the demo opens and shows a **comparison**: the method's output beside ground truth, a baseline, or the input it was conditioned on.

| demo | what it shows | open it | built by | log entry |
|---|---|---|---|---|
| E07 scene gap | TRELLIS run on ten real DL3DV rooms, three conditioning modes, each generation beside the photo it came from | `python -m http.server -d outputs/puffin-gsvoxel/e07_scene_gap/viewer` | `gs_playground.gsvoxel.e07_viewer_assets` (assets only; its `index.html` was hand-written and is **not** reproducible) | `LOG.md` E07 |
| E08b WorldGrow | six generated worlds from 3 m to 9 m of floor, orbiting together, beside a real DL3DV interior rendered from its own cameras | `python -m http.server -d outputs/sota-gs/e08b_worldgrow/viewer` | `gs_playground.sota.e08b_demo` | `LOG.md` E08b |

## Rebuilding a viewer

Each builder takes the run directory it summarises and writes a self-contained directory: the renderer vendored in, assets beside it, and a `manifest.json` the page reads.
Nothing in a viewer directory is hand-edited, so a stale demo is regenerated rather than patched.

```bash
python -m gs_playground.gsvoxel.e07_viewer_assets \
    --generation outputs/puffin-gsvoxel/e07_scene_gap/generation.json \
    --scenes-root "$GS_PLAYGROUND_DATA/dl3dv/scenes" \
    --out outputs/puffin-gsvoxel/e07_scene_gap/viewer
```

Anything done to an asset to make it loadable, such as E07's opacity-ranked subsampling, is stated on the page itself.
