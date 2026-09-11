# fetch/

Data and weight acquisition. Every script writes under `$GS_PLAYGROUND_DATA` (default `data/`) and nowhere else - never into a clone.

What Puffin-World needs, and where it comes from:

| what | source | needed by |
|---|---|---|
| `Puffin-World-{Base,Pro,Caption}.pth` | 🤗 `KangLiao/Puffin-World` | inference, eval, checkpoint merging |
| Puffin-Cam-15M (captioned images + camera params) | 🤗 `KangLiao/Puffin-16M` | stages I, II |
| Puffin-Traj-1M | 🤗 `KangLiao/Puffin-16M` | stages III, IV |
| DL3DV, RealEstate10K, HyperSim, MVS-Synth, TartanAir, ScanNet | respective upstreams | stages III, IV |
| `*-Absolute-Camera` annotations | 🤗 `KangLiao/<DATASET>-Absolute-Camera` | stages III, IV (`physical_propagation='offline'`) |

The absolute-camera sets mirror their source dataset's directory layout; download them alongside the raw data and point each `configs/datasets/multi_view/gen_<dataset>.py`'s `camera_caption_root` at them.
Producing them yourself is GPU-days (`documents/ANNOTATION_CAMERA.md`) - download them.

Derived datasets encode their parameters in the directory name (`dl3dv_views=8`, not `dl3dv_v2`).
