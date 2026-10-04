# fetch/

Data and weight acquisition. Every script writes under `$GS_PLAYGROUND_DATA` (default `data/`) or `~/datasets/`, never into a clone, and stages under `.partial` before publishing by rename.

| script | what |
|---|---|
| `fetch_mipnerf360.sh` | the official Mip-NeRF 360 scenes, checksummed, plus `derived/garden_train161_images_4` (garden's 161 training views) |
| `fetch_scannetpp_surflo_inputs.sh` | per ScanNet++ v2 scene, the undistorted DSLR training frames minus `is_bad`, as SuRFLo input |
| `viewer_js.sha256` | checksums of the vendored three.js and Spark files the web viewers copy |

Derived datasets encode their parameters in the directory name (`garden_train161_images_4`, not `garden_v2`).
