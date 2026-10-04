# tools/

Environment and data acquisition only. Nothing here produces a number.

- `setup/clone_upstream.sh` - rebuild `third_party/clones/` from `PINS.tsv` and re-apply patches.
- `setup/puffin_world_env.sh` - build the base micromamba environment (named `puffin-world` after the pinned fork whose requirements it starts from); `setup/surflo_env.sh` and `setup/worldsculpt_env.sh` build the method envs.
- `fetch/` - dataset and checkpoint downloads. Each script writes into `$GS_PLAYGROUND_DATA` (default `data/`) and nowhere else.
- `archive/` - env and data scripts of retired studies; local only.
- `stamp_provenance.sh` - sourced by every runner; writes `repo.sha`/`repo.patch`/`upstream.sha`/`upstream.patch`/`env.txt` into the run directory.
