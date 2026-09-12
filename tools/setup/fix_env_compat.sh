#!/usr/bin/env bash
# Post-install compatibility fixes to the puffin-world env's site-packages.
#
# Upstream pins xtuner==0.1.23 AND transformers==5.3.0, which cannot even be
# imported together from clean installs (their cluster evidently runs a
# locally modified xtuner). The env is disposable and rebuilt by script, so
# the honest place for these shims is here -- one block per incident, with the
# reason. Idempotent; called at the end of puffin_world_env.sh and safe to
# re-run by hand.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHA="$(git -C "$ROOT/third_party/clones/puffin" rev-parse --short HEAD)"
PREFIX="${ENV_PREFIX:-$ROOT/envs/puffin-world-$SHA-py310-cu126}"
SITE="$PREFIX/lib/python3.10/site-packages"

# -- 1. transformers 5.x removed is_safetensors_available; xtuner 0.1.23
#       imports it at module level (utils/handle_moe_load_and_save.py), which
#       kills `import xtuner.engine` entirely. safetensors is a hard dep of
#       transformers 5, so a constant-true fallback is exact.
F="$SITE/xtuner/utils/handle_moe_load_and_save.py"
if grep -q "^from transformers.utils import (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME,$" "$F"; then
    python3 - "$F" <<'PY'
import sys
from pathlib import Path
f = Path(sys.argv[1])
s = f.read_text()
old = """from transformers.utils import (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME,
                                is_safetensors_available)"""
new = """from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME
try:  # removed in transformers 5.x; safetensors is a hard dep there
    from transformers.utils import is_safetensors_available
except ImportError:
    def is_safetensors_available():
        return True"""
assert old in s
f.write_text(s.replace(old, new))
print(f"patched {f}")
PY
else
    echo "already patched: $F"
fi

# -- 2. rmbrualla/pycolmap (gsplat's data parser) predates numpy 2:
#       np.uint64(-1) raises OverflowError there instead of wrapping.
F="$SITE/pycolmap/scene_manager.py"
if [ -f "$F" ] && grep -q "np.uint64(-1)" "$F"; then
    sed -i 's/np.uint64(-1)/np.uint64(0xFFFFFFFFFFFFFFFF)/' "$F"
    echo "patched $F (numpy 2 uint64 overflow)"
else
    echo "already patched or absent: $F"
fi
