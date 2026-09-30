"""E08e grounding: the contract WorldSculpt's crop stage silently depends on.

Upstream never fails on a bad input, it drops the object: 0/1 masks (it
thresholds alpha at >127), a wrong mask name, a mask below its pixel floor, or
a pose in the wrong convention all end with the object missing from
scene.glb. So this builds a small real scene and runs *upstream's own* crop
stage on it.

Skips unless the NCHC sofa data (NCHC_SOFA_RUN, set in .env.local) and the
WorldSculpt clone are present.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("NCHC_SOFA_RUN", "/nonexistent"))
DATA = RUN / "data/editreadygs_video/nchc_sofa_20260727_143647"
MODEL = RUN / "output/editreadygs_video/nchc_sofa_20260727_143647/3dgs_output"
CLONE = ROOT / "third_party/clones/worldsculpt"
SELECTION = ROOT / "experiments/worldsculpt/e08e_objects.json"
#: the L-sofa, a stool and a pillow: large, medium, small
PICK = {5, 6, 56}

pytestmark = pytest.mark.skipif(
    not (DATA.exists() and MODEL.exists() and (CLONE / "prepare_crops_scene.py").exists()),
    reason="NCHC sofa data or WorldSculpt clone not present")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from gs_playground.worldsculpt import ground
    tmp = tmp_path_factory.mktemp("e08e")
    selection = json.loads(SELECTION.read_text())
    for obj in selection["objects"]:
        obj["keep"] = obj["id"] in PICK
    subset = tmp / "selection.json"
    subset.write_text(json.dumps(selection))
    scene_dir = tmp / "scene"
    sys.argv = ["ground", "build", "--data-dir", str(DATA), "--model-dir", str(MODEL),
                "--objects", str(subset), "--out", str(scene_dir), "--stride", "30"]
    ground.main()
    return tmp, scene_dir


def test_poses_round_trip_to_the_cameras(built):
    from gs_playground.gs.render import load_cameras
    from gs_playground.worldsculpt.ground import FLIP
    _, scene_dir = built
    cams = {Path(c.name).stem: c for c in load_cameras(MODEL / "cameras.json")}
    meta = json.loads((scene_dir / "transforms.json").read_text())
    for frame in meta["frames"]:
        stem = Path(frame["source_image"]).stem
        w2c = np.linalg.inv(np.asarray(frame["transform_matrix"]) @ FLIP)
        assert np.allclose(w2c, cams[stem].viewmat.double().numpy(), atol=1e-5), stem


def test_masks_are_0_255_at_full_resolution(built):
    from PIL import Image
    _, scene_dir = built
    meta = json.loads((scene_dir / "transforms.json").read_text())
    paths = sorted((scene_dir / "masks").rglob("*.png"))
    assert paths
    for path in paths[:10]:
        m = np.asarray(Image.open(path))
        assert m.shape == (meta["h"], meta["w"])
        assert set(np.unique(m)) <= {0, 255}


def test_upstream_crop_stage_keeps_every_object(built):
    """Upstream's own crop script, with inference.sh's flags at a tiny crop size."""
    from gs_playground.worldsculpt import ground
    tmp, scene_dir = built
    case = tmp / "case"
    subprocess.run(
        [sys.executable, "prepare_crops_scene.py", "--scene_dir", str(scene_dir),
         "--case_root", str(case), "--crop_resolution", "64",
         "--alpha_erode_kernel", "0", "--alpha_erode_iters", "0",
         "--min_mask_ratio", "0.001", "--max_crop_ratio", "3.0", "--mask_fit_scale"],
        cwd=CLONE, check=True, capture_output=True)
    out = tmp / "checks.json"
    sys.argv = ["ground", "check", "--scene-dir", str(scene_dir), "--case-root", str(case),
                "--out", str(out)]
    ground.main()
    rows = json.loads(out.read_text())["objects"]
    assert {r["object"] for r in rows} == {f"obj{k:02d}" for k in PICK}
    for row in rows:
        assert row["crops"] >= 1, row
        assert row["mask_fit"] is not None and row["mask_fit"] < 1.5, row
