"""E08l: the ScanNet++ frame, the backbone swap, and the metric wrappers.

The frame test runs under pytest anywhere (numpy, the ScanNet++ share). The
other two need SuRFLo's package, which only imports in the surflo env, and
that env has no pytest: there, run `python tests/test_surflo_swap.py`.
"""

import importlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from gs_playground.paths import SCANNETPP

try:
    import pytest
except ImportError:   # the surflo env
    pytest = None

SCENE = SCANNETPP / "data/825d228aec"


def needs(module: str):
    """The module, or a pytest skip when it only exists in the surflo env."""
    try:
        return importlib.import_module(module)
    except ImportError:
        if pytest is None:
            raise
        pytest.skip(f"needs {module} (surflo env)")


def test_dslr_cameras_land_in_the_mesh_frame_exactly():
    """A wrong axis swap still fits Umeyama with a tiny residual, into the wrong world."""
    if not SCENE.exists():
        pytest.skip("needs the ScanNet++ share")
    from scipy.spatial.transform import Rotation
    from gs_playground.surflo.swap import dslr_cameras
    poses, _ = dslr_cameras(SCENE)
    name = sorted(poses)[10]
    line = subprocess.run(["grep", "-m1", f" {name}.JPG$", str(SCENE / "dslr/colmap/images.txt")],
                          capture_output=True, text=True, check=True).stdout.split()
    qw, qx, qy, qz, tx, ty, tz = map(float, line[1:8])
    w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()   # COLMAP: world-to-camera, mesh frame
    assert np.allclose(w2c @ poses[name][:3, :3], np.eye(3), atol=1e-5)
    assert np.allclose(-w2c.T @ np.array([tx, ty, tz]), poses[name][:3, 3], atol=1e-5)


def test_load_vggt_weights_is_strict_and_strips_the_prefix():
    torch = needs("torch")
    loader = needs("surflo.model.loader")
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    other = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    with tempfile.TemporaryDirectory() as tmp:
        full = {f"module.{k}": v for k, v in other.state_dict().items()} | {"module.part_head.w": torch.ones(1)}
        torch.save(full, Path(tmp) / "full.pth")
        assert loader.load_vggt_weights(model, str(Path(tmp) / "full.pth")) == ["part_head.w"]
        assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))
        torch.save({k: v for k, v in full.items() if not k.startswith("module.1.")}, Path(tmp) / "truncated.pth")
        try:
            loader.load_vggt_weights(model, str(Path(tmp) / "truncated.pth"))
        except RuntimeError as error:
            assert "Missing key" in str(error)
        else:
            raise AssertionError("a truncated state dict loaded without complaint")


def test_score_and_agreement_see_a_5cm_threshold():
    torch = needs("torch")
    needs("surflo.metrics.eval_alignment")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from gs_playground.surflo.swap import agreement, score
    grid = np.stack(np.meshgrid(np.linspace(0, 2, 200), np.linspace(0, 2, 200), [0.0]), -1).reshape(-1, 3)
    ref = np.concatenate([grid, grid + [0, 0, 1.0]])   # two parallel planes, 1 m apart
    # predictions offset towards each other, so they stay inside the ground truth's box
    near = torch.from_numpy(np.concatenate([grid + [0, 0, 0.03], grid + [0, 0, 0.97]]))
    far = torch.from_numpy(np.concatenate([grid + [0, 0, 0.10], grid + [0, 0, 0.90]]))
    assert score(near, ref)["f1_5cm"] > 0.99 and score(far, ref)["f1_5cm"] < 0.01
    assert agreement(torch.from_numpy(ref), near, 0.05)["f1"] > 0.99
    assert agreement(torch.from_numpy(ref), far, 0.05)["f1"] < 0.01


if __name__ == "__main__":   # the surflo env
    for test in (test_load_vggt_weights_is_strict_and_strips_the_prefix, test_score_and_agreement_see_a_5cm_threshold):
        test()
        print(f"{test.__name__}: passed")
