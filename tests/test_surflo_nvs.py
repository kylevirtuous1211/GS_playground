"""E08j's camera maths: a wrong sign here moves every test camera."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from gs_playground.surflo.nvs import crop_params, so3_exp, umeyama


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(16, 3))
    R = Rotation.from_rotvec([0.3, -1.1, 0.7]).as_matrix()
    dst = 2.5 * src @ R.T + np.array([1.0, -2.0, 0.5])
    s, R_hat, t = umeyama(src, dst)
    assert np.isclose(s, 2.5)
    assert np.allclose(R_hat, R)
    assert np.allclose(t, [1.0, -2.0, 0.5])


def test_so3_exp_matches_scipy_and_is_identity_at_zero():
    w = torch.tensor([0.2, -0.4, 0.9], dtype=torch.float64)
    assert np.allclose(so3_exp(w).numpy(), Rotation.from_rotvec(w.numpy()).as_matrix())
    assert torch.allclose(so3_exp(torch.zeros(3, dtype=torch.float64)), torch.eye(3, dtype=torch.float64))


@pytest.mark.parametrize("size, expected", [
    ((1297, 840), {"resize": [518, 335], "offset": [0, 6], "size": [518, 322]}),
    ((518, 336), {"resize": [518, 336], "offset": [0, 0], "size": [518, 336]}),   # the authors' sample
])
def test_crop_params(size, expected):
    assert crop_params(*size) == expected


def test_sweep_status_reaches_the_per_n_reading(tmp_path):
    from gs_playground.surflo.nvs import per_n, sweep
    for seed, wall in ((42, 100), (0, 120)):
        scene = tmp_path / "n16" / f"seed{seed}" / "garden"
        scene.mkdir(parents=True)
        (scene / "guided_state.pt").touch()
        (scene.parent / "done").touch()
        (scene.parent / "wall_s").write_text(str(wall))
    capped = tmp_path / "n161" / "seed0"
    capped.mkdir(parents=True)
    (capped / "status").write_text("skipped by the pre-registered cap: seed 42 took 1300 s > 1200 s\n")
    entries = sweep(tmp_path)
    assert [(e["n"], e["seed"], e["status"]) for e in entries] == [
        (16, 0, "done"), (16, 42, "done"), (161, 0, "skipped by the pre-registered cap: seed 42 took 1300 s > 1200 s")]
    keys = ("ode_s", "peak_vram_ode_gib", "peak_alloc_ode_gib", "psnr_raw_median", "psnr_tto_median",
            "ssim_raw_median", "ssim_tto_median")
    runs = {e["dir"]: {**dict.fromkeys(keys, float(e["seed"])), "wall_s": e["wall_s"]} for e in entries[:2]}
    table = per_n(entries, runs)
    assert table["16"]["seeds_done"] == 2 and table["16"]["wall_s"] == {"median": 110.0, "range": [100, 120]}
    assert table["161"] == {"seeds_done": 0, "not_done": {"0": entries[2]["status"]}}
