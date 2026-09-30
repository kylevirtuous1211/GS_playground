"""Higher-order SH through ply I/O and the gsplat renderer.

A wrong f_rest layout does not crash anything: it renders plausible colours
with the view dependence scrambled. So the layout is pinned to INRIA's
explicitly, and the SH render path is checked against the DC path.
"""

import numpy as np
import pytest
import torch
from plyfile import PlyData

from gs_playground.gs.ply import GaussianCloud, load_ply, save_ply

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def cloud(n=500, bands=15, seed=0):
    g = torch.Generator().manual_seed(seed)
    return GaussianCloud(
        means=torch.randn(n, 3, generator=g) * 0.3 + torch.tensor([0.0, 0.0, 3.0]),
        f_dc=torch.rand(n, 3, generator=g) * 2 - 1,        # colours inside (0.2, 0.8)
        opacity_logit=torch.randn(n, generator=g),
        log_scales=torch.randn(n, 3, generator=g) * 0.3 - 3.5,
        quats=torch.randn(n, 4, generator=g),
        f_rest=torch.randn(n, bands, 3, generator=g) * 0.1 if bands else None,
    )


def camera():
    from gs_playground.gs.render import Camera
    K = torch.tensor([[200.0, 0, 96], [0, 200.0, 64], [0, 0, 1]])
    return Camera(name="t", viewmat=torch.eye(4), K=K, width=192, height=128)


def test_f_rest_is_written_channel_major_like_inria(tmp_path):
    c = cloud()
    save_ply(c, tmp_path / "c.ply")
    v = PlyData.read(str(tmp_path / "c.ply"))["vertex"].data
    # INRIA: f_rest_{c * K + k} is channel c, band k
    assert np.allclose(v["f_rest_1"], c.f_rest[:, 1, 0].numpy())
    assert np.allclose(v["f_rest_15"], c.f_rest[:, 0, 1].numpy())
    assert np.allclose(v["f_rest_44"], c.f_rest[:, 14, 2].numpy())


def test_round_trip_keeps_sh_and_dc_only_stays_dc_only(tmp_path):
    for bands in (15, 0):
        c = cloud(bands=bands)
        save_ply(c, tmp_path / "c.ply")
        back = load_ply(tmp_path / "c.ply")
        for name in ("means", "f_dc", "opacity_logit", "log_scales", "quats"):
            assert torch.equal(getattr(back, name), getattr(c, name)), name
        if bands:
            assert torch.equal(back.f_rest, c.f_rest)
        else:
            assert back.f_rest is None


@needs_gpu
def test_default_render_is_unchanged():
    """The old render(): DC colours, default eps2d, composited over white."""
    from gsplat import rasterization
    from gs_playground.gs.render import render
    c, cam = cloud().to("cuda"), camera()
    colors, alphas, _ = rasterization(
        means=c.means, quats=torch.nn.functional.normalize(c.quats, dim=1),
        scales=torch.exp(c.log_scales), opacities=c.opacities, colors=c.colors,
        viewmats=cam.viewmat[None].cuda(), Ks=cam.K[None].cuda(),
        width=cam.width, height=cam.height)
    old = (colors[0] + (1.0 - alphas[0])).clamp(0.0, 1.0)
    assert torch.equal(render(c, cam), old)


@needs_gpu
def test_sh_path_with_zero_bands_equals_dc_path():
    from gs_playground.gs.render import render
    c = cloud()
    c.f_rest = torch.zeros_like(c.f_rest)
    cam = camera()
    dc = render(c, cam, background=(0.0, 0.0, 0.0))
    sh = render(c, cam, sh_degree=3, background=(0.0, 0.0, 0.0))
    assert dc.max() > 0.1                       # something was drawn
    assert torch.allclose(dc, sh, atol=1e-5)
    # and the bands do something when they are not zero
    lit = render(cloud(), cam, sh_degree=3, background=(0.0, 0.0, 0.0))
    assert (lit - dc).abs().max() > 1e-2
