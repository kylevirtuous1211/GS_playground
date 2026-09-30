"""Render a GaussianCloud with gsplat, from 3DGS cameras.json cameras.

cameras.json (INRIA 3DGS convention): per camera `position` (centre, world),
`rotation` (3x3 camera-to-world), `fx fy width height`. gsplat wants
world-to-camera viewmats and an intrinsics matrix.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from gsplat import rasterization

from .ply import GaussianCloud


@dataclass
class Camera:
    name: str
    viewmat: torch.Tensor  # [4, 4] world-to-camera
    K: torch.Tensor        # [3, 3]
    width: int
    height: int


def load_cameras(path: str | Path, max_side: int | None = None) -> list[Camera]:
    cams = []
    for entry in json.loads(Path(path).read_text()):
        c2w = torch.eye(4)
        c2w[:3, :3] = torch.tensor(entry["rotation"], dtype=torch.float32)
        c2w[:3, 3] = torch.tensor(entry["position"], dtype=torch.float32)
        viewmat = torch.linalg.inv(c2w)
        w, h = entry["width"], entry["height"]
        fx, fy = entry["fx"], entry["fy"]
        # INRIA cameras.json carries no principal point, so the image centre
        # is the documented fallback. gsplat's undistortion leaves it a few
        # pixels off centre, so the DL3DV adapter emits cx/cy and those win
        # where present.
        cx = entry.get("cx", w / 2.0)
        cy = entry.get("cy", h / 2.0)
        scale = 1.0
        if max_side is not None and max(w, h) > max_side:
            scale = max_side / max(w, h)
            w, h = round(w * scale), round(h * scale)
        K = torch.tensor([[fx * scale, 0, cx * scale],
                          [0, fy * scale, cy * scale],
                          [0, 0, 1]], dtype=torch.float32)
        cams.append(Camera(name=entry["img_name"], viewmat=viewmat, K=K,
                           width=w, height=h))
    return cams


@torch.no_grad()
def render(cloud: GaussianCloud, cam: Camera, device: str = "cuda", *,
           sh_degree: int | None = None, eps2d: float = 0.3,
           background: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> torch.Tensor:
    """-> [H, W, 3] float in [0, 1].

    Defaults: DC colour only, gsplat's 0.3 px screen-space dilation, white
    background. `sh_degree` evaluates the cloud's higher-order SH bands up to
    that degree (gsplat adds 0.5 and clamps at 0, as INRIA's rasterizer does);
    `eps2d=0` renders without the dilation, for assets trained without it.
    """
    cloud = cloud.to(device)
    if sh_degree is None:
        colors = cloud.colors
    else:
        bands = (sh_degree + 1) ** 2 - 1
        rest = (cloud.f_rest[:, :bands] if cloud.f_rest is not None
                else cloud.f_dc.new_zeros(len(cloud), bands, 3))
        colors = torch.cat([cloud.f_dc[:, None], rest], dim=1)
    rendered, alphas, _ = rasterization(
        means=cloud.means,
        quats=torch.nn.functional.normalize(cloud.quats, dim=1),
        scales=torch.exp(cloud.log_scales),
        opacities=cloud.opacities,
        colors=colors,
        viewmats=cam.viewmat[None].to(device),
        Ks=cam.K[None].to(device),
        width=cam.width,
        height=cam.height,
        sh_degree=sh_degree,
        eps2d=eps2d,
    )
    # composite over the background ourselves; gsplat's `backgrounds` shape
    # contract varies across 1.x versions and this does not
    bg = torch.tensor(background, device=device, dtype=rendered.dtype)
    img = rendered[0] + (1.0 - alphas[0]) * bg
    return img.clamp(0.0, 1.0)


@torch.no_grad()
def render_depth(cloud: GaussianCloud, cam: Camera,
                 device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """-> (expected z-depth [H, W], alpha [H, W]).

    gsplat's "ED" divides accumulated depth by alpha, so a faint layer still
    reports a confident-looking depth; callers gate on alpha.
    """
    cloud = cloud.to(device)
    depth, alphas, _ = rasterization(
        means=cloud.means,
        quats=torch.nn.functional.normalize(cloud.quats, dim=1),
        scales=torch.exp(cloud.log_scales),
        opacities=cloud.opacities,
        colors=cloud.colors,
        viewmats=cam.viewmat[None].to(device),
        Ks=cam.K[None].to(device),
        width=cam.width,
        height=cam.height,
        render_mode="ED",
    )
    return depth[0, ..., 0], alphas[0, ..., 0]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor, window: int = 11,
         sigma: float = 1.5) -> float:
    """Standard single-scale SSIM, gaussian window, on [H, W, 3] in [0,1]."""
    device = a.device
    coords = torch.arange(window, device=device) - window // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = (g[:, None] * g[None, :] / (g.sum() ** 2))[None, None]
    kernel = kernel.expand(3, 1, window, window)

    def filt(x):
        x = x.permute(2, 0, 1)[None]  # [1, 3, H, W]
        return torch.nn.functional.conv2d(x, kernel, groups=3,
                                          padding=window // 2)

    mu_a, mu_b = filt(a), filt(b)
    var_a = filt(a * a) - mu_a ** 2
    var_b = filt(b * b) - mu_b ** 2
    cov = filt(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return score.mean().item()
