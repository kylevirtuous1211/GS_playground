"""Load / save 3DGS point_cloud.ply files (INRIA layout).

Attributes used: x y z, f_dc_{0,1,2}, opacity (logit), scale_{0,1,2} (log),
rot_{0..3} (unnormalised quaternion, wxyz), and higher-order SH f_rest_* when
the file has them. INRIA stores f_rest channel-major (all of red's bands, then
green's, then blue's); in memory it is [N, K, 3], gsplat's order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

#: SH DC basis constant; colour = 0.5 + C0 * f_dc. The paper quotes it too.
SH_C0 = 0.28209479177387814


@dataclass
class GaussianCloud:
    """Raw 3DGS parameters, exactly as stored in the ply (no activations)."""

    means: torch.Tensor       # [N, 3] world
    f_dc: torch.Tensor        # [N, 3] SH DC coefficients
    opacity_logit: torch.Tensor  # [N]
    log_scales: torch.Tensor  # [N, 3]
    quats: torch.Tensor       # [N, 4] wxyz, unnormalised
    f_rest: torch.Tensor | None = None  # [N, K, 3] higher-order SH, or DC only

    def __len__(self) -> int:
        return self.means.shape[0]

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logit)

    @property
    def colors(self) -> torch.Tensor:
        return (0.5 + SH_C0 * self.f_dc).clamp(0.0, 1.0)

    def to(self, device) -> "GaussianCloud":
        return GaussianCloud(*(t.to(device) for t in
                               (self.means, self.f_dc, self.opacity_logit,
                                self.log_scales, self.quats)),
                             f_rest=None if self.f_rest is None else self.f_rest.to(device))

    def select(self, keep: torch.Tensor) -> "GaussianCloud":
        """The subset `keep` (index or mask), higher-order SH included."""
        return GaussianCloud(self.means[keep], self.f_dc[keep], self.opacity_logit[keep],
                             self.log_scales[keep], self.quats[keep],
                             f_rest=None if self.f_rest is None else self.f_rest[keep])


def load_ply(path: str | Path) -> GaussianCloud:
    ply = PlyData.read(str(path))
    v = ply["vertex"].data

    def cols(names):
        return torch.from_numpy(
            np.stack([v[n].astype(np.float32) for n in names], axis=1))

    rest = sorted((n for n in v.dtype.names if n.startswith("f_rest_")),
                  key=lambda n: int(n[len("f_rest_"):]))
    f_rest = None
    if rest:
        f_rest = cols(rest).reshape(len(v), 3, len(rest) // 3).transpose(1, 2).contiguous()
    return GaussianCloud(
        means=cols(["x", "y", "z"]),
        f_dc=cols(["f_dc_0", "f_dc_1", "f_dc_2"]),
        opacity_logit=torch.from_numpy(v["opacity"].astype(np.float32)),
        log_scales=cols(["scale_0", "scale_1", "scale_2"]),
        quats=cols(["rot_0", "rot_1", "rot_2", "rot_3"]),
        f_rest=f_rest,
    )


def save_ply(cloud: GaussianCloud, path: str | Path) -> None:
    n = len(cloud)
    rest = (np.zeros((n, 0), np.float32) if cloud.f_rest is None
            else cloud.f_rest.transpose(1, 2).reshape(n, -1).cpu().numpy())
    names = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([
        cloud.means.cpu().numpy(),
        cloud.f_dc.cpu().numpy(),
        rest,
        cloud.opacity_logit.cpu().numpy()[:, None],
        cloud.log_scales.cpu().numpy(),
        cloud.quats.cpu().numpy(),
    ], axis=1).astype(np.float32)
    arr = np.empty(n, dtype=[(name, "f4") for name in names])
    for i, name in enumerate(names):
        arr[name] = data[:, i]
    PlyElement.describe(arr, "vertex")
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def to_cloud(gaussian) -> GaussianCloud:
    """TRELLIS Gaussian -> our GaussianCloud, via the activated getters."""
    scales = gaussian.get_scaling.detach().float()
    opacity = gaussian.get_opacity.detach().float().reshape(-1)
    return GaussianCloud(
        means=gaussian.get_xyz.detach().float(),
        f_dc=gaussian.get_features.detach().float()[:, 0, :],
        # logit and log: the inverse of what load_ply re-applies
        opacity_logit=torch.log(opacity.clamp(1e-6, 1 - 1e-6)
                                / (1 - opacity.clamp(1e-6, 1 - 1e-6))),
        log_scales=torch.log(scales.clamp_min(1e-8)),
        quats=gaussian.get_rotation.detach().float(),
    )
