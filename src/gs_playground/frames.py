"""Scene frames shared by the demos: which way is up, where the cameras looked."""

from __future__ import annotations

import numpy as np


def display_frame(c2w: np.ndarray) -> np.ndarray:
    """Rotation from world to a y-up frame whose -z is where the cameras looked.

    Up is the normal of the plane the cameras' right vectors span: a handheld
    or rendered camera barely rolls, so this ignores pitch, which a mean of
    their up vectors does not (E08c's cameras pitch 28 degrees down).
    `c2w` are OpenGL camera-to-world matrices, (N, 4, 4).
    """
    right, up, back = c2w[:, :3, 0], c2w[:, :3, 1], c2w[:, :3, 2]
    y = np.linalg.svd(right)[2][-1]
    y *= np.sign(y @ up.mean(0))
    z = back.mean(0)
    if np.linalg.norm(z - (z @ y) * y) < 1e-3:  # cameras all round a circle: any horizontal axis
        z = np.eye(3)[np.argmin(np.abs(y))]
    z = z - (z @ y) * y
    z /= np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z])
