"""The scene viewer's up axis: pitched cameras must not tilt the room."""

import numpy as np
from scipy.spatial.transform import Rotation

from gs_playground.worldsculpt.scene_viewer import display_frame


def cameras(up: np.ndarray, pitch_deg: float, yaws: np.ndarray) -> np.ndarray:
    """OpenGL c2w for level-roll cameras around world `up`, all pitched down."""
    to_world = Rotation.align_vectors([up], [[0, 1, 0]])[0]
    c2w = np.tile(np.eye(4), (len(yaws), 1, 1))
    for k, yaw in enumerate(yaws):
        local = Rotation.from_euler("YX", [yaw, -pitch_deg], degrees=True)
        c2w[k, :3, :3] = (to_world * local).as_matrix()
    return c2w


def test_up_survives_pitch_and_frame_is_a_rotation():
    up = np.array([0.3, -0.9, 0.2])
    up /= np.linalg.norm(up)
    for yaws in (np.linspace(-40, 40, 12), np.linspace(0, 330, 12)):
        rot = display_frame(cameras(up, 28, yaws))
        assert np.allclose(rot @ up, [0, 1, 0], atol=1e-6)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(rot), 1)
