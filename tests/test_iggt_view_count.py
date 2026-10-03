"""E08m: the camera metric (validity check 2), the anchor design, and T-mIoU's matching."""

import numpy as np
from scipy.spatial.transform import Rotation

from gs_playground.iggt.view_count import anchors_and_batches, auc, pair_errors, t_miou


def poses(n=8, seed=0):
    rng = np.random.default_rng(seed)
    R = Rotation.random(n, random_state=seed).as_matrix()
    return R, rng.normal(size=(n, 3))


def test_perfect_cameras_score_one_and_a_known_rotation_is_reported():
    R, t = poses()
    rot, trans = pair_errors(R, t, R, t)
    assert np.allclose(rot, 0, atol=1e-5) and np.allclose(trans, 0, atol=1e-4)
    assert auc(np.maximum(rot, trans)) == 1.0
    turned = R.copy()
    turned[3] = Rotation.from_euler("y", 10, degrees=True).as_matrix() @ R[3]   # camera 3 turned 10 degrees
    rot, _ = pair_errors(turned, t, R, t)
    i, j = np.triu_indices(8, k=1)
    involved = (i == 3) | (j == 3)
    assert np.allclose(rot[involved], 10, atol=1e-4) and np.allclose(rot[~involved], 0, atol=1e-5)


def test_pair_errors_ignore_the_world_frame_and_scale():
    """VGGT's frame is its first camera's, with its own scale: a similarity must not count as error."""
    R, t = poses()
    S = Rotation.from_euler("xyz", [20, -35, 50], degrees=True).as_matrix()
    # world-to-camera under x' = s S x + c: R' = R S^T, t' = s t - R S^T c
    c, s = np.array([1.0, -2.0, 0.5]), 3.7
    R2 = R @ S.T
    t2 = s * t - np.einsum("nab,b->na", R2, c)
    rot, trans = pair_errors(R2, t2, R, t)
    assert np.allclose(rot, 0, atol=1e-4) and np.allclose(trans, 0, atol=1e-3)


def test_auc_counts_errors_below_each_degree():
    assert auc(np.array([0.5, 0.5])) == 1.0
    assert auc(np.array([40.0])) == 0.0
    assert abs(auc(np.array([15.5])) - 15 / 30) < 1e-9   # below the bound from 16 to 30 degrees


def test_anchors_are_fixed_and_batches_grow_around_them():
    names = [f"img{i:03d}.jpg" for i in range(100)]
    anchors, batches = anchors_and_batches(names, 8, (8, 16, 32, 128))
    assert anchors[0] == "img000.jpg" and len(anchors) == 8
    assert batches[(8, 0)] == sorted(anchors) and len([k for k in batches if k[0] == 8]) == 1
    for (n, seed), batch in batches.items():
        assert len(batch) == n and set(anchors) <= set(batch) and batch == sorted(batch) and batch[0] == anchors[0]
    assert (100, 0) in batches and (128, 0) not in batches        # N above the frame count: all frames
    assert batches[(16, 0)] != batches[(16, 1)]
    assert anchors_and_batches(names, 8, (16,))[1] == {k: v for k, v in batches.items() if k[0] == 16}


def test_t_miou_matches_one_to_one_across_views():
    gt = np.zeros((2, 20, 20), dtype=int)
    gt[:, :10] = 1
    gt[:, 10:] = 2
    valid = np.ones_like(gt, dtype=bool)
    assert t_miou(gt + 5, gt, valid)["t_miou"] == 1.0              # any cluster ids, matched
    merged = np.zeros_like(gt)                                       # one cluster for both instances
    assert abs(t_miou(merged, gt, valid)["t_miou"] - 0.25) < 1e-9   # one instance at IoU 0.5, one unmatched
    flipped = gt.copy()
    flipped[1] = 3 - gt[1]                                           # ids swapped in the second view
    assert t_miou(flipped, gt, valid)["t_miou"] < 0.6               # inconsistent across views is penalised
