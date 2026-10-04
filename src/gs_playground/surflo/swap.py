"""E08l scoring: SuRFLo with its VGGT-1B against IGGT's backbone (surflo env).

Pre-registered in experiments/surflo/PREREG_E08l.md.

    python -m gs_playground.surflo.swap score --root outputs/surflo/iggt_backbone \\
        --scannetpp "$GS_PLAYGROUND_SCANNETPP/data" \\
        --out results/surflo/e08l_iggt_backbone.json

Layout read under --root: <scene>/<backbone>/n16/seed{42,0,1} (guided) and
<scene>/<backbone>/plain/seed42[_repeat], each holding one SuRFLo scene dir;
<scene>/drift.json from `backbone drift`; garden/<backbone>/nvs.json from
`nvs eval`. ScanNet++ runs are scored against the GT mesh in its own metric
frame; agreement between two runs is measured in a common frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

from gs_playground.surflo.evaluate import VOXEL_FRAC, aligned_cloud, camera_centres

#: ScanNet++'s nerfstudio world -> its mesh (and DSLR COLMAP) world (ScanNet++'s nerfstudio export swaps x and y and flips z)
MESH_FROM_NERFSTUDIO = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float)
#: pre-registered
TAU_M = 0.05
TAU_FRAC = 0.01
VISIBILITY_SLACK_M = 0.05
#: implementation choices, not protocol
SURFACE_SAMPLES = 2_000_000
RAYCAST_DOWNSCALE = 4
BACKBONES = ("vggt", "iggt")
SEEDS = ("seed42", "seed0", "seed1")


# ---------------------------------------------------------------- ScanNet++
def dslr_cameras(scene_dir: Path) -> tuple[dict, dict]:
    """Image stem -> OpenCV camera-to-world in the mesh frame, and the pinhole intrinsics."""
    t = json.loads((scene_dir / "dslr/nerfstudio/transforms_undistorted.json").read_text())
    gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    poses = {Path(f["file_path"]).stem: MESH_FROM_NERFSTUDIO @ np.asarray(f["transform_matrix"]) @ gl_to_cv
             for f in t["frames"] + t.get("test_frames", [])}
    return poses, {k: t[k] for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")}


def visible_reference(scene_dir: Path, stems: list[str]) -> np.ndarray:
    """Points sampled on the GT mesh surface that at least one input view sees:
    in its image, in front of it, and within VISIBILITY_SLACK_M of the depth ray-cast there."""
    import open3d as o3d
    import trimesh
    mesh = trimesh.load(scene_dir / "scans/mesh_aligned_0.05.ply", process=False)
    samples = np.asarray(trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=0)[0])
    raycast = o3d.t.geometry.RaycastingScene()
    raycast.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                          o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
    poses, K = dslr_cameras(scene_dir)
    w, h = int(K["w"]) // RAYCAST_DOWNSCALE, int(K["h"]) // RAYCAST_DOWNSCALE
    intrinsic = np.array([[K["fl_x"], 0, K["cx"]], [0, K["fl_y"], K["cy"]], [0, 0, 1]]) / RAYCAST_DOWNSCALE
    intrinsic[2, 2] = 1.0
    seen = np.zeros(len(samples), dtype=bool)
    for stem in stems:
        w2c = np.linalg.inv(poses[stem])
        rays = raycast.create_rays_pinhole(intrinsic_matrix=o3d.core.Tensor(intrinsic),
                                           extrinsic_matrix=o3d.core.Tensor(w2c), width_px=w, height_px=h)
        hit = raycast.cast_rays(rays)["t_hit"].numpy()
        r = rays.numpy()
        points = r[..., :3] + hit[..., None] * r[..., 3:]
        with np.errstate(invalid="ignore"):   # misses: inf x 0 -> nan, replaced below
            depth = np.where(np.isfinite(hit), (points @ w2c[:3, :3].T + w2c[:3, 3])[..., 2], np.inf)
        cam = samples @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.floor(intrinsic[0, 0] * cam[:, 0] / z + intrinsic[0, 2]).astype(np.int64)
            v = np.floor(intrinsic[1, 1] * cam[:, 1] / z + intrinsic[1, 2]).astype(np.int64)
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.flatnonzero(inside)
        seen[idx] |= z[idx] <= depth[v[idx], u[idx]] + VISIBILITY_SLACK_M
    return samples[seen]


# ----------------------------------------------------------------- metrics
def score(pred, ref: np.ndarray, device: str = "cuda") -> dict:
    """F1 at TAU_M and at TAU_FRAC x diagonal, Chamfer, of `pred` cropped to `ref`'s box."""
    import torch
    from surflo.metrics.eval_alignment import chamfer_and_fscore, voxel_downsample
    ref_t = torch.from_numpy(ref)
    lo, hi = ref_t.min(0).values, ref_t.max(0).values
    diag = float((hi - lo).norm())
    keep = ((pred >= lo) & (pred <= hi)).all(dim=1)
    pred_ds = voxel_downsample(pred[keep].float().to(device), VOXEL_FRAC * diag)
    ref_ds = voxel_downsample(ref_t.float().to(device), VOXEL_FRAC * diag)
    metric = chamfer_and_fscore(pred_ds, ref_ds, tau=TAU_M)
    relative = chamfer_and_fscore(pred_ds, ref_ds, tau=TAU_FRAC * diag)
    return {"f1_5cm": metric.f_score, "precision_5cm": metric.precision, "recall_5cm": metric.recall,
            "chamfer_m": metric.chamfer_mean, "f1_rel": relative.f_score,
            "pred_in_box": int(keep.sum()), "diag_m": diag}


def agreement(a, b, tau: float, device: str = "cuda") -> dict:
    """F1 and Chamfer between two clouds already in one frame, at `tau`."""
    from surflo.metrics.eval_alignment import chamfer_and_fscore, voxel_downsample
    lo = np.percentile(a.numpy(), 0.5, axis=0)
    hi = np.percentile(a.numpy(), 99.5, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    a_ds = voxel_downsample(a.float().to(device), VOXEL_FRAC * diag)
    b_ds = voxel_downsample(b.float().to(device), VOXEL_FRAC * diag)
    m = chamfer_and_fscore(a_ds, b_ds, tau=tau)
    return {"f1": m.f_score, "chamfer": m.chamfer_mean, "tau": tau}


def scene_dir_of(run: Path) -> Path:
    return next(run.glob("*/_infer_summary.json")).parent


def reading(arms: dict, metrics: tuple[str, ...]) -> dict:
    """The pre-registered rule: changed if |median IGGT - median VGGT| > the VGGT seed range."""
    out = {}
    for key in metrics:
        vggt = [arms["vggt"][s][key] for s in SEEDS]
        iggt = [arms["iggt"][s][key] for s in SEEDS]
        change, noise = median(iggt) - median(vggt), max(vggt) - min(vggt)
        out[key] = {"vggt_median": median(vggt), "vggt_range": [min(vggt), max(vggt)],
                    "iggt_median": median(iggt), "iggt_range": [min(iggt), max(iggt)],
                    "change": change, "vggt_seed_range": noise, "changed": abs(change) > noise}
    return out


def score_scene(root: Path, scene: str, centres_of) -> dict:
    """Agreement (and, through `centres_of`, the frame) for one scene; GT scores for ScanNet++."""
    runs = {b: {s: scene_dir_of(root / scene / b / "n16" / s) for s in SEEDS} for b in BACKBONES}
    plain = {b: scene_dir_of(root / scene / b / "plain" / "seed42") for b in BACKBONES}
    repeat = scene_dir_of(root / scene / "vggt" / "plain" / "seed42_repeat")
    same = hashlib.sha256((plain["vggt"] / "final.ply").read_bytes()).hexdigest() == \
        hashlib.sha256((repeat / "final.ply").read_bytes()).hexdigest()
    stems = sorted(camera_centres(plain["vggt"]))
    for b in BACKBONES:
        assert sorted(camera_centres(plain[b])) == stems
        assert all(sorted(camera_centres(r)) == stems for r in runs[b].values())
    return {"runs": runs, "plain": plain, "stems": stems, "plain_bitwise_reproducible": same,
            "centres": centres_of(runs, stems)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--scannetpp", type=Path, required=True, help="ScanNet++ v2 data/ dir")
    s.add_argument("--scenes", nargs="+", default=["825d228aec", "6115eddb86", "13c3e046d7"])
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = {"entry": "E08l", "prereg": "experiments/surflo/PREREG_E08l.md",
           "weights_check": json.loads(Path("results/surflo/e08l_weights_check.json").read_text()),
           "scenes": {}}
    for scene in ["garden", *args.scenes]:
        row = {"drift": json.loads((args.root / scene / "drift.json").read_text())}
        if scene == "garden":
            # no metric ground truth: every run is mapped into the VGGT seed-42 run's frame
            def centres_of(runs, stems):
                return camera_centres(runs["vggt"]["seed42"])
        else:
            poses, _ = dslr_cameras(args.scannetpp / scene)

            def centres_of(runs, stems, poses=poses):
                return {k: poses[k][:3, 3] for k in stems}
        meta = score_scene(args.root, scene, centres_of)
        row["plain_bitwise_reproducible"] = meta["plain_bitwise_reproducible"]
        clouds = {b: {s: aligned_cloud(r, meta["centres"])[0] for s, r in meta["runs"][b].items()}
                  for b in BACKBONES}
        plain = {b: aligned_cloud(r, meta["centres"])[0] for b, r in meta["plain"].items()}
        if scene == "garden":
            ref_a = clouds["vggt"]["seed42"].numpy()
            lo, hi = np.percentile(ref_a, 0.5, axis=0), np.percentile(ref_a, 99.5, axis=0)
            tau = TAU_FRAC * float(np.linalg.norm(hi - lo))
            nvs = {b: json.loads((args.root / scene / b / "nvs.json").read_text()) for b in BACKBONES}
            row["nvs"] = {b: nvs[b]["per_n"]["16"] for b in BACKBONES}
            row["nvs_sanity_2"] = {b: nvs[b]["sanity_2"] for b in BACKBONES}
            arms = {b: {Path(d).name: r for d, r in nvs[b]["runs"].items()} for b in BACKBONES}
            row["reading"] = reading(arms, ("psnr_tto_median", "psnr_raw_median", "ssim_tto_median"))
        else:
            ref = visible_reference(args.scannetpp / scene, meta["stems"])
            tau = TAU_M
            row["reference_points"] = len(ref)
            row["guided"] = {b: {s: score(c, ref) for s, c in clouds[b].items()} for b in BACKBONES}
            row["plain"] = {b: score(c, ref) for b, c in plain.items()}
            row["reading"] = reading(row["guided"], ("f1_5cm", "chamfer_m", "f1_rel"))
        row["agreement"] = {
            "cross_backbone": {s: agreement(clouds["vggt"][s], clouds["iggt"][s], tau) for s in SEEDS},
            "cross_seed_vggt": {f"seed42_vs_{s}": agreement(clouds["vggt"]["seed42"], clouds["vggt"][s], tau)
                                for s in ("seed0", "seed1")},
            "plain_cross_backbone": agreement(plain["vggt"], plain["iggt"], tau),
        }
        cross = min(v["f1"] for v in row["agreement"]["cross_backbone"].values())
        floor = min(v["f1"] for v in row["agreement"]["cross_seed_vggt"].values())
        row["agreement"]["within_own_variation"] = cross >= floor
        row["compatible"] = (not any(v["changed"] for v in row["reading"].values())
                             and row["agreement"]["within_own_variation"])
        out["scenes"][scene] = row
        print(scene, "compatible" if row["compatible"] else "CHANGED",
              {k: f"{v['change']:+.4f} vs {v['vggt_seed_range']:.4f}" for k, v in row["reading"].items()},
              f"agreement {cross:.4f} vs floor {floor:.4f}", flush=True)
    out["compatible"] = all(r["compatible"] for r in out["scenes"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
