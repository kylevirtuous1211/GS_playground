"""E08h readings, against experiments/surflo/PREREG_E08h.md.

    python -m gs_playground.surflo.evaluate stage-a \
        --root outputs/surflo/garden_sample --out results/surflo/e08h_stage_a.json
    # surflo env: the authors' metric code only imports there
    python -m gs_playground.surflo.evaluate stage-b --root outputs/surflo/nchc_sofa \
        --model-dir <3dgs_output> --depth-dir outputs/worldsculpt/NCHC/ground/depth \
        --out results/surflo/e08h_stage_b.json

Stage A compares each arm's ODE time and peak VRAM with the authors' README
table and checks the promised outputs exist. The fourth criterion (the mesh
shows the table, the pot and the ground) is judged by eye and recorded in the
LOG, so this file reports it as pending rather than guessing.

Stage B scores each sofa run's point cloud against our 3DGS's back-projected
depth, with the authors' own metric functions (Umeyama on camera centres, no
ICP), and applies the pre-registered seed-noise reading rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

#: README runtime table, one H100, 100k points, DA3 priors and meshing excluded.
THEIRS = {"guided_default": {"ode_s": 45.0, "vram_gib": 14.0},
          "plain": {"ode_s": 8.0, "vram_gib": 8.5}}
#: pre-registered tolerances
MAX_TIME_RATIO = 5.0
VRAM_TOLERANCE = 0.30
OUTPUTS = {"guided_default": ("mesh.ply", "point_cloud_normals.ply", "point_cloud_rgb.ply",
                              "mesh_textured.ply"),
           "plain": ("final.ply",)}


def ply_counts(path: Path) -> dict[str, int]:
    """Element counts from a PLY header, without loading the body."""
    counts = {}
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("element"):
                _, name, n = line.split()
                counts[name] = int(n)
            if line == "end_header":
                break
    return counts


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_run(run: Path, arm: str) -> dict:
    summary_path = next(run.glob("*/_infer_summary.json"))
    scene_dir = summary_path.parent
    info = json.loads(summary_path.read_text())["scene"]
    files = {name: scene_dir / name for name in OUTPUTS[arm]}
    present = {name: p.exists() and p.stat().st_size > 0 for name, p in files.items()}
    first = files[OUTPUTS[arm][0]]
    return {
        "run": run.name,
        "ode_s": info["ode_inference_s"],
        "peak_vram_ode_gib": info.get("peak_vram_ode_gib"),
        "outputs_present": present,
        "counts": ply_counts(first) if first.exists() else None,
        "sha_" + first.name: sha(first) if first.exists() else None,
    }


def stage_a(root: Path) -> dict:
    arms = {}
    for arm, theirs in THEIRS.items():
        runs = [read_run(r, arm) for r in sorted((root / arm).glob("run*"))]
        ode = [r["ode_s"] for r in runs]
        vram = [r["peak_vram_ode_gib"] for r in runs]
        key = next(k for k in runs[0] if k.startswith("sha_"))
        ode_ratio = median(ode) / theirs["ode_s"]
        vram_ratio = median(vram) / theirs["vram_gib"]
        arms[arm] = {
            "n": len(runs),
            "runs": runs,
            "theirs_h100": theirs,
            "ode_s_median": median(ode), "ode_s_range": [min(ode), max(ode)],
            "peak_vram_gib_median": median(vram), "peak_vram_gib_range": [min(vram), max(vram)],
            "ode_ratio_to_theirs": ode_ratio,
            "vram_ratio_to_theirs": vram_ratio,
            "identical_outputs_across_runs": len({r[key] for r in runs}) == 1,
            "criteria": {
                "1_outputs_present": all(all(r["outputs_present"].values()) for r in runs),
                "2_time_within_5x": ode_ratio <= MAX_TIME_RATIO,
                "3_vram_within_30pct": abs(vram_ratio - 1) <= VRAM_TOLERANCE,
            },
        }
    return {
        "entry": "E08h", "stage": "A", "prereg": "experiments/surflo/PREREG_E08h.md",
        "arms": arms,
        "criterion_4_mesh_by_eye": "pending: judged by eye, recorded in LOG.md E08h",
        "passes_1_to_3": all(all(a["criteria"].values()) for a in arms.values()),
    }


#: Stage B, pre-registered
REFERENCE_ARM = "guided_default_16"
ALPHA_MIN = 0.95
PIXEL_STRIDE = 2
BOX_PERCENTILES = (0.5, 99.5)
VOXEL_FRAC = 0.001
TAU_FRAC = 0.01
METRICS = ("chamfer_norm", "precision", "recall", "f1")


def reference_points(model_dir: Path, depth_dir: Path) -> "np.ndarray":
    """Our 3DGS's expected depth, back-projected over every frame (as E08e's gather)."""
    import numpy as np
    pts = []
    for c in json.loads((model_dir / "cameras.json").read_text()):
        d = np.load(depth_dir / f"{Path(c['img_name']).stem}.npz")
        depth, alpha = d["depth"], d["alpha"].astype(np.float32) / 255
        h, w = depth.shape
        sx, sy = w / c["width"], h / c["height"]
        fx, fy = c["fx"] * sx, c["fy"] * sy
        cx, cy = c.get("cx", c["width"] / 2) * sx, c.get("cy", c["height"] / 2) * sy
        ys, xs = np.mgrid[0:h:PIXEL_STRIDE, 0:w:PIXEL_STRIDE]
        z = depth[ys, xs]
        ok = (alpha[ys, xs] > ALPHA_MIN) & (z > 0)
        u, v, z = xs[ok] + 0.5, ys[ok] + 0.5, z[ok]
        cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
        pts.append(cam @ np.asarray(c["rotation"]).T + np.asarray(c["position"]))
    return np.concatenate(pts).astype(np.float32)


def camera_centres(scene: Path) -> dict:
    """A run's own camera centres by image stem, from guided_state.pt or plain_state.pt."""
    import numpy as np
    import torch
    guided = (scene / "guided_state.pt").exists()
    state = torch.load(scene / ("guided_state.pt" if guided else "plain_state.pt"),
                       map_location="cpu", weights_only=False)
    return {Path(p).stem: -np.asarray(c["R"]) @ np.asarray(c["T"])
            for p, c in zip(state["selected_images"], state["cameras"])}


def aligned_cloud(scene: Path, centres: dict):
    """A run's point cloud (guided: the opacity-culled centres; plain: final.ply) mapped
    into the frame of `centres` (image stem -> camera centre) by Umeyama on its own
    camera centres. Returns (points [P, 3] float64, Sim(3) scale, camera residuals)."""
    import numpy as np
    import torch
    import trimesh
    from surflo.metrics.eval_alignment import apply_similarity, umeyama_alignment
    guided = (scene / "guided_state.pt").exists()
    cloud = scene / ("point_cloud_normals.ply" if guided else "final.ply")
    pred = torch.as_tensor(np.asarray(trimesh.load(cloud).vertices), dtype=torch.float64)
    own = camera_centres(scene)
    stems = list(own)
    ours = torch.tensor(np.stack([own[s] for s in stems]), dtype=torch.float64)
    theirs = torch.tensor(np.stack([centres[s] for s in stems]), dtype=torch.float64)
    s, R, t = umeyama_alignment(ours, theirs)
    return apply_similarity(pred, s, R, t), float(s), (apply_similarity(ours, s, R, t) - theirs).norm(dim=1)


def score_run(scene: Path, ref, box, colmap_centres: dict, device: str = "cuda") -> dict:
    from surflo.metrics.eval_alignment import chamfer_and_fscore, voxel_downsample
    pred, s, cam_residual = aligned_cloud(scene, colmap_centres)
    stems = list(camera_centres(scene))
    lo, hi = box
    keep = ((pred >= lo) & (pred <= hi)).all(dim=1)
    diag = float((hi - lo).norm())
    pred_ds = voxel_downsample(pred[keep].float().to(device), VOXEL_FRAC * diag)
    ref_ds = voxel_downsample(ref.to(device), VOXEL_FRAC * diag)
    m = chamfer_and_fscore(pred_ds, ref_ds, tau=TAU_FRAC * diag)
    return {
        "scene": str(scene), "views": len(stems), "pred_points": int(pred.shape[0]),
        "pred_in_box": int(keep.sum()), "umeyama_scale": s,
        "camera_residual_over_diag": float(cam_residual.mean() / diag),
        "chamfer_norm": m.chamfer_mean / diag, "precision": m.precision,
        "recall": m.recall, "f1": m.f_score,
    }


def stage_b(root: Path, model_dir: Path, depth_dir: Path) -> dict:
    import numpy as np
    import torch
    cache = root / "reference_points.npz"
    if not cache.exists():
        np.savez_compressed(cache, points=reference_points(model_dir, depth_dir))
    points = np.load(cache)["points"]
    lo_np, hi_np = np.percentile(points, BOX_PERCENTILES, axis=0)   # exact, deterministic
    points = points[((points >= lo_np) & (points <= hi_np)).all(axis=1)]
    ref = torch.from_numpy(points)
    lo, hi = torch.from_numpy(lo_np), torch.from_numpy(hi_np)
    centres = {Path(c["img_name"]).stem: np.asarray(c["position"])
               for c in json.loads((model_dir / "cameras.json").read_text())}
    arms = {}
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and (p / "seed42").exists()):
        rows = {}
        for seed_dir in sorted(arm_dir.glob("seed*")):
            scene = next(p.parent for p in seed_dir.glob("*/_infer_summary.json"))
            rows[seed_dir.name] = score_run(scene, ref, (lo, hi), centres)
        arms[arm_dir.name] = rows
    base = arms[REFERENCE_ARM]
    seed_range = {k: max(r[k] for r in base.values()) - min(r[k] for r in base.values())
                  for k in METRICS}
    reading = {}
    for arm, rows in arms.items():
        if arm == REFERENCE_ARM:
            continue
        reading[arm] = {}
        for k in METRICS:
            change = median(rows[s][k] - base[s][k] for s in rows if s in base)
            reading[arm][k] = {"median_paired_change": change,
                               "reference_seed_range": seed_range[k],
                               "verdict": "changes" if abs(change) > seed_range[k]
                               else "within seed noise"}
    summary = {arm: {k: {"median": median(r[k] for r in rows.values()),
                         "range": [min(r[k] for r in rows.values()), max(r[k] for r in rows.values())]}
                     for k in METRICS} for arm, rows in arms.items()}
    return {"entry": "E08h", "stage": "B", "lane": "exploratory",
            "prereg": "experiments/surflo/PREREG_E08h.md",
            "reference": {"points_in_box": int(len(ref)), "box_lo": lo.tolist(), "box_hi": hi.tolist(),
                          "diag": float((hi - lo).norm())},
            "reference_arm": REFERENCE_ARM, "arms": arms, "summary": summary, "reading": reading}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stage-a")
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("stage-b")
    b.add_argument("--root", type=Path, required=True)
    b.add_argument("--model-dir", type=Path, required=True)
    b.add_argument("--depth-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "stage-b":
        result = stage_b(args.root, args.model_dir, args.depth_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        for arm, stats in result["summary"].items():
            print(arm, {k: f"{v['median']:.4f} [{v['range'][0]:.4f}, {v['range'][1]:.4f}]"
                        for k, v in stats.items()})
        for arm, r in result["reading"].items():
            print(arm, {k: f"{v['median_paired_change']:+.4f} vs {v['reference_seed_range']:.4f}: "
                           f"{v['verdict']}" for k, v in r.items()})
        return
    if args.cmd == "stage-a":
        result = stage_a(args.root)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        for arm, a in result["arms"].items():
            print(f"{arm}: ODE {a['ode_s_median']:.1f}s ({a['ode_ratio_to_theirs']:.2f}x theirs), "
                  f"VRAM {a['peak_vram_gib_median']:.1f} GiB ({a['vram_ratio_to_theirs']:.2f}x), "
                  f"identical={a['identical_outputs_across_runs']}, {a['criteria']}")
        print(f"criteria 1-3 pass: {result['passes_1_to_3']}; criterion 4 by eye: pending")


if __name__ == "__main__":
    main()
