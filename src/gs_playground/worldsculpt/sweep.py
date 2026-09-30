"""E08f: parameters vs how many objects come out, and how complete they are.

Arms, metrics, nulls and reading rules are fixed in
`experiments/worldsculpt/PREREG_E08f.md`; this module only computes them.

    stage-a   grounding parameters, one at a time -> ids kept, rejection
              reasons, overlap with the default set, box completeness.
              Reuses E08e's cached depth; writes the observed surface points
              Stage B scores against.
    score-b   WorldSculpt parameters -> per-object mesh completeness, held-out
              silhouette agreement, fragments, and the seed null.

    python -m gs_playground.worldsculpt.sweep stage-a \
        --data-dir ... --model-dir ... --ground outputs/worldsculpt/NCHC/ground \
        --out results/worldsculpt/e08f_stage_a.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from . import ground as g

#: Stage A arms, one field of ground.Params at a time (PREREG_E08f.md)
STAGE_A = [
    ("support_frac", [0.02, 0.05, 0.10, 0.20]),
    ("min_views", [4, 8, 12, 16]),
    ("speck_frac", [0.0, 0.02, 0.10]),
    ("largest_only", [True]),
    ("stride", [1, 3, 6, 12]),
    ("clip_gate", [True, False]),
    ("struct_extent", [0.3, 0.4, 0.5]),
]
TAU = 0.02          # of the object's box diagonal
SURFACE_SAMPLES = 200_000
HELDOUT_FRAMES = 20


# --------------------------------------------------------------------------
# Stage A


def evaluate(scene, E: float, p: g.Params, evidence: dict) -> dict:
    """The automatic filter at `p`, plus box completeness per kept id."""
    kept, reasons, completeness, centres = [], {}, {}, {}
    for k in sorted(evidence):
        ev = evidence[k]
        est = g.estimate_box(ev["pts"], ev["pfi"], len(ev["frames"]), E, p)
        gates = (g.views_at_stride(scene, k, est["box"], E, p, ev["frames"])
                 if est["box"] is not None and k != 0 else ([], [], {"small": 0}))
        keep, reason, _ = g.decide(k, ev, est, gates, E, p)
        code = reason.split(" (")[0]
        reasons[code] = reasons.get(code, 0) + 1
        if not keep:
            continue
        kept.append(k)
        lo, hi = np.asarray(est["box"][0]), np.asarray(est["box"][1])
        inside = np.all((ev["pts"] >= lo) & (ev["pts"] <= hi), axis=1)
        completeness[k] = float(inside.mean())
        centres[k] = est["centres"]
    return {"kept": kept, "reasons": reasons, "completeness": completeness, "centres": centres}


def stage_a(args) -> None:
    scene = g.load_scene(args.data_dir, args.model_dir)
    E = g.scene_extent(args.data_dir)
    depth_dir = args.ground / "depth"
    default = g.Params()
    arms = [(name, value) for name, values in STAGE_A for value in values]
    # gather() is the slow step and only the speck settings change it
    by_gather: dict[tuple, list] = {}
    for name, value in arms:
        p = dataclasses.replace(default, **{name: value})
        by_gather.setdefault((p.speck_frac, p.largest_only), []).append((name, value, p))

    rows, base = [], None
    for key in sorted(by_gather, key=lambda k: k != (default.speck_frac, default.largest_only)):
        print(f"gather speck_frac={key[0]} largest_only={key[1]}", flush=True)
        evidence = g.gather(scene, depth_dir, dataclasses.replace(default, speck_frac=key[0], largest_only=key[1]))
        for name, value, p in by_gather[key]:
            result = evaluate(scene, E, p, evidence)
            if p == default and base is None:
                base = result
                sanity(args.ground, result)
                np.savez_compressed(args.observed, **{str(k): v for k, v in result["centres"].items()})
            rows.append((name, value, p == default, result))
            print(f"  {name}={value}: {len(result['kept'])} kept", flush=True)
        del evidence

    table = []
    base_set = set(base["kept"])
    for name, value, is_default, r in rows:
        kept = set(r["kept"])
        common = sorted(kept & base_set)
        table.append({
            "param": name, "value": value, "default": is_default,
            "n_kept": len(kept), "reasons": r["reasons"],
            "jaccard_vs_default": len(kept & base_set) / max(1, len(kept | base_set)),
            "added": sorted(kept - base_set), "removed": sorted(base_set - kept),
            "box_completeness_median": float(np.median(list(r["completeness"].values()))) if kept else None,
            "box_completeness_median_common": float(np.median([r["completeness"][k] for k in common])) if common else None,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "entry": "E08f", "stage": "A", "prereg": "experiments/worldsculpt/PREREG_E08f.md",
        "E": E, "defaults": dataclasses.asdict(default), "rows": table}, indent=1) + "\n")
    print(f"wrote {args.out}")


def sanity(ground_dir: Path, result: dict) -> None:
    """Stage A at defaults must reproduce E08e's automatic kept set."""
    candidates = json.loads((ground_dir / "candidates.json").read_text())["candidates"]
    want = sorted(c["id"] for c in candidates if c["keep"])
    if sorted(result["kept"]) != want:
        raise SystemExit(f"Stage A default does not reproduce E08e: {sorted(result['kept'])} vs {want}")
    print(f"  sanity ok: default reproduces E08e's {len(want)} automatic keeps", flush=True)


# --------------------------------------------------------------------------
# Stage B


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """mesh.pt -> world-space vertices and faces."""
    import torch
    pt = torch.load(path, map_location="cpu", weights_only=False)
    verts = np.asarray(pt["vertices"], dtype=np.float64)
    faces = np.asarray(pt["faces"], dtype=np.int64)
    T = np.asarray(pt["T_canon_to_metric"], dtype=np.float64)
    return verts @ T[:3, :3].T + T[:3, 3], faces


def sample_surface(verts: np.ndarray, faces: np.ndarray, n: int) -> np.ndarray:
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(faces), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[pick]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def fragments(faces: np.ndarray) -> tuple[int, int]:
    """(connected components, components holding at least 1% of faces)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = int(faces.max()) + 1
    a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    _, labels = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    per_face = np.bincount(labels[faces[:, 0]])
    per_face = per_face[per_face > 0]
    return int(len(per_face)), int((per_face >= 0.01 * len(faces)).sum())


def silhouette(points: np.ndarray, cam, w: int, h: int) -> np.ndarray:
    import cv2
    from scipy import ndimage
    xy, z = g.project(points, cam)
    ok = (z > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    img = np.zeros((h, w), np.uint8)
    img[xy[ok, 1].astype(int), xy[ok, 0].astype(int)] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    return ndimage.binary_fill_holes(img)


def box_hull(box, cam, w: int, h: int) -> np.ndarray:
    import cv2
    xy, z = g.project(g.corners(box), cam)
    img = np.zeros((h, w), np.uint8)
    if (z > 0).all():
        cv2.fillConvexPoly(img, cv2.convexHull(xy.astype(np.int32)), 1)
    return img.astype(bool)


def heldout_frames(scene, k: int, box, E: float, p: g.Params) -> list[tuple[str, np.ndarray]]:
    """Frames WorldSculpt never got (outside the stride set) where the object passes E08e's gate."""
    w, h = scene.label_wh
    strided = set(scene.stems[:: p.stride])
    scale = (scene.width * scene.height) / (w * h)
    out = []
    for stem in scene.stems:
        if stem in strided:
            continue
        mask = g.clean_mask(scene.label(stem) == k, p)
        if mask.sum() * scale < p.area_min:
            continue
        kept, _ = g.frame_gate(mask, box, g.scaled(scene.cams[stem], w, h), E, p)
        if kept is not None:
            out.append((stem, kept))
    if len(out) > HELDOUT_FRAMES:
        out = [out[round(i * (len(out) - 1) / (HELDOUT_FRAMES - 1))] for i in range(HELDOUT_FRAMES)]
    return out


def score_object(mesh_path: Path, k: int, box, observed: np.ndarray, scene, E: float,
                 p: g.Params, frames) -> dict:
    from scipy.spatial import cKDTree
    if not mesh_path.exists():
        return {"mesh": False}
    verts, faces = load_mesh(mesh_path)
    surface = sample_surface(verts, faces, SURFACE_SAMPLES)
    tau = TAU * float(np.linalg.norm(np.subtract(box[1], box[0])))
    d_obs = cKDTree(surface).query(observed)[0]
    d_mesh = cKDTree(observed).query(surface[::20])[0]
    w, h = scene.label_wh
    recall, iou, hull_iou = [], [], []
    for stem, mask in frames:
        cam = g.scaled(scene.cams[stem], w, h)
        sil = silhouette(surface, cam, w, h)
        hull = box_hull(box, cam, w, h)
        inter = (sil & mask).sum()
        recall.append(inter / mask.sum())
        iou.append(inter / max(1, (sil | mask).sum()))
        hull_iou.append((hull & mask).sum() / max(1, (hull | mask).sum()))
    n_comp, n_big = fragments(faces)
    return {"mesh": True, "completeness": float(np.mean(d_obs < tau)),
            "accuracy_like": float(np.mean(d_mesh < tau)),
            "silhouette_recall": float(np.median(recall)) if recall else None,
            "silhouette_iou": float(np.median(iou)) if iou else None,
            "box_hull_iou": float(np.median(hull_iou)) if hull_iou else None,
            "heldout_frames": len(frames), "components": n_comp, "components_1pct": n_big,
            "faces": int(len(faces)), "vertices": int(len(verts))}


def score_b(args) -> None:
    configs = json.loads(args.configs.read_text())
    selection = json.loads(args.objects.read_text())
    boxes = {o["id"]: o["aabb_world"] for o in selection["objects"]}
    subset = configs["objects"]
    scene = g.load_scene(args.data_dir, args.model_dir)
    E = selection["E"]
    p = g.Params(**selection["params"])
    observed = np.load(args.observed)
    frames = {k: heldout_frames(scene, k, boxes[k], E, p) for k in subset}
    results = {}
    for cfg in configs["configs"]:
        case = Path(configs["cases"][cfg["case"]])
        results[cfg["name"]] = {
            str(k): score_object(case / f"_sweep_{cfg['name']}" / f"obj{k:02d}" / "mesh.pt",
                                 k, boxes[k], observed[str(k)], scene, E, p, frames[k])
            for k in subset}
        done = sum(r["mesh"] for r in results[cfg["name"]].values())
        print(f"  {cfg['name']}: {done}/{len(subset)} meshes", flush=True)

    readings = read_b(results, subset, args.e08e_recon, args.e08e_log, args.sweep_log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "entry": "E08f", "stage": "B", "prereg": "experiments/worldsculpt/PREREG_E08f.md",
        "objects": subset, "tau_of_box_diagonal": TAU, "configs": configs["configs"],
        "results": results, "readings": readings}, indent=1) + "\n")
    print(f"wrote {args.out}")
    print(json.dumps(readings["summary"], indent=1))


def stage1_record(log: Path) -> dict:
    """Per object: Stage 1 occupied voxels and selected views, from a reconstruct log."""
    import re
    out, obj = {}, None
    for line in log.read_text(errors="replace").replace("\r", "\n").splitlines():
        m = re.search(r"\[batch\] =+ \[\d+/\d+\] (obj\d+)", line)
        if m:
            obj = m.group(1)
        m = re.search(r"views: (\[.*?\]) \(anchor sub=(\d+)\)", line)
        if m and obj:
            out.setdefault(obj, {})["views"] = m.group(1) + f" anchor {m.group(2)}"
        m = re.search(r"coords: (\d+) occupied voxels", line)
        if m and obj:
            out.setdefault(obj, {})["occupied"] = int(m.group(1))
    return out


def read_b(results: dict, subset: list[int], e08e_recon: Path | None,
           e08e_log: Path | None = None, sweep_log: Path | None = None) -> dict:
    """The pre-registered reading rule, per config and metric."""
    seeds = ["default_s42", "default_s0", "default_s1"]
    metrics = ["completeness", "silhouette_iou", "silhouette_recall", "components_1pct", "accuracy_like"]
    readings = {"summary": {}, "sanity": {}}
    # amended check (PREREG_E08f.md): Stage 2 is nondeterministic upstream, so
    # default_s42 must match E08e on the deterministic parts: Stage 1 voxels,
    # views and anchor, and the placement transform
    if e08e_recon is not None and e08e_log and sweep_log and e08e_log.exists() and sweep_log.exists():
        import torch
        theirs_log, mine_log = stage1_record(e08e_log), stage1_record(sweep_log)
        match = {}
        for k in subset:
            name = f"obj{k:02d}"
            a = torch.load(e08e_recon / name / "mesh.pt", map_location="cpu", weights_only=False)
            b = torch.load(e08e_recon.parent / "_sweep_default_s42" / name / "mesh.pt",
                           map_location="cpu", weights_only=False)
            match[k] = (theirs_log.get(name) == mine_log.get(name)
                        and bool(np.allclose(a["T_canon_to_metric"], b["T_canon_to_metric"])))
        readings["sanity"]["default_s42_matches_e08e_stage1_views_T"] = match
    for metric in metrics:
        vals = {s: {k: results[s][str(k)].get(metric) for k in subset} for s in seeds}
        ok = [k for k in subset if all(vals[s][k] is not None for s in seeds)]
        mean = {k: float(np.mean([vals[s][k] for s in seeds])) for k in ok}
        spread = float(np.median([max(vals[s][k] for s in seeds) - min(vals[s][k] for s in seeds) for k in ok])) if ok else None
        per_cfg = {}
        for name, res in results.items():
            if name in seeds:
                continue
            deltas = [res[str(k)][metric] - mean[k] for k in ok if res[str(k)].get(metric) is not None]
            if not deltas:
                per_cfg[name] = {"median_delta": None, "verdict": "no meshes"}
                continue
            md = float(np.median(deltas))
            per_cfg[name] = {"median_delta": md, "n": len(deltas),
                             "verdict": "changes" if spread is not None and abs(md) > spread else "within seed noise"}
        readings["summary"][metric] = {"seed_spread_median": spread,
                                       "default_mean_median": float(np.median(list(mean.values()))) if mean else None,
                                       "configs": per_cfg}
    readings["mesh_counts"] = {name: sum(r["mesh"] for r in res.values()) for name, res in results.items()}
    return readings


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stage-a")
    a.add_argument("--data-dir", required=True, type=Path)
    a.add_argument("--model-dir", required=True, type=Path)
    a.add_argument("--ground", required=True, type=Path, help="E08e propose output (depth cache)")
    a.add_argument("--observed", required=True, type=Path, help="where to write observed surface points")
    a.add_argument("--out", required=True, type=Path)
    b = sub.add_parser("score-b")
    b.add_argument("--data-dir", required=True, type=Path)
    b.add_argument("--model-dir", required=True, type=Path)
    b.add_argument("--configs", required=True, type=Path)
    b.add_argument("--objects", required=True, type=Path)
    b.add_argument("--observed", required=True, type=Path)
    b.add_argument("--e08e-recon", type=Path, help="E08e's _recon, for the reproduction check")
    b.add_argument("--e08e-log", type=Path, help="E08e's run.log (Stage 1 record)")
    b.add_argument("--sweep-log", type=Path, help="default_s42's reconstruct log")
    b.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    {"stage-a": stage_a, "score-b": score_b}[args.cmd](args)


if __name__ == "__main__":
    main()
