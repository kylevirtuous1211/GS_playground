"""E08m: do VGGT's cameras and IGGT's instance features degrade with more frames?

Pre-registered in experiments/iggt/PREREG_E08m.md. Fixed anchor frames, growing
random context; only the number of frames a model sees changes.

    # surflo env (torch, pycolmap, open3d; IGGT patched)
    python -m gs_playground.iggt.view_count check  --out results/iggt/e08m_validity.json
    python -m gs_playground.iggt.view_count vggt   --out results/iggt/e08m_vggt_cameras.json
    python -m gs_playground.iggt.view_count iggt   --out-dir outputs/iggt/view_count
    python -m gs_playground.iggt.view_count gt     --out-dir outputs/iggt/view_count
    # puffin env (sklearn, scipy)
    python -m gs_playground.iggt.view_count analyse --out-dir outputs/iggt/view_count \\
        --out results/iggt/e08m_iggt_instances.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MIPNERF360 = Path.home() / "datasets/mipnerf360/360_v2"
SCANNETPP = Path("/mnt/nchc-2603056/datasets/ScanNet++")
MIP_SCENES = ("bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump")
SPP_SCENES = ("825d228aec", "6115eddb86", "13c3e046d7", "09c1414f1b", "1ada7a0617")
#: pre-registered
VGGT_ANCHORS, VGGT_NS = 8, (8, 16, 24, 32, 48, 64, 96, 128)
IGGT_ANCHORS, IGGT_NS = 4, (4, 8, 12, 16, 24, 32)
SEEDS = (0, 1, 2)
VGGT_TRAINED_MAX, IGGT_TRAINED_MAX = 24, 12
AUC_MAX_DEG = 30
MIN_INSTANCE_PIXELS = 200
HIT_FRACTION_MIN = 0.90


# ------------------------------------------------------------- frame choice
def anchors_and_batches(names: list[str], n_anchors: int, ns: tuple[int, ...]):
    """Anchor names (evenly spaced, the first frame included) and, per (N, seed), the batch
    (anchors + N - n_anchors random others, name order). N above len(names) uses all frames."""
    names = sorted(names)
    idx = np.unique(np.linspace(0, len(names) - 1, n_anchors).round().astype(int))
    anchors = [names[i] for i in idx]
    rest = [n for n in names if n not in set(anchors)]
    batches = {}
    for n in ns:
        n_eff = min(n, len(names))
        for seed in (SEEDS if n_eff > n_anchors else (0,)):
            extra = np.random.default_rng(seed).choice(len(rest), n_eff - n_anchors, replace=False)
            batches[(n_eff, seed)] = sorted(anchors + [rest[i] for i in extra])
    return anchors, batches


# ----------------------------------------------------------- camera metrics
def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    """Angle of rotation matrices [..., 3, 3], degrees."""
    cos = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def pair_errors(R_pred, t_pred, R_gt, t_gt) -> tuple[np.ndarray, np.ndarray]:
    """VGGT's camera evaluation on all pairs of world-to-camera poses [S, 3, 3], [S, 3]:
    relative rotation error and relative translation-direction error (degrees), the
    latter with the sign ambiguity VGGT's evaluation allows (min(e, 180 - e))."""
    i, j = np.triu_indices(len(R_gt), k=1)

    def relative(R, t):
        R_ij = R[i] @ np.swapaxes(R[j], -1, -2)
        t_ij = t[i] - np.einsum("pab,pb->pa", R_ij, t[j])
        return R_ij, t_ij

    Rp, tp = relative(R_pred, t_pred)
    Rg, tg = relative(R_gt, t_gt)
    rot = rotation_angle_deg(Rp @ np.swapaxes(Rg, -1, -2))
    # atan2 of |a x b| and a . b: exact near 0, where arccos of a normalised dot product is not
    trans = np.degrees(np.arctan2(np.linalg.norm(np.cross(tp, tg), axis=1), np.einsum("pa,pa->p", tp, tg)))
    return rot, np.minimum(trans, 180.0 - trans)


def auc(errors: np.ndarray, max_deg: int = AUC_MAX_DEG) -> float:
    """VGGT's AUC@max_deg: mean over 1-degree bins of the fraction of errors below each bound."""
    hist, _ = np.histogram(errors, bins=np.arange(max_deg + 1))
    return float(np.mean(np.cumsum(hist) / len(errors)))


# ---------------------------------------------------------------- loaders
def colmap_poses(scene: Path) -> dict:
    """Image name -> (R world-to-camera, t, fx in full-resolution pixels, full width)."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(scene / "sparse/0"))
    out = {}
    for image in rec.images.values():
        pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        camera = rec.cameras[image.camera_id]
        out[image.name] = (np.asarray(pose.rotation.matrix()), np.asarray(pose.translation),
                           float(camera.focal_length_x), int(camera.width))
    return out


def spp_frames(scene: str) -> list[str]:
    t = json.loads((SCANNETPP / "data" / scene / "dslr/nerfstudio/transforms_undistorted.json").read_text())
    return sorted(f["file_path"] for f in t["frames"] if not f.get("is_bad", False))


def spp_image(scene: str, name: str) -> Path:
    local = Path.home() / "datasets/scannetpp/derived/surflo_inputs" / scene / name
    return local if local.exists() else SCANNETPP / "data" / scene / "dslr/resized_undistorted_images" / name


# -------------------------------------------------------------------- VGGT
def vggt_cameras(model, images, dtype):
    """VGGT's pose encoding exactly as its forward computes it (aggregator under autocast, camera head
    with autocast off), without the depth, point and track heads, which never touch it (Amendment 2)."""
    import torch
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        tokens, _ = model.aggregator(images)
        with torch.amp.autocast("cuda", enabled=False):
            return model.camera_head(tokens)[-1]


def run_vggt(out: Path) -> None:
    import time
    import torch
    from surflo.data.utils import load_and_preprocess_images
    from surflo.nn.vggt.models.vggt import VGGT
    from surflo.nn.vggt.utils.pose_enc import pose_encoding_to_extri_intri
    model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    done = {(r["scene"], r["n"], r["seed"]) for r in rows}
    warm = True   # one untimed forward first, so CUDA and kernel set-up is not charged to a row
    for scene in MIP_SCENES:
        gt = colmap_poses(MIPNERF360 / scene)
        images_dir = MIPNERF360 / scene / "images_4"
        names = sorted(p.name for p in images_dir.iterdir() if p.name in gt)
        anchors, batches = anchors_and_batches(names, VGGT_ANCHORS, VGGT_NS)
        for (n, seed), batch in batches.items():
            if (scene, n, seed) in done:
                continue
            images = load_and_preprocess_images([str(images_dir / b) for b in batch], mode="no_stretch",
                                                target_size=518, rotate_portrait=True).cuda()[None]
            if warm:
                vggt_cameras(model, images, dtype)
                warm = False
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            pose_enc = vggt_cameras(model, images, dtype)
            torch.cuda.synchronize()
            seconds = time.time() - t0
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            pos = [batch.index(a) for a in anchors]
            E = extrinsic[0, pos].float().cpu().numpy()
            fx = intrinsic[0, pos, 0, 0].float().cpu().numpy()
            R_gt = np.stack([gt[a][0] for a in anchors])
            t_gt = np.stack([gt[a][1] for a in anchors])
            rot, trans = pair_errors(E[:, :, :3], E[:, :, 3], R_gt, t_gt)
            # COLMAP's focal mapped through the same preprocessing: the 518-wide resize of the full image
            fx_gt = np.array([gt[a][2] * images.shape[-1] / gt[a][3] for a in anchors])
            rows.append({"scene": scene, "n": n, "seed": seed, "frames": batch, "anchors": anchors,
                         "auc30": auc(np.maximum(rot, trans)), "rot_err_median": float(np.median(rot)),
                         "trans_err_median": float(np.median(trans)),
                         "focal_err_pct_median": float(np.median((fx / fx_gt - 1) * 100)),
                         "seconds": seconds, "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
            print(f"{scene} N={n} seed={seed}: AUC@30 {rows[-1]['auc30']:.3f}, rot {rows[-1]['rot_err_median']:.2f} deg,"
                  f" focal {rows[-1]['focal_err_pct_median']:+.2f}%", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"entry": "E08m", "part": 1, "rows": rows}, indent=1))
            del pose_enc
    out.write_text(json.dumps({"entry": "E08m", "part": 1, "rows": rows,
                               "reading": read_vggt(rows)}, indent=1))


def read_vggt(rows: list[dict]) -> dict:
    """The pre-registered reading of part 1."""
    per = {}
    for r in rows:
        per.setdefault(r["scene"], {}).setdefault(r["n"], []).append(r["auc30"])
    scenes = {}
    for scene, by_n in per.items():
        top = max(by_n)
        at24 = by_n[VGGT_TRAINED_MAX]
        scenes[scene] = {"auc30_median": {str(n): median(v) for n, v in sorted(by_n.items())},
                         "largest_n": top, "range_at_24": max(at24) - min(at24),
                         "declines": median(by_n[top]) < median(at24) - (max(at24) - min(at24))}
    # per N, the median and range over scenes; at the largest N each scene contributes its largest
    # batch (stump: all 125 frames), as pre-registered
    def at(s, n):
        return s["auc30_median"][str(min(n, s["largest_n"]))]
    summary = {str(n): {"median": median(at(s, n) for s in scenes.values()),
                        "range": [min(at(s, n) for s in scenes.values()), max(at(s, n) for s in scenes.values())],
                        "scenes_at_fewer_frames": sorted(k for k, s in scenes.items() if s["largest_n"] < n)}
               for n in VGGT_NS if all(str(min(n, s["largest_n"])) in s["auc30_median"] for s in scenes.values())}
    top_median = median(s["auc30_median"][str(s["largest_n"])] for s in scenes.values())
    declining = sum(s["declines"] for s in scenes.values())
    return {"per_scene": scenes, "median_over_scenes": summary,
            "median_over_scenes_at_largest_n": top_median, "scenes_declining": declining,
            "verdict": "degrades beyond its training range"
            if top_median < summary[str(VGGT_TRAINED_MAX)]["median"] and declining >= 5
            else "no decline measured up to 128 frames"}


# -------------------------------------------------------------------- IGGT
def load_iggt():
    import torch
    sys.path.insert(0, str(ROOT / "third_party/clones/iggt"))
    from iggt.models.vggt import IGGT
    model = IGGT()
    state = {k.removeprefix("module."): v for k, v in torch.load(
        ROOT / "data/models/iggt/iggt_checkpoint.pth", map_location="cpu", mmap=True, weights_only=True).items()}
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def iggt_forward(model, paths: list[Path]):
    import torch
    from iggt.utils.load_fn import load_and_preprocess_images
    from gs_playground.iggt.instances import SIZE, unproject
    from iggt.utils.pose_enc import pose_encoding_to_extri_intri
    images = load_and_preprocess_images([str(p) for p in paths], mode="resize", resize_target_size=SIZE).cuda()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        pred = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"][-1], images.shape[-2:])
    points = unproject(pred["depth"][0, ..., 0].float().cpu().numpy(),
                       extrinsic[0].float().cpu().numpy(), intrinsic[0].float().cpu().numpy())
    feat = torch.nn.functional.normalize(pred["part_feat"][0].float(), dim=1).permute(0, 2, 3, 1)
    return images, feat, points


def run_iggt(out_dir: Path) -> None:
    import numpy as np
    model = load_iggt()
    jobs = [(s, [p.name for p in sorted((MIPNERF360 / s / "images_4").iterdir())],
             lambda s, n: MIPNERF360 / s / "images_4" / n) for s in MIP_SCENES]
    jobs += [(s, spp_frames(s), spp_image) for s in SPP_SCENES]
    for scene, names, path_of in jobs:
        anchors, batches = anchors_and_batches(names, IGGT_ANCHORS, IGGT_NS)
        for (n, seed), batch in batches.items():
            target = out_dir / scene / f"n{n}" / f"seed{seed}.npz"
            if target.exists():
                continue
            images, feat, points = iggt_forward(model, [path_of(scene, b) for b in batch])
            pos = [batch.index(a) for a in anchors]
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, anchors=np.array(anchors), frames=np.array(batch),
                                images=(images[pos].permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8),
                                features=feat[pos].cpu().numpy().astype(np.float16), points=points[pos])
            print(f"{scene} N={n} seed={seed}", flush=True)


# ------------------------------------------------------------ ground truth
def instance_of_vertex(scene_dir: Path) -> np.ndarray:
    """Per mesh vertex: 1 + objectId for objects whose label maps to a top-100 instance class, else 0."""
    meta = SCANNETPP / "metadata"
    classes = set((meta / "semantic_benchmark/top100_instance.txt").read_text().split("\n"))
    with open(meta / "semantic_benchmark/map_benchmark.csv") as f:
        mapping = {r["class"]: r["instance_map_to"] for r in csv.DictReader(f)}
    seg = np.asarray(json.loads((scene_dir / "scans/segments.json").read_text())["segIndices"])
    out = np.zeros(len(seg), dtype=np.int32)
    for group in json.loads((scene_dir / "scans/segments_anno.json").read_text())["segGroups"]:
        label = mapping.get(group["label"], "") or (group["label"] if group["label"] in classes else "")
        if label in classes:
            out[np.isin(seg, group["segments"])] = group["objectId"] + 1
    return out


def render_gt(out_dir: Path) -> None:
    import open3d as o3d
    import trimesh
    from PIL import Image
    from gs_playground.iggt.instances import SIZE
    from gs_playground.surflo.swap import dslr_cameras
    for scene in SPP_SCENES:
        scene_dir = SCANNETPP / "data" / scene
        anchors, _ = anchors_and_batches(spp_frames(scene), IGGT_ANCHORS, (IGGT_ANCHORS,))
        mesh = trimesh.load(scene_dir / "scans/mesh_aligned_0.05.ply", process=False)
        vertex_instance = instance_of_vertex(scene_dir)
        raycast = o3d.t.geometry.RaycastingScene()
        raycast.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                              o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
        poses, K = dslr_cameras(scene_dir)
        w, h = SIZE
        scale = w / K["w"]
        assert abs(h / K["h"] - scale) < 1e-6, "ScanNet++ DSLR is 3:2 like IGGT's 504x336"
        intrinsic = np.array([[K["fl_x"] * scale, 0, K["cx"] * scale], [0, K["fl_y"] * scale, K["cy"] * scale], [0, 0, 1]])
        masks, hit_masks, hits = [], [], []
        for name in anchors:
            rays = raycast.create_rays_pinhole(intrinsic_matrix=o3d.core.Tensor(intrinsic),
                                               extrinsic_matrix=o3d.core.Tensor(np.linalg.inv(poses[Path(name).stem])),
                                               width_px=w, height_px=h)
            prim = raycast.cast_rays(rays)["primitive_ids"].numpy().astype(np.int64)
            hit = prim != o3d.t.geometry.RaycastingScene.INVALID_ID
            mask = np.zeros((h, w), dtype=np.int32)
            mask[hit] = vertex_instance[np.asarray(mesh.faces)[prim[hit], 0]]
            masks.append(mask)
            hit_masks.append(hit)
            hits.append(float(hit.mean()))
        target = out_dir / scene / "gt.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, anchors=np.array(anchors), masks=np.stack(masks), hit=np.stack(hit_masks),
                            hit_fraction=np.array(hits))
        rgb = np.asarray(Image.open(spp_image(scene, anchors[0])).convert("RGB").resize((w, h)))
        palette = (np.random.default_rng(1).random((vertex_instance.max() + 1, 3)) * 255).astype(np.uint8)
        palette[0] = 0
        overlay = (0.5 * rgb + 0.5 * palette[masks[0]]).astype(np.uint8)
        Image.fromarray(np.concatenate([rgb, overlay], 1)).save(out_dir / scene / "gt_overlay.png")
        print(f"{scene}: hit fraction {min(hits):.3f}-{max(hits):.3f}, instances {len(np.unique(np.stack(masks))) - 1}")


# ---------------------------------------------------------------- analysis
def cluster_anchors(npz: Path) -> tuple[np.ndarray, np.ndarray]:
    """IGGT's demo instances on the anchor frames: 3D-smoothed features, HDBSCAN, nearest assignment.
    Returns (features [A, H, W, 8], labels [A, H, W])."""
    from sklearn.cluster import HDBSCAN
    from sklearn.neighbors import NearestNeighbors
    from gs_playground.iggt.instances import (CLUSTER_EPS, CLUSTER_STRIDE, KNN, MIN_CLUSTER_SIZE, MIN_SAMPLES,
                                              SMOOTH_STRIDE)
    data = np.load(npz)
    feats, points = data["features"].astype(np.float32), data["points"]
    a, h, w, c = feats.shape
    # KNN neighbours other than the pixel itself (the demo's knn_graph(loop=False)); not re-normalised, as the demo
    _, idx = NearestNeighbors(n_neighbors=KNN + 1).fit(points.reshape(-1, 3)).kneighbors(
        points[:, ::SMOOTH_STRIDE, ::SMOOTH_STRIDE].reshape(-1, 3))
    smooth = feats.reshape(-1, c)[idx[:, 1:]].mean(1)
    hs, ws = feats[:, ::SMOOTH_STRIDE, ::SMOOTH_STRIDE].shape[1:3]
    smooth = smooth.reshape(a, hs, ws, c)
    step = CLUSTER_STRIDE // SMOOTH_STRIDE
    sample = smooth[:, ::step, ::step].reshape(-1, c)
    scale = CLUSTER_STRIDE ** 2
    labels = HDBSCAN(min_cluster_size=max(2, MIN_CLUSTER_SIZE // scale), min_samples=max(1, MIN_SAMPLES // scale),
                     cluster_selection_epsilon=CLUSTER_EPS).fit_predict(sample)
    if (labels < 0).all():   # the demo's fallback: one instance
        labels[:] = 0
    kept = labels >= 0
    _, nearest = NearestNeighbors(n_neighbors=1).fit(sample[kept]).kneighbors(smooth.reshape(-1, c))
    small = labels[kept][nearest[:, 0]].reshape(a, hs, ws)
    full = np.repeat(np.repeat(small, SMOOTH_STRIDE, 1), SMOOTH_STRIDE, 2)[:, :h, :w]
    return feats, full


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a.ravel(), b.ravel()))


def t_miou(labels: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    """Clusters matched one-to-one to GT instances (Hungarian on IoU over all anchor frames jointly)."""
    from scipy.optimize import linear_sum_assignment
    pred, true = labels[valid], gt[valid]
    gt_ids = [g for g in np.unique(true) if g > 0 and (true == g).sum() >= MIN_INSTANCE_PIXELS]
    pred_ids = np.unique(pred)
    size_p = np.bincount(np.searchsorted(pred_ids, pred), minlength=len(pred_ids))
    iou = np.zeros((len(gt_ids), len(pred_ids)))
    for i, g in enumerate(gt_ids):
        in_g = true == g
        inter = np.bincount(np.searchsorted(pred_ids, pred[in_g]), minlength=len(pred_ids))
        iou[i] = inter / (in_g.sum() + size_p - inter)
    rows, cols = linear_sum_assignment(-iou)
    matched = np.zeros(len(gt_ids))
    matched[rows] = iou[rows, cols]
    return {"t_miou": float(matched.mean()) if len(gt_ids) else float("nan"), "gt_instances": len(gt_ids)}


def centroid_miou(features: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    """Clustering-free (Amendment 1): each GT instance's mean feature over the anchor frames is its centroid;
    every pixel of a counted instance goes to the nearest centroid (cosine); mean IoU over those instances."""
    true = gt[valid]
    feats = features[valid].astype(np.float64)
    ids = [g for g in np.unique(true) if g > 0 and (true == g).sum() >= MIN_INSTANCE_PIXELS]
    if not ids:
        return {"centroid_miou": float("nan")}
    keep = np.isin(true, ids)
    true, feats = true[keep], feats[keep]
    centroids = np.stack([feats[true == g].mean(0) for g in ids])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    assigned = np.asarray(ids)[np.argmax(feats @ centroids.T, axis=1)]
    ious = [((assigned == g) & (true == g)).sum() / ((assigned == g) | (true == g)).sum() for g in ids]
    return {"centroid_miou": float(np.mean(ious))}


def analyse(out_dir: Path, out: Path) -> None:
    rows = []
    for scene in MIP_SCENES + SPP_SCENES:
        ref_feat, ref_labels = cluster_anchors(out_dir / scene / f"n{IGGT_ANCHORS}" / "seed0.npz")
        gt = np.load(out_dir / scene / "gt.npz") if scene in SPP_SCENES else None
        for npz in sorted((out_dir / scene).glob("n*/seed*.npz"), key=lambda p: (int(p.parent.name[1:]), p.name)):
            feat, labels = cluster_anchors(npz)
            row = {"scene": scene, "dataset": "scannetpp" if gt is not None else "mipnerf360",
                   "n": int(npz.parent.name[1:]), "seed": int(npz.stem[4:]),
                   "cosine_to_n4": float((feat * ref_feat).sum(-1).mean()),
                   "ari_to_n4": adjusted_rand_index(labels, ref_labels), "clusters": int(labels.max() + 1)}
            if gt is not None:
                assert list(np.load(npz)["anchors"]) == list(gt["anchors"])
                row.update(t_miou(labels, gt["masks"], gt["hit"]))
                row.update(centroid_miou(feat, gt["masks"], gt["hit"]))
            rows.append(row)
            print(scene, row["n"], row["seed"], {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"entry": "E08m", "part": 2, "rows": rows, "reading": read_iggt(rows)}, indent=1))


def read_quality(rows: list[dict], metric: str) -> dict:
    """The pre-registered 2b rule, for one metric (T-mIoU; Amendment 1 adds centroid mIoU)."""
    per = {}
    for r in rows:
        if r["dataset"] == "scannetpp":
            per.setdefault(r["scene"], {}).setdefault(r["n"], []).append(r[metric])
    scenes = {}
    for scene, by_n in per.items():
        at12, at32 = by_n[IGGT_TRAINED_MAX], by_n[max(IGGT_NS)]
        scenes[scene] = {"median": {str(n): median(v) for n, v in sorted(by_n.items())},
                         "range_at_12": max(at12) - min(at12),
                         "declines": median(at32) < median(at12) - (max(at12) - min(at12))}
    m12 = median(s["median"][str(IGGT_TRAINED_MAX)] for s in scenes.values())
    m32 = median(s["median"][str(max(IGGT_NS))] for s in scenes.values())
    declining = sum(s["declines"] for s in scenes.values())
    return {"per_scene": scenes, "median_at_12": m12, "median_at_32": m32, "scenes_declining": declining,
            "verdict": "degrades beyond its training range" if m32 < m12 and declining >= 4
            else "no decline measured up to 32 frames"}


def read_iggt(rows: list[dict]) -> dict:
    stability = {}
    for r in rows:
        stability.setdefault(r["dataset"], {}).setdefault(r["n"], []).append((r["cosine_to_n4"], r["ari_to_n4"]))
    return {"t_miou": read_quality(rows, "t_miou"), "centroid_miou": read_quality(rows, "centroid_miou"),
            "stability_median": {d: {str(n): {"cosine": median(v[0] for v in vals), "ari": median(v[1] for v in vals)}
                                     for n, vals in sorted(by_n.items())} for d, by_n in stability.items()}}


# ------------------------------------------------------------ validity 1
def check(out: Path) -> None:
    """Validity check 1: the many-frames fix leaves <= 12 frames untouched, and above that the
    instance features are permutation-equivariant in frames 2.. (per-frame heads)."""
    import torch
    model = load_iggt()
    names = sorted(p.name for p in (MIPNERF360 / "garden/images_4").iterdir())
    paths = [MIPNERF360 / "garden/images_4" / n for n in names]
    eight = [paths[i] for i in np.linspace(0, len(paths) - 1, 8).round().astype(int)]
    _, patched, _ = iggt_forward(model, eight)
    point_forward, part_forward = model.point_head.forward, model.part_head.forward
    model.point_head.forward = lambda *a, **k: point_forward(*a, **{**k, "frames_chunk_size": 12})
    model.part_head.forward = lambda *a, **k: part_forward(*a, **{**k, "frames_chunk_size": 12})
    _, unpatched, _ = iggt_forward(model, eight)
    model.point_head.forward, model.part_head.forward = point_forward, part_forward
    sixteen = [paths[i] for i in np.linspace(0, len(paths) - 1, 16).round().astype(int)]
    order = [0] + list(np.random.default_rng(0).permutation(np.arange(1, 16)))
    _, base, _ = iggt_forward(model, sixteen)
    _, permuted, _ = iggt_forward(model, [sixteen[i] for i in order])
    cos = torch.stack([(permuted[k] * base[i]).sum(-1).mean() for k, i in enumerate(order)])
    _, patched_again, _ = iggt_forward(model, eight)   # run-to-run determinism, to read the comparison against
    result = {"eight_frames_bitwise_equal": bool(torch.equal(patched, unpatched)),
              "eight_frames_patched_twice_bitwise_equal": bool(torch.equal(patched, patched_again)),
              "eight_frames_max_abs": float((patched - unpatched).abs().max()),
              "sixteen_frames_runs": True,
              "permutation_min_cosine": float(cos.min()), "permutation_cosines": cos.tolist()}
    result["passes"] = result["eight_frames_bitwise_equal"] and result["permutation_min_cosine"] >= 0.999
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "permutation_cosines"}))
    if not result["passes"]:
        raise SystemExit("validity check 1 failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "vggt"):
        sub.add_parser(name).add_argument("--out", type=Path, required=True)
    for name in ("iggt", "gt"):
        sub.add_parser(name).add_argument("--out-dir", type=Path, required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--out-dir", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"check": lambda: check(args.out), "vggt": lambda: run_vggt(args.out),
     "iggt": lambda: run_iggt(args.out_dir), "gt": lambda: render_gt(args.out_dir),
     "analyse": lambda: analyse(args.out_dir, args.out)}[args.cmd]()


if __name__ == "__main__":
    main()
