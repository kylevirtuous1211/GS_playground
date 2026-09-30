"""E08e: ground a captured scene for WorldSculpt, per-object masks and boxes.

WorldSculpt (arXiv 2609.05416) consumes posed frames, a mask per object per
frame and one world-space box per object, and puts recovering the last two
"outside the scope of this work". This module builds them for a scene that
already has COLMAP poses, a trained 3DGS, and instance label maps that some
earlier tool associated across views (for the NCHC sofa: HQ-SAM associated
by Inpaint360GS in the EditReadyGS project).

    propose  render depth from the 3DGS, estimate a box per label id, filter
             with a reason for every rejection, and write a review sheet
    build    the WorldSculpt scene directory from the reviewed selection
    check    how many declared objects survive each upstream stage

**Boxes.** Every label id is back-projected through rendered expected depth,
voxelised, and only voxels seen by several *different* frames are kept, so a
frame or two where the association put the id on the wrong thing cannot
stretch the box. The box is the best-supported connected component.

**Frames.** Everything is keyed by file stem (`frame_000131`), never by
position: `cameras.json` lists the held-out cameras first, and the numbering
has gaps where COLMAP did not register a frame.

    python -m gs_playground.worldsculpt.ground propose \
        --data-dir "$NCHC_SOFA_RUN/data/editreadygs_video/nchc_sofa_20260727_143647" \
        --model-dir "$NCHC_SOFA_RUN/output/editreadygs_video/nchc_sofa_20260727_143647/3dgs_output" \
        --out outputs/worldsculpt/NCHC/ground
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: OpenCV camera axes -> OpenGL/Blender, and back (the matrix is its own inverse)
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

#: box edges as corner-index pairs, corners ordered by (x, y, z) bits
EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]


@dataclass(frozen=True)
class Params:
    """Every threshold in one place; E08f sweeps these one at a time."""
    speck_frac: float = 0.02       # drop 8-connected pieces below this share of the frame's mask
    largest_only: bool = False     # keep only the largest piece instead
    alpha_min: float = 0.95        # ED divides by alpha; faint layers lie
    erode_px: int = 2              # at label resolution (4 px at full), avoids boundary depth
    points_per_frame: int = 1500
    voxel_div: int = 400           # voxel = E / voxel_div
    support_frac: float = 0.05     # a voxel needs this share of the id's frames ...
    support_min: int = 3           # ... and at least this many distinct frames
    min_voxels: int = 30
    struct_extent: float = 0.4     # max box side above this * E: structure
    struct_border: float = 0.6     # share of frames touching >= 2 image borders: structure
    tiny_frac: float = 0.01        # max box side below this * E: fragment
    split_flag: float = 0.30       # second component / first above this: flag, two things share an id
    stride: int = 3                # frames handed to WorldSculpt
    min_views: int = 8             # after every gate, at `stride`
    area_min: int = 910            # full-res pixels, upstream's own floor at 1272x715
    clip_gate: bool = True
    clip_margin: float = 0.15      # projected box bbox grown by this per side
    clip_drop: float = 0.10        # drop the frame if clipping removes more than this
    min_consistent: float = 0.5    # share of visible frames that must survive the clip gate
    in_image: float = 0.70         # share of the projected box bbox inside the frame
    near_frac: float = 0.02        # every box corner this far (* E) in front of the camera


# --------------------------------------------------------------------------
# scene


@dataclass
class Scene:
    data_dir: Path
    model_dir: Path
    stems: list[str]
    cams: dict            # stem -> gs.render.Camera at full resolution
    train_stems: list[str]
    width: int
    height: int
    label_wh: tuple[int, int]
    _labels: dict = dataclasses.field(default_factory=dict, repr=False)

    @property
    def images(self) -> Path:
        return self.data_dir / "images"

    @property
    def labels(self) -> Path:
        return self.data_dir / "associated_hqsam"

    @property
    def ply(self) -> Path:
        return self.model_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"

    def label(self, stem: str) -> np.ndarray:
        # read once: the stride gate revisits every frame for every id
        if stem not in self._labels:
            from PIL import Image
            self._labels[stem] = np.asarray(Image.open(self.labels / f"{stem}.png"))
        return self._labels[stem]

    def image(self, stem: str) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(self.images / f"{stem}.jpg").convert("RGB"))


def load_scene(data_dir: Path, model_dir: Path) -> Scene:
    from ..gs.render import load_cameras

    entries = json.loads((model_dir / "cameras.json").read_text())
    cams = {Path(c.name).stem: c for c in load_cameras(model_dir / "cameras.json")}
    image_stems = {p.stem for p in (data_dir / "images").glob("frame_*.jpg")}
    label_stems = {p.stem for p in (data_dir / "associated_hqsam").glob("frame_*.png")}
    if not (image_stems == label_stems == set(cams)):
        raise SystemExit(
            f"frame sets disagree: images {len(image_stems)}, labels "
            f"{len(label_stems)}, cameras {len(cams)}; "
            f"only in cameras: {sorted(set(cams) - label_stems)[:5]}")
    widths = {(c.width, c.height) for c in cams.values()}
    if len(widths) != 1:
        raise SystemExit(f"cameras disagree on resolution: {widths}")
    (width, height), = widths
    # INRIA writes the held-out cameras first; with eval=True every 8th frame
    # in sorted order is held out, which is what the first block holds
    n_test = sum(1 for i, s in enumerate(sorted(cams)) if i % 8 == 0)
    test = {Path(e["img_name"]).stem for e in entries[:n_test]}
    stems = sorted(cams)
    from PIL import Image
    label_w, label_h = Image.open(data_dir / "associated_hqsam" / f"{stems[0]}.png").size
    return Scene(data_dir, model_dir, stems, cams,
                 [s for s in stems if s not in test], width, height,
                 (label_w, label_h))


def scaled(cam, width: int, height: int):
    """The same camera at another resolution; x and y scale separately."""
    from ..gs.render import Camera
    sx, sy = width / cam.width, height / cam.height
    K = cam.K.clone()
    K[0] *= sx
    K[1] *= sy
    return Camera(cam.name, cam.viewmat, K, width, height)


def c2w(cam) -> np.ndarray:
    """OpenCV camera-to-world."""
    return np.linalg.inv(cam.viewmat.double().numpy())


def scene_extent(data_dir: Path) -> float:
    """E: the largest 2-98 percentile span of the sparse COLMAP points."""
    from plyfile import PlyData
    v = PlyData.read(str(data_dir / "sparse" / "0" / "points3D.ply"))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    lo, hi = np.percentile(xyz, 2, 0), np.percentile(xyz, 98, 0)
    return float((hi - lo).max())


# --------------------------------------------------------------------------
# masks and projection


def clean_mask(mask: np.ndarray, p: Params) -> np.ndarray:
    """Drop specks; occlusion legitimately splits an object, so keep pieces."""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    if p.largest_only:
        return lab == (int(np.argmax(sizes)) + 1)
    keep = np.flatnonzero(sizes >= p.speck_frac * sizes.sum()) + 1
    return np.isin(lab, keep)


def corners(box) -> np.ndarray:
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    return np.array([[(hi if (i >> a) & 1 else lo)[a] for a in range(3)]
                     for i in range(8)])


def project(points: np.ndarray, cam) -> tuple[np.ndarray, np.ndarray]:
    """World points -> (pixel xy, camera z) for a camera at its own resolution."""
    w2c = cam.viewmat.double().numpy()
    cam_pts = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam_pts[:, 2]
    K = cam.K.double().numpy()
    xy = (cam_pts[:, :2] / np.maximum(z, 1e-9)[:, None]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return xy, z


def frame_gate(mask: np.ndarray, box, cam, E: float, p: Params):
    """(kept mask, reason). The mask is at `cam`'s resolution.

    Rejects a frame when the box is not wholly in front of the camera, when
    too little of it is in view (a truncated close-up would win upstream's
    anchor vote, which is the largest raw mask), or when clipping the mask to
    the box removes enough to say the id meant something else here.
    """
    h, w = mask.shape
    xy, z = project(corners(box), cam)
    if (z < p.near_frac * E).any():
        return None, "behind"
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    area = max((x1 - x0) * (y1 - y0), 1e-9)
    inside = max(0.0, min(x1, w) - max(x0, 0)) * max(0.0, min(y1, h) - max(y0, 0))
    if inside / area < p.in_image:
        return None, "truncated"
    if p.clip_gate:
        mx, my = p.clip_margin * (x1 - x0), p.clip_margin * (y1 - y0)
        ys, xs = np.nonzero(mask)
        out = (xs < x0 - mx) | (xs > x1 + mx) | (ys < y0 - my) | (ys > y1 + my)
        if len(xs) and out.mean() > p.clip_drop:
            return None, "clipped"
        mask = mask.copy()
        mask[ys[out], xs[out]] = False
    return mask, "ok"


def border_touches(mask: np.ndarray) -> int:
    return int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())


# --------------------------------------------------------------------------
# depth cache and per-id evidence


def cache_depth(scene: Scene, cloud, out: Path) -> None:
    """Expected depth + alpha at label resolution, one npz per frame."""
    from ..gs.render import render_depth
    out.mkdir(parents=True, exist_ok=True)
    w, h = scene.label_wh
    for stem in scene.stems:
        path = out / f"{stem}.npz"
        if path.exists():
            continue
        depth, alpha = render_depth(cloud, scaled(scene.cams[stem], w, h))
        np.savez_compressed(path, depth=depth.cpu().numpy().astype(np.float32),
                            alpha=(alpha.clamp(0, 1) * 255).round().byte().cpu().numpy())


def gather(scene: Scene, depth_dir: Path, p: Params) -> dict:
    """Per id: frames it appears in, mask statistics, back-projected points."""
    from scipy import ndimage
    w, h = scene.label_wh
    ys, xs = np.mgrid[0:h, 0:w]
    evidence: dict[int, dict] = {}
    erode = np.ones((2 * p.erode_px + 1,) * 2, bool) if p.erode_px else None
    for fi, stem in enumerate(scene.stems):
        label = scene.label(stem)
        d = np.load(depth_dir / f"{stem}.npz")
        depth, alpha = d["depth"], d["alpha"].astype(np.float32) / 255
        cam = scaled(scene.cams[stem], w, h)
        K = cam.K.double().numpy()
        pose = c2w(cam)
        for k in np.unique(label):
            k = int(k)
            mask = clean_mask(label == k, p)
            if not mask.any():
                continue
            ev = evidence.setdefault(k, {"frames": [], "area": [], "touch2": [],
                                         "pieces": [], "pts": [], "pfi": []})
            ev["frames"].append(fi)
            ev["area"].append(int(mask.sum()))
            ev["touch2"].append(border_touches(mask) >= 2)
            ev["pieces"].append(int(ndimage.label(mask)[1]))
            core = ndimage.binary_erosion(mask, erode) if erode is not None else mask
            valid = core & (alpha > p.alpha_min) & (depth > 0)
            n = int(valid.sum())
            if n == 0:
                continue
            idx = np.flatnonzero(valid)
            if n > p.points_per_frame:
                idx = np.random.default_rng(fi * 1000 + k).choice(idx, p.points_per_frame, replace=False)
            u, v, z = xs.flat[idx] + 0.5, ys.flat[idx] + 0.5, depth.flat[idx]
            cam_pts = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
            ev["pts"].append((cam_pts @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
            ev["pfi"].append(np.full(len(idx), fi, np.int32))
        if fi % 50 == 0:
            print(f"  gathered {fi + 1}/{len(scene.stems)} frames", flush=True)
    for ev in evidence.values():
        ev["pts"] = np.concatenate(ev["pts"]) if ev["pts"] else np.zeros((0, 3), np.float32)
        ev["pfi"] = np.concatenate(ev["pfi"]) if ev["pfi"] else np.zeros(0, np.int32)
    return evidence


def estimate_box(pts: np.ndarray, pfi: np.ndarray, n_frames: int, E: float, p: Params) -> dict:
    """Best-supported connected component of multi-frame voxels -> AABB."""
    from scipy import ndimage
    out = {"box": None, "voxels": 0, "split": 0.0, "support_needed": 0}
    if len(pts) == 0:
        return out
    vox = E / p.voxel_div
    keys = np.floor(pts / vox).astype(np.int64)
    origin = keys.min(0)
    keys -= origin
    dims = keys.max(0) + 1
    flat = np.ravel_multi_index(keys.T, dims)
    pairs = np.unique(flat * (int(pfi.max()) + 1) + pfi)
    cells, support = np.unique(pairs // (int(pfi.max()) + 1), return_counts=True)
    need = max(p.support_min, math.ceil(p.support_frac * n_frames))
    out["support_needed"] = need
    keep = support >= need
    cells, support = cells[keep], support[keep]
    if len(cells) < p.min_voxels:
        out["voxels"] = int(len(cells))
        return out
    grid = np.zeros(int(np.prod(dims)), bool)
    grid[cells] = True
    lab, n = ndimage.label(grid.reshape(dims), structure=np.ones((3, 3, 3)))
    comp = lab.ravel()[cells]
    weight = np.bincount(comp, weights=support, minlength=n + 1)[1:]
    order = np.argsort(weight)[::-1]
    best = order[0] + 1
    out["split"] = float(weight[order[1]] / weight[order[0]]) if n > 1 else 0.0
    chosen = cells[comp == best]
    centres = (np.stack(np.unravel_index(chosen, dims), 1) + origin + 0.5) * vox
    lo = np.percentile(centres, 1, 0) - vox
    hi = np.percentile(centres, 99, 0) + vox
    out.update(box=[lo.tolist(), hi.tolist()], voxels=int(len(chosen)))
    return out


def views_at_stride(scene: Scene, k: int, box, E: float, p: Params, frames: list[int]):
    """Gate every frame the id appears in; count survivors at the run stride."""
    w, h = scene.label_wh
    strided = set(range(0, len(scene.stems), p.stride))
    counts = {"ok": 0, "behind": 0, "truncated": 0, "clipped": 0, "small": 0}
    kept_strided, kept_all = [], []
    scale = (scene.width * scene.height) / (w * h)
    for fi in frames:
        stem = scene.stems[fi]
        mask = clean_mask(scene.label(stem) == k, p)
        if mask.sum() * scale < p.area_min:
            counts["small"] += 1
            continue
        kept, reason = frame_gate(mask, box, scaled(scene.cams[stem], w, h), E, p)
        counts[reason] += 1
        if kept is None or kept.sum() * scale < p.area_min:
            continue
        kept_all.append(fi)
        if fi in strided:
            kept_strided.append(fi)
    return kept_strided, kept_all, counts


def decide(k: int, ev: dict, est: dict, gates, E: float, p: Params) -> tuple[bool, str, list[str]]:
    flags = []
    if est["split"] >= p.split_flag:
        flags.append(f"split {est['split']:.2f}")
    if k == 0:
        return False, "unlabelled", flags
    if est["box"] is None:
        return False, f"box_failed ({est['voxels']} voxels)", flags
    extent = np.subtract(est["box"][1], est["box"][0])
    border = float(np.mean(ev["touch2"]))
    if extent.max() > p.struct_extent * E:
        return False, f"structure (extent {extent.max() / E:.2f}E)", flags
    if border > p.struct_border:
        return False, f"structure (2-border {border:.2f})", flags
    if extent.max() < p.tiny_frac * E:
        return False, f"tiny ({extent.max() / E:.3f}E)", flags
    strided, kept_all, counts = gates
    # only the clip gate says the id meant something else in a frame; a frame
    # that cuts a large object off, or has the camera inside its box, is just
    # not a usable view
    judged = counts["ok"] + counts["clipped"]
    if judged and counts["ok"] / judged < p.min_consistent:
        return False, f"inconsistent ({counts['clipped']}/{judged} frames clipped)", flags
    if len(strided) < p.min_views:
        return False, f"too_few_views ({len(strided)} at stride {p.stride})", flags
    return True, "auto: kept", flags


# --------------------------------------------------------------------------
# propose


def alignment_checks(scene: Scene, cloud, depth_dir: Path, E: float) -> dict:
    """Catch a frame, intrinsics or pose-convention mismatch before anything.

    RGB: the render of frame i against photo i, and against photo i+15 (null).
    Depth: sparse COLMAP points projected into camera i, compared with the
    rendered depth of camera i and of camera i+15 (null).
    """
    import torch
    from PIL import Image
    from plyfile import PlyData
    from ..gs.render import psnr, render
    w, h = scene.label_wh

    def photo(stem):
        small = Image.fromarray(scene.image(stem)).resize((w, h))
        return torch.from_numpy(np.asarray(small)).float().cuda() / 255

    v = PlyData.read(str(scene.data_dir / "sparse" / "0" / "points3D.ply"))["vertex"]
    sparse = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    picks = scene.train_stems[:: max(1, len(scene.train_stems) // 10)][:10]
    rows = []
    for stem in picks:
        i = scene.stems.index(stem)
        other = scene.stems[(i + 15) % len(scene.stems)]
        cam = scaled(scene.cams[stem], w, h)
        img = render(cloud, cam)
        xy, z = project(sparse, cam)
        inb = (z > 0.02 * E) & (xy[:, 0] >= 0) & (xy[:, 0] < w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < h - 1)
        px, py = xy[inb, 0].astype(int), xy[inb, 1].astype(int)

        def agree(depth_stem):
            dep = np.load(depth_dir / f"{depth_stem}.npz")["depth"]
            return float(np.mean(np.abs(dep[py, px] / z[inb] - 1) < 0.03))
        rows.append({"frame": stem, "psnr": psnr(img, photo(stem)),
                     "psnr_null": psnr(img, photo(other)),
                     "depth_agree": agree(stem), "depth_agree_null": agree(other),
                     "n_sparse": int(inb.sum())})
    summary = {k: float(np.median([r[k] for r in rows]))
               for k in ("psnr", "psnr_null", "depth_agree", "depth_agree_null")}
    return {"frames": rows, "median": summary}


def propose(args) -> None:
    from ..gs.ply import load_ply
    scene = load_scene(args.data_dir, args.model_dir)
    p = Params()
    E = scene_extent(args.data_dir)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(scene.stems)} frames, {len(scene.train_stems)} trained, "
          f"{scene.width}x{scene.height}, labels {scene.label_wh}, E = {E:.2f}")

    cloud = load_ply(scene.ply)
    depth_dir = out / "depth"
    cache_depth(scene, cloud, depth_dir)
    checks = alignment_checks(scene, cloud, depth_dir, E)
    (out / "alignment.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("alignment (median):", {k: round(v, 3) for k, v in checks["median"].items()})

    evidence = gather(scene, depth_dir, p)
    candidates = []
    for k in sorted(evidence):
        ev = evidence[k]
        est = estimate_box(ev["pts"], ev["pfi"], len(ev["frames"]), E, p)
        gates = (views_at_stride(scene, k, est["box"], E, p, ev["frames"])
                 if est["box"] is not None and k != 0 else ([], [], {"small": 0}))
        keep, reason, flags = decide(k, ev, est, gates, E, p)
        extent = (np.subtract(est["box"][1], est["box"][0]).tolist()
                  if est["box"] is not None else None)
        candidates.append({
            "id": k, "keep": keep, "reason": reason, "flags": flags,
            "aabb_world": est["box"], "extent": extent,
            "extent_over_E": max(extent) / E if extent else None,
            "n_frames": len(ev["frames"]),
            "n_views": len(gates[0]), "n_views_stride1": len(gates[1]),
            "gate_counts": gates[2],
            "two_border_share": float(np.mean(ev["touch2"])),
            "median_area_px": int(np.median(ev["area"])),
            "median_pieces": float(np.median(ev["pieces"])),
            "voxels": est["voxels"], "split": est["split"],
            "view_frames": [scene.stems[i] for i in gates[0]],
        })
    record = {"params": dataclasses.asdict(p), "E": E, "n_frames": len(scene.stems),
              "data_dir": str(args.data_dir), "model_dir": str(args.model_dir),
              "alignment": checks["median"], "candidates": candidates}
    (out / "candidates.json").write_text(json.dumps(record, indent=1) + "\n")
    kept = [c for c in candidates if c["keep"]]
    print(f"{len(candidates)} ids, {len(kept)} kept: {[c['id'] for c in kept]}")
    calibration(candidates)
    review_sheet(scene, record, out, E)


def calibration(candidates: list[dict]) -> None:
    """The thresholds must reject walls/floor/ceiling and keep named furniture."""
    by_id = {c["id"]: c for c in candidates}
    must_reject, must_keep = {3: "walls", 24: "floor", 45: "ceiling"}, \
        {5: "L-sofa", 67: "ottomans", 224: "back sofa", 22: "TV"}
    for k, name in must_reject.items():
        c = by_id.get(k)
        ok = c is not None and not c["keep"]
        print(f"  calibration {'ok ' if ok else 'BAD'} reject {k:3d} {name:10s} -> {c and c['reason']}")
    for k, name in must_keep.items():
        c = by_id.get(k)
        ok = c is not None and c["keep"]
        print(f"  calibration {'ok ' if ok else 'BAD'} keep   {k:3d} {name:10s} -> {c and c['reason']}")


# --------------------------------------------------------------------------
# review sheet


def draw_box(img: np.ndarray, box, cam, color=(255, 0, 255)) -> None:
    import cv2
    xy, z = project(corners(box), cam)
    for a, b in EDGES:
        if z[a] <= 0 or z[b] <= 0:
            continue
        cv2.line(img, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), color, 1, cv2.LINE_AA)


def tile(scene: Scene, c: dict, E: float) -> np.ndarray:
    """Three views of one id: mask tinted, projected box in magenta."""
    import cv2
    from PIL import Image
    w, h = scene.label_wh
    frames = c["view_frames"] or []
    if not frames:
        return None
    # largest mask first, then the 1/3 and 2/3 points of the list
    areas = [(scene.label(s) == c["id"]).sum() for s in frames]
    picks = [frames[int(np.argmax(areas))], frames[len(frames) // 3], frames[2 * len(frames) // 3]]
    panels = []
    for stem in picks:
        img = np.asarray(Image.fromarray(scene.image(stem)).resize((w, h))).copy()
        mask = scene.label(stem) == c["id"]
        img[mask] = (0.45 * img[mask] + 0.55 * np.array([0, 200, 255])).astype(np.uint8)
        if c["aabb_world"] is not None:
            draw_box(img, c["aabb_world"], scaled(scene.cams[stem], w, h))
        panels.append(cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA))
    return np.concatenate(panels, 1)


def review_sheet(scene: Scene, record: dict, out: Path, E: float) -> None:
    """tiles/, index.html and a top-down overview, kept ids first."""
    import cv2
    tiles = out / "tiles"
    if tiles.exists():
        shutil.rmtree(tiles)
    tiles.mkdir()
    rows = []
    shown = sorted((c for c in record["candidates"] if c["n_views"] > 0),
                   key=lambda c: (not c["keep"], -c["n_views"]))
    for c in shown:
        t = tile(scene, c, E)
        if t is None:
            continue
        cv2.imwrite(str(tiles / f"id{c['id']:03d}.jpg"), t[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        ext = f"{c['extent_over_E']:.3f}E" if c["extent_over_E"] else "-"
        flags = f" · {html.escape(', '.join(c['flags']))}" if c["flags"] else ""
        rows.append(
            f'<figure class="{"keep" if c["keep"] else "drop"}"><img src="tiles/id{c["id"]:03d}.jpg" loading="lazy">'
            f'<figcaption><b>id {c["id"]}</b> · {html.escape(c["reason"])} · {c["n_views"]} views · {ext}{flags}</figcaption></figure>')
    n_keep = sum(c["keep"] for c in record["candidates"])
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>E08e grounding review</title>'
        '<style>body{font:13px system-ui;margin:16px;background:#f4f4f2}'
        'figure{display:inline-block;margin:6px;background:#fff;border:3px solid #ccc;vertical-align:top}'
        'figure.keep{border-color:#2a9d4a}figure img{display:block;max-width:960px}'
        'figcaption{padding:4px 6px}</style>'
        f'<h1>E08e grounding review</h1><p>{n_keep} kept of {len(record["candidates"])} ids '
        f'(green border = kept by the automatic filter). Tiles show 3 frames with at least one '
        f'surviving view: mask tinted, estimated box projected in magenta. '
        f'<a href="overview_topdown.png">top-down overview</a></p>' + "\n".join(rows))
    overview(record, out, E)
    print(f"review sheet: {out / 'index.html'} ({len(rows)} tiles)")


def overview(record: dict, out: Path, E: float) -> None:
    """Sparse points and every box, looking down the scene's up axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plyfile import PlyData
    v = PlyData.read(str(Path(record["data_dir"]) / "sparse" / "0" / "points3D.ply"))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(xyz[:, 0], xyz[:, 2], s=0.2, c="#bbbbbb")
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    for c in record["candidates"]:
        if c["aabb_world"] is None or (not c["keep"] and not c["reason"].startswith(("too_few", "inconsistent", "tiny"))):
            continue
        (x0, _, z0), (x1, _, z1) = c["aabb_world"]
        color = "#2a9d4a" if c["keep"] else "#d62728"
        ax.add_patch(plt.Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec=color, lw=1))
        ax.text(x0, z0, str(c["id"]), color=color, fontsize=7)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[2], hi[2])
    ax.set_aspect("equal")
    ax.set_xlabel("x (COLMAP units)")
    ax.set_ylabel("z (COLMAP units)")
    ax.set_title("kept boxes (green) and small-object rejections (red), top-down; up is about -y")
    fig.savefig(out / "overview_topdown.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("propose")
    pr.add_argument("--data-dir", required=True, type=Path)
    pr.add_argument("--model-dir", required=True, type=Path)
    pr.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    {"propose": propose}[args.cmd](args)


if __name__ == "__main__":
    main()
