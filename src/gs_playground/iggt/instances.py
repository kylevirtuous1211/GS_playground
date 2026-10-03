"""IGGT's instance features on a scene's views, rendered for the eye. Exploratory.

    # surflo env (torch, xformers, hydra; IGGT patched by third_party/patches/iggt_inference-shims.patch)
    python -m gs_playground.iggt.instances infer --folder <images> --n-images 12 --out <npz>
    # puffin env (sklearn): PCA colours and HDBSCAN clusters as IGGT's demo makes them
    python -m gs_playground.iggt.instances render --npz <npz> --out <dir>

IGGT (ICLR 2026) predicts an 8-dim, L2-normalised instance feature per pixel
from VGGT's aggregator layers 4/11/17/23 plus its point head. Its own demo runs
<= 12 views stretched to 504x336 (the window attention needs an even patch
grid), smooths the features over 3D neighbours (k = 20) and clusters them with
HDBSCAN; both are done here the same way, on a pixel subsample to fit sklearn.
The weights are loaded strictly (the upstream demo uses strict=False).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CLONE = ROOT / "third_party/clones/iggt"
CKPT = ROOT / "data/models/iggt/iggt_checkpoint.pth"
#: IGGT's demo settings (demo.py: DEFAULT_IMAGE_SIZE, CLUSTERING_CONFIG)
SIZE = (504, 336)
MAX_VIEWS = 12
KNN = 20
MIN_CLUSTER_SIZE, MIN_SAMPLES, CLUSTER_EPS = 500, 100, 0.06   # eps: a feature distance, not scaled
#: pixel strides that keep the CPU smoothing and clustering tractable
SMOOTH_STRIDE, CLUSTER_STRIDE = 2, 4


def unproject(depth: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """VGGT's unprojection (IGGT's copy imports its whole misc.py): depth [S, H, W] with
    world-to-camera extrinsics [S, 3, 4] and intrinsics [S, 3, 3] -> world points [S, H, W, 3]."""
    v, u = np.mgrid[0:depth.shape[1], 0:depth.shape[2]]
    out = []
    for d, E, K in zip(depth, extrinsic, intrinsic):
        cam = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
        out.append((cam - E[:, 3]) @ E[:, :3])   # R^T (x - t)
    return np.stack(out)


def infer(folder: Path, n_images: int, out: Path) -> None:
    import torch
    sys.path.insert(0, str(CLONE))
    from iggt.models.vggt import IGGT
    from iggt.utils.load_fn import load_and_preprocess_images
    from iggt.utils.pose_enc import pose_encoding_to_extri_intri
    assert n_images <= MAX_VIEWS, f"IGGT's instance head breaks above {MAX_VIEWS} views"
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    paths = [paths[i] for i in np.unique(np.linspace(0, len(paths) - 1, n_images).round().astype(int))]
    model = IGGT()
    state = {k.removeprefix("module."): v for k, v in
             torch.load(CKPT, map_location="cpu", mmap=True, weights_only=True).items()}
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    images = load_and_preprocess_images([str(p) for p in paths], mode="resize",
                                        resize_target_size=SIZE).cuda()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        pred = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"][-1], images.shape[-2:])
    depth = pred["depth"][0].float().cpu().numpy()
    points = unproject(depth[..., 0], extrinsic[0].float().cpu().numpy(), intrinsic[0].float().cpu().numpy())
    feat = torch.nn.functional.normalize(pred["part_feat"][0].float(), dim=1).permute(0, 2, 3, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, names=np.array([p.name for p in paths]),
                        images=(images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8),
                        features=feat.cpu().numpy().astype(np.float16), points=points.astype(np.float32),
                        depth_conf=pred["depth_conf"][0].float().cpu().numpy())
    print(f"{len(paths)} views at {SIZE[0]}x{SIZE[1]} -> {out}")


def pca_colours(features: np.ndarray) -> np.ndarray:
    """IGGT's apply_pca_colormap: one PCA over every pixel of every view, 2-98% stretch per channel."""
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    flat = flat - flat.mean(0)
    _, _, vt = np.linalg.svd(flat[:: max(1, len(flat) // 200_000)], full_matrices=False)
    colours = flat @ vt[:3].T
    lo, hi = np.percentile(colours, [2, 98], axis=0)
    return np.clip((colours - lo) / np.maximum(hi - lo, 1e-9), 0, 1).reshape(*features.shape[:-1], 3)


def render(npz: Path, out: Path) -> None:
    from PIL import Image
    from sklearn.cluster import HDBSCAN
    from sklearn.neighbors import NearestNeighbors
    data = np.load(npz)
    images, features, points = data["images"], data["features"].astype(np.float32), data["points"]
    n, h, w, c = features.shape
    # 3D smoothing, as IGGT's demo: average each pixel's feature over its k nearest 3D neighbours
    sub = (slice(None), slice(None, None, SMOOTH_STRIDE), slice(None, None, SMOOTH_STRIDE))
    xyz, feat = points[sub].reshape(-1, 3), features[sub].reshape(-1, c)
    # KNN neighbours other than the point itself, as the demo's knn_graph(loop=False); not re-normalised
    _, idx = NearestNeighbors(n_neighbors=KNN + 1).fit(xyz).kneighbors(xyz)
    smooth = feat[idx[:, 1:]].mean(1)
    hs, ws = features[sub].shape[1:3]
    smooth = smooth.reshape(n, hs, ws, c)
    # HDBSCAN on a coarser subsample; demo sizes scaled by the pixel fraction; every pixel then
    # (noise included, as the demo does) takes the label of the nearest clustered sample
    step = CLUSTER_STRIDE // SMOOTH_STRIDE
    sample = smooth[:, ::step, ::step].reshape(-1, c)
    scale = (SMOOTH_STRIDE * step) ** 2
    labels = HDBSCAN(min_cluster_size=max(2, MIN_CLUSTER_SIZE // scale), min_samples=max(1, MIN_SAMPLES // scale),
                     cluster_selection_epsilon=CLUSTER_EPS).fit_predict(sample)
    if (labels < 0).all():   # the demo's fallback: one instance
        labels[:] = 0
    kept = labels >= 0
    _, nearest = NearestNeighbors(n_neighbors=1).fit(sample[kept]).kneighbors(smooth.reshape(-1, c))
    label_maps = labels[kept][nearest[:, 0]].reshape(n, hs, ws)
    palette = (np.random.default_rng(0).random((labels.max() + 1, 3)) * 200 + 55).astype(np.uint8)
    pca = (pca_colours(features) * 255).astype(np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i, name in enumerate(data["names"]):
        clusters = Image.fromarray(palette[label_maps[i]]).resize((w, h), Image.Resampling.NEAREST)
        column = np.concatenate([images[i], pca[i], np.asarray(clusters)], axis=0)
        Image.fromarray(column).save(out / f"{Path(str(name)).stem}.png")
        tiles.append(column)
    sheet = np.concatenate([np.concatenate(tiles[:6], 1), np.concatenate(tiles[6:12], 1)], 0) \
        if len(tiles) == 12 else np.concatenate(tiles, 1)
    Image.fromarray(sheet).save(out / "contact_sheet.png")
    summary = {"views": [str(x) for x in data["names"]], "size": [w, h], "clusters": int(labels.max() + 1),
               "noise_fraction_of_sample": float((~kept).mean()), "knn": KNN,
               "smooth_stride": SMOOTH_STRIDE, "cluster_stride": CLUSTER_STRIDE,
               "min_cluster_size": max(2, MIN_CLUSTER_SIZE // scale), "min_samples": max(1, MIN_SAMPLES // scale),
               "cluster_selection_epsilon": CLUSTER_EPS}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--folder", type=Path, required=True)
    i.add_argument("--n-images", type=int, default=MAX_VIEWS)
    i.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("render")
    r.add_argument("--npz", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "infer":
        infer(args.folder, args.n_images, args.out)
    else:
        render(args.npz, args.out)


if __name__ == "__main__":
    main()
