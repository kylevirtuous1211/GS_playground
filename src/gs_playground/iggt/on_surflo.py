"""E08n (exploratory): instance and class labels on SuRFLo's Gaussians from one encoding.

SuRFLo runs with IGGT's backbone and saves the tokens of aggregator layers
4/11/17/23 (`save_vggt_tokens=true`); IGGT's instance head reads exactly those,
so no second backbone pass is needed.

    # surflo env: IGGT's heads on SuRFLo's saved tokens (and a check against IGGT's own forward)
    python -m gs_playground.iggt.on_surflo features --scene-dir <surflo scene dir> --out features.npz
    # puffin env: lift onto the Gaussians, group, name with CLIP, build the viewer
    python -m gs_playground.iggt.on_surflo lift  --scene-dir <dir> --features features.npz --out lifted.npz
    python -m gs_playground.iggt.on_surflo group --lifted lifted.npz --out groups.npz
    python -m gs_playground.iggt.on_surflo label --scene-dir <dir> --lifted lifted.npz --groups groups.npz --out labels.json
    python -m gs_playground.iggt.on_surflo demo  --root <out root> --scenes ... --out <viewer dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..paths import SCANNETPP

#: the lift: a Gaussian centre counts as seen by a view when it is in front of the rendered surface
#: by no more than this fraction of the depth, where the render is opaque
DEPTH_SLACK, MIN_ALPHA = 0.05, 0.5   # 0.02 left 37% of opaque Gaussians unseen on the sofa
MIN_OPACITY = 0.1          # SuRFLo's own cull for its point cloud
#: grouping, IGGT's demo settings scaled from its ~2M pixels (12 views at 504x336) to the Gaussian count
KNN = 20
#: IGGT's demo uses 0.06; on the sofa's Gaussians that merged floor, walls, ceiling and sofa into one group
#: holding 58% of them, while 0.03 separates them (chosen by eye on that scene; exploratory)
CLUSTER_EPS = 0.03
DEMO_PIXELS, DEMO_MIN_CLUSTER, DEMO_MIN_SAMPLES = 12 * 504 * 336, 500, 100
MIN_GROUP_FRACTION = 0.002  # groups smaller than this share of the Gaussians are not named
TOP_VIEWS = 3               # views per group whose crops CLIP averages (OpenMask3D's top-k)
SURROUND_FRACTION = 0.1     # a group whose mask has holes worth this share of it surrounds other things
SCANNETPP_TOP100 = SCANNETPP / "metadata/semantic_benchmark/top100.txt"
#: outdoor words added to ScanNet++'s indoor list; "garden table" was dropped after CLIP gave it to any patio
#: crop, the ground included (ScanNet++'s "table" names the table)
OUTDOOR = ("grass", "tree", "bush", "hedge", "flower", "potted plant", "paving stone", "brick wall", "house",
           "fence", "sky")


# ---------------------------------------------------------------- features
def features(scene_dir: Path, out: Path) -> None:
    import torch
    from gs_playground.iggt.view_count import load_iggt
    saved = torch.load(scene_dir / "vggt_tokens.pt", map_location="cpu", weights_only=False)
    model = load_iggt()
    tokens = [None] * 24
    for i, t in saved["layers"].items():
        tokens[i] = t.cuda()
    images, start = saved["images"].cuda(), saved["patch_start_idx"]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        with torch.amp.autocast("cuda", enabled=False):   # as IGGT's forward runs its heads
            _, _, point_feat = model.point_head(tokens, images=images, patch_start_idx=start, frames_chunk_size=None)
            adapted, _ = model.part_adaptor(tokens, images=images, patch_start_idx=start)
            part = model.part_head(list(adapted.values()), point_feature=point_feat, images=images,
                                   patch_start_idx=start, frames_chunk_size=None)
        # check: IGGT's own full forward on the same images gives the same features
        own = model(images)["part_feat"]
    feat = torch.nn.functional.normalize(part[0].float(), dim=1)
    own = torch.nn.functional.normalize(own[0].float(), dim=1)
    cosine = (feat * own).sum(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, names=np.array([Path(p).name for p in saved["selected_images"]]),
                        features=feat.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16),
                        check_cosine_median=float(cosine.median()), check_cosine_min=float(cosine.min()))
    print(f"{feat.shape[0]} views {tuple(feat.shape[-2:])}; against IGGT's own forward: "
          f"cosine median {float(cosine.median()):.6f}, min {float(cosine.min()):.6f}")


# --------------------------------------------------------------------- lift
def lift(scene_dir: Path, features_npz: Path, out: Path) -> None:
    import torch
    from gs_playground.gs.ply import load_ply
    from gs_playground.gs.render import load_cameras, render_depth
    cloud = load_ply(scene_dir / "point_cloud.ply").to("cuda")
    cams = {Path(c.name).stem: c for c in load_cameras(scene_dir / "cameras.json")}
    data = np.load(features_npz)
    means = cloud.means
    acc = torch.zeros(len(means), 8, device="cuda")
    seen = torch.zeros(len(means), device="cuda")
    for name, feat in zip(data["names"], data["features"]):
        cam = cams[Path(str(name)).stem]
        assert feat.shape[:2] == (cam.height, cam.width), (feat.shape, cam.width, cam.height)
        depth, alpha = render_depth(cloud, cam)
        viewmat, K = cam.viewmat.cuda(), cam.K.cuda()
        pc = means @ viewmat[:3, :3].T + viewmat[:3, 3]
        z = pc[:, 2]
        u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).floor().long()
        v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).floor().long()
        ok = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        idx = ok.nonzero()[:, 0]
        uu, vv = u[idx], v[idx]
        front = (alpha[vv, uu] > MIN_ALPHA) & (z[idx] <= depth[vv, uu] * (1 + DEPTH_SLACK))
        idx, uu, vv = idx[front], uu[front], vv[front]
        acc[idx] += torch.from_numpy(feat.astype(np.float32)).cuda()[vv, uu]
        seen[idx] += 1
    lifted = torch.nn.functional.normalize(acc, dim=1)
    C0 = 0.28209479177387814
    np.savez_compressed(out, means=means.cpu().numpy(), features=lifted.cpu().numpy().astype(np.float32),
                        views_seen=seen.cpu().numpy().astype(np.int16),
                        opacity=cloud.opacities.cpu().numpy(),
                        rgb=np.clip(cloud.f_dc.cpu().numpy() * C0 + 0.5, 0, 1))
    print(f"{len(means)} Gaussians; seen by >= 1 view: {float((seen > 0).float().mean()):.3f}, "
          f"median views {float(seen[seen > 0].median()):.0f}")


# -------------------------------------------------------------------- group
def group(lifted_npz: Path, out: Path, eps: float = CLUSTER_EPS) -> None:
    """IGGT's demo grouping, on Gaussians: features averaged over 3D neighbours, HDBSCAN with the
    demo's epsilon and its sizes scaled to the Gaussian count, the rest assigned to the nearest group.
    Opaque Gaussians no view saw take the group of their nearest grouped neighbour in 3D."""
    from sklearn.cluster import HDBSCAN
    from sklearn.neighbors import NearestNeighbors
    d = np.load(lifted_npz)
    use = (d["views_seen"] > 0) & (d["opacity"] >= MIN_OPACITY)
    xyz, feat = d["means"][use], d["features"][use]
    _, idx = NearestNeighbors(n_neighbors=KNN + 1).fit(xyz).kneighbors(xyz)
    smooth = feat[idx[:, 1:]].mean(1)
    scale = len(xyz) / DEMO_PIXELS
    labels = HDBSCAN(min_cluster_size=max(5, round(DEMO_MIN_CLUSTER * scale)),
                     min_samples=max(2, round(DEMO_MIN_SAMPLES * scale)),
                     cluster_selection_epsilon=eps).fit_predict(smooth)
    if (labels < 0).all():
        labels[:] = 0
    kept = labels >= 0
    _, nearest = NearestNeighbors(n_neighbors=1).fit(smooth[kept]).kneighbors(smooth[~kept])
    labels[~kept] = labels[kept][nearest[:, 0]]
    full = np.full(len(d["means"]), -1, dtype=np.int32)
    full[use] = labels
    unseen = (d["views_seen"] == 0) & (d["opacity"] >= MIN_OPACITY)
    if unseen.any():
        _, near = NearestNeighbors(n_neighbors=1).fit(xyz).kneighbors(d["means"][unseen])
        full[unseen] = labels[near[:, 0]]
    sizes = np.bincount(labels)
    np.savez_compressed(out, labels=full)
    print(f"{len(xyz)} Gaussians grouped into {len(sizes)} (eps {eps}); largest {sizes.max() / len(xyz):.2f}; "
          f"noise reassigned {float((~kept).mean()):.2f}; {int(unseen.sum())} unseen given their neighbour's group")


# -------------------------------------------------------------------- label
def visible_by_view(cloud, cams: dict, names) -> dict:
    """Per input view: (indices of the Gaussians it sees, their pixel u, v), the lift's visibility test."""
    import torch
    from gs_playground.gs.render import render_depth
    out = {}
    for name in names:
        cam = cams[Path(str(name)).stem]
        depth, alpha = render_depth(cloud, cam)
        viewmat, K = cam.viewmat.cuda(), cam.K.cuda()
        pc = cloud.means @ viewmat[:3, :3].T + viewmat[:3, 3]
        z = pc[:, 2]
        u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).floor().long()
        v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).floor().long()
        ok = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        idx = ok.nonzero()[:, 0]
        front = (alpha[v[idx], u[idx]] > MIN_ALPHA) & (z[idx] <= depth[v[idx], u[idx]] * (1 + DEPTH_SLACK))
        idx = idx[front]
        out[str(name)] = (idx.cpu().numpy(), u[idx].cpu().numpy(), v[idx].cpu().numpy())
    return out


def embedding(output):
    """CLIP's projected embedding: a tensor before transformers 5, the output's pooler_output from 5 on."""
    return output if hasattr(output, "norm") else output.pooler_output


def vocabulary() -> list[str]:
    return [c for c in SCANNETPP_TOP100.read_text().split("\n") if c.strip()] + list(OUTDOOR)


def label(scene_dir: Path, lifted_npz: Path, groups_npz: Path, out: Path) -> None:
    """Name each group with CLIP ViT-L/14, much as OpenMask3D does with its masks: in the TOP_VIEWS input
    views that see most of the group, the group's own pixels on grey (its Gaussians rendered white against
    the rest in black, so occlusion counts) and the plain box with context; a group that surrounds other
    things instead gets its holes inpainted from itself and no plain box. The mean embedding is matched to
    "a photo of a <class>" over ScanNet++'s top-100 classes plus a few outdoor ones.
    One box crop let garden's ground be named after the table inside its box; masked crops alone made
    large surfaces (floors, walls) look like small objects to CLIP."""
    import cv2
    import torch
    from PIL import Image
    from scipy.ndimage import binary_fill_holes
    from transformers import CLIPModel, CLIPProcessor
    from gs_playground.gs.ply import load_ply
    from gs_playground.gs.render import load_cameras, render
    labels = np.load(groups_npz)["labels"]
    state = torch.load(scene_dir / "guided_state.pt", map_location="cpu", weights_only=False)
    names = [Path(p).name for p in state["selected_images"]]
    images = {n: (im.permute(1, 2, 0).float().numpy() * 255).clip(0, 255).astype(np.uint8)
              for n, im in zip(names, state["supervision_images"])}
    cloud = load_ply(scene_dir / "point_cloud.ply").to("cuda")
    cams = {Path(c.name).stem: c for c in load_cameras(scene_dir / "cameras.json")}
    visible = visible_by_view(cloud, cams, names)
    classes = vocabulary()
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True).cuda().eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    with torch.no_grad():
        text = processor(text=[f"a photo of a {c}" for c in classes], return_tensors="pt", padding=True).to("cuda")
        text_feat = torch.nn.functional.normalize(embedding(model.get_text_features(**text)), dim=-1)
    total = int((labels >= 0).sum())
    result = {"vocabulary": classes, "groups": {}}
    for g in np.unique(labels[labels >= 0]):
        members = labels == g
        if members.sum() < MIN_GROUP_FRACTION * total:
            continue
        counts = sorted(((int(members[idx].sum()), name) for name, (idx, _, _) in visible.items()), reverse=True)
        views = [name for count, name in counts[:TOP_VIEWS] if count > 0]
        crops = []
        for name in views:
            mask = render(coloured(cloud, np.repeat(members[:, None].astype(np.float32), 3, 1)),
                          cams[Path(name).stem], sh_degree=None,
                          background=(0.0, 0.0, 0.0))[..., 0].cpu().numpy() > 0.5
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            h, w = mask.shape
            pad_x, pad_y = 0.2 * (xs.max() - xs.min()) + 4, 0.2 * (ys.max() - ys.min()) + 4
            box = (int(max(0, xs.min() - pad_x)), int(max(0, ys.min() - pad_y)),
                   int(min(w, xs.max() + pad_x + 1)), int(min(h, ys.max() + pad_y + 1)))
            holes = binary_fill_holes(mask) & ~mask
            if holes.sum() > SURROUND_FRACTION * mask.sum():
                # a surface around other things (garden's ground around its table): its holes inpainted from
                # itself, and no plain box, which would hand CLIP the thing in the middle
                filled = cv2.inpaint(np.ascontiguousarray(images[name]), holes.astype(np.uint8), 5, cv2.INPAINT_TELEA)
                masked = np.where((mask | holes)[..., None], filled, 128).astype(np.uint8)
                crops.append(Image.fromarray(masked[box[1]:box[3], box[0]:box[2]]))
            else:
                masked = np.where(mask[..., None], images[name], 128).astype(np.uint8)
                crops.append(Image.fromarray(masked[box[1]:box[3], box[0]:box[2]]))
                crops.append(Image.fromarray(images[name][box[1]:box[3], box[0]:box[2]]))
        crops = [c for c in crops if min(c.size) >= 2]
        if not crops:
            continue
        with torch.no_grad():
            pixels = processor(images=crops, return_tensors="pt").to("cuda")
            image_feat = torch.nn.functional.normalize(embedding(model.get_image_features(**pixels)), dim=-1)
            mean = torch.nn.functional.normalize(image_feat.mean(0, keepdim=True), dim=-1)
            prob = (100 * mean @ text_feat.T).softmax(-1)[0]
        top = prob.topk(3)
        result["groups"][str(int(g))] = {
            "class": classes[int(top.indices[0])], "prob": float(top.values[0]),
            "top3": [[classes[int(i)], float(p)] for p, i in zip(top.values, top.indices)],
            "gaussians": int(members.sum()), "views": views, "crops": len(crops)}
    out.write_text(json.dumps(result, indent=1))
    named = sorted(result["groups"].values(), key=lambda r: -r["gaussians"])
    print(f"{len(named)} groups named:", ", ".join(f"{r['class']} ({r['prob']:.2f})" for r in named[:12]))


# --------------------------------------------------------------------- demo
TITLES = {"nchc_sofa": "NCHC sofa lounge (our capture)",
          "6115eddb86": "ScanNet++ 6115eddb86 (in IGGT's training data)",
          "garden": "Mip-NeRF 360 garden"}


#: twenty well-separated colours for the class panel (Tableau 20); classes beyond them share "other"
QUALITATIVE = np.array([
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189], [140, 86, 75],
    [227, 119, 194], [188, 189, 34], [23, 190, 207], [174, 199, 232], [255, 187, 120], [152, 223, 138],
    [255, 152, 150], [197, 176, 213], [196, 156, 148], [247, 182, 210], [219, 219, 141], [158, 218, 229],
    [57, 59, 121], [99, 121, 57]], dtype=np.float32) / 255
OTHER = np.array([0.35, 0.35, 0.35], dtype=np.float32)


def palette(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((max(n, 1), 3)) * 0.75 + 0.2).astype(np.float32)


def coloured(cloud, rgb: np.ndarray):
    """The same Gaussians with flat colours (DC only)."""
    import dataclasses
    import torch
    C0 = 0.28209479177387814
    return dataclasses.replace(cloud, f_dc=torch.from_numpy((rgb - 0.5) / C0).float(), f_rest=None)


def legend(entries: list[tuple[str, np.ndarray, str]], target: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=20)
    rows = max(1, len(entries))
    img = Image.new("RGB", (900, 34 * rows + 20), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    for i, (name, colour, note) in enumerate(entries):
        y = 10 + 34 * i
        draw.rectangle([14, y + 4, 40, y + 28], fill=tuple(int(c * 255) for c in colour))
        draw.text((54, y + 4), f"{name}   {note}", fill=(30, 30, 30), font=font)
    img.save(target)
    return target


def colourings(root: Path, scene: str) -> dict:
    """SuRFLo's Gaussians for `scene`, the same Gaussians coloured by instance
    group and by class, and the class legend's entries."""
    from gs_playground.gs.ply import load_ply
    scene_dir = next((root / scene / "surflo").glob("*/guided_state.pt")).parent
    cloud = load_ply(scene_dir / "point_cloud.ply")
    labels = np.load(root / scene / "groups.npz")["labels"]
    named = json.loads((root / scene / "labels.json").read_text())["groups"]
    n_groups = int(labels.max()) + 1
    group_rgb = palette(n_groups)
    grey = np.array([0.55, 0.55, 0.55], dtype=np.float32)
    rgb = np.where(labels[:, None] >= 0, group_rgb[np.clip(labels, 0, None)], grey)
    counts = {}
    for r in named.values():
        counts[r["class"]] = counts.get(r["class"], 0) + r["gaussians"]
    classes = sorted(counts, key=lambda c: -counts[c])
    shown = classes[:len(QUALITATIVE) - 1] if len(classes) > len(QUALITATIVE) else classes
    class_rgb = {c: (QUALITATIVE[i] if c in shown else OTHER) for i, c in enumerate(classes)}
    rgb_cls = np.tile(grey, (len(labels), 1))
    for g, r in named.items():
        rgb_cls[labels == int(g)] = class_rgb[r["class"]]
    entries = [(c, class_rgb[c], f"{counts[c]:,} Gaussians") for c in shown]
    rest = [c for c in classes if c not in shown]
    if rest:
        entries.append(("other: " + ", ".join(rest[:8]) + ("..." if len(rest) > 8 else ""), OTHER,
                        f"{sum(counts[c] for c in rest):,} Gaussians"))
    return {"scene_dir": scene_dir, "cloud": cloud,
            "instances": coloured(cloud, rgb), "classes": coloured(cloud, rgb_cls),
            "legend": entries + [("unnamed (groups under 0.2% of the Gaussians)", grey, "")],
            "n_groups": n_groups, "n_named": len(named), "n_classes": len(classes)}


def demo(root: Path, scenes: list[str], out: Path) -> None:
    import tempfile
    from gs_playground.gs.ply import save_ply
    from gs_playground.surflo.demo import rotation_from_cameras, surflo_c2w
    from gs_playground.viewer import build
    tmp = Path(tempfile.mkdtemp(prefix="e08n_demo_"))
    panels = []
    for scene in scenes:
        c = colourings(root, scene)
        scene_dir, n_groups = c["scene_dir"], c["n_groups"]
        title = TITLES.get(scene, scene)
        rotation = rotation_from_cameras(surflo_c2w(scene_dir / "cameras.json"))
        save_ply(c["instances"], tmp / f"{scene}_instances.ply")
        save_ply(c["classes"], tmp / f"{scene}_classes.ply")
        legend(c["legend"], tmp / f"{scene}_legend.png")
        panels += [
            {"label": f"{title}: SuRFLo 3DGS (IGGT backbone)", "ply": str(scene_dir / "point_cloud.ply"),
             "sh": True, "rotation": rotation,
             "caption": "SuRFLo's guided Gaussians, fitted from 16 photos with IGGT's backbone in place of VGGT-1B."},
            {"label": f"{title}: instances", "ply": str(tmp / f"{scene}_instances.ply"), "rotation": rotation,
             "caption": "IGGT's instance features, read off the same encoding, lifted onto the Gaussians through "
                        "SuRFLo's cameras and grouped as IGGT's demo groups pixels, but with HDBSCAN's epsilon at "
                        "0.03 instead of its 0.06 (chosen by eye on the sofa: 0.06 merged the room); one colour "
                        "per group.",
             "meta": {"groups": f"{n_groups}"}},
            {"label": f"{title}: classes", "ply": str(tmp / f"{scene}_classes.ply"), "rotation": rotation,
             "caption": "Each group named by CLIP ViT-L/14 from the mean embedding of its masked and its plain crop "
                        "in the three views that see most of it; a group that surrounds others (the ground around a "
                        "table) is shown to CLIP with those holes inpainted instead.",
             "meta": {"named groups": f"{c['n_named']}", "classes": f"{c['n_classes']}"}},
            {"label": f"{title}: class legend", "image": str(tmp / f"{scene}_legend.png"),
             "caption": "Colour key for the classes panel, by Gaussian count."},
        ]
    spec = {
        "title": "Semantics on SuRFLo from one encoding",
        "eyebrow": "E08n · exploratory · SuRFLo x IGGT",
        "intro": "SuRFLo runs with IGGT's backbone; IGGT's instance head reads the very tokens SuRFLo "
                 "computed, so one backbone pass yields both the surface and per-pixel instance features. "
                 "Those features are carried onto SuRFLo's Gaussians and grouped; CLIP names each group.",
        "panels": panels,
        "notes": [
            "Exploratory: no ground truth is scored on this page. E08m measured IGGT's instances against "
            "ScanNet++ ground truth: its features separate instances well, its demo grouping is the weak step.",
            "Class names come from ScanNet++'s top-100 classes plus a few outdoor words, matched by CLIP; a wrong "
            "name is CLIP's or the group's, not a measured error rate. Misses seen by eye: garden's ground (grass "
            "and paving) is named 'table', after the table it surrounds, even with the table inpainted out of its "
            "crops; white walls often come out 'whiteboard'; ScanNet++ 6115eddb86's groups are fragmented, with "
            "most of its Gaussians seen by no view and given their neighbour's group.",
            "IGGT's backbone changes SuRFLo's geometry (E08l): better on ScanNet++, slightly worse on garden.",
            "A Gaussian takes features from the views whose rendered depth it is no more than 5% behind; opaque "
            "Gaussians no view sees that way (about a fifth on the sofa) take the group of their nearest grouped "
            "neighbour in 3D.",
        ],
    }
    build(spec, out, Path("data/viewer_js"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--scene-dir", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    li = sub.add_parser("lift")
    li.add_argument("--scene-dir", type=Path, required=True)
    li.add_argument("--features", type=Path, required=True)
    li.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("group")
    g.add_argument("--lifted", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--eps", type=float, default=CLUSTER_EPS, help="HDBSCAN cluster_selection_epsilon")
    lb = sub.add_parser("label")
    lb.add_argument("--scene-dir", type=Path, required=True)
    lb.add_argument("--lifted", type=Path, required=True)
    lb.add_argument("--groups", type=Path, required=True)
    lb.add_argument("--out", type=Path, required=True)
    dm = sub.add_parser("demo")
    dm.add_argument("--root", type=Path, required=True)
    dm.add_argument("--scenes", nargs="+", required=True)
    dm.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "label":
        label(args.scene_dir, args.lifted, args.groups, args.out)
    elif args.cmd == "demo":
        demo(args.root, args.scenes, args.out)
    elif args.cmd == "features":
        features(args.scene_dir, args.out)
    elif args.cmd == "lift":
        lift(args.scene_dir, args.features, args.out)
    elif args.cmd == "group":
        group(args.lifted, args.out, args.eps)


if __name__ == "__main__":
    main()
