"""E08j: novel-view PSNR of SuRFLo's fitted 3DGS on garden's held-out views.

Pre-registered in experiments/surflo/PREREG_E08j.md.

    # surflo env: test images exactly as SuRFLo preprocesses its inputs
    python -m gs_playground.surflo.nvs gt --images-dir <garden>/images_4 --out <gt.npz>
    # puffin env: cameras, alignment, pose refinement, PSNR
    python -m gs_playground.surflo.nvs eval --gt <gt.npz> --data-dir <garden> \\
        --sweep-root <outputs/surflo/views_sweep> --baseline-ply <E08i 3DGS ply> --out <json>

SuRFLo has no camera for a view it was not given. Test cameras come from
gsplat's own Parser (the normalised frame E08i's 3DGS was trained in) and are
mapped into SuRFLo's frame by a Sim(3) fit of the input-camera centres; a short
per-view pose refinement, Gaussians frozen, then absorbs what that leaves.

Each model is rendered at the resolution it was fit at and then put through the
ground truth's own resize and crop (Amendment 1): SuRFLo was fit at exactly the
evaluation size, the 3DGS at 1297x840.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import numpy as np

#: pre-registered
EVAL_WIDTH = 518
TTO_STEPS = 200
TTO_LR = 1e-3
SANITY_DB = 1.0
#: Amendment 2: central-difference step for the focal scale, in log scale
FOCAL_FD_STEP = 1e-3


# --------------------------------------------------------------------------- gt
def crop_params(width: int, height: int, target: int = EVAL_WIDTH) -> dict:
    """SuRFLo's `no_stretch` resize-then-crop for one image size, as numbers."""
    ratio = target / max(width, height)
    rw, rh = round(width * ratio), round(height * ratio)
    ow = rw if rw % 14 == 0 else int(width * ratio / 14) * 14
    oh = rh if rh % 14 == 0 else int(height * ratio / 14) * 14
    cx, cy = rw // 2, rh // 2
    return {"resize": [rw, rh], "offset": [cx - ow // 2, cy - oh // 2], "size": [ow, oh]}


def downscale(image, params: dict):
    """SuRFLo's preprocessing of one PIL image: bicubic resize, then the crop."""
    from PIL import Image
    (rw, rh), (x0, y0), (ow, oh) = params["resize"], params["offset"], params["size"]
    return image.convert("RGB").resize((rw, rh), Image.Resampling.BICUBIC).crop((x0, y0, x0 + ow, y0 + oh))


def gt(images_dir: Path, out: Path) -> None:
    """Every frame through SuRFLo's own preprocessing, checked against our numbers."""
    import torch
    from PIL import Image
    from surflo.data.utils import load_and_preprocess_images
    from torchvision.transforms.functional import to_tensor

    names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    first = Image.open(images_dir / names[0])
    params = crop_params(*first.size)
    ow, oh = params["size"]
    images = []
    for name in names:
        theirs = load_and_preprocess_images([str(images_dir / name)], mode="no_stretch",
                                            target_size=EVAL_WIDTH, rotate_portrait=True)[0]
        ours = to_tensor(downscale(Image.open(images_dir / name), params))
        if ours.shape != theirs.shape or float((ours - theirs).abs().max()) != 0.0:
            raise SystemExit(f"sanity 1 failed on {name}: our crop does not reproduce SuRFLo's")
        images.append((theirs.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    np.savez_compressed(out, names=np.array(names), images=np.stack(images),
                        source_size=np.array(first.size), **{k: np.array(v) for k, v in params.items()})
    print(f"{len(names)} frames -> {ow}x{oh}; sanity 1 (preprocessing reproduced exactly): pass")


# ---------------------------------------------------------------------- cameras
def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) with dst ~= s R src + t (Umeyama 1991)."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (xs ** 2).sum(1).mean())
    return s, R, md - s * R @ ms


def so3_exp(w):
    """Axis-angle -> rotation matrix (Rodrigues), differentiable, safe at 0."""
    import torch
    theta = torch.sqrt((w * w).sum() + 1e-12)
    k = w / theta
    K = torch.zeros(3, 3, dtype=w.dtype, device=w.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    return eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def render_c2w(cloud, c2w, K, width: int, height: int, sh_degree: int, eps2d: float):
    """[H, W, 3] on black; differentiable in `c2w` (OpenCV camera-to-world)."""
    import torch
    from gsplat import rasterization
    bands = (sh_degree + 1) ** 2 - 1
    colors = torch.cat([cloud.f_dc[:, None], cloud.f_rest[:, :bands]], dim=1)
    rgb, _, _ = rasterization(
        means=cloud.means, quats=torch.nn.functional.normalize(cloud.quats, dim=1),
        scales=torch.exp(cloud.log_scales), opacities=torch.sigmoid(cloud.opacity_logit),
        colors=colors, viewmats=torch.linalg.inv(c2w)[None], Ks=K[None],
        width=width, height=height, sh_degree=sh_degree, eps2d=eps2d)
    return rgb[0].clamp(0, 1)


def refine(cloud, c2w, K, target, width, height, sh_degree, eps2d, steps=TTO_STEPS):
    """Per-view pose and focal-scale correction, Gaussians frozen, L1 to the test image.

    Returns (c2w, K). gsplat has no gradient for the intrinsics, so the one focal
    scalar's gradient is a central finite difference of the same loss (Amendment 2).
    """
    import torch
    w = torch.zeros(3, device=c2w.device, dtype=c2w.dtype, requires_grad=True)
    dt = torch.zeros(3, device=c2w.device, dtype=c2w.dtype, requires_grad=True)
    log_focal = torch.zeros(1, device=c2w.device, dtype=c2w.dtype, requires_grad=True)
    opt = torch.optim.Adam([w, dt, log_focal], lr=TTO_LR)

    def posed():
        out = c2w.clone()
        out[:3, :3] = so3_exp(w) @ c2w[:3, :3]
        out[:3, 3] = c2w[:3, 3] + dt
        return out

    def focal(log_scale: float):
        out = K.clone()
        out[0, 0] *= math.exp(log_scale)
        out[1, 1] *= math.exp(log_scale)
        return out

    def loss_at(pose, log_scale: float):
        return (render_c2w(cloud, pose, focal(log_scale), width, height, sh_degree, eps2d) - target).abs().mean()

    for _ in range(steps):
        opt.zero_grad()
        current = float(log_focal)
        loss_at(posed(), current).backward()
        with torch.no_grad():
            pose = posed()
            log_focal.grad = ((loss_at(pose, current + FOCAL_FD_STEP) - loss_at(pose, current - FOCAL_FD_STEP))
                              / (2 * FOCAL_FD_STEP)).reshape(1)
        opt.step()
    with torch.no_grad():
        return posed(), focal(float(log_focal))


# ---------------------------------------------------------------------- eval
def evaluate(views, cloud, target_of, sh_degree, eps2d, tto: bool, *, to_eval=None, score_of=None) -> dict:
    """views: list of (name, c2w, K); returns per-view PSNR/SSIM and summaries.

    Renders at, and refines against, `target_of(name)`. With `to_eval`, every
    render is put through it and scored against `score_of(name)` instead.
    """
    import torch
    from gs_playground.gs.render import psnr, ssim
    to_eval = to_eval or (lambda img: img)
    score_of = score_of or target_of
    rows = []
    for name, c2w, K in views:
        target, score = target_of(name), score_of(name)
        h, w = target.shape[:2]
        with torch.no_grad():
            raw = to_eval(render_c2w(cloud, c2w, K, w, h, sh_degree, eps2d))
        row = {"name": name, "psnr_raw": psnr(raw, score), "ssim_raw": ssim(raw, score)}
        if tto:
            fine, fine_K = refine(cloud, c2w, K, target, w, h, sh_degree, eps2d)
            with torch.no_grad():
                img = to_eval(render_c2w(cloud, fine, fine_K, w, h, sh_degree, eps2d))
            row.update(psnr_tto=psnr(img, score), ssim_tto=ssim(img, score))
        rows.append(row)
    keys = [k for k in rows[0] if k != "name"]
    return {"views": rows, **{f"{k}_median": median(r[k] for r in rows) for k in keys},
            **{f"{k}_mean": sum(r[k] for r in rows) / len(rows) for k in keys}}


def sweep(root: Path) -> list[dict]:
    """Every (N, seed) the runner touched, with its status; done runs carry their scene dir."""
    entries = []
    for d in sorted(root.glob("n*/seed*"), key=lambda p: (int(p.parent.name[1:]), int(p.name[4:]))):
        entry = {"n": int(d.parent.name[1:]), "seed": int(d.name[4:]), "dir": str(d)}
        if (d / "done").exists():
            entry["status"] = "done"
            entry["scene"] = str(next(d.glob("*/guided_state.pt")).parent)
        else:
            entry["status"] = (d / "status").read_text().strip() if (d / "status").exists() else "incomplete"
        wall = d / "wall_s"
        entry["wall_s"] = int(wall.read_text()) if wall.exists() else None
        entries.append(entry)
    return entries


def load_run(scene: Path, c2w_p: dict, size: tuple[int, int]) -> SimpleNamespace:
    """One SuRFLo run: Gaussians, own cameras, the Parser -> SuRFLo map, median K, cost."""
    import torch
    from gs_playground.gs.ply import load_ply
    from gs_playground.gs.render import load_cameras
    cloud = load_ply(scene / "point_cloud.ply").to("cuda")
    cams = {Path(c.name).stem: c for c in load_cameras(scene / "cameras.json")}
    state = torch.load(scene / "guided_state.pt", map_location="cpu", weights_only=False)
    inputs = [Path(p).name for p in state["selected_images"]]
    c2w_s = {n: torch.linalg.inv(cams[Path(n).stem].viewmat).cuda() for n in inputs}
    src = np.stack([c2w_s[n][:3, 3].cpu().numpy() for n in inputs])
    dst = np.stack([c2w_p[n][:3, 3].cpu().numpy() for n in inputs])
    s, R, t = umeyama(src, dst)
    Rt = torch.tensor(R.T, dtype=torch.float32).cuda()
    tt = torch.tensor(t, dtype=torch.float32).cuda()

    def into_surflo(c2w):
        m = c2w.clone()
        m[:3, :3] = Rt @ c2w[:3, :3]
        m[:3, 3] = Rt @ (c2w[:3, 3] - tt) / s
        return m

    Ks = [cams[Path(n).stem].K for n in inputs]
    fx, fy = median(float(k[0, 0]) for k in Ks), median(float(k[1, 1]) for k in Ks)
    first = cams[Path(inputs[0]).stem]
    assert (first.width, first.height) == size, (first.width, first.height, size)
    W, H = size
    Kmed = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32).cuda()
    summary = json.loads((scene / "_infer_summary.json").read_text())["scene"]
    cost = {"inputs": len(inputs), "gaussians": len(cloud), "ode_s": summary["ode_inference_s"],
            "peak_vram_ode_gib": summary["peak_vram_ode_gib"], "peak_alloc_ode_gib": summary["peak_alloc_ode_gib"],
            "umeyama_scale": s,
            "centre_residual_median": float(np.median(np.linalg.norm(s * src @ R.T + t - dst, axis=1)))}
    return SimpleNamespace(cloud=cloud, cams=cams, inputs=inputs, c2w_s=c2w_s, into_surflo=into_surflo,
                           Kmed=Kmed, cost=cost)


#: what the pre-registered reading summarises per N
PER_N_KEYS = ("ode_s", "peak_vram_ode_gib", "peak_alloc_ode_gib", "wall_s",
              "psnr_raw_median", "psnr_tto_median", "ssim_raw_median", "ssim_tto_median")


def per_n(entries: list[dict], runs: dict) -> dict:
    """The pre-registered reading: per N, the median and [min, max] over seeds."""
    table = {}
    for n in sorted({e["n"] for e in entries}):
        mine = [e for e in entries if e["n"] == n]
        rows = [runs[e["dir"]] for e in mine if e["dir"] in runs]
        table[str(n)] = {"seeds_done": len(rows),
                         "not_done": {str(e["seed"]): e["status"] for e in mine if e["status"] != "done"}}
        if not rows:
            continue
        for key in PER_N_KEYS:
            values = [r[key] for r in rows]
            table[str(n)][key] = {"median": median(values), "range": [min(values), max(values)]}
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gt")
    g.add_argument("--images-dir", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("eval")
    e.add_argument("--gt", type=Path, required=True)
    e.add_argument("--data-dir", type=Path, required=True)
    e.add_argument("--sweep-root", type=Path, required=True, help="the runner's n<N>/seed<S> tree")
    e.add_argument("--baseline-ply", type=Path)
    e.add_argument("--gsplat-examples", type=Path, default=Path("third_party/clones/gsplat/examples"))
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "gt":
        gt(args.images_dir, args.out)
        return

    import torch
    from PIL import Image
    from gs_playground.gs.ply import load_ply
    sys.path.insert(0, str(args.gsplat_examples.resolve()))
    from datasets.colmap import Parser

    G = np.load(args.gt)
    target = {str(n): torch.from_numpy(img).float().cuda() / 255 for n, img in zip(G["names"], G["images"])}
    params = {k: [int(v) for v in G[k]] for k in ("resize", "offset", "size")}
    W, H = params["size"]
    sw, sh = (int(v) for v in G["source_size"])
    parser = Parser(str(args.data_dir), factor=4, normalize=True, test_every=8)   # as E08i trained
    index = {n: i for i, n in enumerate(parser.image_names)}
    test = [n for i, n in enumerate(parser.image_names) if i % 8 == 0]
    c2w_p = {n: torch.tensor(parser.camtoworlds[index[n]], dtype=torch.float32).cuda() for n in parser.image_names}
    entries = sweep(args.sweep_root)
    done = [e for e in entries if e["status"] == "done"]
    out = {"entry": "E08j", "prereg": "experiments/surflo/PREREG_E08j.md (with Amendment 1)",
           "eval_size": [W, H], "test_views": test,
           "sweep": [{k: v for k, v in e.items() if k != "scene"} for e in entries],
           "sanity_2": {}, "runs": {}}
    capped = args.sweep_root / "n161_capped"
    if capped.exists():
        out["n161_capped"] = capped.read_text().strip()

    def write():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))

    # sanity 2 first, on every seed-42 run; no test view is rendered unless every one passes
    for entry in [e for e in done if e["seed"] == 42]:
        run = load_run(Path(entry["scene"]), c2w_p, (W, H))
        own = evaluate([(n, run.c2w_s[n], run.cams[Path(n).stem].K.cuda()) for n in run.inputs], run.cloud,
                       target.__getitem__, 3, 0.0, tto=False)
        path = evaluate([(n, run.into_surflo(c2w_p[n]), run.Kmed) for n in run.inputs], run.cloud,
                        target.__getitem__, 3, 0.0, tto=True)
        gap = abs(own["psnr_raw_median"] - path["psnr_tto_median"])
        out["sanity_2"][f"n{entry['n']}"] = {"own_cameras_db": own["psnr_raw_median"],
                                              "test_path_db": path["psnr_tto_median"],
                                              "gap_db": gap, "passes": gap <= SANITY_DB}
        print(f"sanity 2, N = {entry['n']}: own cameras {own['psnr_raw_median']:.2f} dB, "
              f"test path {path['psnr_tto_median']:.2f} dB, gap {gap:.2f} dB, passes={gap <= SANITY_DB}")
        del run
        torch.cuda.empty_cache()
    unchecked = {e["n"] for e in done} - {int(k[1:]) for k in out["sanity_2"]}
    if unchecked or not all(v["passes"] for v in out["sanity_2"].values()):
        write()
        raise SystemExit(f"sanity 2 failed or missing (unchecked N: {sorted(unchecked)}); no test PSNR is read")

    if args.baseline_ply:
        Kp = torch.tensor(parser.Ks_dict[parser.camera_ids[0]], dtype=torch.float32)
        iw, ih = parser.imsize_dict[parser.camera_ids[0]]
        assert (iw, ih) == (sw, sh), (iw, ih, sw, sh)
        cloud = load_ply(args.baseline_ply).to("cuda")
        # as pre-registered: natively at the evaluation size (biased low by zoom-out dilation, Amendment 1)
        (rw, rh), (x0, y0) = params["resize"], params["offset"]
        K = Kp.clone()
        K[0] *= rw / iw
        K[1] *= rh / ih
        K[0, 2] -= x0
        K[1, 2] -= y0
        out["baseline_3dgs_native518"] = evaluate([(n, c2w_p[n], K.cuda()) for n in test], cloud,
                                                  target.__getitem__, 3, 0.3, tto=True)
        # Amendment 1: at its training resolution, then the ground truth's own resize and crop
        full = {}
        for n in test:
            image = Image.open(args.data_dir / "images_4" / n).convert("RGB")
            assert image.size == (iw, ih), (n, image.size)
            full[n] = torch.from_numpy(np.array(image)).float().cuda() / 255

        def to_eval(img):
            quantised = Image.fromarray((img * 255).round().byte().cpu().numpy())
            return torch.from_numpy(np.array(downscale(quantised, params))).float().cuda() / 255

        out["baseline_3dgs"] = evaluate([(n, c2w_p[n], Kp.cuda()) for n in test], cloud, full.__getitem__,
                                        3, 0.3, tto=True, to_eval=to_eval, score_of=target.__getitem__)
        for key in ("baseline_3dgs_native518", "baseline_3dgs"):
            print(f"{key}: raw {out[key]['psnr_raw_median']:.2f} dB, refined {out[key]['psnr_tto_median']:.2f} dB")
        del cloud, full
        torch.cuda.empty_cache()

    for entry in done:
        run = load_run(Path(entry["scene"]), c2w_p, (W, H))
        assert not set(run.inputs) & set(test), "a test view was an input"
        row = {**run.cost, "wall_s": entry["wall_s"]}
        row.update(evaluate([(n, run.into_surflo(c2w_p[n]), run.Kmed) for n in test], run.cloud,
                            target.__getitem__, 3, 0.0, tto=True))
        out["runs"][entry["dir"]] = row
        print(f"N = {entry['n']}, seed {entry['seed']}: raw {row['psnr_raw_median']:.2f} dB, "
              f"refined {row['psnr_tto_median']:.2f} dB")
        del run
        torch.cuda.empty_cache()

    out["per_n"] = per_n(entries, out["runs"])
    write()


if __name__ == "__main__":
    main()
