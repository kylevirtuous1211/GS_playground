"""SuRFLo's fitted Gaussians -> a standard 3DGS asset, and the check that it is one.

Guided mode fits a full 3DGS (SH degree 3) against the input photos and never
writes it out. Our patch to upstream's infer.py (`save_guided_state=true`)
dumps it as `guided_state.pt`; this module turns that into an INRIA-layout
`point_cloud.ply` plus `cameras.json`, and checks the conversion by rendering
the export with gsplat and comparing with SuRFLo's own renderer on the same
Gaussians. Pre-registered in experiments/surflo/PREREG_E08h.md.

    # surflo env: SuRFLo's own renders of the fitted Gaussians
    python -m gs_playground.surflo.export reference --run <scene dir>
    # puffin env: the asset, then the check
    python -m gs_playground.surflo.export export --run <scene dir>
    python -m gs_playground.surflo.export check --runs <scene dir> ... --out <json>

Each subcommand imports only what its env has: `reference` needs SuRFLo and
none of ours, `export` and `check` need plyfile and gsplat and none of SuRFLo.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
import torch

#: SH DC basis constant, as SuRFLo's `RGB2SH` and INRIA's use it.
SH_C0 = 0.28209479177387814
SH_DEGREE = 3
#: pre-registered pass threshold and null margin
PASS_DB = 40.0
NULL_MARGIN_DB = 5.0


def load_state(run: Path) -> dict:
    # ours, written by the patched infer.py; carries numpy arrays in the cameras
    return torch.load(run / "guided_state.pt", map_location="cpu", weights_only=False)


def reference(run: Path) -> None:
    """SuRFLo's `render_surflo` at every refined camera: black, no low-pass."""
    from surflo.rendering.gaussians import Gaussians
    from surflo.rendering.surflo import render_surflo
    from surflo.structures.cameras import Camera

    state = load_state(run)
    gs = Gaussians(
        means=state["aux_means"].cuda(), rotations=state["aux_quats"].cuda(),
        scales=state["aux_scales"].cuda(), opacities=state["aux_opacities"].cuda(),
        colors=state["aux_colors"].cuda(), colors_sh=state["aux_colors_sh"].cuda(),
        active_sh_degree=SH_DEGREE,
    )
    renders = []
    with torch.no_grad():
        for i, (c, image) in enumerate(zip(state["cameras"], state["supervision_images"])):
            cam = Camera(colmap_id=i, R=c["R"], T=c["T"], FoVx=c["FoVx"], FoVy=c["FoVy"],
                         image=image, gt_alpha_mask=None, image_name=c["image_name"],
                         uid=i, data_device="cuda")
            pkg = render_surflo(viewpoint_camera=cam, gaussians=gs,
                                bg_color=torch.zeros(3, device="cuda"), kernel_size=0.0,
                                require_depth=False)
            renders.append(pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    np.savez_compressed(run / "reference_renders.npz", renders=np.stack(renders).astype(np.float32))
    print(f"{run}: {len(renders)} reference renders")


def to_cloud(state: dict):
    from gs_playground.gs.ply import GaussianCloud
    opacity = state["aux_opacities"].float().clamp(1e-6, 1 - 1e-6)
    return GaussianCloud(
        means=state["aux_means"].float(),
        f_dc=(state["aux_colors"].float() - 0.5) / SH_C0,   # SuRFLo's RGB2SH
        opacity_logit=torch.log(opacity / (1 - opacity)),
        log_scales=torch.log(state["aux_scales"].float().clamp_min(1e-12)),
        quats=state["aux_quats"].float(),                     # wxyz, both sides
        f_rest=state["aux_colors_sh"].float(),                # [N, 15, 3], both sides
    )


def cameras_json(state: dict) -> list[dict]:
    """SuRFLo cameras (INRIA convention: w2c = [R^T | T]) -> INRIA cameras.json."""
    out = []
    for i, c in enumerate(state["cameras"]):
        R, T = np.asarray(c["R"], dtype=np.float64), np.asarray(c["T"], dtype=np.float64)
        w, h = c["width"], c["height"]
        out.append({
            "id": i, "img_name": Path(state["selected_images"][i]).stem,
            "width": w, "height": h,
            "position": (-R @ T).tolist(), "rotation": R.tolist(),
            "fx": w / (2 * math.tan(c["FoVx"] / 2)), "fy": h / (2 * math.tan(c["FoVy"] / 2)),
        })
    return out


def export(run: Path) -> None:
    from gs_playground.gs.ply import save_ply
    state = load_state(run)
    cloud = to_cloud(state)
    save_ply(cloud, run / "point_cloud.ply")
    (run / "cameras.json").write_text(json.dumps(cameras_json(state), indent=1))
    print(f"{run}: point_cloud.ply ({len(cloud):,} Gaussians, SH {SH_DEGREE}), "
          f"cameras.json ({len(state['cameras'])})")


def check_run(run: Path) -> dict:
    import dataclasses
    from gs_playground.gs.ply import load_ply
    from gs_playground.gs.render import load_cameras, psnr, render

    cloud = load_ply(run / "point_cloud.ply")
    dc_only = dataclasses.replace(cloud, f_rest=torch.zeros_like(cloud.f_rest))
    cams = load_cameras(run / "cameras.json")
    reference = torch.from_numpy(np.load(run / "reference_renders.npz")["renders"])
    state = load_state(run)
    black = (0.0, 0.0, 0.0)
    full, null, fit = [], [], []
    for i, cam in enumerate(cams):
        ours = render(cloud, cam, sh_degree=SH_DEGREE, eps2d=0.0, background=black).cpu()
        full.append(psnr(ours, reference[i]))
        null.append(psnr(render(dc_only, cam, sh_degree=SH_DEGREE, eps2d=0.0,
                                background=black).cpu(), reference[i]))
        # the fit, reported only: through the per-image exposure SuRFLo learned
        target = state["supervision_images"][i].permute(1, 2, 0).float()
        ec = state.get("aux_exposure_coeffs")
        seen = ours if ec is None else (ours * torch.exp(ec[i, 0]) + ec[i, 1]).clamp(0, 1)
        fit.append(psnr(seen, target))
    row = {
        "run": str(run), "gaussians": len(cloud), "cameras": len(cams),
        "psnr_vs_surflo_render_db": {"median": median(full), "min": min(full)},
        "null_dc_only_db": {"median": median(null), "min": min(null)},
        "fit_psnr_vs_inputs_db_reported_only": {"median": median(fit), "min": min(fit)},
    }
    row["passes"] = (row["psnr_vs_surflo_render_db"]["median"] >= PASS_DB
                     and median(full) - median(null) >= NULL_MARGIN_DB)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("reference", "export"):
        sub.add_parser(name).add_argument("--run", type=Path, required=True,
                                          help="the scene dir holding guided_state.pt")
    c = sub.add_parser("check")
    c.add_argument("--runs", type=Path, nargs="+", required=True)
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "reference":
        reference(args.run)
    elif args.cmd == "export":
        export(args.run)
    else:
        rows = [check_run(run) for run in args.runs]
        for r in rows:
            print(f"{r['run']}: {r['psnr_vs_surflo_render_db']['median']:.1f} dB vs SuRFLo "
                  f"(null DC-only {r['null_dc_only_db']['median']:.1f} dB), "
                  f"fit {r['fit_psnr_vs_inputs_db_reported_only']['median']:.1f} dB, "
                  f"passes={r['passes']}")
        result = {"entry": "E08h", "prereg": "experiments/surflo/PREREG_E08h.md",
                  "pass_db": PASS_DB, "null_margin_db": NULL_MARGIN_DB,
                  "runs": rows, "all_pass": all(r["passes"] for r in rows)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # merged by run, so Stage A and Stage B checks share one file
        if args.out.exists():
            old = {r["run"]: r for r in json.loads(args.out.read_text())["runs"]}
            old.update({r["run"]: r for r in rows})
            result["runs"] = list(old.values())
            result["all_pass"] = all(r["passes"] for r in result["runs"])
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
