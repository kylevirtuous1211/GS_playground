"""E08j probe: where sanity check 2's gap comes from, on SuRFLo's input views only.

Exploratory; no test view is rendered. Sanity 2 compares SuRFLo's own cameras
(raw) with the test path (Parser camera -> Umeyama -> median K -> 200-step pose
refinement). This decomposes the gap: intrinsics, refinement budget, and how
far refinement moves an exact pose.

Run once, at the commit that adds it, with the pose-only refinement that
preceded Amendment 2; `refine` later also returns a focal, so check out that
commit to rerun it.

    python probes/surflo/e08j_sanity2_gap.py --sweep-root outputs/surflo/views_sweep \\
        --gt outputs/surflo/views_sweep/gt.npz --data-dir data/mipnerf360/garden --out <json>
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import median

import numpy as np
import torch

from gs_playground.surflo.nvs import evaluate, load_run, refine, render_c2w

ap = argparse.ArgumentParser()
ap.add_argument("--sweep-root", type=Path, required=True)
ap.add_argument("--gt", type=Path, required=True)
ap.add_argument("--data-dir", type=Path, required=True)
ap.add_argument("--gsplat-examples", type=Path, default=Path("third_party/clones/gsplat/examples"))
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
sys.path.insert(0, str(args.gsplat_examples.resolve()))
from datasets.colmap import Parser  # noqa: E402

G = np.load(args.gt)
target = {str(n): torch.from_numpy(img).float().cuda() / 255 for n, img in zip(G["names"], G["images"])}
W, H = (int(v) for v in G["size"])
parser = Parser(str(args.data_dir), factor=4, normalize=True, test_every=8)
c2w_p = {n: torch.tensor(parser.camtoworlds[i], dtype=torch.float32).cuda() for i, n in enumerate(parser.image_names)}

out = {}
for n in (16, 32, 64):
    scene = next((args.sweep_root / f"n{n}" / "seed42").glob("*/guided_state.pt")).parent
    run = load_run(scene, c2w_p, (W, H))
    own_K = {v: run.cams[Path(v).stem].K.cuda() for v in run.inputs}
    path = {v: run.into_surflo(c2w_p[v]) for v in run.inputs}

    def med(views, tto, steps=None):
        if steps is None:
            return evaluate(views, run.cloud, target.__getitem__, 3, 0.0, tto=tto)
        psnrs = []
        from gs_playground.gs.render import psnr
        for v, c2w, K in views:
            fine = refine(run.cloud, c2w, K, target[v], W, H, 3, 0.0, steps=steps)
            with torch.no_grad():
                psnrs.append(psnr(render_c2w(run.cloud, fine, K, W, H, 3, 0.0), target[v]))
        return {"psnr_tto_median": median(psnrs)}

    own_raw = med([(v, run.c2w_s[v], own_K[v]) for v in run.inputs], tto=False)
    own_tto = med([(v, run.c2w_s[v], own_K[v]) for v in run.inputs], tto=True)
    path_kmed = med([(v, path[v], run.Kmed) for v in run.inputs], tto=True)
    path_ownk = med([(v, path[v], own_K[v]) for v in run.inputs], tto=True)
    path_kmed_1000 = med([(v, path[v], run.Kmed) for v in run.inputs], tto=True, steps=1000)
    # how far the Sim(3)-mapped Parser pose sits from SuRFLo's own pose, per view
    rot_deg = [float(torch.rad2deg(torch.arccos(((path[v][:3, :3].T @ run.c2w_s[v][:3, :3]).trace() - 1) / 2
                                                 ).clamp(-1, 1).nan_to_num(0))) for v in run.inputs]
    out[n] = {
        "own_raw": own_raw["psnr_raw_median"],
        "own_tto": own_tto["psnr_tto_median"],
        "path_raw_kmed": path_kmed["psnr_raw_median"],
        "path_tto_kmed": path_kmed["psnr_tto_median"],
        "path_tto_ownK": path_ownk["psnr_tto_median"],
        "path_tto_kmed_1000steps": path_kmed_1000["psnr_tto_median"],
        "umeyama_centre_residual_over_scene_extent": run.cost["centre_residual_median"] / float(
            np.ptp(np.stack([run.c2w_s[v][:3, 3].cpu().numpy() for v in run.inputs]), axis=0).max()),
        "rotation_offset_deg_median": median(rot_deg), "rotation_offset_deg_max": max(rot_deg),
        "per_view_gap_kmed": {v: o["psnr_raw"] - p["psnr_tto"] for v, o, p in
                              zip(run.inputs, own_raw["views"], path_kmed["views"])},
    }
    print(n, {k: round(v, 3) for k, v in out[n].items() if not isinstance(v, dict)}, flush=True)
    del run
    torch.cuda.empty_cache()

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(out, indent=1))
