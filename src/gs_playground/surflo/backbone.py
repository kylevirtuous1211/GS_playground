"""E08l: SuRFLo's VGGT-1B against IGGT's fine-tuned backbone (surflo env).

Pre-registered in experiments/surflo/PREREG_E08l.md.

    # validity check 1: the swap is real and complete
    python -m gs_playground.surflo.backbone check --surflo-ckpt <surflo_v0.pt> \\
        --iggt <iggt_checkpoint.pth> --out <json>
    # metric 1: token, camera and depth drift on one scene's 16 inputs
    python -m gs_playground.surflo.backbone drift --surflo-ckpt <surflo_v0.pt> \\
        --iggt <iggt_checkpoint.pth> --folder <images> --n-images 16 --out <json>

Both run SuRFLo's own code: the backbone is `surflo.nn.vggt`'s VGGT, the swap
is the patched `surflo.model.loader.load_vggt_weights`, and the inputs are
picked and preprocessed exactly as `surflo.data.image_folder` does.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

#: SuRFLo reads these aggregator layers (configs/model/surflo.yaml)
LAYERS = (4, 11, 17, 23)


def shipped_vggt_state(surflo_ckpt: Path) -> dict:
    """The `vggt.*` weights inside SuRFLo's checkpoint (its EMA copy, as inference loads)."""
    import torch
    ckpt = torch.load(surflo_ckpt, map_location="cpu", mmap=True, weights_only=False)
    prefix = "ema_model.vggt."
    return {k[len(prefix):]: v for k, v in ckpt["ema_state"].items() if k.startswith(prefix)}


def backbone(state: dict, iggt: Path | None = None):
    """A VGGT holding `state`, then IGGT's weights if given, through SuRFLo's own loader."""
    from surflo.model.loader import load_vggt_weights
    from surflo.nn.vggt.models.vggt import VGGT
    model = VGGT()
    model.load_state_dict(state, strict=True)
    if iggt is not None:
        load_vggt_weights(model, str(iggt))
    return model.eval()


def check(surflo_ckpt: Path, iggt: Path, out: Path) -> None:
    import torch
    from surflo.nn.vggt.models.vggt import VGGT
    shipped = shipped_vggt_state(surflo_ckpt)
    hub = VGGT.from_pretrained("facebook/VGGT-1B").state_dict()
    raw = {k.removeprefix("module."): v for k, v in
           torch.load(iggt, map_location="cpu", mmap=True, weights_only=True).items()}
    swapped = backbone(shipped, iggt).state_dict()

    def max_diff(a: dict, b: dict) -> dict:
        per_module = {}
        for k in a:
            module = k.split(".")[0]
            d = float((a[k].float() - b[k].float()).abs().max()) if a[k].numel() else 0.0
            per_module[module] = max(per_module.get(module, 0.0), d)
        return per_module

    loaded_equal = all(torch.equal(swapped[k], raw[k].to(swapped[k].dtype)) for k in swapped)
    result = {
        "vggt_tensors": len(swapped),
        "iggt_keys_outside_vggt": sorted({k.split(".")[0] for k in raw if k not in swapped}),
        "iggt_extra_key_count": sum(k not in swapped for k in raw),
        "loaded_equals_checkpoint": loaded_equal,
        "shipped_vs_hub_vggt1b_max_abs": max_diff(shipped, hub),
        "shipped_equals_hub_vggt1b": all(torch.equal(shipped[k], hub[k]) for k in shipped),
        "iggt_vs_shipped_max_abs": max_diff({k: raw[k] for k in shipped}, shipped),
    }
    result["swap_is_real"] = max(result["iggt_vs_shipped_max_abs"].values()) > 0
    result["passes"] = bool(loaded_equal and result["swap_is_real"] and len(swapped) == 1797)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=1))
    if not result["passes"]:
        raise SystemExit("validity check 1 failed")


def run(model, images, autocast: bool):
    """Tokens at LAYERS, the last camera token, cameras and depth, as SuRFLo's preprocess_images runs them."""
    import torch
    from surflo.nn.vggt.utils.pose_enc import pose_encoding_to_extri_intri
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=autocast):
        pred = model(images, return_aggregated_tokens=True)
    tokens, start = pred["aggregated_tokens_list"], pred["patch_start_idx"]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    return {
        "layers": {layer: tokens[layer][:, :, start:].float().cpu() for layer in LAYERS},
        "camera": tokens[-1][:, :, 0].float().cpu(),
        "extrinsics": extrinsics.float().cpu(), "intrinsics": intrinsics.float().cpu(),
        "depth": pred["depth"].float().cpu(),
    }


def compare(a: dict, b: dict) -> dict:
    """b against a: token cosine and relative L2, camera rotation and focal, depth."""
    import torch

    def tokens(x, y):
        return {"cosine_mean": float(torch.nn.functional.cosine_similarity(x, y, dim=-1).mean()),
                "relative_l2": float((x - y).norm() / x.norm())}

    R_a, R_b = a["extrinsics"][0, :, :, :3], b["extrinsics"][0, :, :, :3]
    cos_angle = ((R_a @ R_b.transpose(1, 2)).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
    angles = [math.degrees(math.acos(max(-1.0, min(1.0, float(c))))) for c in cos_angle]
    focal = (b["intrinsics"][0, :, 0, 0] / a["intrinsics"][0, :, 0, 0] - 1) * 100
    depth = ((b["depth"] - a["depth"]).abs() / a["depth"].clamp_min(1e-6)).flatten()
    return {
        "layers": {str(layer): tokens(a["layers"][layer], b["layers"][layer]) for layer in LAYERS},
        "camera_token": tokens(a["camera"], b["camera"]),
        "rotation_deg": {"median": float(torch.tensor(angles).median()), "max": max(angles)},
        "focal_pct": {"median": float(focal.median()), "max_abs": float(focal.abs().max())},
        "depth_relative_median": float(depth.median()),
    }


def drift(surflo_ckpt: Path, iggt: Path, folder: Path, n_images: int, out: Path) -> None:
    import torch
    from surflo.data.image_folder import list_images_in_folder, sample_image_indices
    from surflo.data.utils import load_and_preprocess_images
    paths = list_images_in_folder(str(folder))
    selected = [paths[i] for i in sample_image_indices(len(paths), n_images, "uniform", 42)]
    images = load_and_preprocess_images(selected, mode="no_stretch", target_size=518,
                                        rotate_portrait=True).cuda()[None]
    shipped = shipped_vggt_state(surflo_ckpt)
    vggt = backbone(shipped).cuda()
    reference, fp32 = run(vggt, images, autocast=True), run(vggt, images, autocast=False)
    del vggt
    torch.cuda.empty_cache()
    swapped = run(backbone(shipped, iggt).cuda(), images, autocast=True)
    result = {"folder": str(folder), "inputs": [Path(p).name for p in selected],
              "input_size": list(images.shape[-2:]),
              "iggt_vs_vggt": compare(reference, swapped),
              "autocast_vs_fp32_vggt": compare(fp32, reference)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    for key in ("iggt_vs_vggt", "autocast_vs_fp32_vggt"):
        r = result[key]
        print(f"{key}: layer 23 cosine {r['layers']['23']['cosine_mean']:.6f}, "
              f"rel L2 {r['layers']['23']['relative_l2']:.2e}; rotation {r['rotation_deg']['median']:.4f} deg, "
              f"focal {r['focal_pct']['median']:+.4f}%, depth {r['depth_relative_median']:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "drift"):
        p = sub.add_parser(name)
        p.add_argument("--surflo-ckpt", type=Path, required=True)
        p.add_argument("--iggt", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "drift":
            p.add_argument("--folder", type=Path, required=True)
            p.add_argument("--n-images", type=int, default=16)
    args = ap.parse_args()
    if args.cmd == "check":
        check(args.surflo_ckpt, args.iggt, args.out)
    else:
        drift(args.surflo_ckpt, args.iggt, args.folder, args.n_images, args.out)


if __name__ == "__main__":
    main()
