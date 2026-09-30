"""E08h demo: what SuRFLo makes from a handful of photos, beside what it was given.

    python -m gs_playground.surflo.demo --out outputs/surflo/viewer \\
        --scene garden outputs/surflo/garden_sample/guided_default/run1 \\
        --scene sofa outputs/surflo/nchc_sofa/guided_default_16/seed42 \\
        --reference sofa "$NCHC_SOFA_RUN/output/.../3dgs_output"

Per scene: the input views as one sheet (the only thing SuRFLo saw), its mesh,
and the 3DGS it fits in guided mode and never writes out (our export,
point_cloud.ply). `--reference` adds a 3DGS trained on the whole capture,
labelled as a reference, not ground truth. Built on gs_playground.viewer.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from ..frames import display_frame
from ..viewer import build

#: OpenCV camera axes -> OpenGL, which display_frame expects
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

#: per scene: title, where the frames come from, and the size of the capture
#: they were taken from (the garden sample is 16 files; its capture has 185)
TEXT = {
    "garden": ("Mip-NeRF 360 garden, the authors' sample",
               "The frames the SuRFLo repository ships, spread from the first to the last "
               "frame of the 185-frame Mip-NeRF 360 capture (518x336, downscaled by the "
               "authors), run with their README's first command.", 185),
    "sofa": ("NCHC sofa lounge, our capture",
             "Sampled uniformly by SuRFLo from the 364 COLMAP-registered frames of our "
             "handheld video (1272x715).", None),
}


def rotation_from_cameras(c2w_cv: np.ndarray) -> list[float]:
    """Quaternion (x y z w) turning the scene y-up, from its cameras."""
    return Rotation.from_matrix(display_frame(c2w_cv @ FLIP)).as_quat().tolist()


def surflo_c2w(cameras_json: Path) -> np.ndarray:
    out = []
    for c in json.loads(cameras_json.read_text()):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = c["rotation"], c["position"]
        out.append(m)
    return np.stack(out)


def contact_sheet(paths: list[Path], target: Path, width: int = 1600) -> Path:
    cols = min(8, len(paths))
    rows = math.ceil(len(paths) / cols)
    first = Image.open(paths[0])
    tw = width // cols
    th = round(tw * first.height / first.width)
    sheet = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    for i, p in enumerate(paths):
        sheet.paste(Image.open(p).convert("RGB").resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
    sheet.save(target, quality=88)
    return target


def decimate(source: Path, target: Path, faces: int) -> dict:
    """Quadric decimation for the browser (open3d, in the surflo env), colours kept.

    Reports the surface area kept: a large loss means the source was not one
    surface, which is how E08g's dust would have shown up.
    """
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(source))
    n, area = len(mesh.triangles), mesh.get_surface_area()
    small = mesh.simplify_quadric_decimation(faces) if n > faces else mesh
    o3d.io.write_triangle_mesh(str(target), small, write_ascii=False)
    return {"faces": len(small.triangles), "faces_source": n,
            "area_kept": small.get_surface_area() / area}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scene", nargs=2, action="append", required=True, metavar=("KEY", "RUN_DIR"))
    ap.add_argument("--reference", nargs=2, action="append", default=[], metavar=("KEY", "MODEL_DIR"))
    ap.add_argument("--js-dir", type=Path, default=Path("data/viewer_js"))
    ap.add_argument("--mesh-faces", type=int, default=500_000, help="per mesh, for the browser")
    args = ap.parse_args()
    references = dict(args.reference)

    tmp = Path(tempfile.mkdtemp(prefix="surflo_demo_"))
    panels, counts = [], []
    for key, run in args.scene:
        run = Path(run)
        scene = next(p.parent for p in run.glob("*/_infer_summary.json"))
        summary = json.loads((scene / "_infer_summary.json").read_text())
        inputs = [Path(p) for p in summary["selected_images"]]
        title, blurb, capture = TEXT[key]
        source = Path(summary["source"]["image_folder"])
        capture = capture or sum(1 for p in source.iterdir()
                                 if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rotation = rotation_from_cameras(surflo_c2w(scene / "cameras.json"))
        n_views = len(inputs)
        counts.append(f"{n_views} of {capture} frames for the {title.split(',')[0]}")
        panels.append({
            "label": f"{title}: the {n_views} input frames ({n_views} of {capture})",
            "image": str(contact_sheet(inputs, tmp / f"{key}_inputs.jpg")),
            "caption": f"{blurb} Everything below is made from these {n_views} frames "
                       f"alone: no poses, no depth.",
            "meta": {"frames used": f"{n_views} of {capture}"},
        })
        small = decimate(scene / "mesh_textured.ply", tmp / f"{key}_mesh.ply", args.mesh_faces)
        print(f"{key}: mesh {small['faces_source']:,} -> {small['faces']:,} faces, "
              f"{small['area_kept']:.1%} of the area kept")
        panels.append({
            "label": f"{title}: SuRFLo mesh",
            "mesh": str(tmp / f"{key}_mesh.ply"), "rotation": rotation,
            "caption": "Its mesh output with the vertex colours it writes "
                       "(mesh_textured.ply), decimated for the browser.",
            "meta": {"faces": f"{small['faces']:,} of {small['faces_source']:,}",
                     "area kept": f"{small['area_kept']:.0%}",
                     "ODE time": f"{summary['scene']['ode_inference_s']:.0f} s"},
        })
        panels.append({
            "label": f"{title}: SuRFLo 3DGS (our export)",
            "ply": str(scene / "point_cloud.ply"), "sh": True, "rotation": rotation,
            "caption": "The Gaussians guided mode fits against these photos, which SuRFLo "
                       "itself never saves; exported by us, SH degree 3.",
        })
        if key in references:
            model = Path(references[key])
            frames = len(json.loads((model / "cameras.json").read_text()))
            panels.append({
                "label": f"{title}: reference 3DGS, all {frames} frames",
                "ply": str(model / "point_cloud/iteration_30000/point_cloud.ply"),
                "rotation": rotation_from_cameras(surflo_c2w(model / "cameras.json")),
                "caption": f"Our 30k-iteration 3DGS of the same room from every frame: a "
                           f"reference for the eye, not ground truth, trained on "
                           f"{frames / n_views:.0f}x the views.",
            })

    spec = {
        "title": "SuRFLo: a surface from a handful of photos",
        "eyebrow": "E08h · SuRFLo · arXiv 2606.13644",
        "intro": "SuRFLo takes unposed photos and, in one forward pass through a frozen VGGT "
                 "backbone and a flow-matching decoder, returns an oriented surface and a "
                 "mesh. Its guided mode also fits a 3DGS against the photos to steer that "
                 "surface; we export it. Frames used: " + "; ".join(counts) + ".",
        "panels": panels,
        "notes": [
            "Units are VGGT's: no metric scale. Up is estimated from the cameras.",
            "The splat panels are drawn by Spark, which adds a 0.3 px screen-space "
            "low-pass that SuRFLo's own renderer does not; the export itself was checked "
            "against SuRFLo's renderer (results/surflo/e08h_export_check.json).",
            "The garden inputs are the authors' own downscaled sample; the paper's "
            "accuracy on Mip-NeRF 360 was not reproduced because no ground truth for it "
            "is released.",
            "On the sofa, measured against the reference 3DGS's depth (pseudo-ground "
            "truth, exploratory): the guided output shown here agrees less with it than "
            "SuRFLo's own unguided output does (F1 0.58 against 0.87 at 1% of the room's "
            "diagonal, three seeds each; 32 views guided: 0.63). Why is untested: it is not "
            "how the Gaussians sample the surface, since the mesh scores the same. "
            "LOG.md E08h, results/surflo/e08h_stage_b.json.",
        ],
    }
    build(spec, args.out, args.js_dir)


if __name__ == "__main__":
    main()
