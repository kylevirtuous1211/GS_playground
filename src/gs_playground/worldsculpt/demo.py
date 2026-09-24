"""Build the E08c demo: what WorldSculpt made of a scene, beside its input.

WorldSculpt emits meshes, not splats, so this is an image page rather than the
splat viewer. It leans on the artefact the upstream pipeline already produces:
`_scene/renders/view*.jpg`, each a strip of
`[ original frame | normals | per-instance colour | overlay ]`, which is the
comparison the demo needs and is rendered by their code, not ours.

    python -m gs_playground.worldsculpt.demo \
        --case-root outputs/worldsculpt/Marble/<scene> \
        --scene-dir data/worldsculpt_input/Marble/<scene> \
        --out outputs/worldsculpt/viewer
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from PIL import Image

PAGE = """<meta charset="utf-8">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
  :root { --paper:#EFF1EE; --raised:#FFF; --ink:#151A1E; --ink-dim:#5C666C;
          --rule:#CDD3CE; --dial:#2F7D95;
          --shadow:0 1px 2px rgba(21,26,30,.06), 0 8px 24px rgba(21,26,30,.05); }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
      --paper:#10151A; --raised:#171E24; --ink:#E4E9E5; --ink-dim:#8D989E;
      --rule:#29333A; --dial:#5FB6CE;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3); } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--paper); color:var(--ink);
         font:400 15px/1.55 "IBM Plex Sans", system-ui, sans-serif; }
  .wrap { max-width:1300px; margin:0 auto; padding:32px 24px 64px; }
  header { border-bottom:1px solid var(--rule); padding-bottom:20px; margin-bottom:24px; }
  .eyebrow { font:500 11px/1 "IBM Plex Mono", monospace; letter-spacing:.14em;
             text-transform:uppercase; color:var(--ink-dim); }
  h1 { font:700 clamp(26px,4vw,40px)/1.08 "IBM Plex Sans Condensed", system-ui, sans-serif;
       margin:10px 0 8px; }
  h2 { font:600 17px/1.2 "IBM Plex Sans", system-ui, sans-serif;
       margin:34px 0 6px; }
  .intro, .sub { color:var(--ink-dim); max-width:74ch; }
  .card { background:var(--raised); border:1px solid var(--rule); border-radius:10px;
          box-shadow:var(--shadow); overflow:hidden; margin-top:14px; }
  .card img { display:block; width:100%; height:auto; }
  .card .cap { padding:10px 14px; color:var(--ink-dim); font-size:13px;
               border-top:1px solid var(--rule); }
  .pair { display:grid; gap:14px; grid-template-columns:repeat(auto-fit, minmax(300px,1fr)); }
  .stats { display:flex; flex-wrap:wrap; gap:18px; margin:18px 0 0;
           font:500 12px/1.3 "IBM Plex Mono", monospace; color:var(--ink-dim); }
  .stats b { display:block; color:var(--ink); font-size:16px; font-weight:600; }
  .notes { margin-top:30px; border-top:1px solid var(--rule); padding-top:16px;
           color:var(--ink-dim); font-size:13.5px; max-width:82ch; }
  .notes li { margin-bottom:6px; }
</style>
<div class="wrap">
<header>
  <div class="eyebrow">__EYEBROW__</div>
  <h1>__TITLE__</h1>
  <p class="intro">__INTRO__</p>
  <div class="stats">__STATS__</div>
</header>
__BODY__
<ul class="notes">__NOTES__</ul>
</div>
"""


def copy_image(source: Path, out: Path, max_width: int = 1800) -> str | None:
    if not source.exists():
        return None
    image = Image.open(source)
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, round(image.height * ratio)))
    name = source.name.replace(".jpg", ".png") if image.mode == "RGBA" \
        else source.name
    target = out / "images" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(target, quality=86, optimize=True)
    return f"images/{name}"


def section(title: str, sub: str, cards: list[tuple[str, str]]) -> str:
    if not cards:
        return ""
    body = "".join(
        f'<div class="card"><img src="{src}" alt=""><div class="cap">{cap}</div></div>'
        for src, cap in cards)
    return (f"<h2>{title}</h2><p class='sub'>{sub}</p>"
            f"<div class='pair'>{body}</div>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-root", required=True, type=Path,
                    help="the OUTPUT_ROOT/<scene> directory the run wrote")
    ap.add_argument("--scene-dir", required=True, type=Path,
                    help="the input scene, for its transforms.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--views", type=int, default=3,
                    help="how many per-view strips to show")
    args = ap.parse_args()

    scene_dir = args.case_root / "_scene"
    transforms = json.loads((args.scene_dir / "transforms.json").read_text())
    instances = transforms.get("instances", [])
    frames = transforms.get("frames", [])
    reconstructed = sorted((args.case_root / "_recon").glob("*/mesh.pt"))

    args.out.mkdir(parents=True, exist_ok=True)

    strips = []
    for view in sorted((scene_dir / "renders").glob("view*.jpg"))[:args.views]:
        src = copy_image(view, args.out)
        if src:
            strips.append((src, f"{view.stem}: original frame, normals, "
                                f"per-instance colour, overlay"))

    overview = []
    for name, caption in [
        ("mesh_scene_3d.png", "Composed meshes, perspective."),
        ("mesh_scene_topdown.png", "Composed meshes, top down: the layout."),
        ("cloud_scene_3d.png", "The input point cloud the meshes replaced."),
        ("cloud_scene_topdown.png", "Input point cloud, top down."),
    ]:
        src = copy_image(scene_dir / name, args.out)
        if src:
            overview.append((src, caption))

    glb = scene_dir / "scene.glb"
    stats = [
        ("instances in scene", len(instances)),
        ("reconstructed", len(reconstructed)),
        ("input frames", len(frames)),
    ]
    if glb.exists():
        stats.append(("scene.glb", f"{glb.stat().st_size/1e6:.0f} MB"))

    page = PAGE
    page = page.replace("__TITLE__", "One world, taken apart")
    page = page.replace("__EYEBROW__",
                        "E08c · WorldSculpt · arXiv 2609.05416")
    page = page.replace("__INTRO__", html.escape(
        f"WorldSculpt turns posed frames with per-instance masks and 3D boxes "
        f"into a scene of individual object meshes. This is scene "
        f"'{args.scene_dir.name}' from the authors' own release, run here on "
        f"their released weights. Every strip below is rendered by their "
        f"pipeline: the original frame first, then what the composed meshes "
        f"look like from that same camera."))
    page = page.replace("__STATS__", "".join(
        f"<span>{k}<b>{v}</b></span>" for k, v in stats))
    page = page.replace("__BODY__",
                        section("Per view", "The comparison the method owes: "
                                "its own output, from the camera the input "
                                "frame came from.", strips)
                        + section("The whole scene", "Composition and layout.",
                                  overview))
    page = page.replace("__NOTES__", "".join(f"<li>{n}</li>" for n in [
        "A reproduction arm on the authors' data, not a baseline. No number "
        "here compares methods.",
        "WorldSculpt consumes instance masks and 3D boxes, which their "
        "released scenes carry. Our DL3DV scenes have frames and poses but "
        "neither, so running this on our corpus is a grounding project rather "
        "than a flag on the runner.",
        "Geometry only: the runner passes <code>--no_tex</code>, as the "
        "authors' own example does, so the meshes carry material colour "
        "rather than generated texture.",
    ]))
    (args.out / "index.html").write_text(page)

    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"wrote {args.out/'index.html'}  ({size/1e6:.1f} MB, "
          f"{len(strips)} view strips, {len(overview)} overviews)")


if __name__ == "__main__":
    main()
