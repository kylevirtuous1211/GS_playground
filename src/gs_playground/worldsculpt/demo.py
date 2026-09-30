"""Build a WorldSculpt demo: what it made of a scene, beside its input.

WorldSculpt emits meshes, not splats, so this is an image page rather than the
splat viewer. It leans on the artefact the upstream pipeline already produces:
`_scene/renders/view*.jpg`, each a strip of
`[ original frame | normals | per-instance colour | overlay ]`, which is the
comparison the demo needs and is rendered by their code, not ours.

    python -m gs_playground.worldsculpt.demo \
        --case-root outputs/worldsculpt/Marble/<scene> \
        --scene-dir data/worldsculpt_input/Marble/<scene> \
        --out outputs/worldsculpt/viewer                        # E08c

    python -m gs_playground.worldsculpt.demo --page nchc \
        --case-root outputs/worldsculpt/NCHC/nchc_sofa_20260727_143647 \
        --scene-dir data/worldsculpt_input/NCHC/nchc_sofa_20260727_143647 \
        --objects experiments/worldsculpt/e08e_objects.json \
        --tiles outputs/worldsculpt/NCHC/ground/tiles \
        --checks results/worldsculpt/e08e_checks.json \
        --out outputs/worldsculpt/NCHC/viewer                   # E08e
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


#: page text per run. Each page says on itself what was done to its inputs.
PAGES = {
    "marble": {
        "eyebrow": "E08c · WorldSculpt · arXiv 2609.05416",
        "title": "One world, taken apart",
        "intro": "WorldSculpt turns posed frames with per-instance masks and 3D boxes "
                 "into a scene of individual object meshes. This is scene "
                 "'{scene}' from the authors' own release, run here on "
                 "their released weights. Every strip below is rendered by their "
                 "pipeline: the original frame first, then what the composed meshes "
                 "look like from that same camera.",
        "notes": [
            "A reproduction arm on the authors' data, not a baseline. No number "
            "here compares methods.",
            "WorldSculpt consumes instance masks and 3D boxes, which their "
            "released scenes carry. Our DL3DV scenes have frames and poses but "
            "neither, so running this on our corpus is a grounding project rather "
            "than a flag on the runner.",
            "Geometry only: the runner passes <code>--no_tex</code>, as the "
            "authors' own example does, so the meshes carry material colour "
            "rather than generated texture.",
        ],
    },
    "nchc": {
        "eyebrow": "E08e · WorldSculpt on our own capture · NCHC sofa",
        "title": "Our lounge, taken apart",
        "intro": "WorldSculpt, released weights, run on a handheld video we shot of "
                 "the NCHC sofa lounge. It needs a mask per object per frame and a 3D "
                 "box per object, and its authors leave finding those out of scope, "
                 "so we built them: masks from label maps an earlier project "
                 "associated across views, boxes from our 3DGS of the scene. Every "
                 "strip below is rendered by WorldSculpt's own pipeline: the frame "
                 "first, then its meshes from the same camera.",
        "notes": [
            "<b>Masks are predicted, not annotated.</b> They are HQ-SAM label maps "
            "that Inpaint360GS associated across views in an earlier project "
            "(EditReadyGS), at 636x358, with specks removed, upsampled, clipped to "
            "our box plus a 15% margin, and dropped in frames where the object is "
            "cut off or the mask disagrees with the box. Association errors were "
            "not corrected: the potted plant, for one, carries the walls' id and "
            "so is not here at all.",
            "<b>Boxes are ours, not ground truth</b>: each label id back-projected "
            "through depth rendered from our 3DGS, keeping only voxels seen by "
            "several different frames. WorldSculpt then enlarges a box if the masks "
            "do not fit, as in the authors' script.",
            "<b>Walls, floor and ceiling are missing because we excluded them</b>, "
            "not because of the method: an automatic filter drops structure and "
            "fragments, then objects were reviewed by hand (ceiling lights, ducts, "
            "a pillar, a wall panel and the lawn rejected). Every candidate and its "
            "reason is in <code>experiments/worldsculpt/e08e_objects.json</code>.",
            "Frames: every 3rd of the 364 frames COLMAP registered from a 2 fps "
            "sample of a 4K HDR video (tone-mapped to SDR), 1272x715.",
            "Units are COLMAP units: this capture has no metric scale, so the "
            "\"metric\" in upstream's plot titles is not metres.",
            "Geometry only (<code>--no_tex</code>). Sides no camera saw are "
            "generated by the prior, not reconstructed.",
            "There is no ground truth for this scene, so nothing here is scored; "
            "the strips are the evidence.",
        ],
    },
}


def pick_views(views: list[Path], n: int) -> list[Path]:
    """Evenly spaced along the walk, in numeric order (view100 after view99)."""
    views = sorted(views, key=lambda v: int("".join(ch for ch in v.stem if ch.isdigit()) or 0))
    if len(views) <= n:
        return views
    return [views[round(i * (len(views) - 1) / (n - 1))] for i in range(n)] if n > 1 else views[:1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-root", required=True, type=Path,
                    help="the OUTPUT_ROOT/<scene> directory the run wrote")
    ap.add_argument("--scene-dir", required=True, type=Path,
                    help="the input scene, for its transforms.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--views", type=int, default=3,
                    help="how many per-view strips to show")
    ap.add_argument("--page", choices=sorted(PAGES), default="marble")
    ap.add_argument("--objects", type=Path, help="reviewed selection JSON (E08e)")
    ap.add_argument("--tiles", type=Path, help="grounding review tiles (E08e)")
    ap.add_argument("--checks", type=Path, help="ground check output (E08e)")
    args = ap.parse_args()
    text = PAGES[args.page]

    scene_dir = args.case_root / "_scene"
    transforms = json.loads((args.scene_dir / "transforms.json").read_text())
    instances = transforms.get("instances", [])
    frames = transforms.get("frames", [])
    reconstructed = sorted((args.case_root / "_recon").glob("*/mesh.pt"))

    args.out.mkdir(parents=True, exist_ok=True)

    strips = []
    for view in pick_views(list((scene_dir / "renders").glob("view*.jpg")), args.views):
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

    given = []
    if args.objects:
        selection = json.loads(args.objects.read_text())
        objects = selection["objects"]
        kept = [o for o in objects if o["keep"]]
        manual = [o for o in objects if o["reason"].startswith("manual") and not o["keep"]]
        stats[:1] = [("label ids", len(objects)), ("given to WorldSculpt", len(kept)),
                     ("rejected on review", len(manual))]
        if args.checks and args.checks.exists():
            stages = json.loads(args.checks.read_text())["stages"]
            stats.append(("in scene.glb", f"{stages['in_scene_glb']} / {stages['declared']}"))
        for o in kept:
            if args.tiles and (args.tiles / f"id{o['id']:03d}.jpg").exists():
                src = copy_image(args.tiles / f"id{o['id']:03d}.jpg", args.out)
                given.append((src, f"id {o['id']}: {o['n_views']} views. "
                                   f"{html.escape(o['reason'])}"))

    page = PAGE
    page = page.replace("__TITLE__", text["title"])
    page = page.replace("__EYEBROW__", text["eyebrow"])
    page = page.replace("__INTRO__", html.escape(text["intro"].format(scene=args.scene_dir.name)))
    page = page.replace("__STATS__", "".join(
        f"<span>{k}<b>{v}</b></span>" for k, v in stats))
    page = page.replace("__BODY__",
                        section("Per view", "The comparison the method owes: "
                                "its own output, from the camera the input "
                                "frame came from.", strips)
                        + section("The whole scene", "Composition and layout.",
                                  overview)
                        + section("What it was given", "The input each object was "
                                  "conditioned on: three of its frames with our mask "
                                  "tinted and our box in magenta.", given))
    page = page.replace("__NOTES__", "".join(f"<li>{n}</li>" for n in text["notes"]))
    (args.out / "index.html").write_text(page)

    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"wrote {args.out/'index.html'}  ({size/1e6:.1f} MB, "
          f"{len(strips)} view strips, {len(overview)} overviews)")


if __name__ == "__main__":
    main()
