"""Build a self-contained comparison viewer for splat assets.

The house demo pattern, tracked so a demo is regenerated rather than patched.
E07's viewer was hand-written and lived only under `outputs/`, which made it
unreproducible; this module replaces that with a generator.

    python -m gs_playground.viewer --spec <spec.json> --out <dir>

The spec is JSON::

    {
      "title": "...", "eyebrow": "...", "intro": "one paragraph",
      "panels": [
        {"label": "...", "ply": "path/to.ply", "caption": "...",
         "up": "z", "meta": {"gaussians": 530336},
         "sh": false}   # true keeps the higher-order SH bands
      ],
      "notes": ["stated caveats, shown on the page"]
    }

Every panel is normalised to its own extent before rendering, so a 6 m
generated world and a 12 m real room occupy the same screen area and can be
compared at all. That normalisation is a lie about scale, so each panel prints
its true metric extent underneath, and the page says so in the notes.

Splats are served as plain `.ply` beside the page, which is what
`python -m http.server` wants and keeps the page a third smaller than
base64-in-JSON would. Publishing to artifact hosting, which serves only
standard web media types, would need the bytes wrapped in JSON instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import torch

from .gs.ply import GaussianCloud, load_ply, save_ply

#: vendored beside the page. Spark imports FullScreenQuad from three's addons,
#: which is a separate file from three.module.js, so the addon tree ships too or
#: the page dies on an unresolved module specifier.
JS_FILES = ("spark.module.min.js", "three.module.js", "three.core.js")
JS_TREES = ("addons",)

TEMPLATE = """<meta charset="utf-8">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
  :root {
    --paper:#EFF1EE; --raised:#FFFFFF; --ink:#151A1E; --ink-dim:#5C666C;
    --rule:#CDD3CE; --dial:#2F7D95; --field:#0B0E11;
    --shadow:0 1px 2px rgba(21,26,30,.06), 0 8px 24px rgba(21,26,30,.05);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper:#10151A; --raised:#171E24; --ink:#E4E9E5; --ink-dim:#8D989E;
      --rule:#29333A; --dial:#5FB6CE;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
    }
  }
  :root[data-theme="dark"] {
    --paper:#10151A; --raised:#171E24; --ink:#E4E9E5; --ink-dim:#8D989E;
    --rule:#29333A; --dial:#5FB6CE;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--paper); color:var(--ink);
         font:400 15px/1.55 "IBM Plex Sans", system-ui, sans-serif;
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1400px; margin:0 auto; padding:32px 24px 64px; }
  header { border-bottom:1px solid var(--rule); padding-bottom:20px; margin-bottom:24px; }
  .eyebrow { font:500 11px/1 "IBM Plex Mono", monospace; letter-spacing:.14em;
             text-transform:uppercase; color:var(--ink-dim); }
  h1 { font:700 clamp(26px,4vw,40px)/1.08 "IBM Plex Sans Condensed", system-ui, sans-serif;
       letter-spacing:-.01em; margin:10px 0 8px; text-wrap:balance; }
  .intro { color:var(--ink-dim); max-width:72ch; }
  .grid { display:grid; gap:20px; grid-template-columns:repeat(auto-fit, minmax(340px, 1fr)); }
  .panel { background:var(--raised); border:1px solid var(--rule); border-radius:10px;
           overflow:hidden; box-shadow:var(--shadow); }
  .panel h2 { font:600 15px/1.2 "IBM Plex Sans", system-ui, sans-serif; margin:0;
              padding:12px 14px 10px; border-bottom:1px solid var(--rule); }
  .stage { aspect-ratio:4/3; background:var(--field); position:relative; }
  .stage canvas { display:block; width:100%; height:100%; }
  .loading { position:absolute; inset:0; display:grid; place-items:center;
             color:#8D989E; font:500 12px "IBM Plex Mono", monospace; }
  .caption { padding:12px 14px; color:var(--ink-dim); font-size:13.5px; }
  .stats { display:flex; flex-wrap:wrap; gap:14px; padding:0 14px 14px;
           font:500 12px/1.3 "IBM Plex Mono", monospace; color:var(--ink-dim); }
  .stats b { display:block; color:var(--ink); font-weight:600; font-size:14px; }
  .notes { margin-top:28px; border-top:1px solid var(--rule); padding-top:16px;
           color:var(--ink-dim); font-size:13.5px; max-width:80ch; }
  .notes li { margin-bottom:6px; }
  .bar { display:flex; gap:10px; align-items:center; margin:18px 0 20px;
         font:500 12px "IBM Plex Mono", monospace; color:var(--ink-dim); }
  button { font:500 12px "IBM Plex Mono", monospace; color:var(--ink);
           background:var(--raised); border:1px solid var(--rule);
           border-radius:6px; padding:6px 10px; cursor:pointer; }
  button[aria-pressed="true"] { border-color:var(--dial); color:var(--dial); }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">__EYEBROW__</div>
    <h1>__TITLE__</h1>
    <p class="intro">__INTRO__</p>
  </header>

  <div class="bar">
    <span>up axis</span>
    <button id="zup" aria-pressed="true">Z up</button>
    <button id="yup" aria-pressed="false">Y up</button>
    <span style="margin-left:auto">drag to orbit, scroll to zoom, all panels move together</span>
  </div>

  <div class="grid" id="grid"></div>
  <ul class="notes" id="notes"></ul>
</div>

<script type="importmap">
{ "imports": {
    "three": "./three.module.js",
    "three/addons/": "./addons/",
    "@sparkjsdev/spark": "./spark.module.min.js"
} }
</script>
<script type="module">
import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const manifest = await (await fetch("./manifest.json")).json();
const grid = document.getElementById("grid");
document.getElementById("notes").innerHTML =
  manifest.notes.map(n => `<li>${n}</li>`).join("");

// One orbit state, shared by every panel: a comparison whose panels can drift
// apart is not a comparison.
const orbit = { azimuth: 0.6, elevation: 0.85, radius: 1.9 };
let zUp = true;
const panels = [];

for (const panel of manifest.panels) {
  const card = document.createElement("div");
  card.className = "panel";
  const stats = Object.entries(panel.meta ?? {}).map(
    ([k, v]) => `<span>${k}<b>${v}</b></span>`).join("");
  card.innerHTML = `
    <h2>${panel.label}</h2>
    <div class="stage"><div class="loading">loading splats…</div></div>
    <div class="caption">${panel.caption ?? ""}</div>
    <div class="stats">${stats}</div>`;
  grid.appendChild(card);
  const stage = card.querySelector(".stage");

  if (panel.image) {
    // Stills are sheets of several views; cropping them to 4:3 hid half the
    // views. Show the whole sheet, and give a wide one the whole row.
    stage.style.aspectRatio = "auto";
    stage.innerHTML = `<img src="${panel.image}" alt="" style="width:100%;height:auto;display:block">`;
    stage.querySelector("img").onload = (event) => {
      const img = event.target;
      if (img.naturalWidth > 2 * img.naturalHeight) card.style.gridColumn = "1 / -1";
    };
    continue;
  }
  const renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  stage.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  scene.add(new SparkRenderer({ renderer }));
  new ResizeObserver(() => {
    const { clientWidth: w, clientHeight: h } = stage;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  }).observe(stage);

  const entry = { renderer, scene, camera, mesh: null, up: panel.up ?? null };
  panels.push(entry);
  attach(renderer.domElement);

  (async () => {
    const bytes = new Uint8Array(await (await fetch(panel.splat)).arrayBuffer());
    const mesh = new SplatMesh({ fileBytes: bytes, fileType: "ply" });
    await mesh.initialized;
    entry.mesh = mesh;
    setUp(entry);
    scene.add(mesh);
    stage.querySelector(".loading")?.remove();
  })();
}

function setUp(entry) {
  if (!entry.mesh) return;
  const up = entry.up ?? (zUp ? "z" : "y");
  entry.mesh.quaternion.set(up === "z" ? 1 : 0, 0, 0, up === "z" ? 0 : 1);
}

function attach(element) {
  let dragging = false, lastX = 0, lastY = 0;
  element.addEventListener("pointerdown", (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    element.setPointerCapture(e.pointerId);
  });
  element.addEventListener("pointerup", () => { dragging = false; });
  element.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    orbit.azimuth -= (e.clientX - lastX) * 0.006;
    orbit.elevation = Math.max(-1.4, Math.min(1.4,
      orbit.elevation + (e.clientY - lastY) * 0.005));
    lastX = e.clientX; lastY = e.clientY;
  });
  element.addEventListener("wheel", (e) => {
    e.preventDefault();
    orbit.radius = Math.max(0.4, Math.min(9,
      orbit.radius * (1 + Math.sign(e.deltaY) * 0.1)));
  }, { passive: false });
}

for (const [id, value] of [["zup", true], ["yup", false]]) {
  document.getElementById(id).addEventListener("click", () => {
    zUp = value;
    document.getElementById("zup").setAttribute("aria-pressed", String(zUp));
    document.getElementById("yup").setAttribute("aria-pressed", String(!zUp));
    panels.forEach(setUp);
  });
}

function frame() {
  const cosE = Math.cos(orbit.elevation);
  for (const { renderer, scene, camera } of panels) {
    camera.position.set(
      orbit.radius * cosE * Math.sin(orbit.azimuth),
      orbit.radius * Math.sin(orbit.elevation),
      orbit.radius * cosE * Math.cos(orbit.azimuth));
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
"""


def normalise(cloud: GaussianCloud) -> tuple[GaussianCloud, float]:
    """Centre on the 1-99 percentile box and scale its longest side to 1.

    Percentiles rather than the raw box: one far-field outlier would otherwise
    shrink the asset to a dot, which is the failure E07's prereg amendment
    already recorded.
    """
    lo = torch.quantile(cloud.means, 0.01, dim=0)
    hi = torch.quantile(cloud.means, 0.99, dim=0)
    centre = (lo + hi) / 2
    scale = float((hi - lo).max().clamp_min(1e-6))
    means = (cloud.means - centre) / scale
    return GaussianCloud(
        means=means,
        f_dc=cloud.f_dc,
        opacity_logit=cloud.opacity_logit,
        log_scales=cloud.log_scales - torch.log(torch.tensor(scale)),
        quats=cloud.quats,
        f_rest=cloud.f_rest,  # SH bands are view directions: scale and shift leave them alone
    ), scale


def subsample(cloud: GaussianCloud, limit: int, seed: int = 0) -> GaussianCloud:
    """Uniform random keep, seeded.

    E07 subsampled by opacity, which is right for a generated asset whose
    primitives are uniformly opaque. On a real reconstruction it is not: the
    most opaque primitives cluster in a few sharp regions, and keeping 120k of
    4.2M that way empties the room. Uniform sampling thins the scene evenly
    instead, which is the honest thing to show beside a generated one.
    """
    if cloud.means.shape[0] <= limit:
        return cloud
    generator = torch.Generator().manual_seed(seed)
    keep = torch.randperm(cloud.means.shape[0], generator=generator)[:limit]
    return cloud.select(keep)


def build(spec: dict, out: Path, js_dir: Path,
          max_gaussians: int = 250_000) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "splats").mkdir(exist_ok=True)

    for name in JS_FILES:
        source = js_dir / name
        if not source.exists():
            raise SystemExit(f"missing {source}; see tools/fetch/viewer_js.sha256")
        shutil.copy2(source, out / name)
    for tree in JS_TREES:
        source = js_dir / tree
        if not source.is_dir():
            raise SystemExit(f"missing {source}; see tools/fetch/viewer_js.sha256")
        shutil.copytree(source, out / tree, dirs_exist_ok=True)

    panels = []
    for panel in spec["panels"]:
        if "image" in panel:
            source = Path(panel["image"])
            (out / "images").mkdir(exist_ok=True)
            shutil.copy2(source, out / "images" / source.name)
            panels.append({
                "label": panel["label"],
                "caption": panel.get("caption", ""),
                "image": f"images/{source.name}",
                "meta": dict(panel.get("meta", {})),
            })
            print(f"{panel['label']}: still image")
            continue
        cloud = load_ply(Path(panel["ply"]))
        if not panel.get("sh"):
            # DC only unless a panel asks: SH bands quadruple the file, and every
            # viewer built before load_ply kept them was DC only
            cloud = dataclasses.replace(cloud, f_rest=None)
        total = int(cloud.means.shape[0])
        cloud = subsample(cloud, max_gaussians)
        cloud, scale = normalise(cloud)
        name = (panel["label"].lower().replace(" ", "_").replace(",", "")
                .replace("(", "").replace(")", "").replace("/", "-") + ".ply")
        save_ply(cloud, out / "splats" / name)

        meta = dict(panel.get("meta", {}))
        meta.setdefault("gaussians", f"{total:,}")
        if total > max_gaussians:
            meta["shown"] = f"{max_gaussians:,}"
        panels.append({
            "label": panel["label"],
            "up": panel.get("up"),
            "caption": panel.get("caption", ""),
            "splat": f"splats/{name}",
            "meta": meta,
        })
        print(f"{panel['label']}: {total:,} gaussians, normalised by {scale:.3f}")

    notes = list(spec.get("notes", []))
    notes.append("Every panel is scaled to its own 1-99 percentile extent so the "
                 "panels are comparable on screen; the true extent is in each "
                 "panel's stats, not in what you see.")
    if any(int(str(p["meta"].get("gaussians", "0")).replace(",", "")) > max_gaussians
           for p in panels):
        notes.append(f"Assets over {max_gaussians:,} gaussians are randomly "
                     "subsampled (seed 0) for the page, which thins a scene "
                     "evenly rather than favouring its sharpest regions. The "
                     "full-resolution assets stay on disk.")

    (out / "manifest.json").write_text(json.dumps(
        {"panels": panels, "notes": notes}, indent=2) + "\n")

    html = TEMPLATE
    for key in ("title", "eyebrow", "intro"):
        html = html.replace(f"__{key.upper()}__", spec.get(key, ""))
    (out / "index.html").write_text(html)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out/'index.html'}  ({size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--js-dir", type=Path,
                    default=Path("data/viewer_js"))
    ap.add_argument("--max-gaussians", type=int, default=250_000)
    args = ap.parse_args()
    build(json.loads(args.spec.read_text()), args.out, args.js_dir,
          args.max_gaussians)


if __name__ == "__main__":
    main()
