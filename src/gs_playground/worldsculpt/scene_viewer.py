"""Orbitable WorldSculpt scenes: every object mesh in one frame, one tab per run.

The per-view pages (`demo.py`) compare each output to the frame it came from;
this page is the other half, the whole composed scene in 3D the way the
authors' project page shows it. It reads upstream's own `_scene/scene.glb`,
decimates each object for the browser (cumesh, the tool upstream decimates
with), and writes raw position and index buffers beside a three.js page, so no
glTF loader is needed.

    python -m gs_playground.worldsculpt.scene_viewer \
        --scene marble outputs/worldsculpt/Marble/<scene> data/worldsculpt_input/Marble/<scene> \
        --scene nchc outputs/worldsculpt/NCHC/<scene> data/worldsculpt_input/NCHC/<scene> \
        --out outputs/worldsculpt/viewer3d

Needs the worldsculpt overlay env (cumesh, trimesh) and a GPU.
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

#: three r186, as in gs_playground.viewer, plus the orbit controls addon.
JS_FILES = ("three.module.js", "three.core.js", "addons/controls/OrbitControls.js")

#: page text per run key, shown in the tab and under the stage.
SCENES = {
    "marble": {
        "title": "Marble living room",
        "run": "E08c",
        "notes": [
            "The authors' own released scene, on their weights. Its masks and boxes "
            "come with the release; the masks are SAM3 predictions, not annotations.",
            "Walls and floor are absent because the release gives WorldSculpt "
            "objects only.",
            "Units are the scene's own; up is estimated from the cameras.",
        ],
    },
    "nchc": {
        "title": "NCHC sofa lounge",
        "run": "E08e",
        "notes": [
            "Our handheld capture. Masks are HQ-SAM label maps from an earlier "
            "project, and boxes are ours, estimated from our 3DGS; neither is "
            "ground truth.",
            "Walls, floor and ceiling are absent because we excluded them, and "
            "the potted plant is missing because its mask carries the walls' id.",
            "Units are COLMAP units, not metres; up is estimated from the cameras.",
        ],
    },
}

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WorldSculpt scenes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
  :root { --paper:#EFF1EE; --raised:#FFF; --ink:#151A1E; --ink-dim:#5C666C;
          --rule:#CDD3CE; --dial:#2F7D95; --field:#0B0E11;
          --shadow:0 1px 2px rgba(21,26,30,.06), 0 8px 24px rgba(21,26,30,.05); }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
      --paper:#10151A; --raised:#171E24; --ink:#E4E9E5; --ink-dim:#8D989E;
      --rule:#29333A; --dial:#5FB6CE;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3); } }
  :root[data-theme="dark"] { --paper:#10151A; --raised:#171E24; --ink:#E4E9E5;
      --ink-dim:#8D989E; --rule:#29333A; --dial:#5FB6CE; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--paper); color:var(--ink);
         font:400 15px/1.55 "IBM Plex Sans", system-ui, sans-serif; }
  .wrap { max-width:1300px; margin:0 auto; padding:32px 16px 64px; }
  .eyebrow { font:500 11px/1 "IBM Plex Mono", monospace; letter-spacing:.14em;
             text-transform:uppercase; color:var(--ink-dim); }
  h1 { font:700 clamp(26px,4vw,40px)/1.08 "IBM Plex Sans Condensed", system-ui, sans-serif;
       margin:10px 0 8px; }
  .intro { color:var(--ink-dim); max-width:74ch; margin:0 0 20px; }
  .card { background:var(--raised); border:1px solid var(--rule); border-radius:12px;
          box-shadow:var(--shadow); overflow:hidden; }
  .stage { position:relative; height:min(72vh, 720px); min-height:380px; background:var(--field); }
  .stage canvas { display:block; width:100%; height:100%; touch-action:none; }
  .status { position:absolute; inset:0; display:grid; place-items:center; pointer-events:none;
            color:#8D989E; font:500 12px "IBM Plex Mono", monospace; }
  .hint { position:absolute; left:50%; bottom:16px; transform:translateX(-50%);
          padding:8px 18px; border-radius:999px; border:1px solid rgba(255,255,255,.12);
          background:rgba(11,14,17,.72); color:#A8B1B6; font-size:13px; white-space:nowrap;
          pointer-events:none; max-width:calc(100% - 32px); overflow:hidden; text-overflow:ellipsis; }
  .tip { position:absolute; pointer-events:none; padding:4px 8px; border-radius:6px;
         background:rgba(11,14,17,.88); color:#E4E9E5; font:500 12px "IBM Plex Mono", monospace;
         display:none; white-space:nowrap; }
  .info { position:absolute; top:12px; right:12px; width:min(280px, calc(100% - 24px));
          background:rgba(17,22,27,.92); color:#E4E9E5; border:1px solid rgba(255,255,255,.1);
          border-radius:10px; padding:12px; font-size:13px; display:none; }
  .info h3 { margin:0 0 6px; font:600 14px "IBM Plex Mono", monospace; }
  .info img { display:block; width:100%; max-height:200px; object-fit:contain; margin:8px 0;
              background:repeating-conic-gradient(#1c2329 0 25%, #151b20 0 50%) 0 0/16px 16px;
              border-radius:6px; }
  .info .dim { color:#9AA4AA; }
  .info dl { display:grid; grid-template-columns:auto 1fr; gap:2px 10px; margin:6px 0 0;
             font:400 12px "IBM Plex Mono", monospace; }
  .info dt { color:#9AA4AA; }
  .info dd { margin:0; }
  .tabs { display:flex; flex-wrap:wrap; gap:8px; padding:12px; border-top:1px solid var(--rule); }
  .tabs button, .tools button { font:inherit; color:var(--ink-dim); background:transparent;
          border:1px solid transparent; border-radius:10px; padding:8px 16px; cursor:pointer;
          text-align:center; }
  .tabs button b { display:block; font:500 15px "IBM Plex Sans", sans-serif; color:inherit; }
  .tabs button span { font:400 12px "IBM Plex Mono", monospace; }
  .tabs button[aria-pressed="true"] { color:var(--ink); border-color:var(--rule);
          background:var(--paper); }
  .tools { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; padding:0 12px 12px;
           font:500 12px "IBM Plex Mono", monospace; color:var(--ink-dim); }
  .tools button { padding:4px 10px; border-color:var(--rule); border-radius:6px; }
  .tools label { display:flex; gap:6px; align-items:center; cursor:pointer; }
  .objects { display:flex; flex-wrap:wrap; gap:6px; padding:0 12px 14px; }
  .objects button { display:flex; gap:6px; align-items:center; cursor:pointer;
          font:400 12px "IBM Plex Mono", monospace; color:var(--ink); background:var(--paper);
          border:1px solid var(--rule); border-radius:6px; padding:3px 8px; }
  .objects button[aria-pressed="true"] { border-color:var(--dial); }
  .objects i { width:10px; height:10px; border-radius:3px; display:inline-block; }
  @media (max-width:600px) {
    .info { top:auto; bottom:12px; left:12px; right:12px; width:auto; max-height:55%; overflow:auto; }
    .info img { max-height:110px; }
    .hint { display:none; }
  }
  .notes { margin:28px 0 0; padding:16px 0 0 20px; border-top:1px solid var(--rule);
           color:var(--ink-dim); font-size:13.5px; max-width:82ch; }
  .notes li { margin-bottom:6px; }
</style>
<div class="wrap">
  <div class="eyebrow">E08c + E08e · WorldSculpt · arXiv 2609.05416</div>
  <h1>The worlds it rebuilt, object by object</h1>
  <p class="intro">Every mesh WorldSculpt composed into each scene, in the scene's own frame.
  Colours only tell objects apart: the runs were geometry only, with no texture.
  Hover an object for its name and the box it was given; click it for the image it was conditioned on.</p>
  <div class="card">
    <div class="stage" id="stage">
      <canvas id="canvas"></canvas>
      <div class="status" id="status">loading</div>
      <div class="tip" id="tip"></div>
      <div class="info" id="info"></div>
      <div class="hint">drag to orbit · right-drag to pan · scroll to zoom · hover / click objects</div>
    </div>
    <div class="tabs" id="tabs"></div>
    <div class="tools">
      <label><input type="checkbox" id="boxes"> all input boxes</label>
      <button id="reset">reset view</button>
      <span id="facts"></span>
    </div>
    <div class="objects" id="objects"></div>
  </div>
  <ul class="notes" id="notes"></ul>
</div>
<script type="importmap">
{ "imports": { "three": "./three.module.js", "three/addons/": "./addons/" } }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const manifest = await (await fetch("manifest.json")).json();
const $ = (id) => document.getElementById(id);
const stage = $("stage"), statusEl = $("status"), tip = $("tip"), info = $("info");

const renderer = new THREE.WebGLRenderer({ canvas: $("canvas"), antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e11);
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 500);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2f36, 1.7));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(2, 5, 3);
scene.add(sun);

const EDGES = [];
for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++)
  if ([1, 2, 4].includes(i ^ j)) EDGES.push(i, j);

let current = null, hovered = null, selected = null, home = null;
const pointer = new THREE.Vector2(), ray = new THREE.Raycaster();
let pointerDirty = false, downAt = null;

function boxLines(corners, color) {
  const pos = new Float32Array(EDGES.flatMap((k) => corners.slice(3 * k, 3 * k + 3)));
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const line = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: .9 }));
  line.visible = false;
  return line;
}

function dispose() {
  if (!current) return;
  current.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
  scene.remove(current.group);
  current = null; hovered = null; selected = null;
}

async function load(index) {
  dispose();
  const s = manifest.scenes[index];
  document.querySelectorAll("#tabs button").forEach((b, i) =>
    b.setAttribute("aria-pressed", String(i === index)));
  statusEl.textContent = `loading ${s.title} (${(s.bytes / 1e6).toFixed(0)} MB)`;
  statusEl.style.display = "grid";
  const buf = await (await fetch(s.bin)).arrayBuffer();
  const group = new THREE.Group(), meshes = [];
  for (const o of s.objects) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buf, o.pos[0], o.pos[1]), 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(buf, o.idx[0], o.idx[1]), 1));
    g.computeVertexNormals();
    g.computeBoundingSphere();
    const mat = new THREE.MeshStandardMaterial({ color: o.color, roughness: .85, metalness: 0 });
    const mesh = new THREE.Mesh(g, mat);
    mesh.userData = { o, box: boxLines(o.box, 0xffffff) };
    group.add(mesh, mesh.userData.box);
    meshes.push(mesh);
  }
  scene.add(group);
  current = { s, group, meshes };
  // back the camera off along a fixed 3/4 view until every corner of every
  // object's box is inside the frame (the scene's own box is mostly empty)
  const box = new THREE.Box3().setFromObject(group), center = box.getCenter(new THREE.Vector3());
  const back = new THREE.Vector3(.45, .62, .64).normalize();
  const right = new THREE.Vector3(0, 1, 0).cross(back).normalize(), up = back.clone().cross(right);
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * .9, tanH = tanV * stage.clientWidth / stage.clientHeight;
  let d = 0;
  for (const m of meshes) {
    m.geometry.computeBoundingBox();
    const { min, max } = m.geometry.boundingBox;
    for (let i = 0; i < 8; i++) {
      const p = new THREE.Vector3(i & 1 ? max.x : min.x, i & 2 ? max.y : min.y, i & 4 ? max.z : min.z).sub(center);
      const z = p.dot(back);
      d = Math.max(d, z + Math.abs(p.dot(right)) / tanH, z + Math.abs(p.dot(up)) / tanV);
    }
  }
  home = { target: center, position: center.clone().addScaledVector(back, d) };
  const radius = box.getSize(new THREE.Vector3()).length() / 2;
  camera.near = radius / 200; camera.far = radius * 20; camera.updateProjectionMatrix();
  resetView();
  $("boxes").checked = false;
  $("facts").textContent = `${s.objects.length} objects · ${s.faces_shown.toLocaleString()} faces shown of ` +
                           `${s.faces_source.toLocaleString()} in scene.glb`;
  $("objects").innerHTML = "";
  meshes.forEach((m) => {
    const b = document.createElement("button");
    b.innerHTML = `<i style="background:${m.userData.o.color}"></i>${m.userData.o.name}`;
    b.setAttribute("aria-pressed", "false");
    b.onclick = () => select(selected === m ? null : m);
    m.userData.chip = b;
    $("objects").append(b);
  });
  $("notes").innerHTML = s.notes.concat(manifest.notes).map((n) => `<li>${n}</li>`).join("");
  info.style.display = "none";
  statusEl.style.display = "none";
}

function resetView() {
  camera.position.copy(home.position);
  controls.target.copy(home.target);
  controls.update();
}

function paint() {
  if (!current) return;
  const all = $("boxes").checked;
  for (const m of current.meshes) {
    const on = m === hovered || m === selected;
    m.material.emissive.setHex(on ? 0x3a3a3a : 0x000000);
    const dim = selected && m !== selected;
    if (m.material.transparent !== dim) m.material.needsUpdate = true;  // swaps the shader program
    m.material.transparent = dim; m.material.opacity = dim ? .16 : 1; m.material.depthWrite = !dim;
    m.userData.box.visible = all || on;
    m.userData.chip.setAttribute("aria-pressed", String(m === selected));
  }
}

function select(m) {
  selected = m;
  if (hovered !== m) { hovered = null; tip.style.display = "none"; }
  paint();
  if (!m) { info.style.display = "none"; return; }
  const o = m.userData.o;
  const ext = o.extent.map((v) => v.toFixed(2)).join(" × ");
  info.innerHTML = `<h3>${o.name}</h3>
    <div class="dim">What it was given: its anchor crop, the view with the largest mask, one of ${o.views} it saw.</div>
    ${o.crop ? `<img src="${o.crop}" alt="input crop for ${o.name}">` : ""}
    <dl><dt>label id</dt><dd>${o.label}</dd>
        <dt>input box</dt><dd>${ext}</dd>
        <dt>faces</dt><dd>${o.faces.toLocaleString()} of ${o.faces_source.toLocaleString()}</dd>
        <dt>area kept</dt><dd>${Math.round(o.area_kept * 100)}%</dd></dl>`;
  info.style.display = "block";
}

renderer.domElement.addEventListener("pointermove", (e) => {
  const r = renderer.domElement.getBoundingClientRect();
  pointer.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
  tip.style.left = `${e.clientX - r.left + 14}px`; tip.style.top = `${e.clientY - r.top + 14}px`;
  pointerDirty = true;
});
renderer.domElement.addEventListener("pointerleave", () => { hovered = null; tip.style.display = "none"; paint(); });
renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 5) select(hovered);
  downAt = null;
});
addEventListener("keydown", (e) => { if (e.key === "Escape") select(null); });
$("boxes").onchange = paint;
$("reset").onclick = resetView;

function pick() {
  if (!pointerDirty || !current) return;
  pointerDirty = false;
  ray.setFromCamera(pointer, camera);
  const pool = selected ? [selected] : current.meshes;
  const hit = ray.intersectObjects(pool, false)[0];
  const next = hit ? hit.object : null;
  if (next !== hovered) { hovered = next; paint(); }
  tip.style.display = hovered ? "block" : "none";
  if (hovered) tip.textContent = `${hovered.userData.o.name} · ${hovered.userData.o.views} views`;
}

new ResizeObserver(() => {
  const { clientWidth: w, clientHeight: h } = stage;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}).observe(stage);

manifest.scenes.forEach((s, i) => {
  const b = document.createElement("button");
  b.innerHTML = `<b>${s.title}</b><span>${s.objects.length} objects · ${s.run}</span>`;
  b.onclick = () => load(i);
  $("tabs").append(b);
});

renderer.setAnimationLoop(() => { controls.update(); pick(); renderer.render(scene, camera); });
await load(0);
</script>
"""


def display_frame(c2w: np.ndarray) -> np.ndarray:
    """Rotation from world to a y-up frame whose -z is where the cameras looked.

    Up is the normal of the plane the cameras' right vectors span: a handheld
    or rendered camera barely rolls, so this ignores pitch, which a mean of
    their up vectors does not (E08c's cameras pitch 28 degrees down).
    `c2w` are OpenGL camera-to-world matrices, (N, 4, 4).
    """
    right, up, back = c2w[:, :3, 0], c2w[:, :3, 1], c2w[:, :3, 2]
    y = np.linalg.svd(right)[2][-1]
    y *= np.sign(y @ up.mean(0))
    z = back.mean(0)
    if np.linalg.norm(z - (z @ y) * y) < 1e-3:  # cameras all round a circle: any horizontal axis
        z = np.eye(3)[np.argmin(np.abs(y))]
    z = z - (z @ y) * y
    z /= np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z])


def decimate(vertices: np.ndarray, faces: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) <= target:
        return vertices, faces
    import cumesh
    import torch
    mesh = cumesh.CuMesh()
    mesh.init(torch.from_numpy(vertices).float().cuda().contiguous(),
              torch.from_numpy(faces).int().cuda().contiguous())
    mesh.simplify(target, verbose=False)
    mesh.remove_duplicate_faces()
    v, f = mesh.read()
    return v.cpu().numpy(), f.cpu().numpy()


def area(vertices: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)


def crop_image(source: Path, target: Path, size: int = 360) -> bool:
    """The RGBA crop the model saw, trimmed to its mask and shrunk."""
    if not source.exists():
        return False
    image = Image.open(source).convert("RGBA")
    bbox = image.getchannel("A").getbbox()
    if bbox:
        image = image.crop(bbox)
    image.thumbnail((size, size))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, optimize=True)
    return True


#: OKLCH L 0.80 C 0.10 at twelve hues 30 degrees apart, stepped 150 degrees so
#: consecutive objects sit far apart on the wheel. Even lightness, unlike HLS,
#: where every other step lands in a wide band of greens.
PALETTE = ("#f8a49d", "#71d3ba", "#dfa8e1", "#bbc679", "#9ebdff", "#f0ad7f",
           "#65d0dc", "#f2a3c1", "#95cf96", "#c2b1f8", "#dbb970", "#7ac8f5")


def build_scene(key: str, case_root: Path, scene_dir: Path, out: Path, faces: int) -> dict:
    import trimesh

    transforms = json.loads((scene_dir / "transforms.json").read_text())
    rot = display_frame(np.array([f["transform_matrix"] for f in transforms["frames"]]))
    instances = {f"obj{i['pass_index']:02d}": i for i in transforms["instances"]}
    anchors = json.loads((case_root / "_crops/anchor_views.json").read_text())["anchors"]
    glb = trimesh.load(case_root / "_scene/scene.glb")

    names = sorted(glb.geometry, key=lambda n: int(n[3:]))
    parts = {}
    for name in names:
        mesh = glb.geometry[name]
        v, f = decimate(np.asarray(mesh.vertices), np.asarray(mesh.faces), faces)
        # QEM keeps a surface's area; a large loss means the source was not one
        # surface (spconv-era dust decimated to 19% of its area), so say so
        kept = area(v, f) / area(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        if kept < 0.9:
            print(f"[{key}] WARN {name}: decimation kept {kept:.0%} of the surface area")
        parts[name] = (v @ rot.T, f, len(mesh.faces), kept)
    low = np.min([p[0].min(0) for p in parts.values()], 0)
    high = np.max([p[0].max(0) for p in parts.values()], 0)
    shift = np.array([(low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2])

    blob, objects = bytearray(), []
    for i, name in enumerate(names):
        v, f, n_source, kept = parts[name]
        inst, anchor = instances[name], anchors.get(name, {})
        corners = np.array(list(itertools.product(*zip(*inst["aabb_world"]))))
        crop = f"crops/{key}/{name}.png"
        has_crop = anchor and crop_image(
            case_root / "_crops" / name / anchor["anchor_file"], out / crop)
        pos = (v - shift).astype("<f4").tobytes()
        idx = f.astype("<u4").tobytes()
        objects.append({
            "name": name, "label": inst["label"], "color": PALETTE[i % len(PALETTE)],
            "views": anchor.get("n_views", inst.get("n_mask_frames")),
            "pos": [len(blob), len(pos) // 4], "idx": [len(blob) + len(pos), len(idx) // 4],
            "faces": len(f), "faces_source": n_source, "area_kept": round(kept, 3),
            "extent": (np.subtract(*inst["aabb_world"][::-1])).round(3).tolist(),
            "box": ((corners @ rot.T - shift).astype("<f4")).ravel().round(4).tolist(),
            "crop": crop if has_crop else None,
        })
        blob += pos + idx
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    (out / f"scenes/{key}.bin").write_bytes(blob)
    print(f"[{key}] {len(objects)} objects, "
          f"{sum(o['faces'] for o in objects):,} of {sum(o['faces_source'] for o in objects):,} "
          f"faces, {len(blob)/1e6:.0f} MB, least area kept "
          f"{min(o['area_kept'] for o in objects):.0%}")
    return {"key": key, **SCENES[key], "bin": f"scenes/{key}.bin", "bytes": len(blob),
            "objects": objects,
            "faces_shown": sum(o["faces"] for o in objects),
            "faces_source": sum(o["faces_source"] for o in objects)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", nargs=3, action="append", required=True,
                    metavar=("KEY", "CASE_ROOT", "SCENE_DIR"),
                    help=f"one run; KEY picks its page text, one of {sorted(SCENES)}")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--faces", type=int, default=60_000,
                    help="per-object face budget for the browser")
    ap.add_argument("--js-dir", type=Path, default=Path("data/viewer_js"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in JS_FILES:
        source = args.js_dir / name
        if not source.exists():
            raise SystemExit(f"missing {source}; see tools/fetch/viewer_js.sha256")
        (args.out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, args.out / name)

    scenes = [build_scene(key, Path(case), Path(inp), args.out, args.faces)
              for key, case, inp in args.scene]
    manifest = {"scenes": scenes, "notes": [
        "These are the 2026-09-30 reruns. Before them, both runs decoded every "
        "object with the shape decoder's convolutions at random initialisation, "
        "which turned each object into thousands of millimetre-sized blobs "
        "(<code>LOG.md</code> E08g).",
        f"Each object is decimated for the browser to at most {args.faces:,} faces "
        "with cumesh, the tool upstream decimates with; each object's panel gives "
        "its face counts and the share of surface area the decimation kept. "
        "Shading uses normals recomputed from that mesh.",
        "The white box is the one in the run's <code>transforms.json</code>, before "
        "upstream enlarges it to fit the masks.",
        "The crop is exactly what the model was conditioned on for that view, "
        "trimmed to its mask; the checkerboard is transparency.",
        "The per-view comparisons, each render beside its input frame, are on the "
        "E08c and E08e pages; see <code>results/DEMOS.md</code>.",
    ]}
    for scene in scenes:
        scene["notes"] = [html.escape(n) for n in scene["notes"]]
    (args.out / "manifest.json").write_text(json.dumps(manifest))
    (args.out / "index.html").write_text(PAGE)
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"wrote {args.out/'index.html'}  ({size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
