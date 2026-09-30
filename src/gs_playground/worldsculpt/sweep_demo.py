"""Build the E08f demo: parameters vs how many objects come out, and how complete.

Reads the two tracked results files and draws each pre-registered reading as
it was fixed in `experiments/worldsculpt/PREREG_E08f.md`: Stage A as one row of
small charts per grounding parameter, Stage B as the per-config change against
the seed-noise band, plus the view-count curve and held-out silhouettes.

    python -m gs_playground.worldsculpt.sweep_demo \
        --stage-a results/worldsculpt/e08f_stage_a.json \
        --stage-b results/worldsculpt/e08f_stage_b.json \
        --data-dir ... --model-dir ... --out outputs/worldsculpt/NCHC/sweep/viewer
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

STYLE = """
.viz-root { color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --accent:#2a78d6; --accent-soft:#86b6ef; --neg:#e34948; --band:#f0efec; --trace:#c3c2b7; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --accent:#3987e5; --accent-soft:#184f95; --neg:#e66767; --band:#383835; --trace:#52514e; } }
:root[data-theme="dark"] .viz-root { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --accent:#3987e5; --accent-soft:#184f95; --neg:#e66767; --band:#383835; --trace:#52514e; }
* { box-sizing:border-box; }
body { margin:0; }
.viz-root { background:var(--page); color:var(--ink); min-height:100vh;
  font:400 15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1240px; margin:0 auto; padding:28px 16px 64px; }
.eyebrow { font:500 11px/1 ui-monospace,monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-2); }
h1 { font-size:clamp(24px,4vw,36px); line-height:1.1; margin:10px 0 8px; }
h2 { font-size:18px; margin:40px 0 4px; }
.sub, .intro { color:var(--ink-2); max-width:78ch; margin:4px 0 14px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
.grid2 { display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(min(360px,100%),1fr)); }
.rowlabel { font-weight:600; margin:0 0 6px; }
.rowlabel span { font-weight:400; color:var(--ink-2); font-size:13px; }
svg { display:block; width:100%; height:auto; overflow:visible; }
svg text { fill:var(--muted); font:12px system-ui,-apple-system,"Segoe UI",sans-serif; }
svg text.val { fill:var(--ink-2); }
svg text.ttl { fill:var(--ink-2); font-weight:600; }
.hit { fill:transparent; cursor:default; }
.hit:hover + .mark, .hit:focus + .mark { opacity:.75; }
#tip { position:fixed; pointer-events:none; background:var(--surface); color:var(--ink); border:1px solid var(--border);
  border-radius:6px; padding:6px 9px; font-size:12.5px; box-shadow:0 4px 14px rgba(0,0,0,.12); display:none; max-width:280px; z-index:9; }
#tip b { display:block; font-size:14px; }
.legend { display:flex; gap:16px; flex-wrap:wrap; font-size:12.5px; color:var(--ink-2); margin:6px 0 0; }
.legend i { display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:6px; border-radius:2px; }
.legend i.box { height:10px; }
table { border-collapse:collapse; font-size:13px; width:100%; }
th, td { text-align:left; padding:4px 8px; border-bottom:1px solid var(--grid); font-variant-numeric:tabular-nums; }
th { color:var(--ink-2); font-weight:600; }
details { margin-top:12px; } summary { cursor:pointer; color:var(--ink-2); }
.tablewrap { overflow-x:auto; }
.sil { display:grid; gap:10px; grid-template-columns:repeat(auto-fit,minmax(min(150px,100%),1fr)); margin-bottom:14px; }
.sil figure { margin:0; } .sil img { width:100%; border-radius:6px; display:block; }
.sil figcaption { font-size:12px; color:var(--ink-2); margin-top:3px; }
.notes { margin-top:36px; border-top:1px solid var(--grid); padding-top:14px; color:var(--ink-2); font-size:13.5px; max-width:84ch; }
.notes li { margin-bottom:6px; }
"""

TIP_JS = """
const tip = document.getElementById('tip');
function show(e) {
  const t = e.currentTarget; tip.replaceChildren();
  const b = document.createElement('b'); b.textContent = t.dataset.v; tip.appendChild(b);
  tip.appendChild(document.createTextNode(t.dataset.k)); tip.style.display = 'block';
  const r = t.getBoundingClientRect();
  tip.style.left = Math.min(window.innerWidth - 290, r.left + r.width / 2 + 8) + 'px';
  tip.style.top = Math.max(8, r.top - 44) + 'px';
}
document.querySelectorAll('.hit').forEach(el => {
  el.addEventListener('pointerenter', show); el.addEventListener('focus', show);
  el.addEventListener('pointerleave', () => tip.style.display = 'none');
  el.addEventListener('blur', () => tip.style.display = 'none');
});
"""


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def num(v, spec: str = ".3f") -> str:
    """Blank for a missing value, formatted otherwise."""
    return "" if v is None else format(v, spec)


def hit(x, y, w, h, value, key) -> str:
    return (f'<rect class="hit" tabindex="0" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'data-v="{esc(value)}" data-k="{esc(key)}"/>')


def column_chart(labels, values, highlight, title, fmt="{:d}") -> str:
    """Columns from one baseline; the default arm in the accent, the rest soft."""
    W, H, L, B, T = 460, 170, 30, 26, 44
    vmax = max(values) or 1
    band = (W - L - 8) / len(values)
    bar = min(24, band * 0.6)
    parts = [f'<text class="ttl" x="0" y="12">{esc(title)}</text>',
             f'<line x1="{L}" x2="{W - 8}" y1="{H - B}" y2="{H - B}" stroke="var(--axis)"/>']
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = L + band * (i + 0.5)
        h = (H - B - T) * v / vmax
        y = H - B - h
        color = "var(--accent)" if i == highlight else "var(--accent-soft)"
        r = min(4, h / 2)
        path = (f"M{cx - bar / 2:.1f},{H - B} V{y + r:.1f} Q{cx - bar / 2:.1f},{y:.1f} {cx - bar / 2 + r:.1f},{y:.1f} "
                f"H{cx + bar / 2 - r:.1f} Q{cx + bar / 2:.1f},{y:.1f} {cx + bar / 2:.1f},{y + r:.1f} V{H - B} Z")
        parts.append(hit(cx - band / 2, T - 6, band, H - B - T + 6, fmt.format(v), f"{title}, {lab}"))
        parts.append(f'<path class="mark" d="{path}" fill="{color}"/>')
        parts.append(f'<text class="val" x="{cx:.1f}" y="{y - 5:.1f}" text-anchor="middle">{esc(fmt.format(v))}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{H - B + 16}" text-anchor="middle">{esc(lab)}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">{"".join(parts)}</svg>'


def line_chart(labels, values, highlight, title, lo, hi, fmt="{:.2f}") -> str:
    W, H, L, B, T = 460, 170, 36, 26, 44
    band = (W - L - 8) / len(values)
    y = lambda v: H - B - (H - B - T) * (v - lo) / (hi - lo)
    pts = [(L + band * (i + 0.5), y(v)) for i, v in enumerate(values) if v is not None]
    parts = [f'<text class="ttl" x="0" y="12">{esc(title)}</text>']
    for tick in np.linspace(lo, hi, 3):
        parts.append(f'<line x1="{L}" x2="{W - 8}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="var(--grid)"/>')
        parts.append(f'<text x="{L - 6}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick:.2f}</text>')
    if len(pts) > 1:
        parts.append('<polyline fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" '
                     f'points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}"/>')
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = L + band * (i + 0.5)
        parts.append(f'<text x="{cx:.1f}" y="{H - B + 16}" text-anchor="middle">{esc(lab)}</text>')
        if v is None:
            continue
        parts.append(hit(cx - band / 2, T - 6, band, H - B - T + 6, fmt.format(v), f"{title}, {lab}"))
        rr = 5 if i == highlight else 4
        parts.append(f'<circle class="mark" cx="{cx:.1f}" cy="{y(v):.1f}" r="{rr}" fill="var(--accent)" '
                     f'stroke="var(--surface)" stroke-width="2"/>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">{"".join(parts)}</svg>'


def stage_a_section(a: dict) -> str:
    rows = a["rows"]
    order = []
    for r in rows:
        if r["param"] not in order:
            order.append(r["param"])
    blurbs = {
        "support_frac": "share of an id's frames a voxel must be seen in to count toward its box",
        "min_views": "usable views an object needs at the run stride",
        "speck_frac": "mask pieces smaller than this share of the frame's mask are dropped",
        "largest_only": "keep only the largest mask piece per frame",
        "stride": "every n-th frame is handed to WorldSculpt",
        "clip_gate": "drop frames whose mask spills outside the object's box",
        "struct_extent": "boxes wider than this share of the scene are structure",
    }
    cards = []
    for param in order:
        rs = sorted((r for r in rows if r["param"] == param),
                    key=lambda r: (not r["value"]) if isinstance(r["value"], bool) else r["value"])
        if param == "largest_only":
            base = next(r for r in rows if r["param"] == "speck_frac" and r["default"])
            rs = [dict(base, value=False)] + rs
        labels = [str(r["value"]).replace("True", "on").replace("False", "off") for r in rs]
        default = next((i for i, r in enumerate(rs) if r["default"]), 0)
        if param == "largest_only":
            default = 0
        counts = [r["n_kept"] for r in rs]
        comp = [r["box_completeness_median"] for r in rs]
        cards.append(
            f'<div class="card"><p class="rowlabel">{esc(param)} <span>{esc(blurbs.get(param, ""))}</span></p>'
            f'<div class="grid2">{column_chart(labels, counts, default, "objects kept")}'
            f'{line_chart(labels, comp, default, "box completeness (median)", 0.6, 1.0)}</div></div>')
    table = "".join(
        f"<tr><td>{esc(r['param'])}</td><td>{esc(r['value'])}{' (default)' if r['default'] else ''}</td>"
        f"<td>{r['n_kept']}</td><td>{r['jaccard_vs_default']:.2f}</td>"
        f"<td>{num(r['box_completeness_median'])}</td>"
        f"<td>{esc(', '.join(map(str, r['added'])))}</td><td>{esc(', '.join(map(str, r['removed'])))}</td>"
        f"<td>{esc(', '.join(f'{k} {v}' for k, v in r['reasons'].items() if k != 'auto: kept'))}</td></tr>"
        for r in rows)
    return (
        '<h2>Stage A: grounding parameters, how many objects come out</h2>'
        '<p class="sub">One parameter at a time, the rest at E08e\'s defaults (darker column / larger dot), over all 256 '
        'label ids and the automatic filter only. Box completeness is the share of an object\'s back-projected points '
        'that fall inside its box.</p>'
        + "".join(cards) +
        '<details><summary>Table view: every Stage A setting</summary><div class="tablewrap card"><table>'
        '<tr><th>parameter</th><th>value</th><th>kept</th><th>Jaccard vs default</th><th>box completeness</th>'
        '<th>added ids</th><th>removed ids</th><th>rejections by reason</th></tr>'
        + table + '</table></div></details>')


def reading_chart(summary: dict, metric: str, title: str, results: dict, objects: list) -> str:
    """Median change per config against the seed-noise band: the pre-registered rule, drawn."""
    cfgs = [c for c in summary["configs"]]
    spread = summary["seed_spread_median"] or 0.0
    seeds = ["default_s42", "default_s0", "default_s1"]
    mean = {}
    for k in objects:
        vals = [results[s][str(k)].get(metric) for s in seeds]
        if all(v is not None for v in vals):
            mean[k] = float(np.mean(vals))
    deltas = {c: [results[c][str(k)][metric] - mean[k] for k in mean
                  if results[c][str(k)].get(metric) is not None] for c in cfgs}
    ext = max([abs(v) for d in deltas.values() for v in d] + [spread, 0.05]) * 1.1
    W, rowh, L, R, T = 760, 30, 90, 200, 30
    H = T + rowh * len(cfgs) + 26
    x = lambda v: L + (W - L - R) * (v + ext) / (2 * ext)
    parts = [f'<text class="ttl" x="0" y="14">{esc(title)}</text>',
             f'<rect x="{x(-spread):.1f}" y="{T - 4}" width="{x(spread) - x(-spread):.1f}" height="{rowh * len(cfgs) + 4}" fill="var(--band)"/>',
             f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{T - 4}" y2="{T + rowh * len(cfgs)}" stroke="var(--axis)"/>']
    for tick in (-ext / 1.1, 0, ext / 1.1):
        parts.append(f'<text x="{x(tick):.1f}" y="{H - 6}" text-anchor="middle">{tick:+.2f}</text>')
    for i, c in enumerate(cfgs):
        cy = T + rowh * i + rowh / 2
        md = summary["configs"][c]["median_delta"]
        verdict = summary["configs"][c]["verdict"]
        parts.append(f'<text x="{L - 10}" y="{cy + 4:.1f}" text-anchor="end">{esc(c)}</text>')
        for v in deltas[c]:
            parts.append(f'<circle cx="{x(v):.1f}" cy="{cy:.1f}" r="3" fill="var(--trace)"/>')
        if md is None:
            parts.append(f'<text x="{W - R + 10}" y="{cy + 4:.1f}" class="val">no meshes</text>')
            continue
        x0, x1 = sorted((x(0), x(md)))
        color = "var(--accent)" if md >= 0 else "var(--neg)"
        parts.append(hit(L, cy - rowh / 2, W - L - R, rowh, f"{md:+.3f}",
                         f"{c}: median change vs the default seeds; {verdict} (band ±{spread:.3f})"))
        parts.append(f'<rect class="mark" x="{x0:.1f}" y="{cy - 5:.1f}" width="{max(x1 - x0, 1.5):.1f}" height="10" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{W - R + 10}" y="{cy + 4:.1f}" class="val">{md:+.3f} · {esc(verdict)}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">{"".join(parts)}</svg>'


def views_curve(results: dict, objects: list, metric: str, title: str) -> str:
    """Per object in gray, the median in the accent; seed range at the default 20 views."""
    arms = [("views1", 1), ("views3", 3), ("views6", 6), ("views12", 12), ("default_s42", 20)]
    W, H, L, B, T = 380, 226, 36, 44, 24
    xs = [L + (W - L - 12) * i / (len(arms) - 1) for i in range(len(arms))]
    y = lambda v: H - B - (H - B - T) * v
    parts = [f'<text class="ttl" x="0" y="12">{esc(title)}</text>']
    for tick in (0, 0.5, 1.0):
        parts.append(f'<line x1="{L}" x2="{W - 12}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="var(--grid)"/>')
        parts.append(f'<text x="{L - 6}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick:.1f}</text>')
    seeds = [results[s] for s in ("default_s42", "default_s0", "default_s1")]
    for k in objects:
        vals = [results[a][str(k)].get(metric) for a, _ in arms]
        if any(v is None for v in vals):
            continue
        parts.append('<polyline fill="none" stroke="var(--trace)" stroke-width="1" '
                     f'points="{" ".join(f"{a:.1f},{y(v):.1f}" for a, v in zip(xs, vals))}"/>')
    med = []
    for (a, n), cx in zip(arms, xs):
        vals = [results[a][str(k)].get(metric) for k in objects if results[a][str(k)].get(metric) is not None]
        med.append(float(np.median(vals)) if vals else None)
        parts.append(f'<text x="{cx:.1f}" y="{H - B + 17}" text-anchor="middle">{n}</text>')
    sv = [float(np.median([s[str(k)][metric] for k in objects if s[str(k)].get(metric) is not None])) for s in seeds]
    parts.append(f'<line x1="{xs[-1]:.1f}" x2="{xs[-1]:.1f}" y1="{y(max(sv)):.1f}" y2="{y(min(sv)):.1f}" '
                 f'stroke="var(--ink-2)" stroke-width="6" stroke-linecap="round" opacity=".35"/>')
    parts.append('<polyline fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" '
                 f'points="{" ".join(f"{a:.1f},{y(v):.1f}" for a, v in zip(xs, med) if v is not None)}"/>')
    for (a, n), cx, v in zip(arms, xs, med):
        if v is None:
            continue
        parts.append(hit(cx - 20, T - 6, 40, H - B - T + 6, f"{v:.3f}", f"{title}, {n} view(s), median of {len(objects)} objects"))
        parts.append(f'<circle class="mark" cx="{cx:.1f}" cy="{y(v):.1f}" r="4" fill="var(--accent)" stroke="var(--surface)" stroke-width="2"/>')
    parts.append(f'<text class="val" x="{xs[0] + 9:.1f}" y="{y(med[0]) + 16:.1f}" text-anchor="start">{med[0]:.2f}</text>')
    parts.append(f'<text class="val" x="{xs[-1] - 4:.1f}" y="{y(med[-1]) - 9:.1f}" text-anchor="end">{med[-1]:.2f}</text>')
    parts.append(f'<text x="{(xs[0] + xs[-1]) / 2:.1f}" y="{H - 2}" text-anchor="middle">views given (max_views)</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">{"".join(parts)}</svg>'


def object_window(mask: np.ndarray, w: int, h: int, pad: float = 0.9, min_side: int = 140):
    """A square window around the object, so a small object is legible."""
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(min_side, (1 + pad) * max(xs.max() - xs.min(), ys.max() - ys.min()))
    side = min(side, w, h)
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return x0, y0, int(side)


def crop_to(img: np.ndarray, window) -> np.ndarray:
    import cv2
    x0, y0, side = window
    return cv2.resize(img[y0:y0 + side, x0:x0 + side], (320, 320), interpolation=cv2.INTER_AREA)


def silhouettes(b: dict, args, out: Path, picks: list[int], cfgs: list[str]) -> str:
    """One held-out frame per object: our mask outlined, each config's mesh tinted."""
    import cv2
    from PIL import Image
    from . import ground as g
    from . import sweep as s
    selection = json.loads(args.objects.read_text())
    boxes = {o["id"]: o["aabb_world"] for o in selection["objects"]}
    configs = json.loads(args.configs.read_text())
    scene = g.load_scene(args.data_dir, args.model_dir)
    p = g.Params(**selection["params"])
    w, h = scene.label_wh
    (out / "sil").mkdir(parents=True, exist_ok=True)
    rows = []
    for k in picks:
        frames = s.heldout_frames(scene, k, boxes[k], selection["E"], p)
        if not frames:
            continue
        stem, mask = frames[len(frames) // 2]
        cam = g.scaled(scene.cams[stem], w, h)
        base = np.asarray(Image.fromarray(scene.image(stem)).resize((w, h))).copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crop = object_window(mask, w, h)
        figs = []
        for c in cfgs:
            case = Path(configs["cases"][next(x["case"] for x in configs["configs"] if x["name"] == c)])
            mesh = case / f"_sweep_{c}" / f"obj{k:02d}" / "mesh.pt"
            img = base.copy()
            if mesh.exists():
                verts, faces = s.load_mesh(mesh)
                sil = s.silhouette(s.sample_surface(verts, faces, 120_000), cam, w, h)
                img[sil] = (0.45 * img[sil] + 0.55 * np.array([42, 120, 214])).astype(np.uint8)
            cv2.drawContours(img, contours, -1, (255, 255, 255), 2)
            img = crop_to(img, crop)
            name = f"sil/obj{k:02d}_{c}.jpg"
            Image.fromarray(img).save(out / name, quality=85)
            m = b["results"][c][str(k)]
            iou = m.get("silhouette_iou")
            figs.append(f'<figure><img src="{name}" alt="object {k}, {esc(c)}"><figcaption>{esc(c)} · '
                        f'completeness {num(m.get("completeness"), ".2f")} · held-out IoU {num(iou, ".2f") or "-"}'
                        f'</figcaption></figure>')
        rows.append(f'<p class="rowlabel">object {k} <span>held-out frame {esc(stem)}: white outline = our mask, '
                    f'blue = the mesh\'s silhouette, cropped around the object</span></p><div class="sil">{"".join(figs)}</div>')
    return "".join(rows)


def stage_b_section(b: dict, sil_html: str) -> str:
    objects = b["objects"]
    res = b["results"]
    summ = b["readings"]["summary"]
    counts = b["readings"]["mesh_counts"]
    sanity = b["readings"]["sanity"].get("default_s42_matches_e08e_stage1_views_T", {})
    table_rows = []
    for c, per in res.items():
        for k in objects:
            m = per[str(k)]
            if not m.get("mesh"):
                table_rows.append(f"<tr><td>{esc(c)}</td><td>{k}</td><td colspan='6'>no mesh</td></tr>")
                continue
            table_rows.append(
                f"<tr><td>{esc(c)}</td><td>{k}</td><td>{m['completeness']:.3f}</td>"
                f"<td>{num(m['silhouette_iou'])}</td>"
                f"<td>{num(m['box_hull_iou'])}</td>"
                f"<td>{num(m.get('completeness_posthoc_voxel_tau'))}</td><td>{m['accuracy_like']:.3f}</td><td>{m['heldout_frames']}</td></tr>")
    return (
        '<h2>Stage B: WorldSculpt parameters, how complete each mesh is</h2>'
        f'<p class="sub">Ten objects fixed by rule before any mesh was seen ({esc(", ".join(map(str, objects)))}). '
        'Completeness = share of the object\'s observed surface points within 2% of its box diagonal of the mesh. '
        'A config counts as changing a metric only if its median change over objects leaves the gray band, the '
        'median spread across three seeds of the default.</p>'
        f'<div class="card">{reading_chart(summ["completeness"], "completeness", "completeness: change vs default seeds", res, objects)}'
        '<div class="legend"><span><i class="box" style="background:var(--band)"></i>seed noise (median range over 3 seeds)</span>'
        '<span><i style="background:var(--accent)"></i>median change, up</span><span><i style="background:var(--neg)"></i>median change, down</span>'
        '<span><i style="background:var(--trace);height:6px;width:6px;border-radius:3px"></i>one object</span></div></div>'
        f'<div class="card" style="margin-top:14px">{reading_chart(summ["silhouette_iou"], "silhouette_iou", "held-out silhouette IoU: change vs default seeds", res, objects)}</div>'
        '<h2>More views, more complete?</h2>'
        '<p class="sub">Each gray line is one object; blue is the median. The gray bar at 20 views is the range of '
        'the median across the three default seeds.</p>'
        f'<div class="grid2"><div class="card">{views_curve(res, objects, "completeness", "completeness")}</div>'
        f'<div class="card">{views_curve(res, objects, "silhouette_iou", "held-out silhouette IoU")}</div></div>'
        '<h2>What the numbers look like</h2>'
        '<p class="sub">Frames WorldSculpt never received. The mask is HQ-SAM\'s prediction, not ground truth, and it '
        'only covers what is visible, so parts hidden behind another object count against IoU for every config alike.</p>'
        f'<div class="card">{sil_html}</div>'
        '<details><summary>Table view: every Stage B object and config</summary><div class="tablewrap card"><table>'
        '<tr><th>config</th><th>object</th><th>completeness</th><th>held-out IoU</th><th>box-hull IoU (null)</th>'
        '<th>completeness, voxel tau (post hoc)</th><th>accuracy-like</th><th>held-out frames</th></tr>'
        + "".join(table_rows) + '</table></div></details>'
        f'<p class="sub" style="margin-top:14px">Meshes produced per config: {esc(", ".join(f"{c} {n}" for c, n in counts.items()))}. '
        f'default_s42 matches E08e on Stage 1 voxels, views and placement for {sum(sanity.values())} of {len(sanity)} objects '
        '(the shape stage itself is nondeterministic at a fixed seed; see the notes).</p>')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-a", required=True, type=Path)
    ap.add_argument("--stage-b", type=Path)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--objects", type=Path, default=Path("experiments/worldsculpt/e08e_objects.json"))
    ap.add_argument("--configs", type=Path, default=Path("experiments/worldsculpt/e08f_configs.json"))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    a = json.loads(args.stage_a.read_text())
    body = stage_a_section(a)
    if args.stage_b and args.stage_b.exists():
        b = json.loads(args.stage_b.read_text())
        picks = [k for k in (56, 20, 67) if k in b["objects"]]
        sil = silhouettes(b, args, args.out, picks,
                          ["default_s42", "views1", "views3", "erode4", "dilate4", "nofit"])
        body += stage_b_section(b, sil)
    notes = [
        "<b>Exploratory.</b> Arms, metrics, nulls and reading rules were pre-registered in "
        "<code>experiments/worldsculpt/PREREG_E08f.md</code> before any sweep run. One scene, one capture, ten "
        "objects in Stage B: a large effect would show, a small one would not.",
        "Stage A applies only the automatic filter. E08e's hand review is not re-applied, so its counts include "
        "ceiling lights and ducts that review removed.",
        "Completeness is measured against points back-projected from the same 3DGS depth that the boxes came from, "
        "so it is consistency with our own grounding, not accuracy against ground truth. None exists for this scene.",
        "Held-out IoU compares against HQ-SAM's predicted mask, a reference rather than ground truth; the box-hull "
        "IoU in the table is its null.",
        "Units are COLMAP units; this capture has no metric scale.",
        "<b>WorldSculpt's shape stage is not reproducible at a fixed seed.</b> The same command run twice on one "
        "object gave 120,370 and then 31,136 faces from identical Stage 1 voxels, so the three default seeds "
        "measure run-to-run variation as well as seed variation. Face counts are not read as quality.",
        "Completeness uses a tolerance of 2% of the box diagonal, which for most objects is below the voxel size "
        "of the observed points, so its absolute value is capped well below 1; read it relative to the seeds. A "
        "post-hoc variant with the tolerance floored at half a voxel (in the table) gives the same verdicts. "
        "Mesh fragmentation, pre-registered, could not be measured: the raw output is unwelded patches.",
    ]
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>WorldSculpt Parameter Sweep</title>'
            f'<style>{STYLE}</style><body><div class="viz-root"><div class="wrap">'
            '<div class="eyebrow">E08f · WorldSculpt on NCHC sofa · parameter sweep</div>'
            '<h1>Which knobs change what comes out</h1>'
            '<p class="intro">How each parameter changes how many objects the grounding stage hands to WorldSculpt, '
            'and how complete each object\'s mesh comes back. One parameter moves at a time; everything else stays at '
            'E08e\'s defaults.</p>'
            + body +
            f'<ul class="notes">{"".join(f"<li>{n}</li>" for n in notes)}</ul>'
            f'</div></div><div id="tip" role="tooltip"></div><script>{TIP_JS}</script></body></html>')
    (args.out / "index.html").write_text(page)
    print(f"wrote {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
