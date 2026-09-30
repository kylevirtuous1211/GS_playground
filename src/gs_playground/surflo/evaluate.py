"""E08h readings, against experiments/surflo/PREREG_E08h.md.

    python -m gs_playground.surflo.evaluate stage-a \
        --root outputs/surflo/garden_sample --out results/surflo/e08h_stage_a.json

Stage A compares each arm's ODE time and peak VRAM with the authors' README
table and checks the promised outputs exist. The fourth criterion (the mesh
shows the table, the pot and the ground) is judged by eye and recorded in the
LOG, so this file reports it as pending rather than guessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

#: README runtime table, one H100, 100k points, DA3 priors and meshing excluded.
THEIRS = {"guided_default": {"ode_s": 45.0, "vram_gib": 14.0},
          "plain": {"ode_s": 8.0, "vram_gib": 8.5}}
#: pre-registered tolerances
MAX_TIME_RATIO = 5.0
VRAM_TOLERANCE = 0.30
OUTPUTS = {"guided_default": ("mesh.ply", "point_cloud_normals.ply", "point_cloud_rgb.ply",
                              "mesh_textured.ply"),
           "plain": ("final.ply",)}


def ply_counts(path: Path) -> dict[str, int]:
    """Element counts from a PLY header, without loading the body."""
    counts = {}
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("element"):
                _, name, n = line.split()
                counts[name] = int(n)
            if line == "end_header":
                break
    return counts


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_run(run: Path, arm: str) -> dict:
    summary_path = next(run.glob("*/_infer_summary.json"))
    scene_dir = summary_path.parent
    info = json.loads(summary_path.read_text())["scene"]
    files = {name: scene_dir / name for name in OUTPUTS[arm]}
    present = {name: p.exists() and p.stat().st_size > 0 for name, p in files.items()}
    first = files[OUTPUTS[arm][0]]
    return {
        "run": run.name,
        "ode_s": info["ode_inference_s"],
        "peak_vram_ode_gib": info.get("peak_vram_ode_gib"),
        "outputs_present": present,
        "counts": ply_counts(first) if first.exists() else None,
        "sha_" + first.name: sha(first) if first.exists() else None,
    }


def stage_a(root: Path) -> dict:
    arms = {}
    for arm, theirs in THEIRS.items():
        runs = [read_run(r, arm) for r in sorted((root / arm).glob("run*"))]
        ode = [r["ode_s"] for r in runs]
        vram = [r["peak_vram_ode_gib"] for r in runs]
        key = next(k for k in runs[0] if k.startswith("sha_"))
        ode_ratio = median(ode) / theirs["ode_s"]
        vram_ratio = median(vram) / theirs["vram_gib"]
        arms[arm] = {
            "n": len(runs),
            "runs": runs,
            "theirs_h100": theirs,
            "ode_s_median": median(ode), "ode_s_range": [min(ode), max(ode)],
            "peak_vram_gib_median": median(vram), "peak_vram_gib_range": [min(vram), max(vram)],
            "ode_ratio_to_theirs": ode_ratio,
            "vram_ratio_to_theirs": vram_ratio,
            "identical_outputs_across_runs": len({r[key] for r in runs}) == 1,
            "criteria": {
                "1_outputs_present": all(all(r["outputs_present"].values()) for r in runs),
                "2_time_within_5x": ode_ratio <= MAX_TIME_RATIO,
                "3_vram_within_30pct": abs(vram_ratio - 1) <= VRAM_TOLERANCE,
            },
        }
    return {
        "entry": "E08h", "stage": "A", "prereg": "experiments/surflo/PREREG_E08h.md",
        "arms": arms,
        "criterion_4_mesh_by_eye": "pending: judged by eye, recorded in LOG.md E08h",
        "passes_1_to_3": all(all(a["criteria"].values()) for a in arms.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stage-a")
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "stage-a":
        result = stage_a(args.root)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        for arm, a in result["arms"].items():
            print(f"{arm}: ODE {a['ode_s_median']:.1f}s ({a['ode_ratio_to_theirs']:.2f}x theirs), "
                  f"VRAM {a['peak_vram_gib_median']:.1f} GiB ({a['vram_ratio_to_theirs']:.2f}x), "
                  f"identical={a['identical_outputs_across_runs']}, {a['criteria']}")
        print(f"criteria 1-3 pass: {result['passes_1_to_3']}; criterion 4 by eye: pending")


if __name__ == "__main__":
    main()
