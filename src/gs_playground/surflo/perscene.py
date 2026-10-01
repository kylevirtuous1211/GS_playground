"""E08i reading: per-scene optimisation cost on garden, beside SuRFLo (PREREG_E08i.md).

    python -m gs_playground.surflo.perscene --root outputs/surflo/perscene_garden \\
        --surflo results/surflo/e08h_stage_a.json --out results/surflo/e08i_perscene_garden.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: pre-registered validity floor: below it, a run's cost is not the method's cost
MIN_PSNR_DB = 25.0


def peak_process_mib(vram_csv: Path, pid: int) -> int | None:
    """The trainer's peak in nvidia-smi's per-process samples (pid, MiB per line)."""
    peaks = []
    for line in vram_csv.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            peaks.append(int(parts[1]))
    return max(peaks) if peaks else None


def read_arm(arm_dir: Path) -> dict:
    train = json.loads(next((arm_dir / "stats").glob("train_step29999*.json")).read_text())
    val = json.loads(next((arm_dir / "stats").glob("val_step29999*.json")).read_text())
    pid = int((arm_dir / "pid").read_text())
    smi = peak_process_mib(arm_dir / "vram.csv", pid)
    row = {
        "training_time_s": train["ellipse_time"],
        "wall_clock_s": int((arm_dir / "wall_s").read_text()),
        "peak_alloc_gib": train["mem"],
        "peak_process_gib_nvidia_smi": None if smi is None else smi / 1024,
        "gaussians": train["num_GS"],
        "heldout_psnr_db": val["psnr"],
        "heldout_ssim": val.get("ssim"),
        "heldout_lpips": val.get("lpips"),
    }
    row["valid"] = row["heldout_psnr_db"] >= MIN_PSNR_DB
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--surflo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms = {d.name: read_arm(d) for d in sorted(args.root.iterdir()) if (d / "done").exists()}
    surflo = json.loads(args.surflo.read_text())["arms"]
    beside = {name: {"ode_s_median": a["ode_s_median"],
                     "peak_vram_reserved_gib_median": a["peak_vram_gib_median"]}
              for name, a in surflo.items()}
    result = {"entry": "E08i", "prereg": "experiments/surflo/PREREG_E08i.md",
              "lane": "cost measurement", "gpu": "RTX 6000 Ada", "min_psnr_db": MIN_PSNR_DB,
              "arms": arms, "surflo_same_gpu_e08h": beside}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    for name, a in arms.items():
        smi = a["peak_process_gib_nvidia_smi"]
        print(f"{name}: train {a['training_time_s'] / 60:.1f} min (wall {a['wall_clock_s'] / 60:.1f}), "
              f"alloc {a['peak_alloc_gib']:.2f} GiB, nvidia-smi {'n/a' if smi is None else f'{smi:.2f} GiB'}, "
              f"{a['gaussians']:,} Gaussians, held-out PSNR {a['heldout_psnr_db']:.2f} dB, valid={a['valid']}")


if __name__ == "__main__":
    main()
