"""Build the clips and stills the top-level README shows, from runs already on disk.

    bash tools/build_readme_media.sh                     # both, in the puffin env
    python -m gs_playground.readme_media surflo          # puffin env (gsplat)
    python -m gs_playground.readme_media semantics       # puffin env (gsplat)

Both show Mip-NeRF 360 garden: our own captures are not public.

Everything lands in docs/media/ (tracked). Nothing here is hand-assembled:
rerun the subcommand after its run changes.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .paths import OUTPUTS as OUT, ROOT

MEDIA = ROOT / "docs" / "media"

#: E08j's garden run at N = 16, seed 42 (VGGT-1B backbone)
SURFLO_RUN = OUT / "surflo/views_sweep/n16/seed42/garden_train161_images_4"
E08N = OUT / "iggt/semantics_on_surflo"

#: Two of SuRFLo's 16 garden cameras (indices into cameras.json), 51 degrees
#: apart: DSC07957, low and wide with the table legs and the house behind, to
#: DSC08030, higher and to the side, looking down on the table top. The table
#: and the pot stay in frame the whole way. E08n's garden run has the same 16
#: inputs in the same order, so the indices hold there too.
GARDEN_PATH = [0, 6]
DEG_PER_FRAME = 1.25    # camera turn per frame along the path
STEP_PER_FRAME = 0.012  # camera travel per frame, in SuRFLo's (unit-free) scale
RENDER_SCALE = 1.5      # SuRFLo's cameras are 518 px wide; rendered at 777
CLIP_WIDTH = 777
FPS = 15
QUALITY = 75            # libwebp quality


def write_clip(frames: list[np.ndarray], target: Path, fps: int = FPS,
               width: int = CLIP_WIDTH) -> Path:
    """Frames (H, W, 3 uint8) -> a looping animated WebP.

    WebP, not GIF: GitHub shows both inline, and the first README clip as a GIF
    was 4.5 MB at 480 px wide (128 colours, dithered) against 0.75 MB as WebP at 640.
    """
    import imageio_ffmpeg

    h, w = frames[0].shape[:2]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-vf", f"scale={width}:-2:flags=lanczos", "-c:v", "libwebp_anim",
           "-quality", str(QUALITY), "-compression_level", "4", "-loop", "0", str(target)]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
                   check=True)
    print(f"{target.relative_to(ROOT)}: {len(frames)} frames, {target.stat().st_size / 1e6:.2f} MB")
    return target


def ping_pong(frames: list) -> list:
    """Forward then back, so the loop has no jump; the end frames are not doubled."""
    return frames + frames[-2:0:-1]


def path_through(c2ws: np.ndarray) -> list[np.ndarray]:
    """Camera-to-world 4x4s along a smooth path through the given cameras.

    Centres follow a Catmull-Rom spline (ends clamped), rotations a slerp per
    segment; each segment gets frames in proportion to how far the camera turns
    or travels, whichever is more.
    """
    from scipy.spatial.transform import Rotation, Slerp

    c = c2ws[:, :3, 3]
    out = []
    for i in range(len(c2ws) - 1):
        p0, p1, p2, p3 = c[max(i - 1, 0)], c[i], c[i + 1], c[min(i + 2, len(c) - 1)]
        turn = Rotation.from_matrix(c2ws[i + 1, :3, :3] @ c2ws[i, :3, :3].T).magnitude()
        n = max(4, math.ceil(math.degrees(turn) / DEG_PER_FRAME),
                math.ceil(np.linalg.norm(p2 - p1) / STEP_PER_FRAME))
        slerp = Slerp([0, 1], Rotation.from_matrix(c2ws[i:i + 2, :3, :3]))
        for t in np.arange(n) / n:
            centre = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                            + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
            m = np.eye(4)
            m[:3, :3] = slerp(t).as_matrix()
            m[:3, 3] = centre
            out.append(m)
    out.append(c2ws[-1])
    return out


def flythrough(clouds: list, cameras: Path) -> list[np.ndarray]:
    """Render each GaussianCloud along GARDEN_PATH; frames hold them side by side."""
    import torch

    from .gs.render import Camera, load_cameras, render

    cams = load_cameras(cameras)
    c2ws = np.stack([torch.linalg.inv(cams[i].viewmat).numpy() for i in GARDEN_PATH])
    w, h = round(cams[0].width * RENDER_SCALE), round(cams[0].height * RENDER_SCALE)
    K = cams[0].K.clone()
    K[:2] *= RENDER_SCALE
    frames = []
    for c2w in path_through(c2ws):
        cam = Camera("path", torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32)), K, w, h)
        tiles = [render(cloud, cam, sh_degree=3 if cloud.f_rest is not None else None,
                        background=(0.0, 0.0, 0.0)) for cloud in clouds]
        frames.append((torch.cat(tiles, dim=1).cpu().numpy() * 255).astype(np.uint8))
    return frames


def surflo() -> None:
    """SuRFLo's exported 3DGS of garden from 16 unposed views, flown between two of its input cameras."""
    from .gs.ply import load_ply
    from .surflo.demo import contact_sheet

    frames = flythrough([load_ply(SURFLO_RUN / "point_cloud.ply")], SURFLO_RUN / "cameras.json")
    write_clip(ping_pong(frames), MEDIA / "surflo_garden.webp")
    inputs = json.loads((SURFLO_RUN / "_infer_summary.json").read_text())["selected_images"]
    contact_sheet([Path(p) for p in inputs], MEDIA / "surflo_garden_inputs.jpg")
    print(f"docs/media/surflo_garden_inputs.jpg: the run's {len(inputs)} input photos")


def titled(frame: np.ndarray, titles: list[str]) -> np.ndarray:
    """Write each panel's title in its top-left corner; panels are equal-width columns."""
    from PIL import ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default(size=frame.shape[0] // 14)
    panel = frame.shape[1] // len(titles)
    for i, title in enumerate(titles):
        x0, y0, x1, y1 = draw.textbbox((panel * i + 12, 10), title, font=font)
        draw.rectangle([x0 - 6, y0 - 4, x1 + 6, y1 + 4], fill=(0, 0, 0, 150))
        draw.text((panel * i + 12, 10), title, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def semantics() -> None:
    """E08n on garden: SuRFLo's Gaussians (IGGT backbone) beside the same
    Gaussians coloured by IGGT instance group and by CLIP class."""
    from .iggt.on_surflo import colourings, legend

    garden = colourings(E08N, "garden")
    frames = flythrough([garden["cloud"], garden["instances"], garden["classes"]],
                        garden["scene_dir"] / "cameras.json")
    frames = [titled(f, ["SuRFLo 3DGS", "IGGT instances", "CLIP classes"]) for f in frames]
    write_clip(ping_pong(frames), MEDIA / "iggt_semantics_garden.webp", width=3 * 518)
    legend(garden["legend"], MEDIA / "iggt_semantics_garden_legend.png")
    print("docs/media/iggt_semantics_garden_legend.png: the E08n viewer's class legend")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["surflo", "semantics"])
    args = parser.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)
    globals()[args.what]()


if __name__ == "__main__":
    main()
