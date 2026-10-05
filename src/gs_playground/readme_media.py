"""Build the clips and stills the top-level README shows, from runs already on disk.

    bash tools/build_readme_media.sh                     # all of them, each in its env
    python -m gs_playground.readme_media surflo          # puffin env (gsplat)
    python -m gs_playground.readme_media semantics       # puffin env (gsplat)
    python -m gs_playground.readme_media worldsculpt     # surflo env (nvdiffrast)

SuRFLo and IGGT show Mip-NeRF 360 garden, WorldSculpt the authors' released
Marble living room: our own captures are not public.

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
#: E08g's scene viewer (decimated meshes) and E08c's rerun of the authors' Marble scene
WORLDSCULPT_VIEWER = OUT / "worldsculpt/viewer3d"
WORLDSCULPT_RUN = OUT / "worldsculpt/Marble/marble_serene_living_room_countryside_view/_scene"

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


def look_at_gl(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """World-to-camera, OpenGL axes (camera looks down -z), Y up."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, true_up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def perspective(fov_deg: float, aspect: float, near: float = 0.05, far: float = 100.0) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]])


TURNTABLE_FRAMES = 120   # one full turn; 8 s at FPS
TURNTABLE_SIZE = (1280, 800)  # SuRFLo's garden clip is 1.6:1 too, so the hero row is even
TURNTABLE_FOV = 40.0     # vertical, degrees
TURNTABLE_ELEVATION = 35.0
TURNTABLE_MARGIN = 0.94  # share of the half-frame the outermost vertex may reach
BACKGROUND = (246, 246, 244)


def orbit_axes(back: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right and up of a Y-up camera whose backward (towards the eye) axis is `back`."""
    right = np.cross((0.0, 1.0, 0.0), back)
    right /= np.linalg.norm(right)
    return right, np.cross(back, right)


def fit_orbit(points: np.ndarray, backs: list[np.ndarray], tan_v: float, aspect: float,
              ) -> tuple[float, tuple[float, float]]:
    """Nearest camera distance, and the lens shift (tan units), that keep every
    point (relative to the orbit centre) inside the frame from every direction."""
    axes = np.stack([np.stack([*orbit_axes(b), b]) for b in backs])  # (views, 3, 3)
    cam = np.einsum("vij,pj->vpi", axes, points)                     # right, up, towards the eye

    def spans(d: float) -> np.ndarray:
        t = cam[..., :2] / (d - cam[..., 2:])
        return np.stack([t.min((0, 1)), t.max((0, 1))])

    lo, hi = cam[..., 2].max() + 1e-3, 100.0 * np.abs(points).max()
    for _ in range(50):  # bisection: the spans shrink as the camera backs off
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if np.all(np.diff(spans(mid), axis=0)[0] <= 2 * tan_v * np.array([aspect, 1])) \
            else (mid, hi)
    t = spans(hi)
    return hi, tuple((t[0] + t[1]) / 2)


def worldsculpt() -> None:
    """WorldSculpt's 13 object meshes of the authors' Marble living room, turned once.

    The meshes are the E08g scene viewer's (decimated, Y up, floor at y = 0),
    one flat colour per object as on that page, Lambert-shaded from a light that
    follows the camera. No floor or walls are drawn: WorldSculpt was given
    objects only.
    """
    import nvdiffrast.torch as dr
    import torch

    manifest = json.loads((WORLDSCULPT_VIEWER / "manifest.json").read_text())
    scene = next(s for s in manifest["scenes"] if s["key"] == "marble")
    blob = (WORLDSCULPT_VIEWER / scene["bin"]).read_bytes()
    verts, faces, face_rgb = [], [], []
    base = 0
    for obj in scene["objects"]:
        v = np.frombuffer(blob, "<f4", obj["pos"][1], obj["pos"][0]).reshape(-1, 3)
        f = np.frombuffer(blob, "<u4", obj["idx"][1], obj["idx"][0]).reshape(-1, 3)
        rgb = [int(obj["color"][k:k + 2], 16) / 255 for k in (1, 3, 5)]
        verts.append(v)
        faces.append(f + base)
        face_rgb.append(np.tile(rgb, (len(f), 1)))
        base += len(v)
    verts, faces, face_rgb = np.concatenate(verts), np.concatenate(faces), np.concatenate(face_rgb)
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    dev = "cuda"
    pos = torch.tensor(np.c_[verts, np.ones(len(verts))], dtype=torch.float32, device=dev)
    tri_t = torch.tensor(faces.astype(np.int32), device=dev)
    normals_t = torch.tensor(normals, dtype=torch.float32, device=dev)
    rgb_t = torch.tensor(face_rgb, dtype=torch.float32, device=dev)
    bg = torch.tensor(BACKGROUND, dtype=torch.float32, device=dev) / 255
    ctx = dr.RasterizeCudaContext()

    # The room is two furniture groups along its long axis (z, about 8 units),
    # one under the window and one at the far wall, with no far outliers, so the
    # turn is centred on the box of all 13 objects. One camera distance and one
    # lens shift hold for the whole turn: the nearest that keeps every vertex in
    # frame at every azimuth, so nothing is clipped as the long side swings round.
    # The shift recentres what the nearer furniture pushes towards the bottom.
    centre = (verts.min(0) + verts.max(0)) / 2
    w, h = TURNTABLE_SIZE
    elevation = math.radians(TURNTABLE_ELEVATION)
    backs = [np.array([math.cos(elevation) * math.sin(a), math.sin(elevation), math.cos(elevation) * math.cos(a)])
             for a in 2 * math.pi * np.arange(TURNTABLE_FRAMES) / TURNTABLE_FRAMES]
    dist, shift = fit_orbit(verts[::7] - centre, backs,  # ponytail: every 7th vertex; the margin covers the rest
                            math.tan(math.radians(TURNTABLE_FOV) / 2) * TURNTABLE_MARGIN, w / h)
    f = 1.0 / math.tan(math.radians(TURNTABLE_FOV) / 2)
    proj = perspective(TURNTABLE_FOV, w / h)
    proj[0, 2], proj[1, 2] = f / (w / h) * shift[0], f * shift[1]
    frames = []
    for back in backs:
        view = look_at_gl(centre + dist * back, centre)
        clip = pos @ torch.tensor((proj @ view).T, dtype=torch.float32, device=dev)
        rast, _ = dr.rasterize(ctx, clip[None], tri_t, resolution=[h, w])
        face = rast[0, ..., 3].long() - 1
        hit = face >= 0
        light = torch.tensor(view[:3, :3].T @ np.array([-0.35, 0.55, 1.0]) / np.linalg.norm([-0.35, 0.55, 1.0]),
                             dtype=torch.float32, device=dev)
        shade = 0.35 + 0.65 * (normals_t[face.clamp(min=0)] @ light).abs()
        img = torch.where(hit[..., None], rgb_t[face.clamp(min=0)] * shade[..., None], bg)
        img = dr.antialias(img[None].contiguous(), rast, clip[None], tri_t)[0].flip(0)
        frames.append((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    write_clip(frames, MEDIA / "worldsculpt_marble_turntable.webp", width=960)

    # view02 of upstream's 24 render strips (original | normals | per-instance
    # colour | overlay): looking from the doorway end towards the window, it
    # holds six objects whole (couch, side table and lamp, coffee table,
    # sideboard, armchair, painting) with the room around them; the far-wall
    # views show fewer objects and none of the room's layout
    strip = Image.open(WORLDSCULPT_RUN / "renders/view02.jpg")
    q = strip.width // 4
    panels = [strip.crop((i * q, 0, (i + 1) * q, strip.height)) for i in range(4)]
    grid = Image.new("RGB", (2 * q, 2 * strip.height))
    for i, p in enumerate(panels):
        grid.paste(p, ((i % 2) * q, (i // 2) * strip.height))
    grid.resize((q, strip.height), Image.LANCZOS).save(MEDIA / "worldsculpt_marble_view.jpg", quality=88)
    print("docs/media/worldsculpt_marble_view.jpg: WorldSculpt's own render of view 2, its four panels 2x2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["surflo", "semantics", "worldsculpt"])
    args = parser.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)
    globals()[args.what]()


if __name__ == "__main__":
    main()
