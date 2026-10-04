"""Build the clips and stills the top-level README shows, from runs already on disk.

    bash tools/build_readme_media.sh                     # all of them, each in its env
    python -m gs_playground.readme_media surflo          # puffin env (gsplat)
    python -m gs_playground.readme_media semantics       # puffin env (gsplat)
    python -m gs_playground.readme_media worldsculpt     # surflo env (nvdiffrast)

Everything lands in docs/media/ (tracked). Nothing here is hand-assembled:
rerun the subcommand after its run changes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .paths import OUTPUTS as OUT, ROOT

MEDIA = ROOT / "docs" / "media"

SURFLO_RUN = OUT / "surflo/nchc_sofa/guided_default_16/seed42/images"
E08N = OUT / "iggt/semantics_on_surflo"
WORLDSCULPT_VIEWER = OUT / "worldsculpt/viewer3d"
WORLDSCULPT_RUN = OUT / "worldsculpt/NCHC/nchc_sofa_20260727_143647/_scene"

#: Two of SuRFLo's 16 sofa cameras (indices into cameras.json), 55 degrees
#: apart: the wide view of the sofa and the stools close up. A longer path
#: through more of the inputs flew through the pillar between cameras 8 and 10.
SOFA_PATH = [6, 8]
DEG_PER_FRAME = 1.25    # camera turn per frame along the path
STEP_PER_FRAME = 0.012  # camera travel per frame, in SuRFLo's (unit-free) scale
RENDER_SCALE = 1.5      # SuRFLo's cameras are 518 px wide; rendered at 777
CLIP_WIDTH = 777
FPS = 15
QUALITY = 75            # libwebp quality


def write_clip(frames: list[np.ndarray], target: Path, fps: int = FPS,
               width: int = CLIP_WIDTH) -> Path:
    """Frames (H, W, 3 uint8) -> a looping animated WebP.

    WebP, not GIF: GitHub shows both inline, and the sofa clip as a GIF was
    4.5 MB at 480 px wide (128 colours, dithered) against 0.75 MB as WebP at 640.
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


def sofa_flythrough(clouds: list, cameras: Path) -> list[np.ndarray]:
    """Render each GaussianCloud along SOFA_PATH; frames hold them side by side."""
    import torch

    from .gs.render import Camera, load_cameras, render

    cams = load_cameras(cameras)
    c2ws = np.stack([torch.linalg.inv(cams[i].viewmat).numpy() for i in SOFA_PATH])
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
    """SuRFLo's exported 3DGS of our sofa capture, flown along its own input cameras."""
    from .gs.ply import load_ply

    frames = sofa_flythrough([load_ply(SURFLO_RUN / "point_cloud.ply")], SURFLO_RUN / "cameras.json")
    write_clip(ping_pong(frames), MEDIA / "surflo_sofa.webp")
    shutil.copyfile(OUT / "surflo/viewer/images/sofa_inputs.jpg", MEDIA / "surflo_sofa_inputs.jpg")
    print("docs/media/surflo_sofa_inputs.jpg: copied from the E08h viewer")


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
    """E08n on the sofa: SuRFLo's Gaussians (IGGT backbone) beside the same
    Gaussians coloured by IGGT instance group and by CLIP class."""
    from .iggt.on_surflo import colourings, legend

    sofa = colourings(E08N, "nchc_sofa")
    frames = sofa_flythrough([sofa["cloud"], sofa["instances"], sofa["classes"]],
                             sofa["scene_dir"] / "cameras.json")
    frames = [titled(f, ["SuRFLo 3DGS", "IGGT instances", "CLIP classes"]) for f in frames]
    write_clip(ping_pong(frames), MEDIA / "iggt_semantics_sofa.webp", width=3 * 518)
    legend(sofa["legend"], MEDIA / "iggt_semantics_sofa_legend.png")
    print("docs/media/iggt_semantics_sofa_legend.png: the E08n viewer's class legend")


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
TURNTABLE_SIZE = (1280, 720)
TURNTABLE_ELEVATION = 28.0
BACKGROUND = (246, 246, 244)


def worldsculpt() -> None:
    """WorldSculpt's 24 object meshes of our sofa capture, turned once.

    The meshes are the E08g scene viewer's (decimated, Y up, floor at y = 0),
    one flat colour per object as on that page, Lambert-shaded from a light that
    follows the camera. No floor or walls are drawn: WorldSculpt was given
    objects only.
    """
    import nvdiffrast.torch as dr
    import torch

    manifest = json.loads((WORLDSCULPT_VIEWER / "manifest.json").read_text())
    scene = next(s for s in manifest["scenes"] if s["key"] == "nchc")
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

    # framed on the sofa group: four far objects (a bench and a frame 2 m up,
    # two others 5 m out) would otherwise shrink it to a fifth of the frame;
    # they still pass through the shot as it turns
    lo, hi = np.percentile(verts, 10, axis=0), np.percentile(verts, 90, axis=0)
    centre = (lo + hi) / 2
    radius = 0.95 * np.linalg.norm(hi - lo)
    w, h = TURNTABLE_SIZE
    proj = perspective(40.0, w / h)
    elevation = math.radians(TURNTABLE_ELEVATION)
    frames = []
    for k in range(TURNTABLE_FRAMES):
        azimuth = 2 * math.pi * k / TURNTABLE_FRAMES
        eye = centre + radius * np.array([math.cos(elevation) * math.sin(azimuth), math.sin(elevation),
                                          math.cos(elevation) * math.cos(azimuth)])
        view = look_at_gl(eye, centre)
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
    write_clip(frames, MEDIA / "worldsculpt_sofa_turntable.webp", width=960)

    strip = Image.open(WORLDSCULPT_RUN / "renders/view02.jpg")
    q = strip.width // 4
    panels = [strip.crop((i * q, 0, (i + 1) * q, strip.height)) for i in range(4)]
    grid = Image.new("RGB", (2 * q, 2 * strip.height))
    for i, p in enumerate(panels):
        grid.paste(p, ((i % 2) * q, (i // 2) * strip.height))
    grid.resize((q, strip.height), Image.LANCZOS).save(MEDIA / "worldsculpt_sofa_view.jpg", quality=88)
    print("docs/media/worldsculpt_sofa_view.jpg: WorldSculpt's own render of view 2, its four panels 2x2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["surflo", "semantics", "worldsculpt"])
    args = parser.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)
    globals()[args.what]()


if __name__ == "__main__":
    main()
