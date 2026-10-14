"""Shared helpers for the depthviz tests: headless engines, numpy depth scenes, a real-depth tape built
from saved camera frames (the worker's exact per-frame pipeline: tmix-equivalent, gain, model, mask,
normaliser), and image utilities. Reports go to out/verify/depthviz/.
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audioviz.gl.context import create_headless  # noqa: E402
from audioviz.gl.pingpong import Target  # noqa: E402

from depthviz import preset as presetmod  # noqa: E402
from depthviz.config import Tunables  # noqa: E402
from depthviz.engine import Engine  # noqa: E402

OUT = os.path.join(ROOT, "out", "verify", "depthviz")
DW, DH = 320, 240
SONG = os.environ.get("DEPTHVIZ_TEST_SONG", "C:/Users/Asus/Downloads/Everything In Its Right Place.mp3")


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def write_report(name, rep):
    p = out_path(name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:6000])
    return p


def make_engine(ctx, size=(1920, 1080), seed=7, overrides=None, preset="fog_white", look_fps=12, timers=False, track=None):
    pre = presetmod.apply_overrides(presetmod.load(preset), overrides or [])
    return Engine(ctx, size, pre, seed, Tunables(timers=timers), track, look_fps=look_fps)


def read_rgb(fbo, W, H):
    return np.flipud(np.frombuffer(fbo.read(components=3), np.uint8).reshape(H, W, 3))


def save_rgb(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def scene_figure(fig=0.2, bg=0.92, blur_px=0.0, cx=0.5, w=0.3, top=0.25, mask_sharp=True):
    """Camera-space (top rows first) depth with a rectangular-ish figure, plus its mask (GL rows out)."""
    yy, xx = np.mgrid[0:DH, 0:DW].astype(np.float32)
    inside = (np.abs(xx / DW - cx) < w / 2) & (yy / DH > top)
    d = np.where(inside, fig, bg).astype(np.float32)
    if blur_px > 0:
        d = cv2.GaussianBlur(d, (0, 0), blur_px)
    m = (inside * 255).astype(np.uint8) if mask_sharp else cv2.GaussianBlur((inside * 255).astype(np.uint8), (0, 0), 1.0)
    return np.ascontiguousarray(d[::-1]), np.ascontiguousarray(m[::-1]), inside


def real_depth_tape(tape_dir, frames_dir=None, model=None, ep=None, size=None, device_id=None, loop_to_s=12.0,
                    look_fps=12, dark=False, seed=11):
    """Build a tape from saved camera frames through the worker's own per-frame pipeline. Frames are
    played forward and backward (ping-pong) until loop_to_s. dark=True darkens them like the M0 'monitor' set."""
    from depthviz.config import pick_profile
    from depthviz.depth.bench import FRAME_H, FRAME_W, face_center, load_frames, mp_masks, simulate_dark, to_pipe
    from depthviz.depth.infer import DepthModel, Normaliser, gain_frame
    from depthviz.inputs.depthcam import HI, MpMask, depth_mask
    from depthviz.inputs.tape import TapeWriter

    frames_dir = frames_dir or os.path.join(ROOT, "out", "bench", "frames_lit")
    raw = load_frames(frames_dir)
    if dark:
        rng = np.random.default_rng(seed)
        lit320 = to_pipe(raw, tmix=1)
        masks, _ = mp_masks(lit320[:1])
        fc = face_center(masks[0]) if masks else None
        fc640 = (fc[0] * raw[0].shape[1] / FRAME_W, fc[1] * raw[0].shape[0] / FRAME_H) if fc else None
        raw = [simulate_dark(f, rng, "monitor", fc640) for f in raw]
    seq = to_pipe(raw)
    from depthviz.config import gl_renderer_headless

    prof = pick_profile(gl_renderer_headless())
    dm = DepthModel(model or prof["model"], ep or prof["ep"], size or prof["size"],
                    prof.get("device_id", 0) if device_id is None else device_id)
    try:
        mm = MpMask()
    except Exception:
        mm = None
    norm = Normaliser()
    gst = {}
    n_ticks = int(loop_to_s * look_fps)
    order = list(range(len(seq))) + list(range(len(seq) - 2, 0, -1))
    w = TapeWriter(tape_dir, {"song": None, "look_fps": look_fps, "depth_size": [DW, DH], "seed": None,
                              "source": f"frames:{os.path.basename(frames_dir)}{':dark' if dark else ''}",
                              "model": dm.model_id, "ep": dm.info["ep"]})
    dt = 1.0 / look_fps
    for tick in range(n_ticks):
        f = seq[order[tick % len(order)]]
        g, gi = gain_frame(f, gst, dt)
        z = dm(g)
        m = mm(g) if mm is not None else None
        d = norm(z, dt, m)
        if m is None:
            m = depth_mask(d)
            mk = m > 127
            norm.mask_cov = float(mk.mean())
            if norm.mask_cov > 0.01:
                norm.fig_depth = float(np.median(d[mk]))
        hdr = np.zeros(32)
        hdr[HI["fig_depth"]] = norm.fig_depth
        hdr[HI["mask_cov"]] = norm.mask_cov
        hdr[HI["brightness"]] = gi["brightness"]
        hdr[HI["gain"]] = gi["gain"]
        w.write(tick, tick / look_fps, d[::-1], m[::-1], hdr)
    w.close()
    if mm is not None:
        mm.close()
    return tape_dir, dm.info


def ensure_tape(name, loop_to_s=4.0, dark=False):
    """A real-depth tape under out/verify/depthviz/<name>, built in a SUBPROCESS: running MediaPipe and the
    DirectML session in the test process leaves the moderngl context non-current ('cannot create buffer')."""
    import subprocess

    path = out_path(name)
    if os.path.isfile(os.path.join(path, "meta.json")):
        return path
    code = ("import sys; sys.path.insert(0, %r); from depthviz_common import real_depth_tape; "
            "print(real_depth_tape(%r, loop_to_s=%r, dark=%r)[1])" % (os.path.dirname(os.path.abspath(__file__)),
                                                                   path, float(loop_to_s), bool(dark)))
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, errors="replace")
    if p.returncode != 0 or not os.path.isfile(os.path.join(path, "meta.json")):
        raise RuntimeError(f"tape {name} failed: {p.stderr[-1200:]}")
    print(f"built tape {name}: {p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ''}")
    return path
