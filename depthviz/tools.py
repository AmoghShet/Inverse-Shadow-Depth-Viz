"""Headless helpers: `still` (one frame) and `check` (context, providers, models, MediaPipe)."""
import os
import sys

import cv2
import numpy as np

from . import TITLE
from . import preset as presetmod
from .config import Tunables
from .engine import Engine
from .render.present import VIEWS


def still(args):
    from audioviz.gl.context import create_headless
    from audioviz.gl.pingpong import Target

    from .export.recorder import DepthFeed

    preset = presetmod.apply_overrides(presetmod.load(args.preset), args.set)
    track = None
    if args.song:
        from audioviz.audio.analysis import load_or_analyze
        from audioviz.audio.features import FeatureTrack

        track = FeatureTrack(load_or_analyze(args.song))
    W, H = args.size
    ctx = create_headless()
    look = int(preset["grain"].get("frame_rate", 12))
    eng = Engine(ctx, (W, H), preset, args.seed, Tunables(timers=False), track, look_fps=look)
    feed = DepthFeed(args.source, look)
    tgt = Target(ctx, (W, H), 4, "f1", "nearest")
    dt = 1.0 / 60.0
    t0 = max(0.0, args.t - 3.0)
    n = int(round((args.t - t0) / dt))
    last = None
    for i in range(n + 1):
        t = t0 + i * dt
        eng.sim_step(t, dt)
        tick = int(np.floor(t * look + 1e-9))
        if tick != last:
            feed.apply(eng, tick)
            eng.view = 0
            eng.render(tgt.fbo, tick=tick, t=t)
            last = tick
    eng.view = VIEWS.index(args.view)
    eng.present_only(tgt.fbo)
    img = np.flipud(np.frombuffer(tgt.fbo.read(components=3), np.uint8).reshape(H, W, 3))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    extra = {"song": args.song, "t": args.t, "resolution": [W, H], "view": args.view}
    extra.update(feed.describe())
    presetmod.write_resolved(os.path.splitext(args.out)[0] + ".json", eng.resolved, extra)
    print(f"wrote {args.out}  (seed {eng.seed}, polarity {eng.resolved['polarity']})")


def check():
    """GL window at 4.1 core, onnxruntime providers, depth models present, MediaPipe importable."""
    import time

    from audioviz.gl.context import create_window, gl_info

    ok = True
    window = create_window(title=f"{TITLE} check", size=(640, 360))
    ctx = window.ctx
    for k, v in gl_info(ctx).items():
        print(f"{k:>18}: {v}")
    print(f"{'buffer_size':>18}: {window.buffer_size} (window size {window.size})")
    ok &= ctx.version_code >= 410
    try:
        import onnxruntime as ort

        print(f"{'onnxruntime':>18}: {ort.__version__} {ort.get_available_providers()}")
    except Exception as e:
        ok = False
        print(f"{'onnxruntime':>18}: MISSING ({e}) -> pip install -r requirements-depthviz.txt")
    from .depth import models as M

    for mid in M.MODELS:
        print(f"{mid:>18}: {'present' if os.path.isfile(M.model_path(mid)) else 'missing (fetch-models)'}")
    try:
        import mediapipe  # noqa: F401

        from .inputs.depthcam import SELFIE_MODEL

        print(f"{'mediapipe':>18}: ok, selfie model {'present' if os.path.isfile(SELFIE_MODEL) else 'MISSING'}")
    except Exception as e:
        print(f"{'mediapipe':>18}: not importable ({type(e).__name__}); mask falls back to depth")
    from .config import pick_profile
    from audioviz.gl.context import renderer_name

    print(f"{'profile':>18}: {pick_profile(renderer_name(ctx))}")
    print(f"{'platform':>18}: {Tunables.platform_summary()}")
    t0 = time.perf_counter()
    while not window.is_closing and time.perf_counter() - t0 < 1.0:
        ctx.clear(0.9, 0.9, 0.9)
        window.swap_buffers()
    window.destroy()
    print("check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check())
