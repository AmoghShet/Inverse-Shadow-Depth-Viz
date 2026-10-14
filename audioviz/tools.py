"""Headless helpers: `still` (one frame) and `bench` (per-pass GPU timings per tier)."""
import json
import os
import time

import cv2
import numpy as np

from . import OUT_DIR
from . import preset as presetmod
from .config import TIER_BY_NAME, Tunables
from .engine import Engine
from .gl.context import create_headless, gl_info
from .gl.pingpong import Target
from .render.present import VIEWS


def still(args):
    from .app import apply_overrides
    from .audio.features import FeatureTrack

    preset = presetmod.load(args.preset)
    apply_overrides(preset, args.set)
    track = None
    if args.song:
        from .audio.analysis import load_or_analyze

        track = FeatureTrack(load_or_analyze(args.song))
    W, H = args.size
    ctx = create_headless()
    eng = Engine(ctx, (W, H), preset, args.seed, args.tier, Tunables(timers=False), track)
    tgt = Target(ctx, (W, H), 4, "f1", "nearest")
    pre = float(eng.resolved["params"]["sim"]["prewarm_s"])
    t_start = max(0.0, args.t - 6.0)
    eng.prewarm(pre, t0=t_start)
    dt = 1.0 / 60.0
    n = int(round((args.t - t_start) / dt))
    for i in range(n):
        eng.sim_step(t_start + (i + 1) * dt, dt)
        if i % 2 == 0:
            eng.render(tgt.fbo)  # keeps the exposure meter running
    eng.view = VIEWS.index(args.view)
    eng.render(tgt.fbo)
    img = np.flipud(np.frombuffer(tgt.fbo.read(components=3), np.uint8).reshape(H, W, 3))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    presetmod.write_resolved(os.path.splitext(args.out)[0] + ".json", eng.resolved,
                             {"song": args.song, "t": args.t, "resolution": [W, H], "view": args.view})
    print(f"wrote {args.out}")


def bench(args):
    W, H = args.size
    ctx = create_headless()
    print("GL:", gl_info(ctx)["GL_RENDERER"])
    preset = presetmod.load("ink_cold")
    results = {}
    tgt = Target(ctx, (W, H), 4, "f1", "nearest")
    for tier in args.tiers.split(","):
        eng = Engine(ctx, (W, H), preset, 7, tier, Tunables(timers=True), None)
        dt = 1.0 / 60.0
        eng.prewarm(2.0)
        n = int(args.seconds * 60)
        wall = []
        for i in range(n):
            t0 = time.perf_counter()
            eng.sim_step(2.0 + i * dt, dt)
            eng.render(tgt.fbo)
            ctx.finish()
            wall.append((time.perf_counter() - t0) * 1000.0)
            eng.timers.end_frame()
        summ = eng.timers.summary()
        total = round(sum(summ.values()), 3)
        results[tier] = {"particles": TIER_BY_NAME[tier].particles, "passes_ms": summ, "gpu_total_ms": total,
                         "wall_ms_mean": round(float(np.mean(wall[30:])), 3),
                         "wall_ms_p95": round(float(np.percentile(wall[30:], 95)), 3)}
        print(f"{tier:>5} ({TIER_BY_NAME[tier].particles / 1e6:.2f}M): gpu {total:.2f} ms  "
              f"wall mean {results[tier]['wall_ms_mean']:.2f} p95 {results[tier]['wall_ms_p95']:.2f}  {summ}")
        eng.release()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"bench_{W}x{H}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"gl": gl_info(ctx), "size": [W, H], "results": results}, fh, indent=2, default=str)
    print("wrote", path)
