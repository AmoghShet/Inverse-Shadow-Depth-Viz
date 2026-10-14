"""M3 (one look): the depth fog as a grainy photograph, checked on real camera depth (tapes from saved frames).

    python tests/depthviz_verify_look.py

1. depth gradation: with the artistic light, swirl and vignette off, on-screen tone follows depth (Spearman rank
   correlation between per-pixel depth and the displayed frame > 0.75); with the look as shipped, near (depth
   < 0.15) displays dark (< 0.35) and far (depth > 0.75) light (> 0.7)
2. soft edges: the figure / room transition on screen is soft (10-90% width 6..120 px at 1280x720), as in the
   references, not a hard cut-out
3. temporal consistency: mean tick-to-tick change of the soft fog tone on the dark (noisy) tape < 0.02
4. grain: a held frame is identical, the same tick re-rendered on a still scene is identical, the next tick differs
5. timing at 1080p (median of 3 rounds): tick < 2 ms
6. contact sheets of 12 seeds (lit and dark tapes): every seed is the same look
Report: out/verify/depthviz/report_m3_look.json
"""
import hashlib
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depthviz_common import (DH, DW, Target, create_headless, ensure_tape, make_engine, out_path, read_rgb,  # noqa: E402
                             save_rgb, write_report)

from depthviz.export.recorder import DepthFeed  # noqa: E402
from depthviz.render.present import VIEWS  # noqa: E402

STEP = 1.0 / 60.0


def render_ticks(eng, feed_or_depth, n, tgt, t=1.0, tick0=0):
    for i in range(n):
        for _ in range(5):
            t += STEP
            eng.sim_step(t, STEP)
        if isinstance(feed_or_depth, DepthFeed):
            feed_or_depth.apply(eng, tick0 + i)
        else:
            d, m = feed_or_depth
            eng.upload_depth(d, m)
            eng.set_depth_info(0.3, 0.0, 0.0, True, True)
        eng.render(tgt.fbo, tick=tick0 + i, t=t)
    return t


def frame_and_depth(eng, tgt, W, H):
    """Displayed frame luminance (0..1) and the per-pixel depth on screen (decoded from the depth view)."""
    eng.view = VIEWS.index("frame")
    eng.present_only(tgt.fbo)
    lum = read_rgb(tgt.fbo, W, H).astype(np.float32).mean(axis=2) / 255.0
    eng.view = VIEWS.index("depth")
    eng.present_only(tgt.fbo)
    dv = read_rgb(tgt.fbo, W, H).astype(np.float32).mean(axis=2) / 255.0
    eng.view = 0
    depth = np.power(np.clip(dv, 0, 1), 2.2)      # present gamma-encodes; values above 0.8 are shouldered
    return lum, depth


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def test_gradation(ctx, tape):
    W, H = 1280, 720
    res = []
    flat = ["light.gain=0.0", "light.blob=0.0", "fog.swirl=0.0", "grain.vignette=0.0"]
    for seed in (1, 2, 3):
        idx = np.random.default_rng(0).choice(W * H, 20000, replace=False)
        eng = make_engine(ctx, (W, H), seed, flat)
        tgt = Target(ctx, (W, H), 4, "f1", "nearest")
        render_ticks(eng, DepthFeed("tape:" + tape, 12), 24, tgt)
        lum0, depth0 = frame_and_depth(eng, tgt, W, H)
        rho = spearman(depth0.ravel()[idx], cv2.GaussianBlur(lum0, (0, 0), 6).ravel()[idx])
        tgt.release()
        eng.release()
        eng = make_engine(ctx, (W, H), seed, [])
        tgt = Target(ctx, (W, H), 4, "f1", "nearest")
        render_ticks(eng, DepthFeed("tape:" + tape, 12), 24, tgt)
        lum, depth = frame_and_depth(eng, tgt, W, H)
        soft = cv2.GaussianBlur(lum, (0, 0), 6)            # judge tone, not grain
        near, far = depth < 0.15, depth > 0.75
        res.append({"seed": seed, "spearman_depth_tone": round(rho, 3),
                    "near_display_mean": round(float(soft[near].mean()), 3) if near.any() else None,
                    "far_display_mean": round(float(soft[far].mean()), 3) if far.any() else None,
                    "spearman_with_light": round(spearman(depth.ravel()[idx], soft.ravel()[idx]), 3)})
        if seed == 1:
            save_rgb(out_path("look_gradation_seed1.png"), np.concatenate(
                [np.dstack([lum] * 3), np.dstack([np.power(depth, 1 / 2.2)] * 3)], axis=1)[::2, ::2] * 255)
            # soft edges: a row through the head; transition from the figure to the room on the right side
            row = soft[H // 2]
            drow = depth[H // 2]
            fig_cols = np.nonzero(drow < 0.15)[0]
            edge = None
            if len(fig_cols):
                x0 = int(fig_cols.max())
                seg = row[x0 - 60:x0 + 160]
                lo, hi = float(seg[:20].mean()), float(seg[-20:].mean())
                a, b = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
                i10 = int(np.argmax(seg >= a))
                i90 = int(np.argmax(seg >= b))
                edge = {"width_px": i90 - i10, "from": round(lo, 3), "to": round(hi, 3)}
            res[-1]["edge"] = edge
        tgt.release()
        eng.release()
    ok = all(r["spearman_depth_tone"] > 0.75 and (r["near_display_mean"] or 0) < 0.35
             and (r["far_display_mean"] or 1) > 0.7 for r in res)
    e = res[0]["edge"]
    soft_ok = e is not None and 6 <= e["width_px"] <= 120
    return res, ok, soft_ok


def test_consistency(ctx, tape):
    W, H = 960, 540
    eng = make_engine(ctx, (W, H), 4, ["fog.swirl=0.0", "fog.parallax=0.0"])
    tgt = Target(ctx, (W, H), 4, "f1", "nearest")
    feed = DepthFeed("tape:" + tape, 12)
    t = 1.0
    prev, diffs = None, []
    for i in range(40):
        for _ in range(5):
            t += STEP
            eng.sim_step(t, STEP)
        feed.apply(eng, i)
        eng.render(tgt.fbo, tick=i, t=t)
        soft = eng.read_tex(eng.blur.tex, 1)[..., 0]
        if prev is not None and i > 6:
            diffs.append(float(np.abs(soft - prev).mean()))
        prev = soft
    tgt.release()
    eng.release()
    m = float(np.mean(diffs))
    return {"mean_tick_change": round(m, 4), "p95_tick_change": round(float(np.percentile(diffs, 95)), 4), "pass": m < 0.02}


def md5(fbo):
    return hashlib.md5(fbo.read(components=3)).hexdigest()


def test_grain(ctx):
    d = np.full((DH, DW), 0.9, np.float32)
    d[60:200, 120:200] = 0.2
    m = np.zeros((DH, DW), np.uint8)
    d, m = np.ascontiguousarray(d[::-1]), np.ascontiguousarray(m[::-1])
    eng = make_engine(ctx, (960, 540), 12, ["fog.swirl=0.0", "fog.parallax=0.0"])
    tgt = Target(ctx, eng.fb_size, 4, "f1", "nearest")
    render_ticks(eng, (d, m), 40, tgt)
    eng.present_only(tgt.fbo)
    a = md5(tgt.fbo)
    eng.present_only(tgt.fbo)
    a2 = md5(tgt.fbo)
    st = eng.last_state
    eng.render(tgt.fbo, tick=39, t=st["t"])
    a3 = md5(tgt.fbo)
    eng.render(tgt.fbo, tick=40, t=st["t"])
    b = md5(tgt.fbo)
    tgt.release()
    eng.release()
    return {"held_present_identical": a == a2, "same_tick_rerender_identical": a == a3, "next_tick_differs": a != b,
            "pass": a == a2 and a == a3 and a != b}


def test_timing(ctx, tape):
    eng = make_engine(ctx, (1920, 1080), 5, [], timers=True)
    tgt = Target(ctx, eng.fb_size, 4, "f1", "nearest")
    feed = DepthFeed("tape:" + tape, 12)
    t, tick, rounds = 1.0, 0, []
    for rnd in range(4):
        eng.timers.hist.clear()
        for i in range(80 if rnd else 40):
            for _ in range(5):
                t += STEP
                eng.sim_step(t, STEP)
            feed.apply(eng, tick % 40)
            eng.render(tgt.fbo, tick=tick, t=t)
            tick += 1
            ctx.finish()
            eng.timers.end_frame()
        if rnd:
            rounds.append(eng.timers.summary())
    tgt.release()
    eng.release()
    totals = [sum(r.values()) for r in rounds]
    summ = rounds[int(np.argsort(totals)[1])]
    total = round(sum(summ.values()), 3)
    return {"passes_ms": summ, "tick_total_ms": total, "round_totals_ms": [round(x, 3) for x in totals],
            "pass": total < 2.0}


def contact_sheet(ctx, tape, name, seeds=range(1, 13)):
    W, H = 640, 360
    tiles = []
    for s in seeds:
        eng = make_engine(ctx, (W, H), s, [])
        tgt = Target(ctx, (W, H), 4, "f1", "nearest")
        render_ticks(eng, DepthFeed("tape:" + tape, 12), 24, tgt)
        img = read_rgb(tgt.fbo, W, H).copy()
        cv2.putText(img, f"seed {s}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 60), 1, cv2.LINE_AA)
        tiles.append(img)
        tgt.release()
        eng.release()
    rows = [np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, len(tiles), 3)]
    p = out_path(f"contact_look_{name}.png")
    save_rgb(p, np.concatenate(rows, axis=0))
    return p


if __name__ == "__main__":
    ctx = create_headless()
    lit, dark = ensure_tape("tape_lit", 4.0), ensure_tape("tape_dark", 4.0, dark=True)
    rep = {}
    rep["gradation"], rep["pass_depth_gradation"], rep["pass_soft_edges"] = test_gradation(ctx, lit)
    rep["consistency_dark"] = test_consistency(ctx, dark)
    rep["pass_temporal_consistency"] = rep["consistency_dark"]["pass"]
    rep["grain"] = test_grain(ctx)
    rep["pass_grain_refresh"] = rep["grain"]["pass"]
    rep["timing"] = test_timing(ctx, lit)
    rep["pass_timing"] = rep["timing"]["pass"]
    rep["contact_lit"] = contact_sheet(ctx, lit, "lit")
    rep["contact_dark"] = contact_sheet(ctx, dark, "dark")
    rep["all_pass"] = all(v for k, v in rep.items() if k.startswith("pass_"))
    write_report("report_m3_look.json", rep)
