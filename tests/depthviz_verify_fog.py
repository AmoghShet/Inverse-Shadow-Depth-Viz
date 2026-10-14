"""M2: the fog look (depth -> tone), headless.

    python tests/depthviz_verify_fog.py

1. tones on a synthetic scene (figure d 0.2, room d 0.92): high-key room > 0.8 and figure < 0.45; inverted
   room < 0.15 and figure > 0.5; and a depth ramp (d 0 -> 1 across the frame) maps to a monotonic tone ramp
   spanning > 0.6 (continuous depth gradation, not a near/far switch)
2. stabiliser: a small step (< edge reset) is >= 80% settled within 5 ticks (tau 0.22 s); a big step (> edge reset)
   is >= 85% there after 1 tick; a still scene with noise (std 0.03 per tick) comes out with a
   temporal std < 0.3x the input's
3. mask halo: a 6 px blurred silhouette with a sharp mask -> output edge (10-90%) <= 2 camera texels
4. polarity flip: the tone pair crossfades monotonically; the exposure pole is reached (whiteout
   >= 3.5x and frame mean > 0.9 going to inverted; blackout <= 0.1x and mean < 0.1 going back)
5. portrait is not stretched: a camera-space disc stays round on screen at 16:9 and 9:16
6. PNGs for the user (fog view + depth view, synthetic and real camera depth, lit and dark,
   high-key and inverted) in out/verify/depthviz/fog_*.png
Report: out/verify/depthviz/report_m2_fog.json
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depthviz_common import (DH, DW, ROOT, Target, create_headless, ensure_tape, make_engine, out_path,  # noqa: E402
                             read_rgb, save_rgb, scene_figure, write_report)

STEP = 1.0 / 60.0


def run_ticks(eng, depth, mask, n, t0=1.0, fig=None, cov=None, tick0=0):
    """n picture ticks at the engine's look rate with fixed depth; returns the last time."""
    tgt = Target(eng.ctx, eng.fb_size, 4, "f1", "nearest")
    t = t0
    per = int(round(60 / eng.look_fps))
    m = mask > 127
    for i in range(n):
        for _ in range(per):
            t += STEP
            eng.sim_step(t, STEP)
        eng.upload_depth(depth, mask)
        eng.set_depth_info(fig if fig is not None else (float(np.median(depth[m])) if m.any() else 0.5),
                           cov if cov is not None else float(m.mean()), 0.0, True, True)
        eng.render(tgt.fbo, tick=tick0 + i, t=t)
    tgt.release()
    return t


def fog_v(eng):
    return eng.read_tex(eng.fog.tex, 4)


def test_tones(ctx):
    d, m, _ = scene_figure()
    res = {}
    for pol, ov in (("high", ["random.p_high_key=1.0", "depth.mask=true"]),
                    ("inv", ["random.p_high_key=0.0", "depth.mask=true"])):
        vals = []
        for seed in (1, 2, 3, 4, 5):
            eng = make_engine(ctx, (960, 540), seed, ov)
            run_ticks(eng, d, m, 8)
            f = fog_v(eng)
            fig = f[..., 3] > 0.9
            far = f[..., 3] < 0.05
            vals.append({"seed": seed, "far_mean": round(float(f[..., 0][far].mean()), 3),
                         "figure_mean": round(float(f[..., 0][fig].mean()), 3)})
            eng.release()
        res[pol] = vals
    ok_high = all(v["far_mean"] > 0.8 and v["figure_mean"] < 0.45 for v in res["high"])
    ok_inv = all(v["far_mean"] < 0.15 and v["figure_mean"] > 0.5 for v in res["inv"])
    # depth ramp: d = x across the camera frame -> tone along the screen row must rise monotonically
    ramp = np.ascontiguousarray(np.tile(np.linspace(0.0, 1.0, DW, dtype=np.float32), (DH, 1)))
    eng = make_engine(ctx, (960, 540), 2, ["random.p_high_key=1.0", "fog.swirl=0.0", "fog.parallax=0.0",
                                           "light.gain=0.0", "light.blob=0.0"])
    run_ticks(eng, ramp, np.zeros((DH, DW), np.uint8), 8, cov=0.0)
    row = fog_v(eng)[..., 0].mean(axis=0)
    eng.release()
    prof = [round(float(x), 3) for x in row[::60]]
    mono = bool(np.all(np.diff(row[8:-8]) > -0.004))
    res["ramp"] = {"profile_every_60px": prof, "monotonic": mono, "span": round(float(row[-8] - row[8]), 3)}
    ok_high = ok_high and mono and res["ramp"]["span"] > 0.6
    return res, ok_high, ok_inv


def stab_value(eng, x=0.25, y=0.5):
    s = eng.read_tex(eng.depth.stab.src, 2)
    h, w = s.shape[:2]
    return float(s[int(y * h), int(x * w), 0])


def test_stabiliser(ctx):
    eng = make_engine(ctx, (640, 360), 3, ["depth.mask=false"])
    zeros = np.zeros((DH, DW), np.uint8)
    res = {}
    for name, a, b, n_need, need in (("small_step", 0.5, 0.55, 5, 0.8), ("big_step", 0.3, 0.75, 1, 0.85)):
        run_ticks(eng, np.full((DH, DW), a, np.float32), zeros, 6, cov=0.0)
        resp = []
        for i in range(n_need):
            run_ticks(eng, np.full((DH, DW), b, np.float32), zeros, 1, cov=0.0)
            resp.append(round((stab_value(eng) - a) / (b - a), 3))
        res[name] = {"from": a, "to": b, "response_per_tick": resp, "need": need, "pass": resp[-1] >= need}
    rng = np.random.default_rng(5)
    outs, ins = [], []
    for i in range(36):
        x = np.clip(0.5 + rng.normal(0, 0.03, (DH, DW)), 0, 1).astype(np.float32)
        ins.append(x[60:180, 40:280])
        run_ticks(eng, x, zeros, 1, cov=0.0)
        f = eng.read_tex(eng.depth.tex, 4)[..., 0]
        outs.append(f[120:360, 80:560])
    s_in = float(np.std(np.array(ins[6:]), axis=0).mean())
    s_out = float(np.std(np.array(outs[6:]), axis=0).mean())
    res["noise"] = {"input_temporal_std": round(s_in, 4), "output_temporal_std": round(s_out, 4),
                    "ratio": round(s_out / s_in, 3), "pass": s_out < 0.3 * s_in}
    eng.release()
    return res


def edge_width(profile, lo, hi):
    """10-90% transition width (in samples) of a rising profile."""
    a, b = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
    i10 = int(np.argmax(profile >= a))
    i90 = int(np.argmax(profile >= b))
    return i90 - i10


def test_halo(ctx):
    d_blur, m, _ = scene_figure(blur_px=6.0)
    res = {}
    for name, ov in (("mask_on", ["depth.mask=true"]), ("mask_off", ["depth.mask=false"])):
        eng = make_engine(ctx, (640, 360), 3, ov + ["depth.sigma_s=2.0"])
        run_ticks(eng, d_blur, m, 8, fig=0.2, cov=float((m > 127).mean()))
        f = eng.read_tex(eng.depth.tex, 4)[..., 0]          # 640x480, GL rows
        H2, W2 = f.shape
        row = f[int(H2 * 0.3)]                               # GL row 0.3 = 70% down the image: inside the figure
        # left edge of the figure at x = 0.35 (camera width): walk outward from the inside
        xe = int(0.35 * W2)
        prof = row[xe - 40:xe + 40][::-1]                    # from inside (near) to outside (far)
        w = edge_width(prof, 0.2, 0.92)
        res[name] = {"edge_width_texels_640": int(w), "edge_width_texels_320": w / 2.0}
        eng.release()
    res["pass"] = res["mask_on"]["edge_width_texels_320"] <= 2.0
    return res


def test_flip(ctx):
    d, m, _ = scene_figure()
    eng = make_engine(ctx, (640, 360), 9, ["random.p_high_key=1.0"])
    t = run_ticks(eng, d, m, 6)
    base_exp = eng.last_state["exposure"]
    tgt = Target(ctx, eng.fb_size, 4, "f1", "nearest")
    rec = []
    for leg in ("to_inverted", "to_high"):
        eng.director.flip_now(t)
        tick0 = 1000 if leg == "to_inverted" else 2000
        for i in range(int(3.0 * 12)):
            for _ in range(5):
                t += STEP
                eng.sim_step(t, STEP)
            eng.upload_depth(d, m)
            eng.set_depth_info(0.2, float((m > 127).mean()), 0.0, True, True)
            eng.render(tgt.fbo, tick=tick0 + i, t=t)
            s = eng.last_state
            f = fog_v(eng)
            rec.append({"leg": leg, "t": round(t, 3), "x": round(s["flip_x"], 3), "near": round(s["near"], 4),
                        "far": round(s["far"], 4), "exp": round(s["exposure"], 3), "mean_v": round(float(f[..., 0].mean()), 3)})
    tgt.release()
    eng.release()
    out = {}
    for leg in ("to_inverted", "to_high"):
        r = [x for x in rec if x["leg"] == leg]
        near = [x["near"] for x in r]
        far = [x["far"] for x in r]
        mono = (all(b >= a - 1e-6 for a, b in zip(near, near[1:])) and all(b <= a + 1e-6 for a, b in zip(far, far[1:]))) \
            if leg == "to_inverted" else \
            (all(b <= a + 1e-6 for a, b in zip(near, near[1:])) and all(b >= a - 1e-6 for a, b in zip(far, far[1:])))
        exps = [x["exp"] / base_exp for x in r]
        means = [x["mean_v"] for x in r]
        if leg == "to_inverted":
            pole = max(exps) >= 3.5 and max(means) > 0.9
        else:
            pole = min(exps) <= 0.1 and min(means) < 0.1
        out[leg] = {"monotonic": mono, "pole_reached": pole, "exp_ratio_extreme": round(max(exps) if leg == "to_inverted" else min(exps), 3),
                    "mean_v_extreme": max(means) if leg == "to_inverted" else min(means),
                    "end_near_far": [near[-1], far[-1]]}
    out["trace"] = rec[::3]
    out["pass"] = all(out[k]["monotonic"] and out[k]["pole_reached"] for k in ("to_inverted", "to_high"))
    return out


def test_portrait(ctx):
    yy, xx = np.mgrid[0:DH, 0:DW].astype(np.float32)
    disc = ((xx - DW / 2) ** 2 + (yy - DH / 2) ** 2) < (0.3 * DH) ** 2
    d = np.ascontiguousarray(np.where(disc, 0.2, 0.92).astype(np.float32)[::-1])
    m = np.ascontiguousarray((disc * 255).astype(np.uint8)[::-1])
    res = {}
    for name, size, preset in (("landscape", (1920, 1080), "fog_white"), ("portrait", (1080, 1920), "fog_white_portrait")):
        eng = make_engine(ctx, size, 4, ["fog.parallax=0.0", "depth.mask=true"], preset=preset)
        run_ticks(eng, d, m, 6, fig=0.2, cov=float(disc.mean()))
        f = fog_v(eng)
        fig = f[..., 3] > 0.5
        ys, xs = np.nonzero(fig)
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        cut = xs.min() == 0 or xs.max() == f.shape[1] - 1
        res[name] = {"bbox_w": int(w), "bbox_h": int(h), "aspect": round(w / h, 3), "touches_side": bool(cut)}
        eng.release()
    # the disc is 0.6 of the camera height; the cover crop shows `vis` of the camera height (0.75 at 16:9,
    # 1.0 at 9:16), so its on-screen height is 0.6 / vis of the frame. In portrait the 42% width crop cuts
    # the disc's sides, so only the height is compared there; in landscape the bbox must also be square.
    from depthviz.engine import cover_scale

    ok = True
    for name, size in (("landscape", (1920, 1080)), ("portrait", (1080, 1920))):
        vis = cover_scale(size[0] / size[1])[1]
        exp_h = 0.6 / vis * (size[1] // 2)
        res[name]["expected_h"] = round(exp_h, 1)
        ok = ok and abs(res[name]["bbox_h"] / exp_h - 1.0) < 0.05
    ok = ok and abs(res["landscape"]["aspect"] - 1.0) < 0.05
    res["pass"] = ok
    return res


def pngs(ctx):
    """Fog / depth views for the user to judge against the references."""
    from depthviz.export.recorder import DepthFeed
    from depthviz.inputs.synthetic import Synthetic
    from depthviz.render.present import VIEWS

    shots = []
    W, H = 1280, 720
    tapes = {}
    try:
        tapes["lit"] = ensure_tape("tape_lit", 4.0)
        tapes["dark"] = ensure_tape("tape_dark", 4.0, dark=True)
    except Exception as e:
        print("real-depth tapes skipped:", e)
    sources = [("synthetic", "synthetic")] + [(k, "tape:" + v) for k, v in tapes.items()]
    for sname, src in sources:
        for pol, ov in (("high", ["random.p_high_key=1.0"]), ("inv", ["random.p_high_key=0.0"])):
            eng = make_engine(ctx, (W, H), 21, ov)
            feed = DepthFeed(src, 12)
            tgt = Target(ctx, (W, H), 4, "f1", "nearest")
            t = 0.0
            for tick in range(30):
                for _ in range(5):
                    t += STEP
                    eng.sim_step(t, STEP)
                feed.apply(eng, tick)
                eng.render(tgt.fbo, tick=tick, t=t)
            row = []
            for view in ("fog", "depth", "frame"):
                eng.view = VIEWS.index(view)
                eng.present_only(tgt.fbo)
                row.append(read_rgb(tgt.fbo, W, H))
            img = np.concatenate([cv2.resize(r, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for r in row], axis=1)
            p = out_path(f"fog_{sname}_{pol}.png")
            save_rgb(p, img)
            shots.append(p)
            tgt.release()
            eng.release()
    return shots


if __name__ == "__main__":
    ctx = create_headless()
    rep = {}
    rep["tones"], rep["pass_high_key_tones"], rep["pass_inverted_tones"] = test_tones(ctx)
    rep["stabiliser"] = test_stabiliser(ctx)
    rep["pass_stab_small_step"] = rep["stabiliser"]["small_step"]["pass"]
    rep["pass_stab_big_step"] = rep["stabiliser"]["big_step"]["pass"]
    rep["pass_stab_noise"] = rep["stabiliser"]["noise"]["pass"]
    rep["halo"] = test_halo(ctx)
    rep["pass_mask_halo"] = rep["halo"]["pass"]
    rep["flip"] = test_flip(ctx)
    rep["pass_flip"] = rep["flip"]["pass"]
    rep["portrait"] = test_portrait(ctx)
    rep["pass_portrait_not_stretched"] = rep["portrait"]["pass"]
    if "--no-png" not in sys.argv:
        try:
            rep["pngs"] = pngs(ctx)
        except Exception as e:   # the checks above still get reported
            import traceback

            rep["pngs_error"] = traceback.format_exc()[-1500:]
    rep["all_pass"] = all(v for k, v in rep.items() if k.startswith("pass_"))
    write_report("report_m2_fog.json", rep)
