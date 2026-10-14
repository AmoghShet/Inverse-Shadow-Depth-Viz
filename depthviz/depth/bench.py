"""M0: depth benchmark spike.

    python -m depthviz bench-depth [--frames out/bench/frames_lit] [--dark out/bench/frames_dark] [--quick]

Matrix: model {midas_small, dav2_small fp32 / fp16 / q8 / q4f16} x input size {256 | 252x196, 336x252,
420x308} x EP {dml device 0 / 1 | coreml | cpu}. Every cell runs in its own subprocess (peak RSS and
DirectML state stay per cell). Per cell: session build s, warm-up, mean / p95 inference ms (60 runs),
end-to-end worker ms per frame (tmix-equivalent input, gain, model, normaliser, mask) -> worker fps
and capture-to-shm latency, peak RSS, and the frame p95 of a concurrent 1080p moderngl loop (GPU
contention). Quality, after alignment + normalisation:
  - lit: the real consecutive camera frames;
  - dark_uniform / dark_monitor: the same frames darkened to the user's lights-off brightness (~7/255)
    with shot + read noise and MJPEG re-encoding; "monitor" also lights only the face (background x0.2);
  - static_dark: ONE darkened frame with 30 independent noise realisations (a still scene; the
    temporal std criterion);
  - real dark frames from `camera-test --save-frames` when present.
Metrics: temporal std (static) < 0.04; separation (median depth outside the person mask minus inside,
mask from MediaPipe on the LIT frame) >= 0.2 with gain, also recorded without gain; bounds jumps > 0.2
of the range in < 5% of frames. Writes out/bench/depth_<platform>_<renderer>.json and strips per cell.
"""
import argparse
import ctypes
import glob
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from .. import OUT_DIR, ROOT_DIR
from . import models as M

BENCH_DIR = os.path.join(ROOT_DIR, "out", "bench")
FRAME_W, FRAME_H = 320, 240
DARK_TARGET = 7.0


# ---- utilities -------------------------------------------------------------------------------
def peak_rss_mb():
    try:
        if sys.platform.startswith("win"):
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            fn = k32.K32GetProcessMemoryInfo
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
            fn.restype = ctypes.c_int
            if not fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return -1.0
            return c.PeakWorkingSetSize / 1e6
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1e3
    except Exception:
        return -1.0


def load_frames(d, limit=30):
    ps = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))[:limit]
    return [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in ps]


def to_pipe(frames, tmix=3):
    """What the worker sees: ffmpeg tmix (mean of the last 3 frames) then area-scale to 320x240."""
    out = []
    for i in range(len(frames)):
        lo = max(0, i - tmix + 1)
        acc = np.mean([f.astype(np.float32) for f in frames[lo:i + 1]], axis=0)
        out.append(cv2.resize(np.clip(acc + 0.5, 0, 255).astype(np.uint8), (FRAME_W, FRAME_H),
                              interpolation=cv2.INTER_AREA))
    return out


def simulate_dark(rgb, rng, mode="uniform", face_xy=None, target=DARK_TARGET, jpeg_q=80):
    """Darken a lit frame to the lights-off camera level with sensor noise and MJPEG re-encoding."""
    x = rgb.astype(np.float32)
    h, w = x.shape[:2]
    if mode == "monitor":
        fx, fy = face_xy if face_xy is not None else (0.5 * w, 0.45 * h)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        illum = 0.2 + 0.8 * np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * (0.28 * w) ** 2))
        x = x * illum[..., None]
    x = x * (target / max(float(x.mean()), 1e-3))
    noise = rng.normal(0.0, 1.0, x.shape).astype(np.float32) * np.sqrt(0.8 * x + 1.6 ** 2)
    q = np.clip(np.round(x + noise), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(q, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def mp_masks(frames320):
    """MediaPipe selfie-landscape confidence masks (uint8) for the lit frames, or None."""
    try:
        from ..inputs.depthcam import MpMask

        mm = MpMask()
    except Exception as e:
        print("MediaPipe unavailable:", e)
        return None, None
    masks = [mm(f) for f in frames320]
    t = []
    for _ in range(3):
        for f in frames320[:20]:
            t0 = time.perf_counter()
            mm(f)
            t.append((time.perf_counter() - t0) * 1000.0)
    mm.close()
    return masks, {"mean_ms": round(float(np.mean(t)), 2), "p95_ms": round(float(np.percentile(t, 95)), 2)}


def face_center(mask):
    if mask is None:
        return None
    m = mask > 127
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return None
    top = ys.min()
    sel = ys < top + 0.4 * (ys.max() - top)
    return float(xs[sel].mean()), float(ys[sel].mean())


def build_sets(lit_dir, dark_dir=None, seed=11):
    lit_raw = load_frames(lit_dir)
    if not lit_raw:
        raise SystemExit(f"no frames in {lit_dir}; save some with: python -m depthviz camera-test --save-frames {lit_dir}")
    rng = np.random.default_rng(seed)
    lit320 = to_pipe(lit_raw, tmix=1)
    masks, mp_t = mp_masks(lit320)
    sets = {"lit": to_pipe(lit_raw)}
    fc = face_center(masks[0]) if masks else None
    fc640 = (fc[0] * lit_raw[0].shape[1] / FRAME_W, fc[1] * lit_raw[0].shape[0] / FRAME_H) if fc else None
    sets["dark_uniform"] = to_pipe([simulate_dark(f, rng, "uniform") for f in lit_raw])
    sets["dark_monitor"] = to_pipe([simulate_dark(f, rng, "monitor", fc640) for f in lit_raw])
    sets["static_dark"] = to_pipe([simulate_dark(lit_raw[0], rng, "monitor", fc640) for _ in range(len(lit_raw))])
    if dark_dir and os.path.isdir(dark_dir):
        real = load_frames(dark_dir)
        if real:
            sets["real_dark"] = to_pipe(real)
    return sets, masks, mp_t


# ---- per-cell worker (subprocess) ---------------------------------------------------------------
def _gl_loop(ready, stop, out_q, size=(1920, 1080), passes=4):
    """Concurrent GL load like the live app (several fullscreen passes per frame at 1080p, paced at
    60 Hz); records per-frame render+finish ms."""
    import moderngl

    from audioviz.gl.context import create_headless

    ctx = create_headless()
    prog = ctx.program(vertex_shader="""#version 410 core
in vec2 in_pos; out vec2 v; void main(){ v = in_pos*0.5+0.5; gl_Position = vec4(in_pos,0,1); }""",
                       fragment_shader="""#version 410 core
in vec2 v; out vec4 f; uniform float t;
uint pcg(uint x){ uint s = x*747796405u+2891336453u; uint w = ((s>>((s>>28u)+4u))^s)*277803737u; return (w>>22u)^w; }
void main(){ float a = 0.0; for (int i = 0; i < 24; ++i) { uint h = pcg(uint(gl_FragCoord.x)*1973u ^ uint(gl_FragCoord.y)*9277u ^ uint(i) ^ uint(t*60.0)); a += float(h & 255u)/255.0; }
 f = vec4(a/24.0 * v.x, v.y, 0.5, 1.0); }""")
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])
    tex = ctx.texture(size, 4, dtype="f2")
    fbo = ctx.framebuffer([tex])
    times = []
    ready.set()
    t_next = time.perf_counter()
    k = 0
    while not stop.is_set():
        t0 = time.perf_counter()
        fbo.use()
        for _ in range(passes):
            prog["t"].value = k / 60.0
            vao.render(moderngl.TRIANGLE_STRIP)
        ctx.finish()
        times.append((time.perf_counter() - t0) * 1000.0)
        k += 1
        t_next += 1.0 / 60.0
        dt = t_next - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        else:
            t_next = time.perf_counter()
    out_q.put(times)


def gl_baseline(seconds=4.0):
    import multiprocessing as mp

    c = mp.get_context("spawn")
    ready, stop, q = c.Event(), c.Event(), c.Queue()
    p = c.Process(target=_gl_loop, args=(ready, stop, q), daemon=True)
    p.start()
    ready.wait(30)
    time.sleep(seconds)
    stop.set()
    times = q.get(timeout=30)
    p.join(5)
    t = np.array(times[10:])
    return {"mean_ms": round(float(t.mean()), 2), "p95_ms": round(float(np.percentile(t, 95)), 2), "frames": len(t)}


def run_cell(model_id, size, ep, device_id, sets_path, out_path, runs=60, with_gl=True):
    import multiprocessing as mp

    from .infer import DepthModel, Normaliser, gain_frame
    from ..inputs.depthcam import depth_mask

    data = np.load(sets_path, allow_pickle=False)
    names = [n for n in ("lit", "dark_uniform", "dark_monitor", "static_dark", "real_dark") if n in data.files]
    sets = {n: data[n] for n in names}
    masks = data["masks"] if "masks" in data.files else None
    res = {"model": model_id, "size": size, "ep": ep, "device_id": device_id}
    t0 = time.perf_counter()
    try:
        dm = DepthModel(model_id, ep, size, device_id)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"[:400]
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        return
    res["session"] = dm.info
    res["build_s"] = round(time.perf_counter() - t0, 3)
    frames = sets["dark_monitor"]
    t1 = time.perf_counter()
    dm(frames[0])
    res["first_run_ms"] = round((time.perf_counter() - t1) * 1000.0, 1)
    for f in frames[1:4]:
        dm(f)
    # timed phase, with a concurrent GL loop
    gl = None
    if with_gl:
        c = mp.get_context("spawn")
        ready, stop, q = c.Event(), c.Event(), c.Queue()
        glp = c.Process(target=_gl_loop, args=(ready, stop, q), daemon=True)
        glp.start()
        ready.wait(30)
        time.sleep(0.5)
    inf, e2e = [], []
    norm = Normaliser()
    gst = {}
    for i in range(runs):
        f = frames[i % len(frames)]
        p0 = time.perf_counter()
        g, _ = gain_frame(f, gst, 1 / 30)
        x = dm.preprocess(g)
        z = dm.run(x)
        inf.append(dm.last_ms)
        z = cv2.resize(z.astype(np.float32), (FRAME_W, FRAME_H),
                       interpolation=cv2.INTER_AREA if z.shape[1] > FRAME_W else cv2.INTER_LINEAR)
        d = norm(z, 1 / 30)
        depth_mask(d)
        e2e.append((time.perf_counter() - p0) * 1000.0)
    if with_gl:
        stop.set()
        gl_times = np.array(q.get(timeout=30)[5:])
        glp.join(5)
        gl = {"mean_ms": round(float(gl_times.mean()), 2), "p95_ms": round(float(np.percentile(gl_times, 95)), 2),
              "frames": int(len(gl_times))}
    inf, e2e = np.array(inf[5:]), np.array(e2e[5:])
    res["infer_ms_mean"] = round(float(inf.mean()), 2)
    res["infer_ms_p95"] = round(float(np.percentile(inf, 95)), 2)
    res["e2e_ms_mean"] = round(float(e2e.mean()), 2)
    res["e2e_ms_p95"] = round(float(np.percentile(e2e, 95)), 2)
    res["worker_fps"] = round(min(30.0, 1000.0 / max(res["e2e_ms_mean"], 1e-3)), 1)
    # capture -> shm: tmix delay (1 frame), mean wait for the newest frame (half a frame), processing
    res["latency_ms_est"] = round(1000.0 / 30 * 1.5 + res["e2e_ms_mean"], 1)
    res["gl_concurrent"] = gl
    # quality
    q = {}
    strips = []
    for n in names:
        for gain_on in (True, False):
            if n == "lit" and not gain_on:
                continue
            norm = Normaliser()
            gst = {}
            ds, los, his = [], [], []
            for f in sets[n]:
                g = gain_frame(f, gst, 1 / 30)[0] if gain_on else f
                d = norm(dm(g), 1 / 30)
                ds.append(d)
                los.append(norm.lo)
                his.append(norm.hi)
            ds = np.array(ds)
            entry = {}
            tail = ds[5:]
            entry["temporal_std"] = round(float(tail.std(axis=0).mean()), 4)
            entry["frame_diff"] = round(float(np.abs(np.diff(tail, axis=0)).mean()), 4)
            lo, hi = np.array(los), np.array(his)
            rngs = np.maximum(hi - lo, 1e-6)
            jumps = (np.abs(np.diff(lo)) + np.abs(np.diff(hi))) / rngs[1:]
            entry["bounds_jump_frac"] = round(float((jumps > 0.2).mean()), 3)
            if masks is not None and n != "real_dark":
                seps = []
                for k, d in enumerate(ds):
                    mk = masks[0] if n == "static_dark" else masks[min(k, len(masks) - 1)]
                    inside = mk > 127
                    outside = mk < 40
                    if inside.mean() > 0.02 and outside.mean() > 0.02:
                        seps.append(float(np.median(d[outside]) - np.median(d[inside])))
                entry["separation"] = round(float(np.median(seps)), 3) if seps else None
            q[n + ("" if gain_on else "_nogain")] = entry
            if gain_on:
                strips.append(ds[min(15, len(ds) - 1)])
    res["quality"] = q
    res["peak_rss_mb"] = round(peak_rss_mb(), 1)
    strip = np.concatenate([(s * 255).astype(np.uint8) for s in strips], axis=1)
    sp = os.path.splitext(out_path)[0] + ".png"
    cv2.imwrite(sp, strip)
    res["strip"] = sp
    res["strip_order"] = names
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


def stability_after_d1(model_id, size, ep, device_id, sets_path, look_fps=12, tau=0.15, edge_reset=0.12):
    """Temporal std of a still dark scene at the worker output vs after D1 as drawn: the worker runs at
    the camera's 30 fps, D1 samples the newest depth every 1/look_fps and applies its EMA
    (a = 1 - exp(-dt / tau), 0.85 where the jump exceeds edge_reset). CPU emulation of depth_stab.frag."""
    from .infer import DepthModel, Normaliser, gain_frame

    data = np.load(sets_path, allow_pickle=False)
    frames = list(data["static_dark"]) * 3          # 90 frames = 3 s at 30 fps
    dm = DepthModel(model_id, ep, size, device_id)
    norm = Normaliser()
    gst = {}
    ds = []
    for f in frames:
        g, _ = gain_frame(f, gst, 1 / 30)
        ds.append(norm(dm(g), 1 / 30))
    ds = np.array(ds)
    a = 1.0 - np.exp(-(1.0 / look_fps) / tau)
    ticks = []
    prev = None
    k = 0
    while True:
        idx = int(np.floor(k * 30.0 / look_fps))
        if idx >= len(ds):
            break
        dn = ds[idx]
        if prev is None:
            prev = dn.copy()
        else:
            aa = np.where(np.abs(dn - prev) > edge_reset, np.maximum(a, 0.85), a)
            prev = prev + (dn - prev) * aa
        ticks.append(prev.copy())
        k += 1
    ticks = np.array(ticks)
    return {"model": model_id, "size": size, "ep": ep, "device_id": device_id,
            "worker_temporal_std": round(float(ds[10:].std(axis=0).mean()), 4),
            "after_d1_temporal_std": round(float(ticks[4:].std(axis=0).mean()), 4),
            "ema_a_per_tick": round(float(a), 3), "ticks": int(len(ticks))}


# ---- orchestrator ---------------------------------------------------------------------------------
def matrix(quick=False, eps=None):
    if sys.platform.startswith("win"):
        ep_list = eps or [("dml", 0), ("dml", 1), ("cpu", 0)]
    elif sys.platform == "darwin":
        ep_list = eps or [("coreml", 0), ("cpu", 0)]
    else:
        ep_list = eps or [("cpu", 0)]
    cells = []
    for ep, dev in ep_list:
        cells.append(("midas_small", 256, ep, dev))
        for mid in ("dav2_small", "dav2_small_fp16", "dav2_small_q8", "dav2_small_q4f16"):
            for (w, _) in (M.DAV2_SIZES if not quick else M.DAV2_SIZES[:2]):
                cells.append((mid, w, ep, dev))
    return cells


def accept(r):
    if "error" in r:
        return False, "error"
    why = []
    if r["worker_fps"] < 10:
        why.append("fps<10")
    if r["latency_ms_est"] > 150:
        why.append("latency>150")
    if r.get("gl_concurrent") and r["gl_concurrent"]["p95_ms"] >= 16:
        why.append("gl_p95>=16")
    q = r["quality"]
    if q.get("static_dark", {}).get("temporal_std", 1) >= 0.04:
        why.append("temporal_std>=0.04")
    for n in ("dark_uniform", "dark_monitor"):
        s = q.get(n, {}).get("separation")
        if s is None or s < 0.2:
            why.append(f"sep_{n}<0.2")
        if q.get(n, {}).get("bounds_jump_frac", 1) >= 0.05:
            why.append(f"jumps_{n}>=5%")
    return not why, ",".join(why)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="depthviz bench-depth")
    ap.add_argument("--frames", default=os.path.join(BENCH_DIR, "frames_lit"))
    ap.add_argument("--dark", default=os.path.join(BENCH_DIR, "frames_dark"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of model ids")
    ap.add_argument("--eps", default=None, help="e.g. dml:0,cpu:0")
    ap.add_argument("--no-gl", action="store_true")
    ap.add_argument("--stability", default=None, help="model:size:ep:dev -> temporal std before / after D1")
    ap.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--sets", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.stability:
        mid, size, ep, dev = a.stability.split(":")
        r = stability_after_d1(mid, int(size), ep, int(dev), os.path.join(BENCH_DIR, "depth_cells", "sets.npz"))
        print(json.dumps(r, indent=1))
        with open(os.path.join(BENCH_DIR, f"stability_{mid}_{size}_{ep}{dev}.json"), "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1)
        return 0
    if a.cell:   # subprocess mode
        mid, size, ep, dev = a.cell.split(":")
        run_cell(mid, int(size), ep, int(dev), a.sets, a.out, with_gl=not a.no_gl)
        return 0

    from audioviz.gl.context import create_headless, renderer_name

    ctx = create_headless()
    renderer = renderer_name(ctx)
    ctx.release()
    tag = f"{sys.platform}_{''.join(c if c.isalnum() else '_' for c in renderer.lower())[:40]}"
    cell_dir = os.path.join(BENCH_DIR, "depth_cells")
    os.makedirs(cell_dir, exist_ok=True)
    print(f"GL renderer: {renderer}\nbuilding frame sets from {a.frames} ...")
    sets, masks, mp_t = build_sets(a.frames, a.dark)
    sets_path = os.path.join(cell_dir, "sets.npz")
    payload = {k: np.array(v) for k, v in sets.items()}
    if masks is not None:
        payload["masks"] = np.array(masks)
    np.savez_compressed(sets_path, **payload)
    for k, v in sets.items():
        cv2.imwrite(os.path.join(cell_dir, f"set_{k}.png"), cv2.cvtColor(np.concatenate(v[:4], axis=1), cv2.COLOR_RGB2BGR))
    print("sets:", {k: (len(v), round(float(np.mean(v)), 1)) for k, v in sets.items()}, "mediapipe:", mp_t)
    base = None if a.no_gl else gl_baseline()
    print("GL baseline (no inference):", base)
    eps = [(e.split(":")[0], int(e.split(":")[1])) for e in a.eps.split(",")] if a.eps else None
    cells = matrix(a.quick, eps)
    if a.only:
        keep = set(a.only.split(","))
        cells = [c for c in cells if c[0] in keep]
    results = []
    for (mid, size, ep, dev) in cells:
        if not os.path.isfile(M.model_path(mid)):
            continue
        name = f"{mid}_{size}_{ep}{dev}"
        out = os.path.join(cell_dir, name + ".json")
        cmd = [sys.executable, "-m", "depthviz.depth.bench", "--cell", f"{mid}:{size}:{ep}:{dev}", "--sets", sets_path,
               "--out", out] + (["--no-gl"] if a.no_gl else [])
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, errors="replace", timeout=900)
        try:
            with open(out, encoding="utf-8") as fh:
                r = json.load(fh)
        except Exception:
            r = {"model": mid, "size": size, "ep": ep, "device_id": dev, "error": (p.stderr or p.stdout)[-600:]}
        ok, why = accept(r)
        r["accepted"], r["reject_reasons"] = ok, why
        results.append(r)
        if "error" in r:
            print(f"{name:>28}: ERROR {r['error'][:160]}")
        else:
            q = r["quality"]
            print(f"{name:>28}: build {r['build_s']:5.1f}s inf {r['infer_ms_mean']:6.1f}/{r['infer_ms_p95']:6.1f} ms "
                  f"e2e {r['e2e_ms_mean']:6.1f} fps {r['worker_fps']:4.1f} gl95 {(r['gl_concurrent'] or {}).get('p95_ms', '-')} "
                  f"rss {r['peak_rss_mb']:6.0f} | tstd {q['static_dark']['temporal_std']:.3f} "
                  f"sep lit {q['lit'].get('separation')} du {q['dark_uniform'].get('separation')} "
                  f"dm {q['dark_monitor'].get('separation')} dm_nogain {q['dark_monitor_nogain'].get('separation')} "
                  f"| {'ACCEPT' if ok else 'reject: ' + why}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    rep = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "renderer": renderer, "platform": sys.platform,
           "frames": a.frames, "dark_frames": a.dark if "real_dark" in sets else None,
           "sets": {k: {"n": len(v), "mean": round(float(np.mean(v)), 2)} for k, v in sets.items()},
           "mediapipe_320x240": mp_t, "gl_baseline": base, "results": results}
    os.makedirs(BENCH_DIR, exist_ok=True)
    path = os.path.join(BENCH_DIR, f"depth_{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
