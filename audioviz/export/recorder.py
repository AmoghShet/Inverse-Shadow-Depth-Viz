"""Offline export: fixed-step loop on a headless context, PBO readback, ffmpeg pipe with audio mux.

Frame n shows song time t = start + n / fps. The simulation always steps at 60 Hz through a
fixed-step accumulator (a 24 fps export runs 2-3 steps per frame), never reads the wall clock, and
the same (preset, seed, song, extractor version, resolution, fps, look fps) renders the same
frames. With a look frame rate below the file rate, frames are held (repeated).
Readback is double-buffered through two PBOs; a writer thread feeds ffmpeg so encoding overlaps
rendering.
"""
import hashlib
import json
import os
import queue
import subprocess
import threading
import time

import cv2
import numpy as np

from .. import preset as presetmod
from ..audio.analysis import EXTRACTOR_VERSION, load_or_analyze
from ..audio.features import FeatureTrack
from ..config import Tunables
from ..engine import Engine
from ..gl.context import create_headless, gl_info
from ..gl.pingpong import Target


def ffmpeg_exe():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _writer(proc, q, stats):
    while True:
        item = q.get()
        if item is None:
            break
        t0 = time.perf_counter()
        try:
            proc.stdin.write(item)
        except (BrokenPipeError, OSError) as e:
            stats["error"] = str(e)
            break
        stats["write_s"] += time.perf_counter() - t0


def export(song, seed, preset_name="ink_cold", size=(1920, 1080), fps=None, out=None, start=0.0, duration=None,
           tier="high", dump_frames=(), crf=18, x264_preset="veryfast", overrides=None, prewarm=None,
           frames_only=False, verbose=True, look_fps=None):
    from ..app import apply_overrides

    W, H = int(size[0]), int(size[1])
    t_begin = time.perf_counter()
    preset = presetmod.load(preset_name)
    apply_overrides(preset, overrides)
    feats = load_or_analyze(song, verbose=verbose) if song else None
    track = FeatureTrack(feats) if feats else None
    song_dur = feats["duration"] if feats else 60.0
    if duration is None:
        duration = max(0.0, song_dur - start)
    look_fps = int(look_fps or preset["grain"].get("frame_rate", 60))
    fps = int(fps or min(60, look_fps))
    hold = look_fps < fps        # e.g. a 24 fps look in a 60 fps file: frames are repeated
    n_frames = int(round(duration * fps))
    dt = 1.0 / 60.0              # the simulation always steps at 60 Hz (fixed-step accumulator)
    spf = round(60.0 / fps, 3)

    ctx = create_headless()
    tun = Tunables(timers=False)
    eng = Engine(ctx, (W, H), preset, seed, tier, tun, track)
    tgt = Target(ctx, (W, H), 4, "f1", "nearest")
    frame_bytes = W * H * 3
    pbos = [ctx.buffer(reserve=frame_bytes) for _ in range(2)]

    out = out or os.path.join("out", "exports", f"seed{seed}_{W}x{H}_{fps}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    base = os.path.splitext(out)[0]
    proc = None
    q = None
    stats = {"write_s": 0.0, "error": None}
    if not frames_only:
        cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
        if song:
            cmd += ["-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-i", song, "-map", "0:v", "-map", "1:a",
                    "-c:a", "aac", "-b:a", "320k"]
        cmd += ["-vf", "vflip", "-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-t", f"{duration:.6f}", "-movflags", "+faststart", out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        q = queue.Queue(maxsize=6)
        wt = threading.Thread(target=_writer, args=(proc, q, stats), daemon=True)
        wt.start()

    pre = float(eng.resolved["params"]["sim"]["prewarm_s"]) if prewarm is None else float(prewarm)
    eng.prewarm(pre, t0=start, dt=1.0 / 60.0)

    dumps = {}
    dump_set = set(int(d) for d in dump_frames)
    t_render = 0.0
    last_print = time.perf_counter()
    sim_t = start - dt            # prewarm ended one step before `start`
    last_tick, last_raw = None, None
    for n in range(n_frames):
        r0 = time.perf_counter()
        t_frame = start + n / fps
        while sim_t + dt <= t_frame + 1e-9:
            sim_t += dt
            eng.sim_step(sim_t, dt)
        if hold:
            tick = int(np.floor((t_frame - start) * look_fps + 1e-9))
            if tick != last_tick:
                eng.render(tgt.fbo)
                last_raw = tgt.fbo.read(components=3, alignment=1)
                last_tick = tick
            _emit(last_raw, n, q, dump_set, dumps, base, W, H)
        else:
            eng.render(tgt.fbo)
            tgt.fbo.read_into(pbos[n % 2], components=3, alignment=1)
            if n >= 1:
                prev = pbos[(n - 1) % 2].read()
                _emit(prev, n - 1, q, dump_set, dumps, base, W, H)
        t_render += time.perf_counter() - r0
        if verbose and time.perf_counter() - last_print > 5.0:
            last_print = time.perf_counter()
            el = last_print - t_begin
            print(f"  frame {n}/{n_frames}  {n / max(el, 1e-6):.1f} fps")
    if n_frames and not hold:
        _emit(pbos[(n_frames - 1) % 2].read(), n_frames - 1, q, dump_set, dumps, base, W, H)

    ret = 0
    err = ""
    if proc is not None:
        q.put(None)
        wt.join()
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        ret = proc.wait()

    wall = time.perf_counter() - t_begin
    extra = {
        "song": os.path.abspath(song) if song else None,
        "song_sha1": os.path.basename(feats["cache"]).split("_")[0] if feats else None,
        "feature_extractor_version": EXTRACTOR_VERSION,
        "resolution": [W, H], "fps": fps, "look_fps": look_fps, "steps_per_frame": spf, "start": start, "duration": duration,
        "frames": n_frames, "tier": tier, "prewarm_s": pre, "overrides": overrides or [],
        "output": os.path.abspath(out) if not frames_only else None,
        "gl": gl_info(ctx)["GL_RENDERER"], "frame_md5": dumps,
        "x264": {"preset": x264_preset, "crf": crf},
    }
    params_path = base + "_resolved_params.json"
    presetmod.write_resolved(params_path, eng.resolved, extra)
    result = {"out": out if not frames_only else None, "frames": n_frames, "wall_s": wall,
              "fps_achieved": n_frames / max(wall, 1e-6), "render_s": t_render, "pipe_write_s": stats["write_s"],
              "ffmpeg_ret": ret, "ffmpeg_err": err.strip()[:2000], "dumps": dumps, "params": params_path}
    for b in pbos:
        b.release()
    tgt.release()
    eng.release()
    ctx.release()
    if verbose:
        print(json.dumps({k: v for k, v in result.items() if k != "ffmpeg_err"}, indent=2))
        if err.strip():
            print("ffmpeg:", err.strip()[:800])
    return result


def _emit(raw, n, q, dump_set, dumps, base, W, H):
    if n in dump_set:
        md5 = hashlib.md5(raw).hexdigest()
        dumps[n] = md5
        img = np.flipud(np.frombuffer(raw, np.uint8).reshape(H, W, 3))
        cv2.imwrite(f"{base}_frame{n:05d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if q is not None:
        q.put(raw)


def export_from_params(path, out=None, **kw):
    """Re-run an export from a resolved_params.json (reproducibility check)."""
    with open(path, "r", encoding="utf-8") as fh:
        p = json.load(fh)
    return export(p["song"], p["seed"], p["preset"], tuple(p["resolution"]), p["fps"], out=out,
                  start=p.get("start", 0.0), duration=p.get("duration"), tier=p.get("tier", "high"),
                  overrides=p.get("overrides"), prewarm=p.get("prewarm_s"), look_fps=p.get("look_fps"), **kw)
