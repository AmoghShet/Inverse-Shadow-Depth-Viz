"""Offline export (pattern-copied from audioviz/export/recorder.py): headless context, fixed-step
director at 60 Hz, picture ticks at the look frame rate, double-buffered PBO readback, a writer
thread feeding ffmpeg (x264 + AAC mux).

Depth input is deterministic: `synthetic` (the moving SDF figure) or `tape:DIR` (a depth tape recorded
live with --tape-out). Frame n shows song time t = start + n / fps; picture tick = floor(t * look_fps),
the same tick index the live app uses, so a tape replays tick-exactly. The same (preset, seed, song,
extractor version, source, resolution, fps, look fps) renders the same frames.
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

from audioviz.audio.analysis import EXTRACTOR_VERSION, load_or_analyze
from audioviz.audio.features import FeatureTrack
from audioviz.gl.context import create_headless, gl_info
from audioviz.gl.pingpong import Target
from audioviz.inputs.camera import ffmpeg_exe

from .. import OUT_DIR
from .. import preset as presetmod
from ..config import Tunables
from ..engine import Engine


class DepthFeed:
    """Deterministic depth per tick from the synthetic source or a tape."""

    def __init__(self, source, look_fps):
        self.source = source or "synthetic"
        self.look_fps = int(look_fps)
        self.tape = None
        self.synth = None
        self.last_key = None
        if self.source.startswith("tape:"):
            from ..inputs.tape import TapeReader

            self.tape = TapeReader(self.source[5:])
        else:
            from ..inputs.synthetic import Synthetic

            self.synth = Synthetic()

    def describe(self):
        if self.tape is not None:
            m = self.tape.meta
            return {"depth_source": self.source, "tape_records": self.tape.records,
                    "tape_span_s": [round(self.tape.t_first, 3), round(self.tape.t_last, 3)],
                    "tape_sha1": self.tape.sha1(), "model": m.get("model"), "ep": m.get("ep"),
                    "tape_look_fps": m.get("look_fps")}
        return {"depth_source": "synthetic", "model": None, "ep": None}

    def apply(self, eng, tick):
        """Upload the depth for this tick into the engine (only when it changed)."""
        t = tick / float(self.look_fps)
        if self.tape is not None:
            r = self.tape.at_time(max(t, self.tape.t_first))
            if r is None:
                eng.set_depth_info(on=False, new=False)
                return
            depth, mask, hdr, key = r
            if key != self.last_key:
                eng.upload_depth(depth, mask)
                from ..inputs.depthcam import HI

                eng.set_depth_info(fig_depth=hdr[HI["fig_depth"]], mask_cov=hdr[HI["mask_cov"]], on=True, new=True)
                self.last_key = key
            else:
                eng.set_depth_info(on=True, new=False)
            return
        depth, mask = self.synth.frame(t)
        h = self.synth.header(depth, mask)
        eng.upload_depth(depth, mask)
        eng.set_depth_info(fig_depth=h["fig_depth"], mask_cov=h["mask_cov"], on=True, new=True)


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


def export(song, seed, preset_name="fog_white", size=(1920, 1080), fps=None, out=None, start=0.0, duration=None,
           source="synthetic", look_fps=None, dump_frames=(), crf=18, x264_preset="veryfast", overrides=None,
           prewarm=2.0, frames_only=False, verbose=True, view="frame"):
    W, H = int(size[0]), int(size[1])
    t_begin = time.perf_counter()
    preset = presetmod.apply_overrides(presetmod.load(preset_name), overrides)
    feats = load_or_analyze(song, verbose=verbose) if song else None
    track = FeatureTrack(feats) if feats else None
    look_fps = int(look_fps or preset["grain"].get("frame_rate", 12))
    feed = DepthFeed(source, look_fps)
    song_dur = feats["duration"] if feats else 30.0
    if duration is None:
        end = song_dur
        if feed.tape is not None:
            end = min(song_dur, feed.tape.t_last + 1.0 / look_fps) if feats else feed.tape.t_last + 1.0 / look_fps
        duration = max(0.0, end - start)
    fps = int(fps or min(60, look_fps))
    n_frames = int(round(duration * fps))
    dt = 1.0 / 60.0

    ctx = create_headless()
    eng = Engine(ctx, (W, H), preset, seed, Tunables(timers=False), track, look_fps=look_fps)
    from ..render.present import VIEWS

    eng.view = VIEWS.index(view)
    tgt = Target(ctx, (W, H), 4, "f1", "nearest")
    frame_bytes = W * H * 3
    pbos = [ctx.buffer(reserve=frame_bytes) for _ in range(2)]

    out = out or os.path.join(OUT_DIR, "exports", f"seed{seed}_{W}x{H}_{fps}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    base = os.path.splitext(out)[0]
    proc, q, wt = None, None, None
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

    # prewarm: director envelopes and the depth stabiliser settle before `start` (deterministic)
    n_pre = int(round(prewarm * 60))
    sim_t = start - (n_pre + 1) * dt
    last_tick = None
    for i in range(n_pre):
        sim_t += dt
        eng.sim_step(sim_t, dt)
        tick = int(np.floor(sim_t * look_fps + 1e-9))
        if tick != last_tick:
            feed.apply(eng, tick)
            eng.render(tgt.fbo, tick=tick, t=sim_t)
            last_tick = tick
    last_tick, last_raw = None, None
    hold = look_fps < fps
    dumps = {}
    dump_set = set(int(x) for x in dump_frames)
    t_render = 0.0
    last_print = time.perf_counter()
    for n in range(n_frames):
        r0 = time.perf_counter()
        t_frame = start + n / fps
        while sim_t + dt <= t_frame + 1e-9:
            sim_t += dt
            eng.sim_step(sim_t, dt)
        tick = int(np.floor(t_frame * look_fps + 1e-9))
        if hold:
            if tick != last_tick:
                feed.apply(eng, tick)
                eng.render(tgt.fbo, tick=tick, t=t_frame)
                last_raw = tgt.fbo.read(components=3, alignment=1)
                last_tick = tick
            _emit(last_raw, n, q, dump_set, dumps, base, W, H)
        else:
            feed.apply(eng, tick)
            eng.render(tgt.fbo, tick=tick if tick != last_tick else tick, t=t_frame)
            last_tick = tick
            tgt.fbo.read_into(pbos[n % 2], components=3, alignment=1)
            if n >= 1:
                _emit(pbos[(n - 1) % 2].read(), n - 1, q, dump_set, dumps, base, W, H)
        t_render += time.perf_counter() - r0
        if verbose and time.perf_counter() - last_print > 5.0:
            last_print = time.perf_counter()
            print(f"  frame {n}/{n_frames}  {n / max(last_print - t_begin, 1e-6):.1f} fps")
    if n_frames and not hold:
        _emit(pbos[(n_frames - 1) % 2].read(), n_frames - 1, q, dump_set, dumps, base, W, H)

    ret, err = 0, ""
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
        "resolution": [W, H], "fps": fps, "look_fps": look_fps, "start": start, "duration": duration,
        "frames": n_frames, "prewarm_s": prewarm, "overrides": overrides or [], "source": source, "view": view,
        "output": os.path.abspath(out) if not frames_only else None, "gl": gl_info(ctx)["GL_RENDERER"],
        "frame_md5": dumps, "x264": {"preset": x264_preset, "crf": crf},
        "flips": eng.director.flip_count, "sections_applied": eng.director.section,
    }
    extra.update(feed.describe())
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
        dumps[n] = hashlib.md5(raw).hexdigest()
        img = np.flipud(np.frombuffer(raw, np.uint8).reshape(H, W, 3))
        cv2.imwrite(f"{base}_frame{n:05d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if q is not None:
        q.put(raw)


def export_from_params(path, out=None, **kw):
    """Re-run an export from a resolved_params.json (reproducibility check)."""
    with open(path, "r", encoding="utf-8") as fh:
        p = json.load(fh)
    return export(p["song"], p["seed"], p["preset"], tuple(p["resolution"]), p["fps"], out=out,
                  start=p.get("start", 0.0), duration=p.get("duration"), source=p.get("source", "synthetic"),
                  look_fps=p.get("look_fps"), overrides=p.get("overrides"), prewarm=p.get("prewarm_s", 2.0),
                  view=p.get("view", "frame"), **kw)
