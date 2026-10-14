"""Live mode: window, song clock, fixed-step director at 60 Hz, picture ticks at the look rate
(12 fps default; frames held in between), depth source (camera worker / synthetic / tape), keys, HUD,
screenshots, recording, depth tape, hot reload.

Keys: R reseed, [ ] seed -/+1, Space pause, S screenshot + sidecar JSON, Ctrl+S save preset,
A cycle view (frame / depth / fog / mask), B halation, H HUD, F fullscreen, P look fps 12/24/60,
1-9 presets, Left/Right scrub 5 s, , . audio offset -/+10 ms, E export (from the current tape when
--tape-out is on, else synthetic), C camera on/off (off holds the last depth), V polarity flip by hand,
Esc quit.
"""
import collections
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from audioviz.audio.features import FeatureTrack
from audioviz.gl.context import create_window, gl_info, renderer_name
from audioviz.gl.pingpong import Target
from audioviz.render.present import Hud

from . import OUT_DIR, ROOT_DIR, TITLE
from . import preset as presetmod
from . import seed as seedmod
from .config import Tunables, pick_profile
from .engine import Engine
from .inputs.depthcam import EPS, HI
from .inputs.synthetic import Synthetic
from .render.present import VIEWS


def _bar(v, n=10):
    k = int(round(np.clip(v, 0, 1) * n))
    return "#" * k + "-" * (n - k)


def _fmt_t(t):
    if not np.isfinite(t):
        return "--:--.-"
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:04.1f}"


def window_size_for(aspect_wh, frac=0.82):
    """Logical window size that fits the primary monitor at the requested aspect (darwin: the video
    mode is already in points, so no content-scale division there)."""
    import glfw

    try:
        glfw.init()
        mon = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(mon)
        if sys.platform == "darwin":
            sw, sh = mode.size.width, mode.size.height
        else:
            sx, sy = glfw.get_monitor_content_scale(mon)
            sw, sh = mode.size.width / sx, mode.size.height / sy
    except Exception:
        sw, sh = 1536, 864
    aw, ah = aspect_wh
    h = sh * frac
    w = h * aw / ah
    if w > sw * frac:
        w = sw * frac
        h = w * ah / aw
    return int(w), int(h)


class App:
    def __init__(self, args):
        self.args = args
        self.tun = Tunables()
        self.preset_name = args.preset
        self.preset = presetmod.apply_overrides(presetmod.load(args.preset), args.set)
        aspect = args.aspect or self.preset["meta"].get("aspect", "16:9")
        aw, ah = (int(v) for v in aspect.split(":"))
        size = tuple(int(v) for v in args.size.split("x")) if args.size else window_size_for((aw, ah))
        self.window = create_window(title=TITLE, size=size, fullscreen=args.fullscreen, vsync=not args.no_vsync)
        self.ctx = self.window.ctx
        info = gl_info(self.ctx)
        self.renderer = renderer_name(self.ctx)
        print(f"GL: {info['GL_RENDERER']} | {info['GL_VERSION']} | buffer {self.window.buffer_size}")

        self.song = args.song
        self.track = None
        self.features = None
        if self.song:
            from audioviz.audio.analysis import load_audio, load_or_analyze
            from audioviz.audio.player import Player

            print("analysing song (cached after the first run)...")
            self.features = load_or_analyze(self.song)
            self.track = FeatureTrack(self.features)
            pcm, sr = load_audio(self.song)
            self.clock = Player(pcm, sr, loop=not args.no_loop, offset_s=args.offset_ms / 1000.0, mute=args.mute)
        else:
            from audioviz.audio.player import WallClock

            self.clock = WallClock()

        self.look_fps = int(args.look_fps or self.preset["grain"].get("frame_rate", 12))
        seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(3), "little")
        self.engine = Engine(self.ctx, self.window.buffer_size, self.preset, seed, self.tun, self.track,
                             look_fps=self.look_fps)
        self.engine.view = VIEWS.index(args.debug_view)
        self.hud = Hud(self.ctx, self.engine.lib, self.engine.geom, self.window.buffer_size[1])
        self.hud_on = bool(getattr(args, "hud", False)) and not args.no_hud   # hidden by default; H toggles

        # depth source
        self.source = args.source or "device"
        self.synth = Synthetic()
        self.cam = None
        self.cam_on = True
        self.tape_in = None
        self.tape_key = None
        self.profile = None
        if self.source.startswith("tape:"):
            from .inputs.tape import TapeReader

            self.tape_in = TapeReader(self.source[5:])
            print(f"tape: {self.tape_in.records} records, {self.tape_in.t_first:.1f}-{self.tape_in.t_last:.1f} s")
        elif self.source == "device":
            self._start_camera()
        elif self.source != "synthetic":
            raise SystemExit(f"--source must be device, synthetic or tape:DIR (got {self.source!r})")
        self.tape_out = None
        if args.tape_out:
            self._start_tape()

        self.sim_t = 0.0
        self.dt = 1.0 / self.tun.sim_hz
        self._last_tick = None
        self.frame_times = collections.deque(maxlen=120)
        self.frame_times_all = []
        self.gpu_all = []
        self.last_change = ""
        self.export_proc = None
        self.pending_resize = None
        self.steps_last = 0
        self.dropped_steps = 0
        self.ticks_rendered = 0
        self.depth_updates = 0
        self.quit = False
        self.recorder = None
        self._rec_tick = None
        self._fix_restore = False
        self._windowed_buffer = tuple(self.window.buffer_size)
        self._last_errors = {}
        self.auto_keys = parse_auto_keys(getattr(args, "auto_keys", ""))
        self.keys_pressed = []
        self.window.key_event_func = self.on_key
        self.window.resize_func = self.on_resize
        self.window.close_func = self.on_close

    # -- depth sources ---------------------------------------------------------------------------
    def _start_camera(self):
        from .inputs.depthcam import DepthCam

        a = self.args
        prof = pick_profile(self.renderer)
        model = prof["model"] if a.model in (None, "auto") else a.model
        ep = a.ep or prof["ep"]
        size = a.model_size or (prof["size"] if model == prof["model"] else None)
        dev_id = a.device_id if a.device_id is not None else prof.get("device_id", 0)
        self.profile = {"model": model, "ep": ep, "size": size, "device_id": dev_id, "mask": a.mask,
                        "profile_note": prof.get("note", "")}
        try:
            self.cam = DepthCam(self.ctx, device=a.camera_device, model=model, ep=ep, size=size, device_id=dev_id,
                                mask=a.mask, exposure=a.cam_exposure)
            self.cam.start()
            self.cam.prime()
            self.cam_on = True
            self.last_change = f"camera: {self.cam.describe()}"
            print(self.last_change)
        except Exception as e:
            import traceback

            traceback.print_exc()
            if self.cam is not None:
                self.cam.stop()
            self.cam = None
            self.last_change = f"camera failed: {e} (synthetic depth instead)"
            print(self.last_change)

    def _start_tape(self):
        from .inputs.tape import TapeWriter

        out = self.args.tape_out
        if out == "auto":
            out = os.path.join(OUT_DIR, "tapes", time.strftime("%Y%m%d_%H%M%S"))
        meta = {"song": os.path.abspath(self.song) if self.song else None,
                "song_sha1": os.path.basename(self.features["cache"]).split("_")[0] if self.features else None,
                "look_fps": self.look_fps, "depth_size": [320, 240], "seed": self.engine.seed,
                "source": self.source, "model": (self.profile or {}).get("model"), "ep": (self.profile or {}).get("ep"),
                "device_id": (self.profile or {}).get("device_id"), "renderer": self.renderer}
        self.tape_out = TapeWriter(out, meta)
        print(f"depth tape -> {out}")

    def _depth_for_tick(self, tick):
        """Feed the engine the depth for this picture tick (and record it to the tape)."""
        eng = self.engine
        t = tick / float(self.look_fps)
        rec = None
        if self.tape_in is not None:
            r = self.tape_in.at_time(max(t, self.tape_in.t_first))
            if r is not None:
                depth, mask, hdr, key = r
                if key != self.tape_key:
                    eng.upload_depth(depth, mask)
                    eng.set_depth_info(hdr[HI["fig_depth"]], hdr[HI["mask_cov"]], 0.0, True, True)
                    self.tape_key = key
                    self.depth_updates += 1
                else:
                    eng.set_depth_info(on=True, new=False)
            return
        if self.cam is not None:
            if not self.cam_on:
                eng.set_depth_info(stale=1.0, on=eng.depth_on, new=False)
                return
            new, hdr = self.cam.update()
            if self.cam.have_frame:
                eng.set_depth_textures(self.cam.depth_tex, self.cam.mask_tex)
                if new:
                    eng.set_depth_info(hdr[HI["fig_depth"]], hdr[HI["mask_cov"]], 0.0, True, True)
                    self.depth_updates += 1
                    rec = (self.cam._depth, self.cam._mask, self.cam._hdr)
                else:
                    eng.set_depth_info(stale=1.0 if self.cam.stalled() else 0.0, on=True, new=False)
            else:   # camera still opening, or every frame DARK so far: an empty room of fog (no stand-in)
                eng.set_depth_info(on=False, new=False)
        else:
            depth, mask = self.synth.frame(t)
            h = self.synth.header(depth, mask)
            eng.upload_depth(depth, mask)
            eng.set_depth_info(h["fig_depth"], h["mask_cov"], 0.0, True, True)
            self.depth_updates += 1
            rec = (depth, mask, np.zeros(32))
        if rec is not None and self.tape_out is not None:
            self.tape_out.write(tick, t, rec[0], rec[1], rec[2])

    # -- events ------------------------------------------------------------------------------
    def on_close(self):
        self.quit = True

    def on_resize(self, w, h):
        self.pending_resize = True

    def on_key(self, key, action, modifiers):
        k = self.window.keys
        if action != k.ACTION_PRESS:
            return
        eng = self.engine
        ctrl = getattr(modifiers, "ctrl", False)
        if key == k.ESCAPE or key == k.Q:
            self.quit = True
        elif key == k.R:
            eng.set_seed(int.from_bytes(os.urandom(3), "little"))
            self.last_change = f"seed {eng.seed}"
        elif key == k.LEFT_BRACKET:
            eng.set_seed(eng.seed - 1)
            self.last_change = f"seed {eng.seed}"
        elif key == k.RIGHT_BRACKET:
            eng.set_seed(eng.seed + 1)
            self.last_change = f"seed {eng.seed}"
        elif key == k.SPACE:
            self.clock.set_paused(not self.clock.paused)
            self.last_change = "paused" if self.clock.paused else "playing"
        elif key == k.S and ctrl:
            path = presetmod.save_user_preset(seedmod.frozen_preset(eng.resolved, self.preset),
                                              {"preset": eng.resolved["preset"], "seed": eng.seed})
            self.last_change = f"saved preset {os.path.basename(path)}"
        elif key == k.S:
            self.screenshot()
        elif key == k.A:
            eng.view = (eng.view + 1) % len(VIEWS)
            self._last_tick = None
            self.last_change = f"view {VIEWS[eng.view]}"
        elif key == k.B:
            eng.halo_on = not eng.halo_on
            self._last_tick = None
            self.last_change = f"halation {'on' if eng.halo_on else 'off'}"
        elif key == k.H:
            self.hud_on = not self.hud_on
        elif key == k.F:
            going_fs = not self.window.fullscreen
            self.window.fullscreen = going_fs
            self.pending_resize = True
            self._fix_restore = not going_fs
        elif key == k.P:
            rates = [12, 24, 60]
            self.look_fps = rates[(rates.index(self.look_fps) + 1) % 3] if self.look_fps in rates else 12
            eng.look_fps = self.look_fps
            self._last_tick = None
            self.last_change = f"look {self.look_fps} fps"
        elif key == k.C:
            if self.cam is None and self.source == "device":
                self._start_camera()
            elif self.cam is not None:
                self.cam_on = not self.cam_on
                self.last_change = f"camera {'on' if self.cam_on else 'off (holding the last depth)'}"
            else:
                self.last_change = f"no camera (source {self.source})"
        elif key == k.V:
            eng.director.flip_now(self.sim_t)
            self.last_change = eng.director.last_event
        elif key == k.LEFT:
            self._seek(-5.0)
        elif key == k.RIGHT:
            self._seek(5.0)
        elif key == k.COMMA:
            self.clock.offset -= 0.010
            self.last_change = f"audio offset {self.clock.offset * 1000:+.0f} ms"
        elif key == k.PERIOD:
            self.clock.offset += 0.010
            self.last_change = f"audio offset {self.clock.offset * 1000:+.0f} ms"
        elif key == k.E:
            self.start_export()
        else:
            for i, num in enumerate((k.NUMBER_1, k.NUMBER_2, k.NUMBER_3, k.NUMBER_4, k.NUMBER_5,
                                     k.NUMBER_6, k.NUMBER_7, k.NUMBER_8, k.NUMBER_9)):
                if key == num:
                    self.load_preset_slot(i)
        self.keys_pressed.append(self.last_change)
        if self.args.verbose and self.last_change:
            print(f"[key] {key} -> {self.last_change}")

    def _press(self, name, ctrl=False):
        class _Mods:
            pass

        m = _Mods()
        m.ctrl, m.shift, m.alt = ctrl, False, False
        k = self.window.keys
        self.on_key(getattr(k, name), k.ACTION_PRESS, m)
        print(f"[auto-key] {'ctrl+' if ctrl else ''}{name} -> {self.last_change}")

    def _restore_windowed_size(self):
        try:
            import glfw

            handle = self.window._window
            bw, bh = glfw.get_framebuffer_size(handle)
            tw, th = self._windowed_buffer
            if (bw, bh) != (tw, th) and bw > 0 and bh > 0:
                ww, wh = glfw.get_window_size(handle)
                glfw.set_window_size(handle, int(round(ww * tw / bw)), int(round(wh * th / bh)))
                self.pending_resize = True
        except Exception as e:
            print("window restore:", e)

    def _seek(self, delta):
        t = max(0.0, self.clock.time() - self.clock.offset + delta)
        self.clock.seek(t)
        self.sim_t = self.clock.time()
        self.engine.director.t_prev = None
        self._last_tick = None
        self.last_change = f"seek {_fmt_t(t)}"

    def load_preset_slot(self, i):
        names = presetmod.list_presets()
        if i >= len(names):
            self.last_change = f"no preset in slot {i + 1}"
            return
        self.preset_name = names[i]
        self.preset = presetmod.load(names[i])
        self.engine.set_preset(self.preset)
        self._last_tick = None
        self.last_change = f"preset {names[i]} (slot {i + 1})"

    # -- outputs -----------------------------------------------------------------------------
    def screenshot(self):
        eng = self.engine
        os.makedirs(os.path.join(OUT_DIR, "screens"), exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(OUT_DIR, "screens", f"{stamp}_seed{eng.seed}_{VIEWS[eng.view]}")
        W, H = eng.fb_size
        tgt = Target(self.ctx, (W, H), 4, "f1", "nearest")
        eng.present_only(tgt.fbo)       # the held frame exactly as shown
        shots = [(base + ".png", tgt.fbo.read(components=3))]
        if self.hud_on:
            self.hud.draw(tgt.fbo, (W, H))
            shots.append((base + "_hud.png", tgt.fbo.read(components=3)))
        tgt.release()

        def write():
            for path, raw in shots:
                img = np.flipud(np.frombuffer(raw, np.uint8).reshape(H, W, 3))
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        import threading

        threading.Thread(target=write, daemon=True).start()
        presetmod.write_resolved(base + ".json", eng.resolved, self.run_extra())
        self.last_change = f"screenshot {os.path.basename(base)}.png"

    def run_extra(self):
        from audioviz.audio.analysis import EXTRACTOR_VERSION

        eng = self.engine
        d = eng.director
        extra = {
            "song": os.path.abspath(self.song) if self.song else None,
            "song_sha1": os.path.basename(self.features["cache"]).split("_")[0] if self.features else None,
            "feature_extractor_version": EXTRACTOR_VERSION, "resolution": list(eng.fb_size), "fps": "live",
            "look_fps": self.look_fps, "song_time": round(self.clock.time(), 3), "depth_source": self.source,
            "polarity_now": d.polarity, "section": d.section, "flips": d.flip_count,
        }
        if self.profile:
            extra.update({"model": self.profile["model"], "ep": self.profile["ep"],
                          "model_size": self.profile["size"], "device_id": self.profile["device_id"]})
        if self.tape_out is not None:
            extra["tape"] = self.tape_out.dir
        return extra

    def start_export(self):
        if self.export_proc is not None and self.export_proc.poll() is None:
            self.last_change = "export already running"
            return
        if not self.song:
            self.last_change = "export needs a song"
            return
        W, H = self.engine.fb_size
        size = "1920x1080" if W >= H else "1080x1920"
        os.makedirs(os.path.join(OUT_DIR, "exports"), exist_ok=True)
        out = os.path.join(OUT_DIR, "exports", f"{time.strftime('%Y%m%d_%H%M%S')}_seed{self.engine.seed}.mp4")
        cmd = [sys.executable, "-m", "depthviz", "export", self.song, "--seed", str(self.engine.seed),
               "--preset", self.preset_name, "--size", size, "--look-fps", str(self.look_fps), "--out", out]
        if self.tape_out is not None:
            self.tape_out.flush()
            cmd += ["--tape", self.tape_out.dir]
            src = "tape"
        else:
            cmd += ["--source", "synthetic"]
            src = "synthetic"
        self.export_proc = subprocess.Popen(cmd, cwd=ROOT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.last_change = f"exporting ({src}) {os.path.basename(out)}"

    # -- HUD ---------------------------------------------------------------------------------
    def hud_lines(self, fps):
        eng = self.engine
        s = eng.last_state or {}
        e = s.get("env", {})
        tm = eng.timers
        d = eng.director
        ci = self.clock.info()
        dur = getattr(self.clock, "duration", float("inf"))
        lines = [
            (f"{TITLE}  seed {eng.seed}  preset {self.preset_name}  {eng.fb_size[0]}x{eng.fb_size[1]}  "
             f"view {VIEWS[eng.view]}  source {self.source}", 2),
            (f"fps {fps:5.1f} (look {self.look_fps})  gpu {tm.total():5.2f} ms  steps/frame {self.steps_last}  "
             f"song {_fmt_t(self.clock.time())} / {_fmt_t(dur)}{'  [paused]' if self.clock.paused else ''}", 0),
            (f"offset {ci['offset_ms']:+.0f} ms  latency {ci['latency_ms']:.0f} ms  underflows {ci['underflows']}  "
             f"section {d.section}  polarity {d.polarity}{' (flipping)' if s.get('flip_kind') else ''}  "
             f"halation {'on' if eng.halo_on else 'off'}", 0),
            ("sub " + _bar(e.get("sub", 0)) + "  bass " + _bar(e.get("bass", 0)) + "  high " + _bar(e.get("high", 0))
             + "  onset " + _bar(e.get("onset", 0)) + "  loud " + _bar(e.get("loudness", 0)), 0),
            ("cent " + _bar(e.get("centroid", 0)) + "  flux " + _bar(e.get("flux", 0)) + "  harm "
             + _bar(e.get("harmonic", 0)) + f"  k {s.get('k', 0):.2f}  exp {s.get('exposure', 0):.2f}  "
             f"flash {s.get('flash', 0):.2f}  grain {s.get('grain', 0):.2f}", 0),
            ("  ".join(f"{n.split()[0]} {tm.mean(n):.2f}" for n in tm.order), 0),
            (f"last: {self.last_change}   event: {d.last_event}", 2),
        ]
        if self.cam is not None:
            lines.append((f"camera: {self.cam.status()}  {'on' if self.cam_on else 'OFF (holding)'}", 0))
        elif self.tape_in is not None:
            lines.append((f"tape in: {self.tape_in.dir}  {self.tape_in.records} records", 0))
        if self.tape_out is not None:
            lines.append((f"tape out: {os.path.basename(self.tape_out.dir)}  {self.tape_out.records} records  "
                          f"{self.tape_out.bytes / 1e6:.1f} MB", 0))
        if self.recorder is not None:
            lines.append((f"recording: {os.path.basename(self.recorder.out)}  {self.recorder.frames} frames", 1))
        for key, err in eng.lib.errors.items():
            for ln in err.splitlines()[:3]:
                lines.append((f"[{key}] {ln[:150]}", 1))
        return lines

    # -- main loop ---------------------------------------------------------------------------
    def run(self):
        eng = self.engine
        print("priming the pipeline...")
        # compile every program and bind the depth textures before the clock starts
        tgt = Target(self.ctx, (64, 36), 4, "f1", "nearest")
        if self.cam is not None:
            eng.set_depth_textures(self.cam.depth_tex, self.cam.mask_tex)
        for i in range(3):
            eng.sim_step(-0.1 + i * self.dt, self.dt)
            eng.render(tgt.fbo, tick=-1)
        eng.set_depth_textures(None, None)
        eng.view = VIEWS.index(self.args.debug_view)
        self.ctx.finish()
        tgt.release()
        vp = (0, 0) + tuple(self.window.buffer_size)
        self._depth_for_tick(0)
        eng.render(self.window.fbo, tick=0, viewport=vp)
        self.hud.set_lines(self.hud_lines(0.0))
        self.window.swap_buffers()
        self.ctx.finish()
        eng.director.t_prev = None
        self.clock.start()
        if self.args.record:
            from audioviz.export.live_recorder import LiveRecorder

            rec_fps = self.look_fps if self.look_fps < 60 else 30
            out = self.args.record
            if out == "auto":
                os.makedirs(os.path.join(OUT_DIR, "recordings"), exist_ok=True)
                out = os.path.join(OUT_DIR, "recordings", f"{time.strftime('%Y%m%d_%H%M%S')}_seed{eng.seed}.mp4")
            self.recorder = LiveRecorder(self.ctx, self.window.buffer_size, rec_fps, out, self.song)
            print(f"recording to {out} at {rec_fps} fps (HUD included)")
        self.sim_t = 0.0
        last = time.perf_counter()
        hud_counter = 0
        fps = 0.0
        t_wall0 = time.perf_counter()
        while not self.window.is_closing and not self.quit:
            now = time.perf_counter()
            while self.auto_keys and now - t_wall0 >= self.auto_keys[0][0]:
                _, name, ctrl = self.auto_keys.pop(0)
                self._press(name, ctrl)
            self.frame_times.append(now - last)
            self.frame_times_all.append(now - last)
            last = now
            if self._fix_restore:
                self._fix_restore = False
                self._restore_windowed_size()
            if self.pending_resize:
                self.pending_resize = None
                if eng.resize(self.window.buffer_size):
                    self.hud = Hud(self.ctx, eng.lib, eng.geom, self.window.buffer_size[1])
                    self._last_tick = None
            # fixed-step director chasing the song clock
            target = self.clock.time()
            steps = 0
            if not self.clock.paused:
                if target < self.sim_t - 0.25 or target > self.sim_t + 2.0:   # seek / loop / long stall
                    if self.args.verbose:
                        print(f"[resync] sim {self.sim_t:.3f} -> song {target:.3f}")
                    self.sim_t = target - self.dt
                    eng.director.t_prev = None
                while self.sim_t + self.dt <= target and steps < self.tun.max_steps_per_frame:
                    self.sim_t += self.dt
                    eng.sim_step(self.sim_t, self.dt)
                    steps += 1
                if target - self.sim_t > self.dt * self.tun.max_steps_per_frame:
                    n_drop = int((target - self.sim_t) / self.dt)
                    self.dropped_steps += n_drop
                    if self.args.verbose:
                        print(f"[drop] {n_drop} steps at song {target:.3f} (frame dt {self.frame_times[-1] * 1000:.1f} ms)")
                    self.sim_t = target - self.dt
            self.steps_last = steps
            vp = (0, 0) + tuple(self.window.buffer_size)
            rate = self.look_fps if self.look_fps < 60 else 60
            tick = int(np.floor(self.sim_t * rate))
            if tick != self._last_tick:
                self._depth_for_tick(tick)
                eng.render(self.window.fbo, tick=tick, viewport=vp)
                self._last_tick = tick
                self.ticks_rendered += 1
            else:
                eng.present_only(self.window.fbo, viewport=vp)
            if self.hud_on:
                hud_counter += 1
                if hud_counter % 6 == 1:
                    ft = np.mean(self.frame_times) if self.frame_times else 1 / 60
                    fps = 1.0 / max(ft, 1e-6)
                    self.hud.set_lines(self.hud_lines(fps))
                self.hud.draw(self.window.fbo, self.window.buffer_size)
            rec_tick = self._last_tick if self.look_fps < 60 else int(np.floor(self.sim_t * 30))
            if self.recorder is not None and rec_tick != self._rec_tick:
                if tuple(self.window.buffer_size) != self.recorder.size:
                    self.recorder.stop()
                    print(f"recording stopped (window resized): {self.recorder.out}")
                else:
                    self.recorder.capture(self.window.fbo, rec_tick)
                self._rec_tick = rec_tick
            self.window.swap_buffers()
            eng.timers.end_frame()
            total = eng.timers.total()
            if total > 0:
                self.gpu_all.append(total)
            if eng.poll_shaders():
                print("[hot-reload] shaders recompiled")
                self._last_tick = None
            errs = dict(eng.lib.errors)
            if errs != self._last_errors:
                for key, err in errs.items():
                    print(f"[hot-reload] {key} failed, keeping the old program: {err.splitlines()[0][:160]}")
                if not errs and self._last_errors:
                    print("[hot-reload] all shader errors cleared")
                self._last_errors = errs
            if getattr(self.clock, "ended", False) and self.args.no_loop:
                break
        self.shutdown()

    def write_stats(self):
        ft = np.array(self.frame_times_all[1:]) * 1000.0 if len(self.frame_times_all) > 1 else np.zeros(1)
        gpu = np.array(self.gpu_all) if self.gpu_all else np.zeros(1)
        eng = self.engine
        d = eng.director
        stats = {
            "song": self.song, "seed": eng.seed, "fb_size": list(eng.fb_size), "source": self.source,
            "look_fps": self.look_fps, "frames": len(ft), "ticks_rendered": self.ticks_rendered,
            "depth_updates": self.depth_updates, "song_time_end": round(self.clock.time(), 2),
            "frame_ms_p50": round(float(np.percentile(ft, 50)), 2), "frame_ms_p95": round(float(np.percentile(ft, 95)), 2),
            "frame_ms_max": round(float(ft.max()), 2), "gpu_ms_mean": round(float(gpu.mean()), 2),
            "gpu_ms_p95": round(float(np.percentile(gpu, 95)), 2), "dropped_steps": self.dropped_steps,
            "timers": eng.timers.summary(), "clock": self.clock.info(), "kicks": d.kick_count, "hats": d.hat_count,
            "sections_applied": d.section, "flips": d.flip_count, "keys": self.keys_pressed,
            "shader_errors": dict(eng.lib.errors), "profile": self.profile,
        }
        if self.cam is not None:
            stats["camera"] = {k: round(self.cam.h(k), 3) for k in ("frames", "fps", "cam_fps", "depth_ms", "mask_ms",
                                                                     "proc_ms", "brightness", "gain", "mask_cov",
                                                                     "latency_ms", "first_frame_s", "session_s", "ep_id")}
            stats["camera"]["ep"] = EPS[int(self.cam.h("ep_id"))] if self.cam.h("ep_id") >= 0 else None
            stats["camera"]["error"] = self.cam.error
            stats["camera"]["dark_skipped"] = getattr(self.cam, "dark_skipped", 0)
        if self.tape_out is not None:
            stats["tape_out"] = {"dir": self.tape_out.dir, "records": self.tape_out.records}
        if self.recorder is not None:
            stats["recording"] = {"out": self.recorder.out, "frames": self.recorder.frames}
        path = self.args.stats_out or os.path.join(OUT_DIR, "live_stats.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, default=str)
        return stats

    def shutdown(self):
        if self.recorder is not None:
            self.recorder.stop()
            print(f"recording saved: {self.recorder.out} ({self.recorder.frames} frames)")
        if self.tape_out is not None:
            self.tape_out.close()
            print(f"tape saved: {self.tape_out.dir} ({self.tape_out.records} records)")
        if self.cam is not None:
            self.cam.poll_errors()
        try:
            self.write_stats()
        except Exception as e:
            print("stats failed:", e)
        try:
            ci = self.clock.info()
            print("timers (ms):", self.engine.timers.summary(), "underflows:", ci["underflows"],
                  "after start:", ci.get("underflows_after_start"), "dropped steps:", self.dropped_steps)
        except Exception:
            pass
        self.clock.stop()
        if self.cam is not None:
            self.cam.stop()
        self.window.destroy()


def parse_auto_keys(spec):
    out = []
    for item in [x for x in (spec or "").split(",") if x.strip()]:
        t, key = item.split(":", 1)
        ctrl = key.lower().startswith("ctrl+")
        name = key[5:] if ctrl else key
        name = {"[": "LEFT_BRACKET", "]": "RIGHT_BRACKET", ",": "COMMA", ".": "PERIOD", " ": "SPACE"}.get(name, name)
        if name.isdigit():
            name = "NUMBER_" + name
        out.append((float(t), name.upper(), ctrl))
    return sorted(out)
