"""Live mode: window, song clock, fixed-step simulation, keys, HUD, tier controller, hot reload.

Keys: R reseed, [ ] seed -/+1, Space pause, S screenshot + sidecar JSON, A cycle view
(ASCII / raw density / flow arrows), B bloom, G grid overlay, 1-9 preset slots, Left/Right scrub
5 s, , . audio offset -/+10 ms, E export this song/seed in the background, Ctrl+S save preset,
H HUD, F fullscreen, T next tier down, C camera stir on/off, V camera background on/off,
P frame-rate look 12/24/60, Esc quit.
"""
import collections
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from . import OUT_DIR, ROOT_DIR
from . import preset as presetmod
from . import seed as seedmod
from .audio.features import FeatureTrack
from .config import TIERS, Tunables, detect_tier
from .engine import Engine
from .gl.context import create_window, gl_info, renderer_name
from .gl.pingpong import Target
from .render.present import VIEWS, Hud


def _bar(v, n=10):
    k = int(round(np.clip(v, 0, 1) * n))
    return "#" * k + "-" * (n - k)


def _fmt_t(t):
    if not np.isfinite(t):
        return "--:--.-"
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:04.1f}"


def window_size_for(aspect_wh, frac=0.82):
    """Logical window size that fits the primary monitor at the requested aspect."""
    import glfw

    try:
        glfw.init()
        mon = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(mon)
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
        self.preset = presetmod.load(args.preset)
        apply_overrides(self.preset, args.set)
        aspect = args.aspect or self.preset["meta"].get("aspect", "16:9")
        aw, ah = (int(v) for v in aspect.split(":"))
        size = tuple(int(v) for v in args.size.split("x")) if args.size else window_size_for((aw, ah))
        self.window = create_window(title="ASCII Ink Visualizer", size=size, fullscreen=args.fullscreen, vsync=not args.no_vsync)
        self.ctx = self.window.ctx
        info = gl_info(self.ctx)
        print(f"GL: {info['GL_RENDERER']} | {info['GL_VERSION']} | buffer {self.window.buffer_size}")

        self.song = args.song
        self.track = None
        if self.song:
            from .audio.analysis import load_audio, load_or_analyze
            from .audio.player import Player

            print("analysing song (cached after the first run)...")
            self.features = load_or_analyze(self.song)
            self.track = FeatureTrack(self.features)
            pcm, sr = load_audio(self.song)
            self.clock = Player(pcm, sr, loop=not args.no_loop, offset_s=args.offset_ms / 1000.0, mute=args.mute)
        else:
            from .audio.player import WallClock

            self.features = None
            self.clock = WallClock()

        tier = args.tier or detect_tier(renderer_name(self.ctx))
        seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(3), "little")
        self.engine = Engine(self.ctx, self.window.buffer_size, self.preset, seed, tier, self.tun, self.track)
        self.engine.view = VIEWS.index(args.debug_view)
        self.hud = Hud(self.ctx, self.engine.lib, self.engine.geom, self.window.buffer_size[1])
        self.hud_on = not args.no_hud
        # frame hold: the sim runs at 60 Hz, the picture only changes look_fps times a second
        self.look_fps = int(args.look_fps or self.preset["grain"].get("frame_rate", 60))
        self._last_tick = None
        self.click_test = args.click_test

        self.camera = None
        self.camera_on = False
        if args.camera:
            self._start_camera()

        self.sim_t = 0.0
        self.dt = 1.0 / self.tun.sim_hz
        self.frame_times = collections.deque(maxlen=120)
        self.frame_times_all = []
        self.gpu_all = []
        self.gpu_hist = collections.deque(maxlen=self.tun.tier_drop_frames)
        self.tier_cooldown_until = 0.0
        self.tier_warmup_until = 0.0
        self.last_change = ""
        self.export_proc = None
        self.pending_resize = None
        self.steps_last = 0
        self.dropped_steps = 0
        self.quit = False
        self.recorder = None
        self._rec_tick = None
        self._fix_restore = False
        self._windowed_buffer = tuple(self.window.buffer_size)
        self.extra_steps = 0
        self._last_errors = {}
        if getattr(args, "tier_drop_ms", None):
            self.tun.tier_drop_ms = args.tier_drop_ms

        self.auto_keys = parse_auto_keys(getattr(args, "auto_keys", ""))
        self.window.key_event_func = self.on_key
        self.window.resize_func = self.on_resize
        self.window.close_func = self.on_close

    # -- camera (M7) -------------------------------------------------------------------------
    def _start_camera(self):
        from .inputs.camera import CameraStir

        try:
            self.camera = CameraStir(self.ctx, device=self.args.camera_device, synthetic=self.args.camera == "synthetic",
                                     gain=self.preset["sim"].get("ext_flow_gain", 0.6))
            self.camera.start()
            ftex, btex, sc = self.camera.prime(self.engine.aspect if hasattr(self, "engine") else None)
            if hasattr(self, "engine"):
                self._prime_camera(ftex, btex, sc)
            self.camera_on = True
            self.last_change = f"camera: {self.camera.describe()}"
        except Exception as e:
            import traceback

            traceback.print_exc()
            if self.camera is not None:
                self.camera.stop()
            self.camera = None
            self.camera_on = False
            self.last_change = f"camera failed: {e}"
            print(self.last_change)

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
            self._after_seed()
        elif key == k.LEFT_BRACKET:
            eng.set_seed(eng.seed - 1)
            self._after_seed()
        elif key == k.RIGHT_BRACKET:
            eng.set_seed(eng.seed + 1)
            self._after_seed()
        elif key == k.SPACE:
            self.clock.set_paused(not self.clock.paused)
            self.last_change = "paused" if self.clock.paused else "playing"
        elif key == k.S and ctrl:
            path = presetmod.save_user_preset(seedmod.frozen_preset(eng.resolved),
                                              {"preset": eng.resolved["preset"], "seed": eng.seed})
            self.last_change = f"saved preset {os.path.basename(path)}"
        elif key == k.S:
            self.screenshot()
        elif key == k.A:
            eng.view = (eng.view + 1) % len(VIEWS)
            self.last_change = f"view {VIEWS[eng.view]}"
        elif key == k.B:
            eng.bloom_on = not eng.bloom_on
            self.last_change = f"bloom {'on' if eng.bloom_on else 'off'}"
        elif key == k.G:
            eng.grid_overlay = not eng.grid_overlay
            self.last_change = f"grid overlay {'on' if eng.grid_overlay else 'off'}"
        elif key == k.H:
            self.hud_on = not self.hud_on
        elif key == k.F:
            going_fs = not self.window.fullscreen
            self.window.fullscreen = going_fs
            self.pending_resize = True
            self._fix_restore = not going_fs
        elif key == k.T:
            if eng.lower_tier():
                self.last_change = f"tier -> {eng.tier.name}"
        elif key == k.C:
            if self.camera is None:
                self._start_camera()
            else:
                self.camera_on = not self.camera_on
                self.last_change = f"camera stir {'on' if self.camera_on else 'off'}"
        elif key == k.V:
            eng.cam_bg_on = not eng.cam_bg_on
            note = "" if self.camera else " (no camera running: start it with C)"
            self.last_change = "camera background " + ("on" if eng.cam_bg_on else "off") + note
        elif key == k.P:
            rates = [12, 24, 60]
            self.look_fps = rates[(rates.index(self.look_fps) + 1) % 3] if self.look_fps in rates else 24
            self._last_tick = None
            self.last_change = f"frame rate look {self.look_fps} fps"
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
        """moderngl-window leaves fullscreen with the logical size as raw pixels on Windows at
        125% scaling (2098x1180 came back as 1679x944). Rescale until the framebuffer matches the
        original windowed one; platform-agnostic, so macOS points are left alone."""
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

    def _after_seed(self):
        """Refill the frame after a reseed without blocking: extra sim steps spread over the next
        frames. (A blocking prewarm plus its synchronous meter read held the GIL for ~190 ms and
        starved the audio callback.)"""
        self.last_change = f"seed {self.engine.seed}"
        self.engine.warm_atlases()
        self.extra_steps = int(1.0 * self.tun.sim_hz)

    def _seek(self, delta):
        t = max(0.0, self.clock.time() - self.clock.offset + delta)
        self.clock.seek(t)
        self.sim_t = self.clock.time()
        self.engine.director.t_prev = None
        self.last_change = f"seek {_fmt_t(t)}"

    def load_preset_slot(self, i):
        names = presetmod.list_presets()
        if i >= len(names):
            self.last_change = f"no preset in slot {i + 1}"
            return
        self.preset_name = names[i]
        self.preset = presetmod.load(names[i])
        self.engine.set_preset(self.preset)
        self.engine.warm_atlases()
        self.extra_steps = int(1.0 * self.tun.sim_hz)
        self.last_change = f"preset {names[i]} (slot {i + 1})"

    # -- outputs -----------------------------------------------------------------------------
    def screenshot(self):
        eng = self.engine
        os.makedirs(os.path.join(OUT_DIR, "screens"), exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(OUT_DIR, "screens", f"{stamp}_seed{eng.seed}_{VIEWS[eng.view]}")
        W, H = eng.fb_size
        tgt = Target(self.ctx, (W, H), 4, "f1", "nearest")
        eng.render(tgt.fbo)
        shots = [(base + ".png", tgt.fbo.read(components=3))]
        if self.hud_on:
            self.hud.draw(tgt.fbo, (W, H))
            shots.append((base + "_hud.png", tgt.fbo.read(components=3)))
        tgt.release()

        def write():   # PNG encoding takes ~150 ms at 2K: keep it off the render thread
            for path, raw in shots:
                img = np.flipud(np.frombuffer(raw, np.uint8).reshape(H, W, 3))
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        import threading

        threading.Thread(target=write, daemon=True).start()
        presetmod.write_resolved(base + ".json", eng.resolved, self.run_extra())
        self.last_change = f"screenshot {os.path.basename(base)}.png"

    def run_extra(self):
        from .audio.analysis import EXTRACTOR_VERSION

        return {
            "song": os.path.abspath(self.song) if self.song else None,
            "song_sha1": os.path.basename(self.features["cache"]).split("_")[0] if self.features else None,
            "feature_extractor_version": EXTRACTOR_VERSION,
            "resolution": list(self.engine.fb_size), "fps": "live", "tier": self.engine.tier.name,
            "song_time": round(self.clock.time(), 3), "grid_pair_now": self.engine.director.grid_pair,
            "palette_now": self.engine.director.palette_variant,
        }

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
        cmd = [sys.executable, "-m", "audioviz", "export", self.song, "--seed", str(self.engine.seed),
               "--preset", self.preset_name, "--size", size, "--fps", str(min(60, self.look_fps)),
               "--look-fps", str(self.look_fps), "--out", out]
        self.export_proc = subprocess.Popen(cmd, cwd=ROOT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.last_change = f"exporting {os.path.basename(out)}"

    # -- HUD ---------------------------------------------------------------------------------
    def hud_lines(self, fps):
        eng = self.engine
        s = eng.last_state or {}
        e = s.get("env", {})
        tm = eng.timers
        gpu = tm.total()
        d = eng.director
        ci = self.clock.info()
        dur = getattr(self.clock, "duration", float("inf"))
        lines = [
            (f"ASCII Ink Visualizer  seed {eng.seed}  preset {self.preset_name}  tier {eng.tier.name} ({eng.particles.N / 1e6:.2f}M)  "
             f"{eng.fb_size[0]}x{eng.fb_size[1]}  view {VIEWS[eng.view]}", 2),
            (f"fps {fps:5.1f} (look {self.look_fps})  gpu {gpu:5.2f} ms  steps/frame {self.steps_last}  song {_fmt_t(self.clock.time())} / "
             f"{_fmt_t(dur)}{'  [paused]' if self.clock.paused else ''}", 0),
            (f"offset {ci['offset_ms']:+.0f} ms  latency {ci['latency_ms']:.0f} ms  underflows {ci['underflows']}  "
             f"emitters {s.get('num_emitters', 0)}  section {d.section}  grid {d.grid_pair}  palette {d.palette_variant}  "
             f"glyphs {eng.resolved['glyph_set']}  auto-exp {d.auto:.2f}", 0),
            ("sub " + _bar(e.get("sub", 0)) + "  bass " + _bar(e.get("bass", 0)) + "  mid " + _bar(e.get("mid", 0))
             + "  high " + _bar(e.get("high", 0)) + "  onset " + _bar(e.get("onset", 0)), 0),
            ("cent " + _bar(e.get("centroid", 0)) + "  flux " + _bar(e.get("flux", 0)) + "  harm " + _bar(e.get("harmonic", 0))
             + "  perc " + _bar(e.get("percussive", 0)) + "  loud " + _bar(e.get("loudness", 0)), 0),
            ("  ".join(f"{n.split()[0]} {tm.mean(n):.2f}" for n in tm.order), 0),
            (f"last: {self.last_change}   event: {d.last_event}", 2),
        ]
        if self.camera is not None:
            lines.append((f"camera: {self.camera.status()}  stir {'on' if self.camera_on else 'off'}  "
                          f"background {'on' if eng.cam_bg_on else 'off'}", 0))
        for key, err in eng.lib.errors.items():
            for ln in err.splitlines()[:3]:
                lines.append((f"[{key}] {ln[:150]}", 1))
        return lines

    # -- main loop ---------------------------------------------------------------------------
    def run(self):
        eng = self.engine
        pre = float(eng.resolved["params"]["sim"]["prewarm_s"])
        if self.camera is not None:   # bind the camera textures before prewarm (driver warm-up)
            ftex, btex, sc = self.camera.prime(eng.aspect)
            self._prime_camera(ftex, btex, sc)
        print("prewarming...")
        eng.prewarm(pre, t0=0.0)
        # build glyph atlases, HUD text and the GPU timers before the audio clock starts
        eng.render(self.window.fbo, viewport=(0, 0) + tuple(self.window.buffer_size))
        self.hud.set_lines(self.hud_lines(0.0))
        self.window.swap_buffers()
        self.ctx.finish()
        self.clock.start()
        if self.args.record:
            from .export.live_recorder import LiveRecorder

            rec_fps = self.look_fps if self.look_fps < 60 else 30
            out = self.args.record
            if out == "auto":
                os.makedirs(os.path.join(OUT_DIR, "recordings"), exist_ok=True)
                out = os.path.join(OUT_DIR, "recordings", f"{time.strftime('%Y%m%d_%H%M%S')}_seed{eng.seed}.mp4")
            self.recorder = LiveRecorder(self.ctx, self.window.buffer_size, rec_fps, out, self.song)
            print(f"recording to {out} at {rec_fps} fps (HUD included)")
        self.tier_warmup_until = time.perf_counter() + 5.0
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
                    eng.warm_atlases()

            # fixed-step accumulator chasing the song clock
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
                    self._camera_update()
                    eng.sim_step(self.sim_t, self.dt)
                    steps += 1
                if target - self.sim_t > self.dt * self.tun.max_steps_per_frame:
                    n_drop = int((target - self.sim_t) / self.dt)
                    self.dropped_steps += n_drop
                    if self.args.verbose:
                        print(f"[drop] {n_drop} steps at song {target:.3f} (frame dt {self.frame_times[-1] * 1000:.1f} ms)")
                    self.sim_t = target - self.dt
            if self.extra_steps > 0:   # refill after a reseed, a few steps per frame
                k = min(4, self.extra_steps)
                for _ in range(k):
                    eng.sim_step(self.sim_t, self.dt)
                self.extra_steps -= k
            self.steps_last = steps

            vp = (0, 0) + tuple(self.window.buffer_size)
            eng.present.pass_.set("u_flash", float(eng.last_state["pulse"]) if (self.click_test and eng.last_state) else 0.0)
            tick = int(np.floor(self.sim_t * self.look_fps)) if self.look_fps < 60 else None
            if tick is None and self.recorder is not None:
                self._last_tick = int(np.floor(self.sim_t * 30))
            if tick is None or tick != self._last_tick:
                if self.camera is not None:
                    tex, sc, lo, hi = self.camera.background(eng.aspect)
                    eng.set_camera_background(tex, sc, lo, hi)
                eng.render(self.window.fbo, viewport=vp)
                self._last_tick = tick
            else:
                eng.present_only(self.window.fbo, viewport=vp)
            if self.hud_on:
                hud_counter += 1
                if hud_counter % 6 == 1:
                    ft = np.mean(self.frame_times) if self.frame_times else 1 / 60
                    fps = 1.0 / max(ft, 1e-6)
                    self.hud.set_lines(self.hud_lines(fps))
                self.hud.draw(self.window.fbo, self.window.buffer_size)
            if self.recorder is not None and self._last_tick != self._rec_tick:
                if tuple(self.window.buffer_size) != self.recorder.size:   # resized: finish the file
                    self.recorder.stop()
                    print(f"recording stopped (window resized): {self.recorder.out}")
                else:
                    self.recorder.capture(self.window.fbo, self._last_tick)
                self._rec_tick = self._last_tick
            self.window.swap_buffers()
            eng.timers.end_frame()
            if eng.poll_shaders():
                print("[hot-reload] shaders recompiled")
            errs = dict(eng.lib.errors)
            if errs != self._last_errors:
                for key, err in errs.items():
                    print(f"[hot-reload] {key} failed, keeping the old program: {err.splitlines()[0][:160]}")
                if not errs and self._last_errors:
                    print("[hot-reload] all shader errors cleared")
                self._last_errors = errs
            self._tier_control()
            if getattr(self.clock, "ended", False) and self.args.no_loop:
                break
        self.shutdown()

    def _prime_camera(self, ftex, btex, sc):
        eng = self.engine
        eng.set_external_flow(ftex, sc, 0.0)
        eng.set_camera_background(btex, sc, 0.0, 1.0)
        tgt = Target(self.ctx, (64, 36), 4, "f1", "nearest")
        eng.sim_step(-1.0, 1.0 / self.tun.sim_hz)
        eng.render(tgt.fbo)
        self.ctx.finish()
        tgt.release()

    def _camera_update(self):
        if self.camera is None:
            return
        if self.camera_on:
            tex, scale, gain = self.camera.update(self.engine.aspect)
            self.engine.set_external_flow(tex, scale, gain)
        else:
            self.engine.set_external_flow(None, (1.0, 1.0), 0.0)

    def _tier_control(self):
        eng = self.engine
        total = eng.timers.total()
        if total <= 0:
            return
        self.gpu_hist.append(total)
        self.gpu_all.append(total)
        now = time.perf_counter()
        if now < self.tier_warmup_until:   # startup spikes (camera process start, first uploads) are not load
            self.gpu_hist.clear()
            return
        if now < self.tier_cooldown_until or len(self.gpu_hist) < self.gpu_hist.maxlen:
            return
        if np.percentile(self.gpu_hist, 95) > self.tun.tier_drop_ms:
            if eng.lower_tier():
                self.last_change = f"auto tier drop -> {eng.tier.name} (p95 gpu > {self.tun.tier_drop_ms} ms)"
                print(self.last_change)
            self.gpu_hist.clear()
            self.tier_cooldown_until = now + 5.0

    def write_stats(self):
        ft = np.array(self.frame_times_all[1:]) * 1000.0 if len(self.frame_times_all) > 1 else np.zeros(1)
        gpu = np.array(self.gpu_all) if self.gpu_all else np.zeros(1)
        stats = {
            "song": self.song, "seed": self.engine.seed, "fb_size": list(self.engine.fb_size),
            "tier_end": self.engine.tier.name, "frames": len(ft), "song_time_end": round(self.clock.time(), 2),
            "frame_ms_p50": round(float(np.percentile(ft, 50)), 2), "frame_ms_p95": round(float(np.percentile(ft, 95)), 2),
            "frame_ms_max": round(float(ft.max()), 2), "gpu_ms_mean": round(float(gpu.mean()), 2),
            "gpu_ms_p95": round(float(np.percentile(gpu, 95)), 2), "dropped_steps": self.dropped_steps,
            "timers": self.engine.timers.summary(), "clock": self.clock.info(),
            "kicks": self.engine.director.kick_count, "hats": self.engine.director.hat_count,
            "sections_applied": self.engine.director.section,
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "live_stats.json"), "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, default=str)
        return stats

    def shutdown(self):
        if self.recorder is not None:
            self.recorder.stop()
            print(f"recording saved: {self.recorder.out} ({self.recorder.frames} frames)")
        try:
            self.write_stats()
        except Exception as e:
            print("stats failed:", e)
        try:
            ci = self.clock.info()
            print("timers (ms):", self.engine.timers.summary(), "underflows:", ci["underflows"],
                  "after start:", ci.get("underflows_after_start"), "at callbacks:", ci.get("underflow_callbacks"),
                  "dropped steps:", self.dropped_steps)
        except Exception:
            pass
        self.clock.stop()
        if self.camera is not None:
            self.camera.stop()
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


def apply_overrides(preset, sets):
    """--set sim.splat=0.004 (value parsed as JSON when possible)."""
    for item in sets or []:
        if "=" not in item:
            raise SystemExit(f"--set needs key=value, got {item!r}")
        key, val = item.split("=", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        d = preset
        parts = key.split(".")
        for p in parts[:-1]:
            d = d[p]
        if parts[-1] not in d:
            raise SystemExit(f"--set: unknown key {key}")
        d[parts[-1]] = val
    return preset
