"""Camera stir (M7): webcam -> DIS optical flow -> `extVel` texture summed into the flow field (P0).

The camera image is never shown. Capture goes through the ffmpeg binary bundled with imageio-ffmpeg
(MJPEG over DirectShow on Windows, AVFoundation on macOS), because OpenCV's DirectShow backend
cannot negotiate MJPEG on the Logitech C270 and requesting 1280x720 without MJPEG wedges that
camera until it is replugged. This module only ever asks for 640x480 MJPEG.

Threads: a capture thread reads frames from the ffmpeg pipe and computes flow (DIS ULTRAFAST at
320x240, ~1-3 ms); the render thread only uploads the newest flow field. A watchdog zeroes the
stir if no frame arrives for 2 s; dark frames (lights off) give zero flow instead of sensor noise.

`synthetic=True` replaces the webcam with a generated moving bright blob on a dark background,
pushed through the same flow path, so the stir can be tested with the lights off.
"""
import math
import re
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

CAP_SIZE = (640, 480)
CAP_FPS = 30
FLOW_SIZE = (240, 180)   # 4:3 like the capture; DIS ULTRAFAST here is ~1 ms
BG_SIZE = (480, 360)     # what ffmpeg delivers: drawn into the smoke; downsized 2x for the flow
MIN_LEVEL_RANGE = 0.06   # auto-levels never stretch a range narrower than this (0..1 units)
BG_SMOOTH = 0.5          # temporal smoothing of the drawn image (weight of the newest frame)
BG_SHARPEN = 0.8         # unsharp mask on the drawn image (clarity); 0 = off
DARK_MEAN = 8.0      # 0..255; below this (and low contrast) the frame is treated as "lights off"
DARK_STD = 3.0
STALL_S = 2.0
NOISE_PX = 0.25      # flow below this (px/frame at FLOW_SIZE, after smoothing) is treated as noise


def ffmpeg_exe():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def list_video_devices():
    """Names of video capture devices (Windows dshow) or indices (macOS avfoundation)."""
    exe = ffmpeg_exe()
    if sys.platform.startswith("win"):
        cmd = [exe, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
        out = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=15).stderr
        names = []
        for line in out.splitlines():
            m = re.search(r'"([^"]+)"\s*\(video\)', line)
            if m:
                names.append(m.group(1))
        if not names:  # older ffmpeg prints a "DirectShow video devices" section instead
            sect = False
            for line in out.splitlines():
                if "video devices" in line:
                    sect = True
                elif "audio devices" in line:
                    sect = False
                elif sect:
                    m = re.search(r'"([^"]+)"', line)
                    if m and "@device" not in m.group(1):
                        names.append(m.group(1))
        return names
    if sys.platform == "darwin":
        cmd = [exe, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
        out = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=15).stderr
        names, sect = [], False
        for line in out.splitlines():
            if "video devices" in line:
                sect = True
            elif "audio devices" in line:
                sect = False
            elif sect:
                m = re.search(r"\[(\d+)\]\s+(.+)$", line)
                if m and "Capture screen" not in m.group(2):
                    names.append(m.group(1))
        return names
    return ["/dev/video0"]


def capture_cmd(device, w=CAP_SIZE[0], h=CAP_SIZE[1], fps=CAP_FPS):
    exe = ffmpeg_exe()
    common = [exe, "-hide_banner", "-loglevel", "error", "-fflags", "nobuffer", "-flags", "low_delay"]
    if sys.platform.startswith("win"):
        inp = ["-f", "dshow", "-rtbufsize", "64M", "-vcodec", "mjpeg", "-video_size", f"{w}x{h}",
               "-framerate", str(fps), "-i", f"video={device}"]
    elif sys.platform == "darwin":
        inp = ["-f", "avfoundation", "-framerate", str(fps), "-video_size", f"{w}x{h}", "-i", f"{device}:none"]
    else:
        inp = ["-f", "v4l2", "-input_format", "mjpeg", "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", device]
    vf = f"hflip,scale={BG_SIZE[0]}:{BG_SIZE[1]}:flags=area"
    return common + inp + ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"]


class _FfmpegSource:
    def __init__(self, device):
        self.device = device
        self.proc = subprocess.Popen(capture_cmd(device), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        self.nbytes = BG_SIZE[0] * BG_SIZE[1]   # gray, mirrored, BG_SIZE

    def read(self):
        buf = bytearray()
        while len(buf) < self.nbytes:
            chunk = self.proc.stdout.read(self.nbytes - len(buf))
            if not chunk:
                return None
            buf += chunk
        return np.frombuffer(bytes(buf), np.uint8).reshape(BG_SIZE[1], BG_SIZE[0])

    def error_text(self):
        try:
            return self.proc.stderr.read().decode(errors="replace").strip()[:500]
        except Exception:
            return ""

    def close(self):
        """Ask ffmpeg to quit ('q' on stdin) so the device is released cleanly; hard kills are what
        wedged the C270 during the research spikes."""
        p = self.proc
        if p.poll() is None:
            # keep draining stdout so ffmpeg is not blocked on a full pipe and can read the 'q'
            def drain():
                try:
                    while p.stdout.read(65536):
                        pass
                except Exception:
                    pass

            threading.Thread(target=drain, daemon=True).start()
            try:
                p.stdin.write(b"q")
                p.stdin.flush()
            except Exception:
                pass
            try:
                p.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    p.kill()


class _SyntheticSource:
    """A bright blob (a 'hand') sweeping across a dark, slightly noisy frame at 30 fps."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.n = 0
        rng = np.random.default_rng(0)
        R = 128
        yy, xx = np.mgrid[-R:R, -R:R].astype(np.float32)
        blob = 200.0 * np.exp(-(xx ** 2 + yy ** 2) / (2 * 45.0 ** 2))
        tex = 30.0 * np.sin(xx * 0.15) * np.sin(yy * 0.15) * (blob > 20)
        self.sprite = (blob + tex).astype(np.float32)
        self.R = R
        self.noise = [rng.normal(25.0, 2.0, size=(CAP_SIZE[1], CAP_SIZE[0])).astype(np.float32) for _ in range(8)]

    def read(self):
        target = self.t0 + self.n / CAP_FPS
        dt = target - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        elif dt < -2.0 / CAP_FPS:      # fell behind (thread starved): drop frames like a real camera
            self.t0 = time.perf_counter() - self.n / CAP_FPS
        t = self.n / CAP_FPS
        self.n += 1
        cx = int(CAP_SIZE[0] * (0.5 + 0.35 * math.sin(2 * math.pi * 0.25 * t)))
        cy = int(CAP_SIZE[1] * (0.5 + 0.25 * math.sin(2 * math.pi * 0.17 * t + 1.0)))
        img = self.noise[self.n % len(self.noise)].copy()
        R = self.R
        x0, y0 = cx - R, cy - R
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(CAP_SIZE[0], x0 + 2 * R), min(CAP_SIZE[1], y0 + 2 * R)
        img[dy0:dy1, dx0:dx1] += self.sprite[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
        g = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.flip(cv2.resize(g, BG_SIZE, interpolation=cv2.INTER_AREA), 1)

    def error_text(self):
        return ""

    def close(self):
        pass


# shared-memory layout written by the worker process: a float64 header, then the flow field, then
# the latest small gray frame (debug only)
_H = ["seq", "fps", "brightness", "dark", "flow_ms", "motion", "active", "frames", "first_frame_s",
      "last_frame_wall", "running", "p05", "p95"]
_HDR_BYTES = 16 * 8
_FLOW_BYTES = FLOW_SIZE[0] * FLOW_SIZE[1] * 2 * 4
_FRAME_BYTES = BG_SIZE[0] * BG_SIZE[1]


def _views(buf):
    hdr = np.ndarray((16,), np.float64, buffer=buf, offset=0)
    flow = np.ndarray((FLOW_SIZE[1], FLOW_SIZE[0], 2), np.float32, buffer=buf, offset=_HDR_BYTES)
    frame = np.ndarray((BG_SIZE[1], BG_SIZE[0]), np.uint8, buffer=buf, offset=_HDR_BYTES + _FLOW_BYTES)
    return hdr, flow, frame


def _worker(shm_name, device, synthetic, stop_evt, err_q, t_start_wall):
    """Camera process: capture -> optical flow -> shared memory. It runs outside the render process
    so the render loop (which holds the GIL while it waits on the GPU) cannot starve it; as a
    thread inside the app it fell to 9 fps with 100 ms per flow frame."""
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(name=shm_name)
    hdr, out_flow, out_frame = _views(shm.buf)
    I = {k: i for i, k in enumerate(_H)}
    src = None
    try:
        src = _SyntheticSource() if synthetic else _FfmpegSource(device)
        hdr[I["running"]] = 1
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        prev, ema_px = None, None
        bg_acc = None
        stamps = []
        frames = 0
        while not stop_evt.is_set():
            big = src.read()                        # gray, mirrored (selfie view), BG_SIZE
            if big is None:
                err_q.put("camera stream ended: " + src.error_text())
                break
            now = time.time()
            frames += 1
            if frames == 1:
                hdr[I["first_frame_s"]] = now - t_start_wall
            stamps = (stamps + [now])[-31:]
            t0 = time.perf_counter()
            small = cv2.resize(big, FLOW_SIZE, interpolation=cv2.INTER_AREA)
            m, sd = cv2.meanStdDev(small)
            dark = float(m[0, 0]) < DARK_MEAN and float(sd[0, 0]) < DARK_STD
            if prev is None or dark:
                ema_px = None
                flow = np.zeros((FLOW_SIZE[1], FLOW_SIZE[0], 2), np.float32)
            else:
                f = dis.calc(prev, small, None)                     # pixels per frame, y down
                # DIS fills flat regions by extrapolating nearby motion; only trust it where the
                # image actually changed. Threshold adapts to the sensor noise (median difference).
                diff = cv2.GaussianBlur(cv2.absdiff(small, prev), (7, 7), 1.5).astype(np.float32)
                noise = float(np.median(diff))
                lo = 2.5 * noise + 2.0
                change = np.clip((diff - lo) / (lo + 4.0), 0.0, 1.0)
                change = cv2.GaussianBlur(cv2.dilate(change, np.ones((5, 5), np.uint8)), (9, 9), 2.0)
                f = cv2.GaussianBlur(f, (9, 9), 2.0) * change[..., None]
                # smooth in time first: sensor noise averages out, real motion persists
                ema_px = f if ema_px is None else cv2.addWeighted(ema_px, 0.6, f, 0.4, 0.0)
                fx, fy = cv2.split(ema_px)
                mag = cv2.magnitude(fx, fy)
                gate = np.clip((mag - NOISE_PX) * (1.0 / NOISE_PX), 0.0, 1.0)
                k = CAP_FPS / FLOW_SIZE[1]          # px/frame -> camera heights per second (fixed-rate capture)
                flow = cv2.merge([fx * gate * k, fy * gate * -k])   # y up
                flow = cv2.flip(flow, 0)                             # GL rows: bottom first
            prev = small
            flow_ms = (time.perf_counter() - t0) * 1000.0
            mag_all = cv2.magnitude(np.ascontiguousarray(flow[..., 0]), np.ascontiguousarray(flow[..., 1]))
            # seqlock publish: odd while writing, even when consistent
            hdr[I["seq"]] += 1
            out_flow[...] = flow
            # drawn into the smoke: averaged over a few frames, which quiets sensor noise in low light
            bf = big.astype(np.float32)
            bg_acc = bf if bg_acc is None else cv2.addWeighted(bg_acc, 1.0 - BG_SMOOTH, bf, BG_SMOOTH, 0.0)
            shown = bg_acc
            if BG_SHARPEN > 0:
                soft = cv2.GaussianBlur(bg_acc, (0, 0), 2.0)
                shown = cv2.addWeighted(bg_acc, 1.0 + BG_SHARPEN, soft, -BG_SHARPEN, 0.0)
            out_frame[...] = np.clip(shown + 0.5, 0, 255).astype(np.uint8)
            hdr[I["fps"]] = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-6) if len(stamps) > 1 else 0.0
            hdr[I["brightness"]] = float(m[0, 0])
            hdr[I["dark"]] = 1.0 if dark else 0.0
            hdr[I["flow_ms"]] = flow_ms
            hdr[I["motion"]] = float(mag_all.mean())
            hdr[I["active"]] = float((mag_all > 0.02).mean())
            hdr[I["frames"]] = frames
            hdr[I["last_frame_wall"]] = now
            if frames % 5 == 1:   # auto-levels for the background layer
                lo, hi = np.percentile(small[::2, ::2], (5, 95))
                hdr[I["p05"]], hdr[I["p95"]] = float(lo), float(hi)
            hdr[I["seq"]] += 1
    except Exception as e:  # report, then let the parent show it
        err_q.put(f"camera worker: {e}")
    finally:
        if src is not None:
            src.close()
        hdr[I["running"]] = 0
        del hdr, out_flow, out_frame
        shm.close()


class CameraStir:
    """Main-process side: starts the worker, reads the newest flow from shared memory, uploads it."""

    def __init__(self, ctx=None, device=None, synthetic=False, gain=None, mirror=True):
        self.ctx = ctx
        self.synthetic = synthetic
        self.device = device
        self.mirror = mirror     # ffmpeg / the synthetic source mirror the image (selfie view)
        self.gain = 0.6 if gain is None else float(gain)
        self.tex = None
        self.proc = None
        self.shm = None
        self.error = ""
        self.uploaded_seq = -1
        self.bg_seq = -1
        self.bg_tex = None
        self._flow = np.zeros((FLOW_SIZE[1], FLOW_SIZE[0], 2), np.float32)
        self.t_start_wall = None

    def describe(self):
        return "synthetic blob" if self.synthetic else f"device {self.device!r} (ffmpeg MJPEG 640x480)"

    def start(self):
        import multiprocessing as mp
        from multiprocessing import shared_memory

        if not self.synthetic and self.device is None:
            devs = list_video_devices()
            if not devs:
                raise RuntimeError("no video capture device found")
            pref = [d for d in devs if "c270" in d.lower() or "webcam" in d.lower() or "camera" in d.lower()]
            self.device = (pref or devs)[0]
        self.shm = shared_memory.SharedMemory(create=True, size=_HDR_BYTES + _FLOW_BYTES + _FRAME_BYTES)
        self.hdr, self.flow_view, self.frame_view = _views(self.shm.buf)
        self.hdr[:] = 0
        self.I = {k: i for i, k in enumerate(_H)}
        ctx = mp.get_context("spawn")
        self.stop_evt = ctx.Event()
        self.err_q = ctx.Queue()
        self.t_start_wall = time.time()
        self.proc = ctx.Process(target=_worker, name="audioviz-camera", daemon=True,
                                args=(self.shm.name, self.device, self.synthetic, self.stop_evt, self.err_q,
                                      self.t_start_wall))
        self.proc.start()

    # -- stats (read from the shared header) --------------------------------------------------
    def _h(self, k):
        return float(self.hdr[self.I[k]]) if self.shm is not None else 0.0

    @property
    def frames(self):
        return int(self._h("frames"))

    @property
    def fps(self):
        return self._h("fps")

    @property
    def brightness(self):
        return self._h("brightness")

    @property
    def dark(self):
        return self._h("dark") > 0.5

    @property
    def flow_ms(self):
        return self._h("flow_ms")

    @property
    def flow_mag(self):
        return self._h("motion")

    @property
    def active_frac(self):
        return self._h("active")

    @property
    def first_frame_s(self):
        v = self._h("first_frame_s")
        return v if v > 0 else None

    @property
    def running(self):
        return self.proc is not None and self.proc.is_alive()

    @property
    def last_frame(self):
        return None if self.shm is None or self.frames < 15 else self.frame_view.copy()

    def _poll_errors(self):
        try:
            while True:
                self.error = self.err_q.get_nowait()
        except Exception:
            pass

    def stalled(self):
        last = self._h("last_frame_wall")
        return last <= 0 or time.time() - last > STALL_S

    def read_flow(self):
        """Consistent copy of the newest flow (seqlock), or (None, -1)."""
        if self.shm is None:
            return None, -1
        for _ in range(4):
            s1 = int(self.hdr[0])
            if s1 % 2 == 1:
                continue
            np.copyto(self._flow, self.flow_view)
            if int(self.hdr[0]) == s1:
                return self._flow, s1
        return None, -1

    def update(self, screen_aspect=None):
        """Main thread: upload the newest flow. Returns (texture or None, uv scale, gain)."""
        if self.ctx is None or self.shm is None or self.stalled() or self.dark:
            return None, (1.0, 1.0), 0.0
        seq = int(self.hdr[0])
        if self.tex is None:
            from ..gl.pingpong import make_texture

            self.tex = make_texture(self.ctx, FLOW_SIZE, 2, "f4", "linear")
        if seq != self.uploaded_seq:
            flow, s = self.read_flow()
            if flow is not None:
                self.tex.write(flow.tobytes())
                self.uploaded_seq = s
        return self.tex, self.cover_scale(screen_aspect), self.gain

    def prime(self, screen_aspect=None):
        """Create the camera textures (blank) right away so the app can bind them during prewarm:
        the first bind of these textures cost a ~0.5 s driver hitch when it happened mid-song."""
        if self.ctx is None:
            return None, None, (1.0, 1.0)
        import moderngl

        from ..gl.pingpong import make_texture

        if self.tex is None:
            self.tex = make_texture(self.ctx, FLOW_SIZE, 2, "f4", "linear")
        if self.bg_tex is None:
            self.bg_tex = make_texture(self.ctx, BG_SIZE, 1, "f1", "linear")
            self.bg_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.bg_tex.build_mipmaps()
        return self.tex, self.bg_tex, self.cover_scale(screen_aspect)

    def background(self, screen_aspect=None):
        """Main thread: newest gray frame as an R8 texture for the dim background layer.
        Returns (texture or None, uv scale, black point, white point) in 0..1 units."""
        if self.ctx is None or self.shm is None or self.stalled() or self.dark or self.frames < 2:
            return None, (1.0, 1.0), 0.0, 1.0
        if self.bg_tex is None:
            from ..gl.pingpong import make_texture

            import moderngl

            self.bg_tex = make_texture(self.ctx, BG_SIZE, 1, "f1", "linear")
            self.bg_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)   # mip blur
        seq = int(self.hdr[0])
        if seq != self.bg_seq and seq % 2 == 0:
            img = np.flipud(self.frame_view).copy()        # GL rows: bottom first
            if int(self.hdr[0]) == seq:
                self.bg_tex.write(img.tobytes())
                self.bg_tex.build_mipmaps()
                self.bg_seq = seq
        lo, hi = self._h("p05") / 255.0, self._h("p95") / 255.0
        # stretch up to ~16x so a face lit only by a monitor (values 10-25 of 255) still reads; the
        # camera process smooths the image over time so the stretched sensor noise stays quiet
        return self.bg_tex, self.cover_scale(screen_aspect), lo, max(hi, lo + MIN_LEVEL_RANGE)

    @staticmethod
    def cover_scale(screen_aspect):
        """uv scale so the camera covers the screen (crop, no stretch)."""
        cam = FLOW_SIZE[0] / FLOW_SIZE[1]
        if screen_aspect is None:
            return (1.0, 1.0)
        if screen_aspect >= cam:
            return (1.0, cam / screen_aspect)
        return (screen_aspect / cam, 1.0)

    def status(self):
        self._poll_errors()
        if self.error:
            return self.error[:120]
        if self.frames == 0:
            return "opening..." if self.running else "camera process exited"
        st = "STALLED" if self.stalled() else ("dark (lights off?)" if self.dark else "ok")
        return (f"{st}  {self.fps:4.1f} fps  brightness {self.brightness:5.1f}  flow {self.flow_ms:4.1f} ms  "
                f"motion {self.flow_mag:.3f}  active {self.active_frac * 100:4.1f}%")

    def stop(self):
        if self.proc is not None:
            self.stop_evt.set()
            self.proc.join(timeout=6.0)     # the worker asks ffmpeg to quit ('q'); terminate only as a fallback
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2.0)
            self.proc = None
        if self.tex is not None:
            self.tex.release()
            self.tex = None
        if self.bg_tex is not None:
            self.bg_tex.release()
            self.bg_tex = None
        if self.shm is not None:
            self._poll_errors()
            del self.hdr, self.flow_view, self.frame_view
            self.shm.close()
            self.shm.unlink()
            self.shm = None


def camera_test(args):
    """Headless plumbing check: frames arrive, fps, brightness, flow cost, motion, clean shutdown."""
    import json
    import os

    from .. import OUT_DIR

    cam = CameraStir(None, device=args.device, synthetic=args.synthetic)
    try:
        cam.start()
    except Exception as e:
        print("camera start failed:", e)
        return 1
    print("camera:", cam.describe())
    t0 = time.perf_counter()
    samples = []
    while time.perf_counter() - t0 < args.seconds and (cam.running or time.perf_counter() - t0 < 3):
        time.sleep(0.5)
        samples.append({"t": round(time.perf_counter() - t0, 1), "status": cam.status()})
        print(" ", samples[-1]["status"])
        if not cam.running and cam.error:
            break
    report = {
        "describe": cam.describe(), "frames": cam.frames, "fps": round(cam.fps, 1),
        "first_frame_s": round(cam.first_frame_s, 2) if cam.first_frame_s else None,
        "brightness": round(cam.brightness, 1), "dark": cam.dark, "flow_ms": round(cam.flow_ms, 2),
        "motion": round(cam.flow_mag, 4), "active_frac": round(cam.active_frac, 4), "error": cam.error,
        "samples": samples,
    }
    os.makedirs(os.path.join(OUT_DIR, "camera"), exist_ok=True)
    tag = "synthetic" if args.synthetic else "device"
    if cam.last_frame is not None:
        cv2.imwrite(os.path.join(OUT_DIR, "camera", f"last_frame_{tag}.png"), cam.last_frame)
    flow, _ = cam.read_flow()
    flow = None if flow is None else flow.copy()
    if flow is not None:
        f = np.flipud(flow)
        mag = np.sqrt((f ** 2).sum(axis=2))
        ang = np.arctan2(f[..., 1], f[..., 0])
        hsv = np.zeros(f.shape[:2] + (3,), np.uint8)
        hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(mag / max(mag.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, "camera", f"flow_{tag}.png"), cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    t_stop = time.perf_counter()
    cam.stop()
    report["stop_s"] = round(time.perf_counter() - t_stop, 2)
    with open(os.path.join(OUT_DIR, "camera", f"camera_test_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=1))
    return 0
