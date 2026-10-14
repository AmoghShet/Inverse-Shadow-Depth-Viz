"""Camera -> dark-room gain -> depth model -> mask -> shared memory, in a worker process.

Process layout (multiprocessing spawn, daemon): a capture thread drains the ffmpeg pipe and keeps
only the newest frame (a slow model never backs the pipe up; latency stays one camera frame); the
model loop takes the newest frame, runs gain + depth (+ mask) + normalisation and publishes; frames
that arrive while it is busy are dropped. The ORT session is built while the camera starts (3-4.6 s).

ffmpeg: Windows `-f dshow -rtbufsize 64M -vcodec mjpeg -video_size 640x480 -framerate 30` (never ask
the C270 for 720p without MJPEG: it wedges until replugged); macOS AVFoundation 1280x720 centre-cropped
to 4:3. Filter chain `[crop,]hflip,tmix=frames=3,scale=320:240:flags=area`, rgb24 raw pipe.

Shared memory (one block): header 32 x float64, depth float32 240x320 (GL row order: bottom row
first), mask uint8 240x320 (GL rows), gained gray uint8 240x320 (top row first, debug view).
Seqlock: `seq` is odd while writing and even when consistent; the reader copies and rechecks.
"""
import math
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from .. import ROOT_DIR

DEPTH_W, DEPTH_H = 320, 240
CAP_W, CAP_H, CAP_FPS = 640, 480, 30
STALL_S = 2.0
DARK_RANGE = 3.0 / 255.0     # p1..p99 spread below this = no usable signal (true black / lens covered)
MASK_MODES = ("off", "mp", "depth")
EPS = ("cpu", "dml", "coreml")
SELFIE_MODEL = os.path.join(ROOT_DIR, "models", "selfie_segmenter_landscape.tflite")

HDR = ["seq", "fps", "cam_fps", "depth_ms", "mask_ms", "brightness", "gain", "dark", "lo", "hi", "align_a",
       "align_b", "fig_depth", "mask_cov", "frames", "first_frame_s", "last_frame_wall", "running", "model_id",
       "ep_id", "in_w", "in_h", "mask_mode", "stale", "latency_ms", "depth_frames", "session_s", "cam_frames",
       "gain_lo", "gain_hi", "proc_ms", "error"]
HI = {k: i for i, k in enumerate(HDR)}
_HDR_BYTES = 32 * 8
_DEPTH_BYTES = DEPTH_W * DEPTH_H * 4
_PLANE = DEPTH_W * DEPTH_H
SHM_BYTES = _HDR_BYTES + _DEPTH_BYTES + 2 * _PLANE
assert len(HDR) <= 32


def views(buf):
    hdr = np.ndarray((32,), np.float64, buffer=buf, offset=0)
    depth = np.ndarray((DEPTH_H, DEPTH_W), np.float32, buffer=buf, offset=_HDR_BYTES)
    mask = np.ndarray((DEPTH_H, DEPTH_W), np.uint8, buffer=buf, offset=_HDR_BYTES + _DEPTH_BYTES)
    gray = np.ndarray((DEPTH_H, DEPTH_W), np.uint8, buffer=buf, offset=_HDR_BYTES + _DEPTH_BYTES + _PLANE)
    return hdr, depth, mask, gray


# ---- devices and capture -----------------------------------------------------------------------
def list_devices():
    from audioviz.inputs.camera import list_video_devices

    return list_video_devices()


def pick_device(device=None):
    if device is not None:
        return device
    devs = list_devices()
    if not devs:
        raise RuntimeError("no video capture device found")
    pref = [d for d in devs if any(k in d.lower() for k in ("c270", "webcam", "camera", "facetime"))]
    return (pref or devs)[0]


def capture_cmd(device, out_w=DEPTH_W, out_h=DEPTH_H, tmix=3):
    from audioviz.inputs.camera import ffmpeg_exe

    exe = ffmpeg_exe()
    common = [exe, "-hide_banner", "-loglevel", "error", "-fflags", "nobuffer", "-flags", "low_delay"]
    crop = ""
    if sys.platform.startswith("win"):
        inp = ["-f", "dshow", "-rtbufsize", "64M", "-vcodec", "mjpeg", "-video_size", f"{CAP_W}x{CAP_H}",
               "-framerate", str(CAP_FPS), "-i", f"video={device}"]
    elif sys.platform == "darwin":   # Mac cameras have no 640x480 mode: 720p, centre-cropped to 4:3
        inp = ["-f", "avfoundation", "-framerate", str(CAP_FPS), "-video_size", "1280x720", "-i", f"{device}:none"]
        crop = "crop=ih*4/3:ih,"
    else:
        inp = ["-f", "v4l2", "-input_format", "mjpeg", "-video_size", f"{CAP_W}x{CAP_H}", "-framerate", str(CAP_FPS),
               "-i", device]
    vf = f"{crop}hflip,{f'tmix=frames={tmix},' if tmix > 1 else ''}scale={out_w}:{out_h}:flags=area"
    return common + inp + ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def set_exposure(device_name, value):
    """One-shot manual exposure through DirectShow at 640x480 (log2 seconds: -5 = 31 ms). The value
    persists in the camera after release (measured in the research spike). Windows only."""
    if not sys.platform.startswith("win"):
        return False
    devs = list_devices()
    idx = devs.index(device_name) if device_name in devs else 0
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if value is None or str(value).lower() == "auto":
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        cap.read()
        return True
    finally:
        cap.release()


class FfmpegSource:
    """ffmpeg rgb24 pipe with a capture thread that keeps only the newest frame."""

    def __init__(self, device, out_size=(DEPTH_W, DEPTH_H), tmix=3):
        self.device = device
        self.w, self.h = out_size
        self.nbytes = self.w * self.h * 3
        self.proc = subprocess.Popen(capture_cmd(device, self.w, self.h, tmix), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.latest = None
        self.latest_n = 0
        self.latest_wall = 0.0
        self.ended = False
        self.stamps = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        buf = bytearray(self.nbytes)
        mv = memoryview(buf)
        while True:
            got = 0
            while got < self.nbytes:
                n = self.proc.stdout.readinto(mv[got:])
                if not n:
                    with self.cond:
                        self.ended = True
                        self.cond.notify_all()
                    return
                got += n
            frame = np.frombuffer(bytes(buf), np.uint8).reshape(self.h, self.w, 3)
            now = time.time()
            with self.cond:
                self.latest = frame
                self.latest_n += 1
                self.latest_wall = now
                self.stamps = (self.stamps + [now])[-31:]
                self.cond.notify_all()

    def fps(self):
        with self.lock:
            s = list(self.stamps)
        return (len(s) - 1) / max(s[-1] - s[0], 1e-6) if len(s) > 1 else 0.0

    def wait_newer(self, n_seen, timeout=0.5):
        """Newest frame with a counter above n_seen, or (None, n_seen) on timeout / end."""
        with self.cond:
            if self.latest_n <= n_seen and not self.ended:
                self.cond.wait(timeout)
            if self.latest_n <= n_seen:
                return None, n_seen, 0.0
            return self.latest, self.latest_n, self.latest_wall

    def error_text(self):
        try:
            return self.proc.stderr.read().decode(errors="replace").strip()[:500]
        except Exception:
            return ""

    def close(self):
        """'q' on stdin so ffmpeg releases the device cleanly (hard kills wedged the C270). Idempotent."""
        if getattr(self, "_closing", False):
            if self.proc.poll() is None:
                try:
                    self.proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
            return
        self._closing = True
        p = self.proc
        if p.poll() is None:
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


class RgbSynthetic:
    """A dim 'person' (head + torso) swaying in front of a lit back wall, at 30 fps, with sensor
    noise: exercises the model path with the lights off. Brightness ~8/255 like the user's room."""

    def __init__(self, out_size=(DEPTH_W, DEPTH_H), level=1.0):
        self.w, self.h = out_size
        self.n = 0
        self.t0 = time.time()
        self.level = level
        self.rng = np.random.default_rng(3)
        self.latest_n = 0
        self.ended = False

    def fps(self):
        return float(CAP_FPS)

    def frame_at(self, t):
        w, h = self.w, self.h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        wall = 0.55 + 0.25 * np.exp(-((xx - 0.72 * w) ** 2 + (yy - 0.35 * h) ** 2) / (2 * (0.25 * w) ** 2))
        img = np.repeat(wall[..., None], 3, axis=2) * np.array([1.0, 0.97, 0.92], np.float32)
        img[int(0.78 * h):] *= 0.6                                     # floor
        cx = w * (0.45 + 0.12 * math.sin(2 * math.pi * 0.11 * t))
        body = ((xx - cx) / (0.17 * w)) ** 2 + ((yy - 0.95 * h) / (0.42 * h)) ** 2 < 1.0
        head = ((xx - cx) / (0.075 * w)) ** 2 + ((yy - 0.36 * h) / (0.11 * h)) ** 2 < 1.0
        fig = body | head
        lit = 0.35 + 0.25 * np.clip((xx - cx) / (0.2 * w), -1, 1)       # side light from a monitor
        img[fig] = (lit[fig, None] * np.array([0.9, 0.95, 1.0], np.float32))
        img = img * (8.0 / 255.0 / 0.6) * self.level                     # dark room
        noise = self.rng.normal(0.0, 2.2 / 255.0, img.shape).astype(np.float32)
        return np.clip((img + noise) * 255.0, 0, 255).astype(np.uint8)

    def wait_newer(self, n_seen, timeout=0.5):
        target = self.t0 + self.n / CAP_FPS
        dt = target - time.time()
        if dt > 0:
            time.sleep(min(dt, timeout))
        f = self.frame_at(self.n / CAP_FPS)
        self.n += 1
        self.latest_n = self.n
        return f, self.n, time.time()

    def error_text(self):
        return ""

    def close(self):
        pass


# ---- mask ------------------------------------------------------------------------------------
class MpMask:
    """MediaPipe selfie-landscape confidence mask at the frame size (CPU, VIDEO mode)."""

    def __init__(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.isfile(SELFIE_MODEL):
            raise FileNotFoundError(SELFIE_MODEL)
        opts = vision.ImageSegmenterOptions(base_options=mp_python.BaseOptions(model_asset_path=SELFIE_MODEL),
                                            running_mode=vision.RunningMode.VIDEO, output_category_mask=False,
                                            output_confidence_masks=True)
        self.seg = vision.ImageSegmenter.create_from_options(opts)
        self.ts = 0

    def __call__(self, rgb):
        import mediapipe as mp

        self.ts += 33
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.seg.segment_for_video(img, self.ts)
        m = res.confidence_masks[0].numpy_view()
        return (np.clip(m, 0, 1) * 255.0 + 0.5).astype(np.uint8).reshape(rgb.shape[0], rgb.shape[1])

    def close(self):
        try:
            self.seg.close()
        except Exception:
            pass


def depth_mask(d):
    """Fallback mask: Otsu on nearness (1 - d), 5x5 open/close, 3 px feather. d: 0 near .. 1 far."""
    near = (np.clip(1.0 - d, 0.0, 1.0) * 255.0).astype(np.uint8)
    _, m = cv2.threshold(near, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return cv2.GaussianBlur(m, (0, 0), 3.0)


def make_mask(mode):
    """mode auto|mp|depth|off -> (callable or None, resolved mode)."""
    if mode in ("auto", "mp"):
        try:
            return MpMask(), "mp"
        except Exception as e:
            if mode == "mp":
                raise
            print(f"[depthcam] MediaPipe mask unavailable ({type(e).__name__}: {e}); using the depth mask")
            return None, "depth"
    return None, mode


# ---- worker process -----------------------------------------------------------------------------
def _worker(shm_name, cfg, stop_evt, err_q, t_start_wall):
    from multiprocessing import shared_memory

    from ..depth import models as M
    from ..depth.infer import DepthModel, Normaliser, gain_frame

    shm = shared_memory.SharedMemory(name=shm_name)
    hdr, o_depth, o_mask, o_gray = views(shm.buf)
    src = None
    masker = None
    try:
        hdr[HI["running"]] = 1
        if cfg.get("synthetic_rgb"):
            src = RgbSynthetic()
        else:
            if cfg.get("exposure") is not None:
                try:
                    set_exposure(cfg["device"], cfg["exposure"])
                except Exception as e:
                    err_q.put(f"exposure set failed: {e}")
            src = FfmpegSource(cfg["device"], tmix=cfg.get("tmix", 3))

            def release_on_stop(s=src):   # release the camera the moment stop is set, even mid-inference
                import multiprocessing as mp

                parent = mp.parent_process()
                while not stop_evt.wait(0.5):
                    if parent is not None and not parent.is_alive():   # app killed: free the webcam anyway
                        stop_evt.set()
                s.close()

            threading.Thread(target=release_on_stop, daemon=True).start()
        t0 = time.perf_counter()
        try:
            model = DepthModel(cfg["model"], cfg["ep"], cfg.get("size"), cfg.get("device_id", 0), cfg.get("threads"))
        except Exception as e:   # no DirectML / CoreML, or a broken session: MiDaS on the CPU still works
            from ..config import CPU_FALLBACK as fb

            err_q.put(f"depth session failed ({type(e).__name__}: {str(e)[:120]}); CPU fallback {fb['model']}")
            model = DepthModel(fb["model"], fb["ep"], fb["size"], 0, cfg.get("threads"))
        hdr[HI["session_s"]] = time.perf_counter() - t0
        hdr[HI["model_id"]] = list(M.MODELS).index(cfg["model"])
        hdr[HI["ep_id"]] = EPS.index(model.info["ep"]) if model.info["ep"] in EPS else -1
        hdr[HI["in_w"]], hdr[HI["in_h"]] = model.in_w, model.in_h
        model(np.zeros((DEPTH_H, DEPTH_W, 3), np.uint8))   # warm-up (DML compiles on first run)
        # MediaPipe takes 4-12 s to import and build in a fresh process: build it on a thread and use the
        # depth mask until it is ready, so depth starts with the first camera frame
        want = cfg.get("mask", "auto")
        mp_box = {}
        mask_mode = "off" if want == "off" else "depth"
        if want in ("auto", "mp"):
            def build_mask():
                try:
                    mp_box["masker"] = MpMask()
                except Exception as e:
                    mp_box["error"] = f"{type(e).__name__}: {e}"

            threading.Thread(target=build_mask, daemon=True).start()
        hdr[HI["mask_mode"]] = MASK_MODES.index(mask_mode)
        norm = Normaliser()
        gstate = {}
        seen = 0
        frames = 0
        stamps = []
        prev_wall = None
        while not stop_evt.is_set():
            rgb, n, wall = src.wait_newer(seen, 0.5)
            if rgb is None:
                if src.ended:
                    err_q.put("camera stream ended: " + src.error_text())
                    break
                continue
            seen = n
            frames += 1
            if masker is None and "masker" in mp_box:
                masker, mask_mode = mp_box.pop("masker"), "mp"
                hdr[HI["mask_mode"]] = MASK_MODES.index(mask_mode)
            elif "error" in mp_box:
                err_q.put(f"MediaPipe mask unavailable ({mp_box.pop('error')[:120]}); using the depth mask")
                if want == "mp":
                    mask_mode = "off"
                    hdr[HI["mask_mode"]] = 0
            now = time.time()
            if frames == 1:
                hdr[HI["first_frame_s"]] = now - t_start_wall
            stamps = (stamps + [now])[-31:]
            dt = 1.0 / CAP_FPS if prev_wall is None else min(max(now - prev_wall, 1e-3), 1.0)
            prev_wall = now
            p0 = time.perf_counter()
            g, gi = gain_frame(rgb, gstate, dt)
            dark = (gi["hi"] - gi["lo"]) < DARK_RANGE
            z = model(g)
            dms = model.last_ms
            m0 = time.perf_counter()
            mraw = masker(g) if masker is not None else None
            d = norm(z, dt, mraw)
            if mask_mode == "depth":
                mraw = depth_mask(d)
                m = mraw > 127
                norm.mask_cov = float(m.mean())
                if norm.mask_cov > 0.01:
                    norm.fig_depth = float(np.median(d[m]))
            mms = (time.perf_counter() - m0) * 1000.0 if mask_mode != "off" else 0.0
            proc = (time.perf_counter() - p0) * 1000.0
            # seqlock publish
            hdr[HI["seq"]] += 1
            o_depth[...] = d[::-1]
            if mraw is not None:
                o_mask[...] = mraw[::-1]
            else:
                o_mask[...] = 0
            o_gray[...] = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
            hdr[HI["fps"]] = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-6) if len(stamps) > 1 else 0.0
            hdr[HI["cam_fps"]] = src.fps()
            hdr[HI["depth_ms"]] = dms
            hdr[HI["mask_ms"]] = mms
            hdr[HI["proc_ms"]] = proc
            hdr[HI["brightness"]] = gi["brightness"]
            hdr[HI["gain"]] = gi["gain"]
            hdr[HI["gain_lo"]], hdr[HI["gain_hi"]] = gi["lo"], gi["hi"]
            hdr[HI["dark"]] = 1.0 if dark else 0.0
            hdr[HI["lo"]], hdr[HI["hi"]] = norm.lo, norm.hi
            hdr[HI["align_a"]], hdr[HI["align_b"]] = norm.a, norm.b
            hdr[HI["fig_depth"]] = norm.fig_depth
            hdr[HI["mask_cov"]] = norm.mask_cov
            hdr[HI["frames"]] = frames
            hdr[HI["cam_frames"]] = n
            hdr[HI["depth_frames"]] = frames
            hdr[HI["last_frame_wall"]] = time.time()
            hdr[HI["latency_ms"]] = (time.time() - wall) * 1000.0 + 1000.0 / CAP_FPS * 0.5
            hdr[HI["seq"]] += 1
    except Exception as e:
        import traceback

        err_q.put(f"depth worker: {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")
        hdr[HI["error"]] = 1
    finally:
        if src is not None:
            src.close()           # releases the camera first ('q' to ffmpeg)
        hdr[HI["running"]] = 0
        del hdr, o_depth, o_mask, o_gray
        shm.close()
        try:
            err_q.close()
            err_q.join_thread()   # flush queued messages before the hard exit
        except Exception:
            pass
        # skip interpreter teardown: onnxruntime / MediaPipe finalisers (and a MediaPipe build still running
        # on its thread) made the worker take 1.5-6 s to exit; everything that matters is released above
        os._exit(0)


class DepthCam:
    """Main-process side: starts the worker, reads the newest depth/mask (seqlock), uploads textures."""

    def __init__(self, ctx=None, device=None, model="dav2_small", ep="auto", size=None, device_id=0, mask="off",
                 exposure=None, synthetic_rgb=False, threads=None, tmix=3):
        self.ctx = ctx
        self.cfg = {"device": device, "model": model, "ep": ep, "size": size, "device_id": device_id, "mask": mask,
                    "exposure": exposure, "synthetic_rgb": synthetic_rgb, "threads": threads, "tmix": tmix}
        self.proc = None
        self.shm = None
        self.error = ""
        self.depth_tex = None
        self.mask_tex = None
        self.uploaded_seq = -1
        self.t_start_wall = None
        self._depth = np.zeros((DEPTH_H, DEPTH_W), np.float32)
        self._mask = np.zeros((DEPTH_H, DEPTH_W), np.uint8)
        self._hdr = np.zeros(32, np.float64)
        self.have_frame = False

    def describe(self):
        c = self.cfg
        src = "synthetic dark RGB" if c["synthetic_rgb"] else f"{c['device']!r}"
        return f"{src} -> {c['model']} {c['size'] or ''} on {c['ep']} mask {c['mask']}"

    def start(self):
        import multiprocessing as mp
        from multiprocessing import shared_memory

        if not self.cfg["synthetic_rgb"]:
            self.cfg["device"] = pick_device(self.cfg["device"])
        self.shm = shared_memory.SharedMemory(create=True, size=SHM_BYTES)
        self.hdr, self.depth_view, self.mask_view, self.gray_view = views(self.shm.buf)
        self.hdr[:] = 0
        ctx = mp.get_context("spawn")
        self.stop_evt = ctx.Event()
        self.err_q = ctx.Queue()
        self.t_start_wall = time.time()
        self.proc = ctx.Process(target=_worker, name="depthviz-camera", daemon=True,
                                args=(self.shm.name, dict(self.cfg), self.stop_evt, self.err_q, self.t_start_wall))
        self.proc.start()

    # -- header -----------------------------------------------------------------------------------
    def h(self, k):
        return float(self.hdr[HI[k]]) if self.shm is not None else 0.0

    @property
    def frames(self):
        return int(self.h("frames"))

    @property
    def running(self):
        return self.proc is not None and self.proc.is_alive()

    @property
    def dark(self):
        return self.h("dark") > 0.5

    def poll_errors(self):
        try:
            while True:
                self.error = self.err_q.get_nowait()
        except Exception:
            pass

    def stalled(self):
        last = self.h("last_frame_wall")
        return last <= 0 or time.time() - last > STALL_S

    def read(self):
        """Consistent copy (depth GL rows, mask GL rows, header) of the newest result, or None."""
        if self.shm is None:
            return None
        for _ in range(4):
            s1 = int(self.hdr[0])
            if s1 % 2 == 1 or s1 == 0:
                if s1 == 0:
                    return None
                time.sleep(0.0005)
                continue
            np.copyto(self._depth, self.depth_view)
            np.copyto(self._mask, self.mask_view)
            np.copyto(self._hdr, self.hdr)
            if int(self.hdr[0]) == s1:
                return self._depth, self._mask, self._hdr
        return None

    def gray(self):
        return None if self.shm is None or self.frames < 1 else self.gray_view.copy()

    # -- GL ---------------------------------------------------------------------------------------
    def prime(self):
        """Create the textures blank before prewarm (a first bind mid-song cost ~0.5 s in audioviz)."""
        if self.ctx is None:
            return None, None
        from audioviz.gl.pingpong import make_texture

        if self.depth_tex is None:
            self.depth_tex = make_texture(self.ctx, (DEPTH_W, DEPTH_H), 1, "f2", "linear",   # R16F (Apple-safe)
                                          data=np.full(DEPTH_W * DEPTH_H, 1.0, np.float16).tobytes())
            self.mask_tex = make_texture(self.ctx, (DEPTH_W, DEPTH_H), 1, "f1", "linear",
                                         data=bytes(DEPTH_W * DEPTH_H))
        return self.depth_tex, self.mask_tex

    def update(self, hold_if_dark=True):
        """Upload the newest result if it changed. Returns (new: bool, header copy or None).
        A frame flagged DARK (no signal at all: lens covered, true black) is not uploaded, so the last
        good depth is held; the user's lights-off room (brightness 4-10) is not DARK, the gain handles it."""
        if self.shm is None:
            return False, None
        seq = int(self.hdr[0])
        if seq == self.uploaded_seq or seq % 2 == 1 or seq == 0:
            return False, None
        r = self.read()
        if r is None:
            return False, None
        d, m, hdr = r
        if hold_if_dark and hdr[HI["dark"]] > 0.5:
            self.uploaded_seq = int(hdr[0])
            self.dark_skipped = getattr(self, "dark_skipped", 0) + 1
            return False, hdr
        if self.ctx is not None:
            self.prime()
            self.depth_tex.write(d.astype(np.float16).tobytes())
            self.mask_tex.write(m.tobytes())
        self.uploaded_seq = int(hdr[0])
        self.have_frame = True
        return True, hdr

    def status(self):
        self.poll_errors()
        if self.error:
            return self.error.splitlines()[0][:140]
        if self.frames == 0:
            return "opening..." if self.running else "camera process exited"
        st = "STALLED" if self.stalled() else ("DARK (no signal)" if self.dark else "ok")
        return (f"{st} cam {self.h('cam_fps'):4.1f} fps depth {self.h('fps'):4.1f} fps {self.h('depth_ms'):5.1f} ms "
                f"bright {self.h('brightness'):5.1f} gain {self.h('gain'):4.1f}x lo/hi {self.h('lo'):.2f}/{self.h('hi'):.2f} "
                f"mask {self.h('mask_cov') * 100:4.1f}% fig {self.h('fig_depth'):.2f} {EPS[int(self.h('ep_id'))] if self.h('ep_id') >= 0 else '?'}")

    def stop(self):
        if self.proc is not None:
            self.stop_evt.set()
            self.proc.join(timeout=6.0)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2.0)
            self.proc = None
        for t in (self.depth_tex, self.mask_tex):
            if t is not None:
                t.release()
        self.depth_tex = self.mask_tex = None
        if self.shm is not None:
            self.poll_errors()
            del self.hdr, self.depth_view, self.mask_view, self.gray_view
            self.shm.close()
            self.shm.unlink()
            self.shm = None


def save_frames(device, out_dir, n=30, every=1, warmup_s=1.0, tmix=1, size=(CAP_W, CAP_H)):
    """Save n consecutive raw camera frames (no gain) as PNG for the M0 benchmark. Returns paths."""
    os.makedirs(out_dir, exist_ok=True)
    src = FfmpegSource(pick_device(device), out_size=size, tmix=tmix)
    paths = []
    try:
        seen = 0
        t0 = time.time()
        while time.time() - t0 < 15.0:
            f, seen, _ = src.wait_newer(seen, 1.0)
            if f is not None:
                break
        if f is None:
            raise RuntimeError("no camera frames in 15 s: " + src.error_text())
        t_first = time.time() - t0
        t1 = time.time()
        while time.time() - t1 < warmup_s:
            f, seen, _ = src.wait_newer(seen, 1.0)
        k = 0
        while len(paths) < n:
            f, seen, _ = src.wait_newer(seen, 1.0)
            if f is None:
                if src.ended:
                    break
                continue
            k += 1
            if k % every:
                continue
            p = os.path.join(out_dir, f"frame_{len(paths):03d}.png")
            cv2.imwrite(p, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            paths.append(p)
        print(f"saved {len(paths)} frames to {out_dir} (first frame after {t_first:.1f} s, cam {src.fps():.1f} fps)")
    finally:
        src.close()
    return paths


def camera_test(args):
    """Headless worker check: fps, depth ms, gain, mask coverage, clean stop; optional frame dump."""
    import json

    from .. import OUT_DIR

    if getattr(args, "save_frames", None):
        save_frames(args.device, args.save_frames, n=args.n_frames)
        return 0
    from ..config import gl_renderer_headless, pick_profile

    prof = pick_profile(gl_renderer_headless())
    cam = DepthCam(None, device=args.device, model=args.model or prof["model"], ep=args.ep or prof["ep"],
                   size=args.model_size or prof["size"], device_id=prof.get("device_id", 0), mask=args.mask,
                   exposure=args.cam_exposure, synthetic_rgb=args.synthetic)
    t_open = time.perf_counter()
    cam.start()
    print("camera:", cam.describe())
    samples = []
    first_frame = None
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        time.sleep(0.5)
        if first_frame is None and cam.frames > 0:
            first_frame = round(time.perf_counter() - t_open, 2)
        samples.append({"t": round(time.perf_counter() - t0, 1), "status": cam.status()})
        print(" ", samples[-1]["status"])
        if not cam.running:
            break
    snap = {k: round(cam.h(k), 3) for k in HDR}
    os.makedirs(os.path.join(OUT_DIR, "camera"), exist_ok=True)
    tag = "synthetic" if args.synthetic else "device"
    r = cam.read()
    if r is not None:
        d, m, _ = r
        cv2.imwrite(os.path.join(OUT_DIR, "camera", f"depth_{tag}.png"), (np.flipud(d) * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(OUT_DIR, "camera", f"mask_{tag}.png"), np.flipud(m))
        g = cam.gray()
        if g is not None:
            cv2.imwrite(os.path.join(OUT_DIR, "camera", f"gained_{tag}.png"), g)
    t_stop = time.perf_counter()
    cam.stop()
    stop_s = round(time.perf_counter() - t_stop, 2)
    rep = {"describe": cam.describe(), "header": snap, "first_result_s": first_frame, "stop_s": stop_s,
           "error": cam.error, "samples": samples}
    with open(os.path.join(OUT_DIR, "camera", f"camera_test_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "samples"}, indent=1))
    return 0 if snap["frames"] > 0 and not cam.error else 1
