"""Depth inference (session / execution-provider selection, pre- and post-processing) and the
per-frame normaliser that turns relative inverse depth into a stable 0..1 depth (0 = near, 1 = far).

Execution providers (`ep`): "dml" (DirectML, Windows), "coreml" (macOS), "cpu", or "auto"
(dml on Windows, coreml on macOS, cpu elsewhere; always with CPU as the fallback provider).
"""
import os
import sys
import time

import cv2
import numpy as np

from . import models as M

OUT_SIZE = (320, 240)   # what the worker publishes (4:3, camera space, mirrored)


def physical_cores():
    n = os.cpu_count() or 4
    return max(1, n // 2) if n >= 4 else n


def resolve_ep(ep="auto"):
    if ep != "auto":
        return ep
    if sys.platform.startswith("win"):
        return "dml"
    if sys.platform == "darwin":
        return "coreml"
    return "cpu"


def make_session(model_id, ep="auto", device_id=0, threads=None, opt=None, input_wh=None):
    """onnxruntime InferenceSession for a model on an EP. Returns (session, info dict).

    Dynamic-shape models (DA-V2) get free-dimension overrides for the chosen input size: with the
    dims left free, DirectML runs DA-V2 small at 252x196 in ~145 ms on a GTX 1650 Ti; pinned, 26 ms
    (M0, 2026-10-14)."""
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    path = M.ensure(model_id)
    spec = M.MODELS[model_id]
    ep = resolve_ep(ep)
    levels = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
              "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}
    tries = [opt or spec["opt"]]
    if tries[0] == "all":
        tries.append("basic")
    avail = ort.get_available_providers()
    last_err = None
    for lvl in tries:
        so = ort.SessionOptions()
        so.graph_optimization_level = levels[lvl]
        so.intra_op_num_threads = int(threads or physical_cores())
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if input_wh is not None and not spec["fixed"]:
            so.add_free_dimension_override_by_name("batch_size", 1)
            so.add_free_dimension_override_by_name("height", int(input_wh[1]))
            so.add_free_dimension_override_by_name("width", int(input_wh[0]))
        if ep == "dml":
            so.enable_mem_pattern = False   # required by the DirectML EP
            if str(device_id) == "hp":   # let DirectML pick the high-performance GPU
                dml_opts = {"performance_preference": "high_performance", "device_filter": "gpu"}
            else:
                dml_opts = {"device_id": int(device_id)}
            providers = [("DmlExecutionProvider", dml_opts), "CPUExecutionProvider"]
            if "DmlExecutionProvider" not in avail:
                raise RuntimeError(f"DirectML EP not available (have {avail}); install onnxruntime-directml")
        elif ep == "coreml":
            providers = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
                         "CPUExecutionProvider"]
            if "CoreMLExecutionProvider" not in avail:
                raise RuntimeError(f"CoreML EP not available (have {avail})")
        else:
            providers = ["CPUExecutionProvider"]
        t0 = time.perf_counter()
        try:
            sess = ort.InferenceSession(path, so, providers=providers)
        except Exception as e:   # fp16 exports: fusion bug at "all"; retry at "basic"
            last_err = e
            continue
        info = {"model": model_id, "ep": ep, "device_id": device_id, "opt": lvl, "fixed_dims": input_wh is not None,
                "providers": sess.get_providers(), "build_s": round(time.perf_counter() - t0, 3),
                "threads": so.intra_op_num_threads}
        return sess, info
    raise RuntimeError(f"could not build a session for {model_id} on {ep}: {last_err}")


class DepthModel:
    """rgb uint8 (H, W, 3) -> relative inverse depth float32 at OUT_SIZE (larger = nearer)."""

    def __init__(self, model_id="dav2_small", ep="auto", size=None, device_id=0, threads=None):
        self.model_id = model_id
        self.spec = M.MODELS[model_id]
        self.in_w, self.in_h = M.input_size(model_id, size)
        self.sess, self.info = make_session(model_id, ep, device_id, threads, input_wh=(self.in_w, self.in_h))
        self.inp = self.sess.get_inputs()[0].name
        self.mean = np.array(M.MEAN, np.float32).reshape(1, 1, 3)
        self.inv_std = (1.0 / np.array(M.STD, np.float32)).reshape(1, 1, 3)
        self.last_ms = 0.0
        self.info.update({"input": [self.in_w, self.in_h]})

    def preprocess(self, rgb):
        h, w = rgb.shape[:2]
        interp = cv2.INTER_AREA if (w > self.in_w or h > self.in_h) else cv2.INTER_LINEAR
        x = cv2.resize(rgb, (self.in_w, self.in_h), interpolation=interp).astype(np.float32) * (1.0 / 255.0)
        x = (x - self.mean) * self.inv_std
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def run(self, x):
        t0 = time.perf_counter()
        y = self.sess.run(None, {self.inp: x})[0]
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return y.reshape(y.shape[-2], y.shape[-1])

    def __call__(self, rgb, out_size=OUT_SIZE):
        z = self.run(self.preprocess(rgb))
        interp = cv2.INTER_AREA if (z.shape[1] > out_size[0]) else cv2.INTER_LINEAR
        return cv2.resize(z.astype(np.float32), out_size, interpolation=interp)


def _ema_k(dt, tau):
    return 1.0 - float(np.exp(-max(dt, 0.0) / max(tau, 1e-4)))


class Normaliser:
    """Plan step 1 (worker side). Removes the per-frame scale/shift ambiguity of relative depth:

    z' = a z + b, (a, b) matching the mean and std of z onto a running reference z_ref (4x subsampled; the plan's
    least squares collapsed the range, see __call__), z_ref <- EMA(z', 0.5 s);
    robust bounds lo/hi = p2/p98 of z' with an asymmetric EMA (0.3 s widening, 2 s narrowing);
    d = 1 - clamp((z' - lo) / (hi - lo), 0, 1): 0 = near, 1 = far. fig_depth = median d inside the mask.
    """

    def __init__(self, tau_ref=0.5, tau_widen=0.3, tau_narrow=2.0, sub=4, min_range=1e-3):
        self.tau_ref, self.tau_widen, self.tau_narrow, self.sub = tau_ref, tau_widen, tau_narrow, sub
        self.min_range = min_range
        self.reset()

    def reset(self):
        self.ref = None
        self.lo = None
        self.hi = None
        self.a, self.b = 1.0, 0.0
        self.fig_depth = 0.5
        self.mask_cov = 0.0

    def __call__(self, z, dt, mask=None):
        s = self.sub
        zs = z[::s, ::s].astype(np.float64)
        # first frame (or after a reset): normalise to its own robust range so the reference is 0..1
        if self.ref is None:
            p2, p98 = np.percentile(zs, (2, 98))
            rng = max(p98 - p2, 1e-6)
            self.a, self.b = 1.0 / rng, -p2 / rng
            self.ref = zs * self.a + self.b
            self.ref_mean, self.ref_std = float(self.ref.mean()), max(float(self.ref.std()), 1e-6)
        else:
            # Moment matching (mean and std of this frame onto the reference), NOT the plan's least squares:
            # a least-squares slope is attenuated by noise (regression dilution), and feeding it back through
            # the reference EMA shrank the depth range every frame (live: bounds 0.00/0.88 -> 0.25/0.30 in
            # 20 s, the depth turned into a flickering threshold). The correlation still guards bad frames.
            x = zs.ravel()
            y = self.ref.ravel()
            xm, ym = x.mean(), y.mean()
            sx, sy = x.std(), y.std()
            corr = float(((x - xm) * (y - ym)).mean() / max(sx * sy, 1e-12))
            if sx > 1e-9 and corr > 0.3:
                a = sy / sx
                b = ym - a * xm
            else:            # degenerate frame (black, pure noise, a cut): keep the previous mapping
                a, b = self.a, self.b
            self.a, self.b = float(a), float(b)
            k = _ema_k(dt, self.tau_ref)
            self.ref = self.ref + (zs * self.a + self.b - self.ref) * k
            # anchor the reference's mean / std to the first frame's: an EMA of frames with moving content
            # loses spatial variance, and the alignment then shrank the range every frame (live: a -> 0.02)
            m, sd = float(self.ref.mean()), float(self.ref.std())
            if sd > 1e-9:
                self.ref = (self.ref - m) * (self.ref_std / sd) + self.ref_mean
        zp = z * np.float32(self.a) + np.float32(self.b)
        lo, hi = np.percentile(zp[::s, ::s], (2, 98))
        if self.lo is None:
            self.lo, self.hi = float(lo), float(hi)
        else:
            kw, kn = _ema_k(dt, self.tau_widen), _ema_k(dt, self.tau_narrow)
            self.lo += (lo - self.lo) * (kw if lo < self.lo else kn)
            self.hi += (hi - self.hi) * (kw if hi > self.hi else kn)
        rng = max(self.hi - self.lo, self.min_range)
        d = 1.0 - np.clip((zp - np.float32(self.lo)) / np.float32(rng), 0.0, 1.0)
        d = d.astype(np.float32)
        if mask is not None:
            m = mask > (127 if mask.dtype == np.uint8 else 0.5)
            self.mask_cov = float(m.mean())
            if self.mask_cov > 0.01:
                self.fig_depth = float(np.median(d[m]))
        return d


def gain_frame(rgb, state, dt, tau=1.0, gamma=1.6, max_gain=24.0, min_range=0.04):
    """Dark-room gain: percentile stretch (p1/p99, EMA tau 1 s) + gamma 1/1.6, capped at 24x.

    `state` is a dict carried between frames. Returns (gained uint8 rgb, info dict)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lo, hi = np.percentile(gray[::2, ::2], (1, 99))
    lo, hi = float(lo) / 255.0, float(hi) / 255.0
    if "lo" not in state:
        state["lo"], state["hi"] = lo, hi
    else:
        k = _ema_k(dt, tau)
        state["lo"] += (lo - state["lo"]) * k
        state["hi"] += (hi - state["hi"]) * k
    lo, hi = state["lo"], state["hi"]
    rng = max(hi - lo, min_range, 1.0 / max_gain)
    lut = state.get("lut")
    key = (round(lo, 4), round(rng, 4))
    if lut is None or state.get("lut_key") != key:
        x = np.arange(256, dtype=np.float32) / 255.0
        y = np.clip((x - lo) / rng, 0.0, 1.0) ** (1.0 / gamma)
        lut = (y * 255.0 + 0.5).astype(np.uint8)
        state["lut"], state["lut_key"] = lut, key
    out = cv2.LUT(rgb, lut)
    return out, {"brightness": float(gray.mean()), "gain": 1.0 / rng, "lo": lo, "hi": hi}
