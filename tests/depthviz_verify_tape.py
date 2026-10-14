"""M1: worker + plumbing.

    python tests/depthviz_verify_tape.py [--no-camera]

1. tape round trip (unit): 10 s of the synthetic source at 12 fps written with a 5-tick gap and one
   backwards record -> every tick reads back tick-exact (depth within uint16 quantisation, mask exact),
   gap ticks hold the previous record, the backwards record is ignored, 64-record chunks
2. live recording: `play --source synthetic --tape-out` for ~10 s, read back: every record equals the
   synthetic source at its own tick time (tick-exact), times = tick / look_fps, ticks contiguous
3. camera (skipped with --no-camera): `camera-test` three times in a row -> frames arrive every time
   (no wedge), cam fps, depth fps / ms, gain, mask coverage reported, clean stop < 1 s
Report: out/verify/depthviz/report_m1_tape.json
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depthviz_common import OUT, ROOT, out_path, write_report  # noqa: E402

from depthviz.inputs.synthetic import Synthetic  # noqa: E402
from depthviz.inputs.tape import TapeReader, TapeWriter  # noqa: E402

PY = sys.executable
Q = 1.0 / 65535.0


def unit_roundtrip():
    d = out_path("tape_unit")
    shutil.rmtree(d, ignore_errors=True)
    syn = Synthetic(flicker=0.01, seed=3)
    w = TapeWriter(d, {"look_fps": 12, "source": "unit"})
    written = {}
    for tick in range(120):
        if 50 <= tick < 55:
            continue
        t = tick / 12.0
        depth, mask = syn.frame(t)
        w.write(tick, t, depth, mask, np.full(32, tick, np.float64))
        written[tick] = (depth, mask)
        if tick == 80:
            back_ok = not w.write(79, 79 / 12.0, depth, mask)   # backwards: ignored
    w.close()
    r = TapeReader(d)
    errs, mask_ok, hold_ok = [], True, True
    for tick in range(120):
        got = r.at_tick(tick, 12)
        src_tick = tick if tick in written else max(k for k in written if k < tick)
        depth, mask = written[src_tick]
        errs.append(float(np.abs(got[0] - depth).max()))
        mask_ok &= bool(np.array_equal(got[1], mask))
        hold_ok &= int(got[2][0]) == src_tick
    return {"records": r.records, "chunks": len(r.chunks), "max_depth_err": max(errs), "mask_exact": mask_ok,
            "gap_holds_previous": hold_ok, "backwards_ignored": back_ok, "skipped_backwards": r.meta.get("skipped_backwards"),
            "pass": max(errs) <= Q * 0.5 + 1e-7 and mask_ok and hold_ok and back_ok and r.records == 115
            and len(r.chunks) == 2}


def live_recording(seconds=10.0):
    d = out_path("tape_live_synthetic")
    shutil.rmtree(d, ignore_errors=True)
    stats = out_path("live_tape_stats.json")
    cmd = [PY, "-m", "depthviz", "play", "--source", "synthetic", "--tape-out", d, "--quit-after", str(seconds + 1.0),
           "--mute", "--no-hud", "--size", "640x360", "--stats-out", stats, "--seed", "5"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=180)
    if p.returncode != 0:
        return {"pass": False, "rc": p.returncode, "stderr": p.stderr[-1500:]}
    r = TapeReader(d)
    syn = Synthetic()
    lf = r.look_fps
    ticks, times, errs = [], [], []
    for ci, path in enumerate(r.chunks):
        with np.load(path) as z:
            for k in range(len(z["ticks"])):
                tick, t = int(z["ticks"][k]), float(z["times"][k])
                ticks.append(tick)
                times.append(t)
                ref, refm = syn.frame(tick / lf)
                errs.append(float(np.abs(z["depth"][k].astype(np.float32) * Q - ref).max()))
    ticks = np.array(ticks)
    contiguous = bool(np.all(np.diff(ticks) == 1))
    times_ok = bool(np.allclose(np.array(times), ticks / lf))
    return {"records": r.records, "first_tick": int(ticks[0]), "last_tick": int(ticks[-1]), "look_fps": lf,
            "contiguous_ticks": contiguous, "gaps": int(np.sum(np.diff(ticks) > 1)), "times_are_tick_over_fps": times_ok,
            "max_err_vs_source": max(errs),
            "pass": r.records >= seconds * lf * 0.9 and max(errs) <= Q * 0.5 + 1e-7 and times_ok}


def camera_runs(n=3, seconds=15.0):
    runs = []
    rep_path = os.path.join(ROOT, "out", "depthviz", "camera", "camera_test_device.json")
    for i in range(n):
        t0 = time.time()
        p = subprocess.run([PY, "-m", "depthviz", "camera-test", "--seconds", str(seconds)], cwd=ROOT,
                           capture_output=True, text=True, errors="replace", timeout=120)
        try:
            if os.path.getmtime(rep_path) < t0:
                raise FileNotFoundError("report not refreshed")
            with open(rep_path, encoding="utf-8") as fh:
                rep = json.load(fh)
        except Exception as e:
            runs.append({"run": i + 1, "pass": False, "error": str(e), "stderr": p.stderr[-800:]})
            continue
        h = rep["header"]
        runs.append({"run": i + 1, "frames": h["frames"], "cam_fps": h["cam_fps"], "depth_fps": h["fps"],
                     "depth_ms": h["depth_ms"], "mask_ms": h["mask_ms"], "proc_ms": h["proc_ms"],
                     "brightness": h["brightness"], "gain": h["gain"], "mask_cov": h["mask_cov"],
                     "latency_ms": h["latency_ms"], "first_frame_s": h["first_frame_s"], "session_s": h["session_s"],
                     "stop_s": rep["stop_s"], "error": rep["error"], "describe": rep["describe"],
                     "pass": h["frames"] > 10 and rep["stop_s"] < 1.0 and not rep["error"]})
        time.sleep(1.0)
    return runs


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rep = {"when": time.strftime("%Y-%m-%d %H:%M:%S")}
    rep["unit"] = unit_roundtrip()
    rep["pass_tape_roundtrip"] = rep["unit"]["pass"]
    rep["live"] = live_recording()
    rep["pass_live_tape_tick_exact"] = rep["live"]["pass"]
    if "--no-camera" not in sys.argv:
        rep["camera"] = camera_runs()
        rep["pass_camera_3_opens"] = all(r["pass"] for r in rep["camera"])
    rep["all_pass"] = all(v for k, v in rep.items() if k.startswith("pass_"))
    write_report("report_m1_tape.json", rep)
