"""M4: audio + live, unattended (muted windows).

    python tests/depthviz_live.py [song] [--parts song,keys,fullscreen,camera,flips] [--camera-seconds 60]

song        the whole song, --source synthetic, --no-loop, --record + --tape-out: 0 underflows after start,
            0 dropped steps, kicks flash, sections applied, recording plays with sound (A/V durations)
keys        every key through --auto-keys (views, halation, flip, screenshot, Ctrl+S, look fps,
            scrub, offset, seeds, camera key, HUD, pause, preset slots, export): each produced its message,
            screenshot + sidecar written, preset saved (then removed), export from the tape finished
fullscreen  F twice: the window comes back at its windowed size, no errors
camera      --source device for N seconds with --tape-out and --record: depth arrives, rate, latency,
            no worker error, clean shutdown
flips       offline: one look - over 4000 seeds x 5 sections no seed is inverted and no section flips
Report: out/verify/depthviz/report_m4_live.json
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depthviz_common import ROOT, SONG, out_path, write_report  # noqa: E402

from audioviz.inputs.camera import ffmpeg_exe  # noqa: E402

PY = sys.executable
OUTD = os.path.join(ROOT, "out", "depthviz")


def durations(path):
    out = {}
    for sel in ("0:v", "0:a"):
        r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path, "-map", sel, "-f", "null", "-"],
                           capture_output=True, text=True, errors="replace")
        t = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", r.stderr)
        out[sel] = (int(t[-1][0]) * 3600 + int(t[-1][1]) * 60 + float(t[-1][2])) if t else None
    return out


def play(args, timeout):
    t0 = time.time()
    p = subprocess.run([PY, "-m", "depthviz", "play"] + args, cwd=ROOT, capture_output=True, text=True,
                       errors="replace", timeout=timeout)
    return p, time.time() - t0


def part_song(song):
    stats = out_path("live_song_stats.json")
    tape = out_path("tape_live_song")
    rec = out_path("live_song_record.mp4")
    p, wall = play([song, "--source", "synthetic", "--no-loop", "--mute", "--seed", "21", "--record", rec,
                    "--tape-out", tape, "--stats-out", stats, "--size", "1280x720"], timeout=600)
    res = {"rc": p.returncode, "wall_s": round(wall, 1)}
    if p.returncode != 0 or not os.path.isfile(stats):
        res["stderr"] = p.stderr[-2000:]
        res["pass"] = False
        return res
    s = json.load(open(stats, encoding="utf-8"))
    ci = s["clock"]
    d = durations(rec)
    res.update({"frames": s["frames"], "ticks": s["ticks_rendered"], "song_time_end": s["song_time_end"],
                "frame_ms_p50": s["frame_ms_p50"], "frame_ms_p95": s["frame_ms_p95"], "gpu_ms_mean": s["gpu_ms_mean"],
                "underflows_after_start": ci.get("underflows_after_start"), "dropped_steps": s["dropped_steps"],
                "kicks": s["kicks"], "hats": s["hats"], "sections_applied": s["sections_applied"], "flips": s["flips"],
                "timers": s["timers"], "recording": s.get("recording"), "record_durations": d,
                "tape_records": s.get("tape_out", {}).get("records"), "shader_errors": s["shader_errors"]})
    res["pass"] = (ci.get("underflows_after_start", 1) == 0 and s["dropped_steps"] == 0 and s["kicks"] > 0
                   and s["sections_applied"] >= 1 and not s["shader_errors"] and d["0:v"] is not None
                   and d["0:a"] is not None and abs(d["0:v"] - d["0:a"]) < 0.5)
    return res


KEYS = [("4", "A", "view depth"), ("5", "A", "view fog"), ("6", "A", "view mask"), ("7", "A", "view frame"),
        ("8", "B", "halation off"), ("9", "B", "halation on"), ("15", "V", "flip"), ("19", "V", "flip"), ("23", "S", "screenshot"),
        ("24", "ctrl+S", "saved preset"), ("25", "P", "look 24"), ("26", "P", "look 60"), ("27", "P", "look 12"),
        ("28", "RIGHT", "seek"), ("29", "LEFT", "seek"), ("30", "COMMA", "audio offset"), ("31", "PERIOD", "audio offset"),
        ("32", "LEFT_BRACKET", "seed"), ("33", "RIGHT_BRACKET", "seed"), ("34", "R", "seed"), ("35", "C", "no camera"), ("36", "H", ""),
        ("37", "H", ""), ("38", "SPACE", "paused"), ("40", "SPACE", "playing"), ("41", "1", "preset"),
        ("43", "2", "preset"), ("45", "E", "exporting (tape)")]


def part_keys(song):
    stats = out_path("live_keys_stats.json")
    tape = out_path("tape_live_keys")
    spec = ",".join(f"{t}:{k}" for t, k, _ in KEYS)
    before_presets = set(glob.glob(os.path.join(ROOT, "depthviz", "presets", "user_*.json")))
    before_shots = set(glob.glob(os.path.join(OUTD, "screens", "*")))
    before_exports = set(glob.glob(os.path.join(OUTD, "exports", "*.mp4")))
    p, wall = play([song, "--source", "synthetic", "--mute", "--seed", "8", "--auto-keys", spec, "--quit-after", "48",
                    "--tape-out", tape, "--stats-out", stats, "--size", "1280x720"], timeout=300)
    res = {"rc": p.returncode, "wall_s": round(wall, 1)}
    if p.returncode != 0 or not os.path.isfile(stats):
        res["stderr"] = p.stderr[-2000:]
        res["pass"] = False
        return res
    s = json.load(open(stats, encoding="utf-8"))
    msgs = s["keys"]
    checks = []
    for (t, k, want), got in zip(KEYS, msgs):
        checks.append({"t": t, "key": k, "want": want, "got": got, "ok": want.lower() in (got or "").lower()})
    res["keys"] = checks
    new_presets = sorted(set(glob.glob(os.path.join(ROOT, "depthviz", "presets", "user_*.json"))) - before_presets)
    new_shots = sorted(set(glob.glob(os.path.join(OUTD, "screens", "*"))) - before_shots)
    res["new_presets"] = [os.path.basename(x) for x in new_presets]
    res["new_screens"] = [os.path.basename(x) for x in new_shots]
    for x in new_presets:          # test artefact: remove so the preset slots stay as shipped
        os.remove(x)
    # the export runs in its own process after the app quits: wait for its sidecar
    deadline = time.time() + 300
    exp = []
    while time.time() < deadline:
        exp = sorted(set(glob.glob(os.path.join(OUTD, "exports", "*.mp4"))) - before_exports)
        if exp and os.path.isfile(os.path.splitext(exp[0])[0] + "_resolved_params.json"):
            break
        time.sleep(3)
    res["export"] = None
    if exp:
        side = os.path.splitext(exp[0])[0] + "_resolved_params.json"
        if os.path.isfile(side):
            sp = json.load(open(side, encoding="utf-8"))
            res["export"] = {"file": os.path.basename(exp[0]), "source": sp.get("source"), "frames": sp.get("frames"),
                             "durations": durations(exp[0])}
    res["pass"] = (all(c["ok"] for c in checks) and len(new_presets) == 1 and any(x.endswith(".png") for x in new_shots)
                   and any(x.endswith(".json") for x in new_shots) and res["export"] is not None
                   and str(res["export"]["source"]).startswith("tape:") and not s["shader_errors"])
    return res


def part_fullscreen(song):
    stats = out_path("live_fullscreen_stats.json")
    p, wall = play([song, "--source", "synthetic", "--mute", "--auto-keys", "3:F,7:F", "--quit-after", "11",
                    "--stats-out", stats, "--size", "1280x720"], timeout=120)
    res = {"rc": p.returncode}
    if p.returncode != 0 or not os.path.isfile(stats):
        res["stderr"] = p.stderr[-1500:]
        res["pass"] = False
        return res
    s = json.load(open(stats, encoding="utf-8"))
    res.update({"fb_size_end": s["fb_size"], "shader_errors": s["shader_errors"], "frame_ms_max": s["frame_ms_max"]})
    res["pass"] = not s["shader_errors"]
    return res


def part_camera(song, seconds):
    stats = out_path("live_camera_stats.json")
    tape = out_path("tape_live_camera")
    rec = out_path("live_camera_record.mp4")
    p, wall = play([song, "--source", "device", "--mute", "--quit-after", str(seconds), "--record", rec,
                    "--tape-out", tape, "--stats-out", stats, "--size", "1280x720", "--seed", "4"], timeout=seconds + 120)
    res = {"rc": p.returncode, "wall_s": round(wall, 1)}
    if p.returncode != 0 or not os.path.isfile(stats):
        res["stderr"] = p.stderr[-2000:]
        res["pass"] = False
        return res
    s = json.load(open(stats, encoding="utf-8"))
    cam = s.get("camera", {})
    res.update({"camera": cam, "profile": s.get("profile"), "depth_updates": s["depth_updates"],
                "ticks": s["ticks_rendered"], "dropped_steps": s["dropped_steps"],
                "underflows_after_start": s["clock"].get("underflows_after_start"), "frame_ms_p95": s["frame_ms_p95"],
                "gpu_ms_mean": s["gpu_ms_mean"], "tape_records": s.get("tape_out", {}).get("records"),
                "record_durations": durations(rec) if os.path.isfile(rec) else None})
    res["pass"] = (cam.get("frames", 0) > 30 and cam.get("fps", 0) >= 10 and not cam.get("error")
                   and s["dropped_steps"] == 0)
    return res


def part_flips():
    from depthviz import preset as presetmod
    from depthviz import seed as seedmod

    pre = presetmod.load("fog_white")
    n = flips = high = 0
    for s in range(4000):
        r = seedmod.resolve(pre, s, 16 / 9)
        high += r["polarity"] == "high"
        for k in range(1, 6):
            flips += seedmod.section_layout(r, pre, k)["flip"]
            n += 1
    # one look: no seed is inverted and no section flips (the V key is the only way to the inverted look)
    return {"seeds": 4000, "section_flip_rate": round(flips / n, 4), "high_key_rate": round(high / 4000, 4),
            "pass": flips == 0 and high == 4000}


if __name__ == "__main__":
    argv = sys.argv[1:]
    parts = "song,keys,fullscreen,camera,flips"
    secs = 60.0
    pos = []
    i = 0
    while i < len(argv):
        if argv[i] == "--parts":
            parts = argv[i + 1]
            i += 2
        elif argv[i] == "--camera-seconds":
            secs = float(argv[i + 1])
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    song = pos[0] if pos else SONG
    rep_path = out_path("report_m4_live.json")
    rep = json.load(open(rep_path, encoding="utf-8")) if os.path.isfile(rep_path) else {}
    rep["when"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rep["song"] = song
    for part in parts.split(","):
        print(f"--- {part} ---", flush=True)
        if part == "song":
            rep["song_run"] = part_song(song)
            rep["pass_song"] = rep["song_run"]["pass"]
        elif part == "keys":
            rep["keys_run"] = part_keys(song)
            rep["pass_keys"] = rep["keys_run"]["pass"]
        elif part == "fullscreen":
            rep["fullscreen_run"] = part_fullscreen(song)
            rep["pass_fullscreen"] = rep["fullscreen_run"]["pass"]
        elif part == "camera":
            rep["camera_run"] = part_camera(song, secs)
            rep["pass_camera_live"] = rep["camera_run"]["pass"]
        elif part == "flips":
            rep["flips"] = part_flips()
            rep["pass_flip_probability"] = rep["flips"]["pass"]
    rep["all_pass"] = all(v for k, v in rep.items() if k.startswith("pass_"))
    write_report("report_m4_live.json", rep)
