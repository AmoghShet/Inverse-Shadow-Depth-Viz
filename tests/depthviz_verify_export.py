"""M5: export + determinism.

    python tests/depthviz_verify_export.py ["C:/Users/Asus/Downloads/Everything In Its Right Place.mp3"]

1. export twice from one depth tape (real camera depth, 12 s) -> frame 300 md5 identical (1080p60)
2. re-run from the first run's resolved_params.json -> same md5
3. synthetic export twice -> same md5
4. 1080p60 synthetic export speed vs real time (frames per wall second > 60)
5. audio and video stream durations in the mp4 match
6. 9:16 export (1080x1920 at 30 fps) works
Report: out/verify/depthviz/report_m5_export.json
"""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depthviz_common import SONG, ensure_tape, out_path, write_report  # noqa: E402

from audioviz.inputs.camera import ffmpeg_exe  # noqa: E402

from depthviz.export.recorder import export, export_from_params  # noqa: E402


def stream_duration(path, sel):
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path, "-map", sel, "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    times = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", r.stderr)
    if not times:
        return None
    h, m, s = times[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def frame_count(path):
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path, "-map", "0:v", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    fr = re.findall(r"frame=\s*(\d+)", r.stderr)
    return int(fr[-1]) if fr else None


def main(song):
    rep = {"song": song, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    tape = ensure_tape("tape_lit_12s", 12.0)
    src = "tape:" + tape
    runs = []
    for k in (1, 2):
        r = export(song, 7, "fog_white", (1920, 1080), 60, out=out_path(f"export_tape_run{k}.mp4"), start=0.0,
                   duration=6.0, source=src, dump_frames=[300], verbose=False)
        runs.append(r)
    rep["tape_md5"] = [r["dumps"].get(300) for r in runs]
    rep["pass_tape_deterministic"] = runs[0]["dumps"].get(300) is not None and len(set(rep["tape_md5"])) == 1
    r3 = export_from_params(runs[0]["params"], out=out_path("export_tape_from_params.mp4"), dump_frames=[300], verbose=False)
    rep["tape_md5_from_params"] = r3["dumps"].get(300)
    rep["pass_params_reproduce"] = rep["tape_md5_from_params"] == rep["tape_md5"][0]
    syn = []
    for k in (1, 2):
        r = export(song, 11, "fog_white", (1280, 720), 24, out=out_path(f"export_syn_run{k}.mp4"), start=30.0,
                   duration=5.0, source="synthetic", dump_frames=[60], verbose=False, look_fps=24)
        syn.append(r["dumps"].get(60))
    rep["synthetic_md5"] = syn
    rep["pass_synthetic_deterministic"] = syn[0] is not None and syn[0] == syn[1]
    r4 = export(song, 3, "fog_white", (1920, 1080), 60, out=out_path("export_speed_1080p60.mp4"), start=60.0,
                duration=20.0, source="synthetic", verbose=False)
    rep["speed_1080p60"] = {"frames": r4["frames"], "wall_s": round(r4["wall_s"], 2), "fps": round(r4["fps_achieved"], 1),
                            "render_loop_s": round(r4["render_s"], 2), "pipe_write_s": round(r4["pipe_write_s"], 2)}
    rep["pass_faster_than_realtime"] = r4["fps_achieved"] > 60.0
    v, a = stream_duration(r4["out"], "0:v"), stream_duration(r4["out"], "0:a")
    rep["durations"] = {"video_s": v, "audio_s": a, "frames": frame_count(r4["out"]), "expected_s": 20.0}
    rep["pass_av_durations_equal"] = v is not None and a is not None and abs(v - a) <= 0.05
    r5 = export(song, 5, "fog_white_portrait", (1080, 1920), 30, out=out_path("export_portrait_1080x1920_30.mp4"),
                start=90.0, duration=8.0, source=src, dump_frames=[120], verbose=False)
    rep["portrait"] = {"ret": r5["ffmpeg_ret"], "fps": round(r5["fps_achieved"], 1), "frames": frame_count(r5["out"])}
    rep["pass_portrait_export"] = r5["ffmpeg_ret"] == 0 and rep["portrait"]["frames"] == 240
    rep["ffmpeg_errors"] = [r["ffmpeg_err"] for r in runs + [r3, r4, r5] if r["ffmpeg_err"]]
    rep["all_pass"] = all(v for k, v in rep.items() if k.startswith("pass_"))
    write_report("report_m5_export.json", rep)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SONG)
