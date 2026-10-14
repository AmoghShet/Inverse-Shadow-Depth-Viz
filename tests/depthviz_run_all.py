"""Run every depthviz milestone check and collect one summary.

    python tests/depthviz_run_all.py [song] [--no-camera]

Each check runs as a subprocess; its JSON report is read back from out/verify/depthviz/. Writes
out/verify/depthviz/summary.json and prints a table. Live checks open muted windows; the camera parts use the webcam.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
OUT = os.path.join(ROOT, "out", "verify", "depthviz")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
SONG = args[0] if args else "C:/Users/Asus/Downloads/Everything In Its Right Place.mp3"
NO_CAM = "--no-camera" in sys.argv

CHECKS = [
    ("check", [PY, "-m", "depthviz", "check"], None),
    ("M1 tape + camera", [PY, "tests/depthviz_verify_tape.py"] + (["--no-camera"] if NO_CAM else []), "report_m1_tape.json"),
    ("M2 fog", [PY, "tests/depthviz_verify_fog.py"], "report_m2_fog.json"),
    ("M3 look", [PY, "tests/depthviz_verify_look.py"], "report_m3_look.json"),
    ("M4 live", [PY, "tests/depthviz_live.py", SONG, "--parts",
                 "song,keys,fullscreen,flips" + ("" if NO_CAM else ",camera")], "report_m4_live.json"),
    ("M5 export", [PY, "tests/depthviz_verify_export.py", SONG], "report_m5_export.json"),
]


def passes(report):
    return {k: v for k, v in report.items() if k.startswith("pass") and isinstance(v, bool)}


def main():
    os.makedirs(OUT, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    results = []
    for name, cmd, report in CHECKS:
        t0 = time.time()
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace")
        entry = {"check": name, "rc": p.returncode, "seconds": round(time.time() - t0)}
        if report:
            path = os.path.join(OUT, report)
            try:
                if os.path.getmtime(path) < t0:
                    raise FileNotFoundError("report not refreshed")
                with open(path, encoding="utf-8") as fh:
                    entry["passes"] = passes(json.load(fh))
            except Exception as e:
                entry["passes"], entry["error"] = {}, str(e)
        else:
            entry["passes"] = {"pass": p.returncode == 0}
        if p.returncode != 0:
            entry["stderr_tail"] = (p.stderr or p.stdout)[-1500:]
        entry["ok"] = p.returncode == 0 and bool(entry["passes"]) and all(entry["passes"].values())
        results.append(entry)
        failed = [k for k, v in entry["passes"].items() if not v]
        print(f"{'PASS' if entry['ok'] else 'FAIL'}  {name:<18} {entry['seconds']:>4}s  {len(entry['passes'])} checks"
              + (f"  failed: {failed}" if failed else "") + (f"  rc={p.returncode}" if p.returncode else ""), flush=True)
    summary = {"started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "song": SONG,
               "all_ok": all(r["ok"] for r in results), "results": results}
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("ALL PASS" if summary["all_ok"] else "SOME CHECKS FAILED", "->", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
