"""Depth Fog command line.

  python -m depthviz play [song.mp3] [--seed N] [--preset fog_white] [--aspect 16:9|9:16]
                          [--source device|synthetic|tape:DIR] [--record] [--tape-out [DIR]] ...
  python -m depthviz export song.mp3 (--tape DIR | --source synthetic) --seed N [--size WxH] [--fps] [--out]
  python -m depthviz export --params out/depthviz/exports/x_resolved_params.json --out again.mp4
  python -m depthviz still [song.mp3] --seed N --t 30 [--source synthetic|tape:DIR] --out frame.png
  python -m depthviz analyze song.mp3
  python -m depthviz check                          GL context, onnxruntime providers, models, MediaPipe
  python -m depthviz fetch-models [--model ID]      download + sha256-verify depth models
  python -m depthviz bench-depth [--quick]          M0 depth benchmark
  python -m depthviz camera-test [--seconds 10] [--save-frames DIR]
"""
import argparse
import sys


def _size(s):
    w, h = (int(v) for v in s.lower().split("x"))
    return w, h


def main(argv=None):
    ap = argparse.ArgumentParser(prog="depthviz", description="Depth Fog: the webcam's depth map as a grainy fog "
                                                               "photograph, driven by a song")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="live window")
    p.add_argument("song", nargs="?")
    p.add_argument("--seed", type=int)
    p.add_argument("--preset", default="fog_white")
    p.add_argument("--aspect", default=None, choices=["16:9", "9:16"], help="default: the preset's meta.aspect")
    p.add_argument("--size", help="window size WxH in screen units (default: fit the monitor)")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--source", default="device", help="device | synthetic | tape:DIR")
    p.add_argument("--camera-device", default=None)
    p.add_argument("--model", default="auto", help="auto (per-platform profile) or a model id")
    p.add_argument("--ep", default=None, choices=["auto", "dml", "coreml", "cpu"], help="default: the profile's")
    p.add_argument("--device-id", type=int, default=None, help="DirectML adapter index (default: the profile's)")
    p.add_argument("--model-size", type=int, default=None, help="model input width (DA-V2: 252 / 336 / 420)")
    p.add_argument("--mask", default="off", choices=["auto", "mp", "depth", "off"],
                   help="person mask to sharpen the silhouette (default off: edges come from the depth itself)")
    p.add_argument("--cam-exposure", default=None, help="one-shot DirectShow exposure (log2 s, e.g. -5) or 'auto'")
    p.add_argument("--look-fps", type=int, choices=[12, 24, 60], help="picture update rate (default: preset)")
    p.add_argument("--record", nargs="?", const="auto", default=None,
                   help="record the window (HUD included) with the song to mp4 (default out/depthviz/recordings/)")
    p.add_argument("--tape-out", nargs="?", const="auto", default=None,
                   help="record the depth per tick to a tape dir (default out/depthviz/tapes/<time>)")
    p.add_argument("--mute", action="store_true", help="run the audio clock but output silence")
    p.add_argument("--hud", action="store_true", help="start with the HUD shown (default hidden; H toggles it)")
    p.add_argument("--no-hud", action="store_true", help=argparse.SUPPRESS)   # the default now; kept for old commands
    p.add_argument("--offset-ms", type=float, default=0.0)
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--debug-view", default="frame", choices=["frame", "depth", "fog", "mask"])
    p.add_argument("--no-vsync", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quit-after", type=float, default=None, help="seconds; for unattended tests")
    p.add_argument("--auto-keys", default="", help="unattended key presses, e.g. '5:S,8:A,12:ctrl+S'")
    p.add_argument("--stats-out", default=None, help="where to write the run stats JSON")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="preset override, e.g. fog.k=3")

    e = sub.add_parser("export", help="offline render to mp4 (deterministic: synthetic source or a depth tape)")
    e.add_argument("song", nargs="?")
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--preset", default="fog_white")
    e.add_argument("--tape", default=None, help="depth tape directory (from play --tape-out)")
    e.add_argument("--source", default="synthetic", help="synthetic | tape:DIR (--tape DIR is the same)")
    e.add_argument("--size", type=_size, default=(1920, 1080))
    e.add_argument("--fps", type=int, default=None, help="file frame rate (default: the look frame rate)")
    e.add_argument("--look-fps", type=int, default=None)
    e.add_argument("--out")
    e.add_argument("--start", type=float, default=0.0)
    e.add_argument("--duration", type=float)
    e.add_argument("--crf", type=int, default=18)
    e.add_argument("--x264-preset", default="veryfast")
    e.add_argument("--dump-frame", type=int, action="append", default=[], help="also save this frame as PNG + md5")
    e.add_argument("--frames-only", action="store_true")
    e.add_argument("--view", default="frame", choices=["frame", "depth", "fog", "mask"])
    e.add_argument("--params", help="re-run from a resolved_params.json")
    e.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    s = sub.add_parser("still", help="render one frame headless")
    s.add_argument("song", nargs="?")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--preset", default="fog_white")
    s.add_argument("--source", default="synthetic", help="synthetic | tape:DIR")
    s.add_argument("--t", type=float, default=20.0, help="song time of the frame")
    s.add_argument("--size", type=_size, default=(1920, 1080))
    s.add_argument("--view", default="frame", choices=["frame", "depth", "fog", "mask"])
    s.add_argument("--out", default="out/depthviz/still.png")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    a = sub.add_parser("analyze", help="analyse a song and cache its features (shared cache with audioviz)")
    a.add_argument("song")

    sub.add_parser("check", help="GL context, onnxruntime providers, depth models, MediaPipe")

    fm = sub.add_parser("fetch-models", help="download + verify depth models into models/depth/")
    fm.add_argument("--model", action="append", default=None, help="model id (repeatable); default: all")

    b = sub.add_parser("bench-depth", help="M0 depth benchmark (see depthviz/depth/bench.py)")
    b.add_argument("rest", nargs=argparse.REMAINDER)

    c = sub.add_parser("camera-test", help="camera + depth worker check (headless)")
    c.add_argument("--seconds", type=float, default=10.0)
    c.add_argument("--device", default=None)
    c.add_argument("--synthetic", action="store_true", help="synthetic dark RGB instead of the webcam")
    c.add_argument("--model", default=None)
    c.add_argument("--ep", default=None)
    c.add_argument("--model-size", type=int, default=None)
    c.add_argument("--mask", default="off", choices=["auto", "mp", "depth", "off"])
    c.add_argument("--cam-exposure", default=None)
    c.add_argument("--save-frames", default=None, metavar="DIR", help="only save raw camera frames (M0 input)")
    c.add_argument("--n-frames", type=int, default=30)

    args = ap.parse_args(argv)

    if args.cmd == "play":
        from .app import App

        app = App(args)
        if args.quit_after:
            import threading

            threading.Timer(args.quit_after, lambda: setattr(app, "quit", True)).start()
        app.run()
    elif args.cmd == "export":
        from .export.recorder import export, export_from_params

        if args.params:
            r = export_from_params(args.params, out=args.out, dump_frames=args.dump_frame, frames_only=args.frames_only)
        else:
            source = f"tape:{args.tape}" if args.tape else args.source
            r = export(args.song, args.seed, args.preset, args.size, args.fps, out=args.out, start=args.start,
                       duration=args.duration, source=source, look_fps=args.look_fps, dump_frames=args.dump_frame,
                       crf=args.crf, x264_preset=args.x264_preset, overrides=args.set, frames_only=args.frames_only,
                       view=args.view)
        return 0 if r["ffmpeg_ret"] == 0 else 1
    elif args.cmd == "still":
        from .tools import still

        still(args)
    elif args.cmd == "analyze":
        from audioviz.audio.analysis import CONTINUOUS, load_or_analyze

        d = load_or_analyze(args.song)
        print(f"tempo {d['tempo']:.1f}  beats {len(d['beats'])}  kicks {len(d['kicks'])}  hats {len(d['hats'])}  "
              f"sections {[round(x, 1) for x in d['sections']]}")
        for k in CONTINUOUS:
            print(f"  {k:>11}: mean {d[k].mean():.2f}")
    elif args.cmd == "check":
        from .tools import check

        return check()
    elif args.cmd == "fetch-models":
        from .depth.models import fetch

        fetch(args.model)
    elif args.cmd == "bench-depth":
        from .depth.bench import main as bench_main

        return bench_main(args.rest)
    elif args.cmd == "camera-test":
        from .inputs.depthcam import camera_test

        return camera_test(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
