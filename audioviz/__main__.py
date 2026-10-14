"""ASCII Ink Visualizer command line.

  python -m audioviz play [song.mp3] [--seed N] [--preset ink_cold] [--aspect 16:9|9:16] [--camera]
  python -m audioviz export song.mp3 --seed N --size 1920x1080 [--fps 24] --out file.mp4
  python -m audioviz export --params out/exports/x_resolved_params.json --out again.mp4
  python -m audioviz still [song.mp3] --seed N --t 30 --size 1920x1080 --out frame.png
  python -m audioviz analyze song.mp3
  python -m audioviz bench [--seconds 8]           per-pass GPU timings for every tier
  python -m audioviz check                         M0 context check
  python -m audioviz camera-test [--seconds 10]    M7 camera plumbing check
"""
import argparse
import sys


def _size(s):
    w, h = (int(v) for v in s.lower().split("x"))
    return w, h


def main(argv=None):
    ap = argparse.ArgumentParser(prog="audioviz", description="ASCII Ink Visualizer: an ASCII/fluid-based, audio-reactive visualizer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="live window")
    p.add_argument("song", nargs="?")
    p.add_argument("--seed", type=int)
    p.add_argument("--preset", default="ink_cold")
    p.add_argument("--aspect", default=None, choices=["16:9", "9:16"], help="default: the preset's meta.aspect, else 16:9")
    p.add_argument("--size", help="window size WxH in screen units (default: fit the monitor)")
    p.add_argument("--tier", choices=["high", "mid", "low", "min"])
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--offset-ms", type=float, default=0.0, help="audio/visual offset")
    p.add_argument("--camera", nargs="?", const="device", choices=["device", "synthetic"],
                   help="camera stir (M7); 'synthetic' uses a generated moving blob instead of the webcam")
    p.add_argument("--camera-device", default=None)
    p.add_argument("--click-test", action="store_true", help="flash on every beat to check sync by ear")
    p.add_argument("--debug-view", default="ascii", choices=["ascii", "density", "flow"])
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--record", nargs="?", const="auto", default=None,
                   help="record the window (HUD included) with the song to mp4 (default out/recordings/)")
    p.add_argument("--mute", action="store_true", help="run the audio clock but output silence")
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--verbose", action="store_true", help="log clock resyncs and dropped steps")
    p.add_argument("--tier-drop-ms", type=float, help="override the GPU p95 that triggers a tier drop (testing)")
    p.add_argument("--look-fps", type=int, choices=[12, 24, 60], help="picture update rate (default: preset grain.frame_rate)")
    p.add_argument("--no-vsync", action="store_true")
    p.add_argument("--quit-after", type=float, default=None, help="seconds; for unattended tests")
    p.add_argument("--auto-keys", default="", help="unattended key presses, e.g. '5:S,8:A,9:A,12:ctrl+S'")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="preset override, e.g. sim.splat=0.004")

    e = sub.add_parser("export", help="offline render to mp4")
    e.add_argument("song", nargs="?")
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--preset", default="ink_cold")
    e.add_argument("--size", type=_size, default=(1920, 1080))
    e.add_argument("--fps", type=int, default=None, help="file frame rate (default: the look frame rate)")
    e.add_argument("--look-fps", type=int, default=None, help="picture update rate; frames are held in between")
    e.add_argument("--out")
    e.add_argument("--start", type=float, default=0.0)
    e.add_argument("--duration", type=float)
    e.add_argument("--tier", default="high", choices=["high", "mid", "low", "min"])
    e.add_argument("--crf", type=int, default=18)
    e.add_argument("--x264-preset", default="veryfast")
    e.add_argument("--dump-frame", type=int, action="append", default=[], help="also save this frame as PNG + md5")
    e.add_argument("--frames-only", action="store_true", help="no mp4, only --dump-frame PNGs")
    e.add_argument("--params", help="re-run from a resolved_params.json")
    e.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    s = sub.add_parser("still", help="render one frame headless")
    s.add_argument("song", nargs="?")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--preset", default="ink_cold")
    s.add_argument("--t", type=float, default=20.0, help="song time of the frame")
    s.add_argument("--size", type=_size, default=(1920, 1080))
    s.add_argument("--view", default="ascii", choices=["ascii", "density", "flow"])
    s.add_argument("--tier", default="high", choices=["high", "mid", "low", "min"])
    s.add_argument("--out", default="out/still.png")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    a = sub.add_parser("analyze", help="analyse a song and cache its features")
    a.add_argument("song")

    b = sub.add_parser("bench", help="per-pass GPU timings")
    b.add_argument("--seconds", type=float, default=6.0)
    b.add_argument("--size", type=_size, default=(1920, 1080))
    b.add_argument("--tiers", default="high,mid,low")

    sub.add_parser("check", help="M0 GL context check")

    c = sub.add_parser("camera-test", help="M7 camera plumbing check")
    c.add_argument("--seconds", type=float, default=10.0)
    c.add_argument("--device", default=None)
    c.add_argument("--synthetic", action="store_true")

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
            if not args.song:
                ap.error("export needs a song (or --params)")
            r = export(args.song, args.seed, args.preset, args.size, args.fps, out=args.out, start=args.start,
                       duration=args.duration, tier=args.tier, dump_frames=args.dump_frame, crf=args.crf,
                       x264_preset=args.x264_preset, overrides=args.set, frames_only=args.frames_only,
                       look_fps=args.look_fps)
        return 0 if r["ffmpeg_ret"] == 0 else 1
    elif args.cmd == "still":
        from .tools import still

        still(args)
    elif args.cmd == "analyze":
        from .audio.analysis import CONTINUOUS, load_or_analyze

        d = load_or_analyze(args.song)
        print(f"tempo {d['tempo']:.1f}  beats {len(d['beats'])}  kicks {len(d['kicks'])}  hats {len(d['hats'])}  "
              f"sections {[round(x, 1) for x in d['sections']]}")
        for k in CONTINUOUS:
            print(f"  {k:>11}: mean {d[k].mean():.2f}")
    elif args.cmd == "bench":
        from .tools import bench

        bench(args)
    elif args.cmd == "check":
        from .gl.context import _m0_check

        return _m0_check()
    elif args.cmd == "camera-test":
        from .inputs.camera import camera_test

        return camera_test(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
