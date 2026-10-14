# Inverse Shadow Depth Viz

The webcam's depth map as a grainy fog photograph, driven by a song. It has one look, the reference photographs.
A monocular depth model turns every camera frame into relative depth, and the depth maps straight to tone:
- **Near:** things are dark gray.
- **Further:** things get lighter.
- **The room:** dissolves into white haze.

On top of that come soft edges, heavy monochrome film grain and gentle halation. The song moves the look but
never switches it: fog density, exposure, light flashes on kicks and the grain amount. A seed only changes
details inside the look: where the soft light behind you sits, and small amounts of fog, grain and blur.

The app is the `depthviz` package. `audioviz/` is included because `depthviz` reuses part of the ASCII Ink
Visualizer unchanged: the GL context, audio analysis and player, HUD text atlas and live recorder.

| Path | Contents |
|---|---|
| `depthviz/` | the app: camera + depth worker, fog / film shaders, director, live app, exporter |
| `audioviz/` | shared infrastructure from the ASCII Ink Visualizer (unchanged) |
| `tests/depthviz_*.py` | milestone checks (`tests/depthviz_run_all.py` runs them all) |
| `docs/PLAN_depthviz.md` | the original build plan |
| `docs/bench/` | the M0 depth-model benchmark on a GTX 1650 Ti laptop |
| `NOTES_depthviz_log.md` | timestamped build log: every decision and deviation from the plan |

## Setup

You need Python 3.12 and a webcam. Windows uses DirectML for the depth model on any DirectX 12 GPU; macOS uses
CoreML, which is untested so far.

```bash
git clone https://github.com/AmoghShet/Inverse-Shadow-Depth-Viz.git
```

```bash
cd Inverse-Shadow-Depth-Viz
```

```bash
python -m uv venv --python 3.12 .venv
```

```bash
python -m uv pip install --python .venv/Scripts/python.exe -r requirements-depthviz.txt
```

```bash
.venv/Scripts/python.exe -m depthviz fetch-models
```

`fetch-models` downloads MiDaS v2.1 small and Depth Anything V2 small (fp32 / fp16 / int8 / q4f16) into
`models/depth/` and checks their sha256. The runtime never downloads anything. Without uv, the usual
`py -3.12 -m venv .venv` and `.venv/Scripts/python.exe -m pip install -r requirements-depthviz.txt` work too.
On macOS use `.venv/bin/python`.

The optional person mask (`--mask mp`) also needs MediaPipe's `selfie_segmenter_landscape.tflite` in `models/`.
The mask is off by default.

Check the install. It reports the GL context, the onnxruntime providers, the models and the depth profile picked
for this machine:

```bash
.venv/Scripts/python.exe -m depthviz check
```

## Run

```bash
.venv/Scripts/python.exe -m depthviz play "path/to/song.mp3"
```

Useful options:

| Option | What it does |
|---|---|
| `--seed N` | a fixed seed (default random) |
| `--aspect 9:16` or `--preset fog_white_portrait` | portrait |
| `--source synthetic` | a generated moving figure instead of the camera |
| `--source tape:DIR` | replay a recorded depth tape |
| `--tape-out [DIR]` | record the depth per picture tick (makes a camera run exportable and reproducible) |
| `--record [FILE]` | record the window with the song |
| `--hud` | start with the HUD shown (hidden by default; H toggles it) |
| `--debug-view depth\|fog\|mask` | start in a debug view |
| `--model`, `--ep`, `--device-id`, `--model-size` | override the per-machine depth profile |
| `--mask off\|auto\|mp\|depth` | default off (edges come from the depth); a person mask sharpens the silhouette |
| `--cam-exposure -5` | one-shot manual camera exposure (Windows, DirectShow log2 seconds) |
| `--look-fps 12\|24\|60` | picture rate; the default 12 holds frames like the ink app |
| `--set fog.k=3.5` | override any preset value |

The camera opens in 3-5 s. Until the first depth arrives the screen shows empty fog (the HUD, if switched on with H, says "opening...").
With the room lights off and only the monitor lighting you, the worker's gain stretches the dark frame before the
depth model. A truly black frame (lens covered) holds the last depth instead of drawing noise.

## Keys

| Key | Action |
|---|---|
| R, [ , ] | new random seed, seed -1, seed +1 |
| Space | pause |
| S | screenshot + `resolved_params` sidecar (out/depthviz/screens) |
| Ctrl+S | freeze the current look as `presets/user_NN.json` |
| A | view: frame / depth / fog / mask (debug) |
| B | halation on/off |
| H | HUD on/off (hidden by default) |
| F | fullscreen |
| P | look fps 12 / 24 / 60 |
| 1-9 | preset slots |
| Left / Right | scrub 5 s |
| , . | audio offset -/+ 10 ms |
| E | export to mp4 in the background (from the current tape with `--tape-out`, else synthetic) |
| C | camera on/off (off holds the last depth) |
| V | flip to the inverted look and back by hand (it never happens on its own) |
| Esc | quit |

## Export

Exports are deterministic: the depth comes from a tape or from the synthetic figure, never from a live camera.

```bash
.venv/Scripts/python.exe -m depthviz play SONG --tape-out out/depthviz/tapes/take1
```

```bash
.venv/Scripts/python.exe -m depthviz export SONG --tape out/depthviz/tapes/take1 --seed 42 --size 1920x1080 --fps 60
```

```bash
.venv/Scripts/python.exe -m depthviz export --params out/depthviz/exports/x_resolved_params.json --out again.mp4
```

A tape is one forward pass through the song (about 100-200 MB for 3 minutes). Ticks with no new depth hold the
previous one; scrubbing back or a song loop stops adding records until time passes the last one.

## Other commands

| Command | Purpose |
|---|---|
| `still [SONG] --seed N --t 30 --source synthetic\|tape:DIR --view frame --out f.png` | one frame, headless |
| `camera-test [--seconds 10] [--synthetic] [--save-frames DIR]` | the camera + depth worker on its own |
| `bench-depth [--eps dml:0,cpu:0] [--quick]` | the M0 model / execution-provider benchmark |
| `analyze SONG` | analyse and cache a song's features |

## Tests

| Script | Milestone |
|---|---|
| `python -m depthviz bench-depth` | M0 depth models on this machine |
| `tests/depthviz_verify_tape.py` | M1 tape round trip, live tape tick-exact, camera opens 3x |
| `tests/depthviz_verify_fog.py` | M2 fog tones, stabiliser, mask halo, flips, portrait, PNGs |
| `tests/depthviz_verify_look.py` | M3 the look: depth gradation, soft edges, consistency, grain, timing, seed sheets |
| `tests/depthviz_live.py` | M4 full song live, every key, fullscreen, camera, flip probability |
| `tests/depthviz_verify_export.py` | M5 determinism, params re-run, speed, A/V sync, portrait |

`python tests/depthviz_run_all.py` runs them all. Reports and images go to `out/verify/depthviz/`.

## How it works

```
camera (ffmpeg MJPEG 640x480, hflip, tmix=3, 320x240 rgb24)
  -> worker process: dark-room gain -> depth model (onnxruntime: DirectML / CoreML / CPU)
     -> optional person mask (off by default) -> mean/std alignment + robust bounds (0 near .. 1 far)
     -> shared memory (seqlock)
render process, per picture tick:
  D1 stabilise (EMA with edge reset)  D1b cross-bilateral filter + gradient
  D2 fog (parallax, depth -> tone curve, light behind the figure, exposure)  D2b halation
  soft blur (half res)  D4 film (print range, halation, film grain, dust, flicker, vignette)  D5 present
director (60 Hz, song features): fog density, exposure, kick flashes, grain, halation (one look; no switching)
```

## Mac (M6)

There is no Mac patch for this app: it ships the fixes that the ink app needed on Apple Silicon. That means no
`noise3` name, a darwin window-size branch, 720p capture cropped to 4:3, and a dummy-free fullscreen quad. On a Mac,
run these from Terminal.app, because the camera permission prompt only appears there:

1. `python -m depthviz check`: CoreML must be listed among the onnxruntime providers. Note whether MediaPipe imports.
2. `python -m depthviz camera-test --save-frames out/bench/frames_lit` with the lights on. Then run
   `python -m depthviz bench-depth --eps coreml:0,cpu:0` and put the winner in `depthviz/config.py`
   `DEPTH_PROFILES` (`platform: "darwin"`, `renderer: "apple a18"` or `"apple m"`).
3. `python tests/depthviz_verify_tape.py`, then `python tests/depthviz_live.py --parts song,camera`. Note the memory
   of both processes on the 8 GB MacBook Neo.
