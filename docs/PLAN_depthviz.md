# Plan: "Depth fog in ASCII" — a separate camera-depth audio visualizer (`depthviz/`)

## Context

The user shared reference photos made by a phone app that renders a machine-learned depth map *as* the photograph: monochrome, heavy film grain, soft blurred shapes, near things dark and the far room dissolving into white haze (a minority inverted: black void, pale shapes), figures standing in fog with soft halated edges. They want a "random" (seeded, every run different, always inside the look) audio-reactive visualizer that produces this from their webcam, drawn entirely in ASCII glyphs on the GPU, driven by a song file, on the Windows dev box (Ryzen 5 4600HS, GTX 1650 Ti, Logitech C270) and Apple Silicon Macs (M1+, MacBook Neo A18 Pro 8 GB), 16:9 and 9:16.

Decisions the user made in this session (binding):
- **Its own thing.** New package `depthviz/` next to `audioviz/`. No particle smoke. **No file under `audioviz/` is modified** (the ASCII Ink Visualizer stays exactly as verified; its Mac patch must keep applying).
- **Subject = the camera only**, no generated geometry.
- **Depth = a real monocular depth model** on the webcam frame (Depth Anything V2 small or MiDaS v2.1 small via onnxruntime: DirectML on Windows, CoreML on macOS, CPU fallback). MediaPipe selfie segmentation only as edge cleanup. Cost is unmeasured, so milestone 0 is a benchmark spike.
- `depthviz/` **shares `.venv`** and may **import unchanged infrastructure** from audioviz (GL context, audio analysis/player/feature primitives, glyph atlas builder, HUD, live recorder); anything tied to the ink look's uniform block is copied, not imported.
- **Polarity seeded**: ~75% high-key, ~25% inverted; sections may flip through a whiteout/blackout.
- Keep the conventions the user liked in the ink app: 12 fps frame hold, HUD, screenshot + sidecar JSON, `--record`, same keys.

Verified facts this plan relies on (2026-10-14): `onnxruntime-directml` 1.24.4 ships a CPython 3.12 win_amd64 wheel (Python >=3.11); `onnxruntime` 1.31.0 ships `cp312 macosx_14_0_arm64`; `onnx-community/depth-anything-v2-small` has `onnx/model.onnx` (99 MB), `model_fp16.onnx` (50 MB), `model_q4f16.onnx` (19 MB), `model_int8/uint8/quantized.onnx` (27 MB); DA-V2 preprocessing is ImageNet mean/std, NCHW, sides multiples of 14, output relative inverse depth. The venv already has mediapipe 1.1.0 (Windows only; not in requirements.txt), opencv-contrib 5.0, moderngl 5.12, imageio-ffmpeg 0.6, librosa 1.0; `models/` is gitignored and already holds `selfie_segmenter_landscape.tflite`. Measured on the dev box (research doc §9): fullscreen pass 0.2-0.3 ms, ASCII pass 0.3 ms, selfie mask 7-12 ms at 320x180, camera first frame 3-4.6 s, the C270 wedges if opened at 720p without MJPEG, the user tests with the room lights off (camera brightness 4-10/255).

## Package layout (`D:\AudioViz\depthviz\`)

| Module | Status | Notes |
|---|---|---|
| `__init__.py` | new | `ROOT_DIR` (repo), `SHADER_DIR`, `PRESET_DIR`, `MODEL_DIR = ROOT/models/depth`, `OUT_DIR = ROOT/out/depthviz`, reuse `ROOT/.cache` for song features |
| `__main__.py`, `app.py`, `engine.py`, `tools.py` | new | CLI, live loop, pass wiring; pattern-copied from `audioviz/app.py` + `engine.py` (fixed-step director at 60 Hz, picture ticks at `look_fps`, `present_only` between ticks, keys, HUD lines, screenshot with `resolved_params.json`, `--record`) |
| `config.py` | new | `Tunables` (sim_hz 60, max_steps 3, ref_height 1080, depth sizes, stall timeouts) and `DEPTH_PROFILES` (platform/renderer -> model, input size, EP, device id), filled in by M0. No particle tiers, no tier controller |
| `gl/passes.py` | **copied** from `audioviz/gl/passes.py` | `ShaderLib._load` hardwires `#include "frame_ubo"` to `audioviz.gl.uniforms.glsl_block()` and the audioviz shader dir; the copy points both at depthviz. Otherwise identical (hot reload, errors dict, `FullscreenPass`, `Geometry`) |
| `gl/uniforms.py` | **copied**, own `FIELDS` | same `_dtype / glsl_block / FrameUBO` code, depthviz's block (below) |
| `gl/pingpong.py`, `gl/timing.py` | imported from `audioviz.gl` | pure moderngl helpers; sharing one `Target` type keeps the imported `Hud`/`LiveRecorder` compatible |
| `audioviz.gl.context` | imported | `create_window(title=...)`, `create_headless`, `gl_info`, `renderer_name` |
| `audioviz.audio.{analysis, player, features}` | imported | `load_audio`, `load_or_analyze`, `EXTRACTOR_VERSION`; `Player`, `WallClock`; `FeatureTrack`, `Envelope`, `EventCursor`, `lerp`, `ENV_KEYS`. **Not** `Director` (ink-specific) |
| `audioviz.render.ascii` | imported | `GlyphRenderer`, `atlas_layout`, `build_atlas`, `build_hud_atlas`. **Not** `AsciiPass` (bound to `ascii.frag` semantics) |
| `audioviz.render.grade.grid_geometry` | imported | pure function |
| `audioviz.render.present.Hud` | imported | duck-types `lib.program()`/`geom.fullscreen_vao()`; needs `hud.frag` + `fullscreen.vert` present in depthviz's shader dir (copied verbatim, they have no includes) |
| `audioviz.export.live_recorder.LiveRecorder` | imported | `--record` (HUD included, song muxed), unchanged |
| `audioviz.preset.deep_merge` | imported; `is_jitter`/`resolve_jitter` **copied** (20 lines) | `is_jitter` reads audioviz's module-global `VECTOR_KEYS`; depthviz has its own (`tint`, `safe_area`, `light_dir`). Own `load/list_presets/save_user_preset/write_resolved`; `code_hash` covers depthviz **and** the imported audioviz modules |
| `audioviz.inputs.camera.{list_video_devices, ffmpeg_exe}` | imported | device enumeration (incl. the Mac-patch name preference, since the patch is applied on Macs anyway) |
| `seed.py`, `preset.py`, `director.py` | new | own draw order, own sections, own audio mapping |
| `inputs/depthcam.py` | new (modelled on `audioviz/inputs/camera.py`, own code) | ffmpeg rgb24 -> gain -> depth model -> mask -> shared memory worker; main-side `DepthCam` uploads textures |
| `inputs/synthetic.py` | new | deterministic moving SDF figure (capsule torso, head, pillar, back wall) `frame(t) -> depth, mask` |
| `inputs/tape.py` | new | `TapeWriter` (queue + thread, npz chunks + `meta.json`), `TapeReader(tick)` |
| `depth/models.py`, `depth/infer.py`, `depth/bench.py` | new | pinned model table + `fetch`, session/EP selection, pre/post-processing, alignment + normalisation, the M0 spike |
| `render/fog.py`, `render/cells.py`, `render/present.py` | new | D1-D2b, D3-D4 + atlas cache (uses the imported atlas builder), D5 + debug views |
| `export/recorder.py` | new | pattern-copied from `audioviz/export/recorder.py`: headless, fixed-step, 2 PBOs, writer thread, x264 + AAC mux; input = tape or synthetic |
| `shaders/` | new: `common.glsl`, `depth_stab.frag`, `depth_filter.frag`, `fog.frag`, `halo_pre/down/blur/comb.frag`, `grade.frag`, `ascii.frag`, `present.frag`; copied verbatim: `fullscreen.vert`, `hud.frag` | `common.glsl`: `pcg`/`hash*`, **`vnoise3`** (never the name `noise3`: GLSL 4.1 built-in clash on Apple, which is why the Mac patch renames it), `uv_to_p`, `palette`; `#version 410 core` everywhere; integer hashes only |
| `presets/fog_white.json`, `fog_white_portrait.json` | new | sections `meta, grid, glyphs, depth, fog, light, halo, palette, grain, audio_map, random` |
| `tests/depthviz_*.py` | new | `bench_depth`, `verify_fog`, `verify_ascii`, `verify_tape`, `verify_export`, `live` (pattern of `tests/verify_camera_smoke.py`: headless context, synthetic textures, report JSON in `out/verify/`) |
| `requirements-depthviz.txt` | new | `onnxruntime-directml==1.24.4; sys_platform == "win32"`, `onnxruntime; sys_platform == "darwin"` (same import name, never both), `mediapipe; sys_platform == "win32"` (optional elsewhere), `huggingface_hub` not needed (plain HTTPS download with sha256) |
| `README_depthviz.md`, `NOTES_depthviz_log.md` | new | setup/run/keys; timestamped build log in the style of `NOTES_build_log.md` |

## Camera + depth worker (`inputs/depthcam.py`)

- **Process**: `multiprocessing` spawn, daemon, stop `Event`, error `Queue`; lifecycle as `CameraStir` (graceful `q` to ffmpeg, drain stdout, join 6 s; camera start errors print a traceback). Inside the worker: a capture thread drains the ffmpeg pipe and keeps only the newest frame (a slow model never backs the pipe up; latency stays one camera frame); the model loop takes the newest frame, runs depth (+ mask), publishes; frames that arrive while busy are dropped.
- **ffmpeg**: Windows `-f dshow -rtbufsize 64M -vcodec mjpeg -video_size 640x480 -framerate 30`; macOS `-f avfoundation -video_size 1280x720` + `crop=ih*4/3:ih` (built in, no patch needed); **never 720p without MJPEG**. Filter chain `[crop,]hflip,tmix=frames=3,scale=320:240:flags=area`, `-pix_fmt rgb24` raw pipe (tmix: 3-frame temporal average, free, quiets sensor noise before the model). The ORT session is built during the 3-4.6 s camera startup wait.
- **Dark-room gain** (numpy, per frame): `lo, hi = percentile(gray, 1, 99)` with EMA (tau 1 s); `g = clip(((rgb/255 - lo) / max(hi - lo, 0.04)) ** (1/1.6), 0, 1)` applied to all channels, gain capped at 24x; raw brightness and applied gain go into the header. Camera auto-exposure stays on (in the dark the C270 drops to 62 ms exposure / 15 fps, more light per frame and still above the 12 fps look). Optional `--cam-exposure N`: one-shot DirectShow set through a 640x480 `cv2.VideoCapture` before ffmpeg starts (the research doc measured that the setting persists across processes); default off.
- **Model input**: DA-V2 small accepts 4:3 at multiples of 14 (252x196, 336x252, 420x308; dynamic-shape support verified at M0); MiDaS small is fixed 256x256. ImageNet mean/std, NCHW float32 (fp16/q8 variants where the EP benefits). Output: relative inverse depth, resized (`INTER_AREA`) to 320x240.
- **Mask**: MediaPipe selfie-landscape confidence mask at the model frame size when importable and `--mask auto|mp`; fallback `--mask depth`: Otsu threshold on the normalised inverse depth, 5x5 open/close, 3 px feather; `off`. Published as uint8 320x240.
- **Shared memory** (one block): header 32 x float64 (`seq, fps, cam_fps, depth_ms, mask_ms, brightness, gain, dark, lo, hi, align_a, align_b, fig_depth, mask_cov, frames, first_frame_s, last_frame_wall, running, model_id, ep_id, in_w, in_h, mask_mode, stale`), then depth float32 240x320 (GL row order), mask uint8 240x320, gained gray uint8 240x320 (debug view). Seqlock exactly as `camera.py`: `seq` odd while writing, even when consistent; reader copies and rechecks, 4 retries.
- **Main side `DepthCam`**: `prime()` creates the depth (R32F) and mask (R8) textures blank before prewarm (the driver-hitch fix from audioviz); `update()` uploads on an even, changed `seq`; `status()` for the HUD; `stalled()` after 2 s; on stall or `dark` the engine holds the last stabilised depth (never flashes to black).
- **Model files**: `depth/models.py` `MODELS` table with pinned URL + sha256 + input spec. `python -m depthviz fetch-models [--model ...]` downloads into `models/depth/` (gitignored) and verifies; the runtime never downloads (clear error naming the command). Candidates to pin at M0: MiDaS v2.1 `model-small.onnx` (isl-org/MiDaS release) and the `onnx-community/depth-anything-v2-small` exports (fp32/fp16/q8/q4f16).
- **EP selection** (`--ep auto|dml|coreml|cpu`): Windows `DmlExecutionProvider` with `device_id` from `DEPTH_PROFILES` (hybrid laptop: adapter 0 may be the AMD iGPU; M0 benchmarks ids 0 and 1), `enable_mem_pattern=False`, sequential execution; macOS `CoreMLExecutionProvider` (MLProgram, all compute units) then CPU; CPU fallback everywhere with `intra_op_num_threads` = physical cores.

## Depth normalisation and stabilisation

1. **Worker, per model frame**: raw inverse depth `z`. Affine alignment to a running reference removes the per-frame scale/shift ambiguity of relative-depth models: `(a, b) = argmin sum (a z + b - z_ref)^2` (closed form over a 4x-subsampled frame), `z' = a z + b`, `z_ref <- EMA(z', tau 0.5 s)`. Robust bounds `lo, hi = percentile(z', 2, 98)` with an asymmetric EMA (tau 0.3 s widening, 2 s narrowing). Published `d = 1 - clamp((z' - lo)/(hi - lo), 0, 1)`: 0 = near, 1 = far. `fig_depth` = median of `d` inside the mask.
2. **D1 stabilise (GPU, per tick)**: `a = 1 - exp(-dt_tick / tau)`, tau 0.15 s (about 0.43 per tick at 12 fps); where `|d_new - d_prev| > 0.12` (an edge moved) `a = 0.85` so motion does not smear. Figure edge comes from the mask, not the depth: `d = mix(max(d, fig_depth + 0.08), d, smoothstep(0.35, 0.65, m))` pushes any depth halo around the figure behind it; the sharp mask edge becomes the silhouette.
3. **D1b filter**: 5x5 cross-bilateral (sigma_s 2 texels, sigma_r 0.08 depth units, mask edges as a guide), Sobel gradient `(gx, gy)`, mask feathered 1.5 texels. Output RGBA16F `(d, gx, gy, m)`.

## GPU passes per picture tick (1080p-equivalent; costs are estimates from the measured 0.2-0.3 ms fullscreen and 0.3 ms ASCII passes)

| Pass | Resolution / format | Reads -> writes | Est. | Shader |
|---|---|---|---|---|
| D0 upload | depth R32F + mask R8, 320x240 (camera space, 4:3, mirrored) | shm -> textures | 0.05 ms CPU | - |
| D1 stabilise | 640x480 R16F ping-pong | depth, mask, prev -> EMA depth | 0.05 ms | `depth_stab.frag` |
| D1b filter | 640x480 RGBA16F | D1 -> (d, gx, gy, m) | 0.1 ms | `depth_filter.frag` |
| D2 fog | 1/2 screen RGBA16F | D1b (cover-crop + parallax) -> (v, gx, gy, m) | 0.1 ms | `fog.frag` |
| D2b halation | 1/4, 1/8, 1/16 screen R16F | D2 -> blurred highlights | 0.2 ms | `halo_*.frag` (bloom-chain pattern, own files) |
| D3 grade F / C | grid res RGBA16F (r, angle, grad, m) | D2, D2b -> per-cell ramp, contour angle, gradient, figure share | <0.1 ms | `grade.frag` |
| D4 ascii | full res RGBA16F linear | cellF, cellC, atlasF, atlasC -> glyph frame, grain, flicker, dust, vignette | 0.3 ms | `ascii.frag` |
| D5 present | window / export FBO | frame (+ debug views depth / fog / mask) -> shoulder + gamma 2.2 | 0.1 ms | `present.frag` |

About 1 ms GPU per tick; between ticks only D5 (`present_only`). D1/D1b use `dt_tick = 1/look_fps` (deterministic from a tape); the director still runs per 60 Hz sim step like audioviz.

## Fog, tone, polarity (`fog.frag`)

- Cover-crop uv as in audioviz (`cover_scale`: 4:3 into 16:9 or 9:16; portrait shows the centre 42% of the camera width, fine for a standing figure).
- **Parallax** (the still scene breathes in 2.5D): `uv_c = uv + (pivot - d0) * u_fogD.xy`, `d0` a first depth tap; `u_fogD.xy` = slow seeded drift (0.004-0.012 screen heights, period 20-40 s) + beat wobble `w sin(2 pi phase)`; one re-sample.
- **Transmittance** `T = exp(-k d)`, k seeded 1.2-3.5 and audio-modulated. **Light field** `L(uv) = ambient + gain * smoothstep(-w, w, dot(p - c, dir)) + blob * exp(-|p - c|^2 / (2 s^2))`: a seeded window/door-like bright region behind the figure (centre in the safe area, direction, width, blob radius).
- **Tone** `v = mix(near_tone, far_tone * L(uv), 1 - T)`. High-key: near 0.04-0.12, far 0.9-1.0. Inverted: near 0.55-0.8, far 0.0-0.04. Polarity is only the pair `(near_tone, far_tone)`, so a **flip** is a 2-beat crossfade of the pair plus an exposure push through the pole: `x` 0->1, `pair = mix(A, B, smoothstep(x))`, `E = E0 (1 + 3 sin(pi x))` when flipping to inverted (whiteout), `E0 (1 - 0.95 sin(pi x))` when flipping to high-key (blackout).
- **Exposure** `v = 1 - exp(-E v)`, then `pow(v, gamma)`. **Halation** after D2b: `v += halo_gain * H`, `H` = blur chain of `max(v - 0.6, 0)`, gain 0.25-0.6. Vignette and grain are applied in D4. `m` passes through for D3.

## ASCII mapping (D3 + D4): what differs from the ink app

- D3 supersamples `v` over each cell exactly like `audioviz/shaders/grade.frag` (nx x ny taps, `r = mix(mean, max, peak_mix)`); stores the **contour angle** `atan(gy, gx) + pi/2` folded to [0, pi) (glyphs lie along iso-depth contours), `grad = |(gx, gy)|` and `m`.
- Nested grids: coarse wins in flat haze, fine at edges: `w = smoothstep(T - B, T + B, r_C) * (1 - smoothstep(g0, g1, grad_C))`. Oriented atlas rows only where `grad > dir_threshold` (preset `glyphs.oriented` 0..1, default 0.6); highlights isotropic.
- **Pure-glyph mode** (seed weight 0.65): black between glyphs, column `floor(r (levels - 1) + jitter)`, colour `palette(r)` from a 4-stop neutral / warm / cold gray gradient. High-key frames are dense bright glyphs with the figure as sparse or empty cells; inverted frames the opposite.
- **Paper mode** (weight 0.35, key `T`): the cell is filled with the fog tone as paper and the glyph is ink the other way: column `floor((1 - r)(levels - 1) + jitter)`, colour `mix(bg, ink_tone, ink)` with a polarity-aware ink tone. Reads like a print of the depth map; grain hits the paper too.
- **Grain** (the look's main texture; all hashes use the tick index so held frames are identical and every tick reseeds): per-cell ramp jitter 0.4-0.9 levels; per-pixel ink noise `col *= 1 + grain (hash - 0.5) 2 (0.4 + 0.6 r)` weighted to the whites (the references are grainiest there), 0.08-0.18; dust specks `hash(cell, tick) < dust` (1e-4..6e-4), bright or dark; flicker 0.02-0.05.
- Grid pairs as audioviz (`default` 6x12/12x24, `furry`, `readable`); glyph sets reuse the 10-char iso rows + 4 directional rows; atlases built with the imported `build_atlas`, cached per (set, cell, font), all warmed at startup.

## UBO `FIELDS` (`depthviz/gl/uniforms.py`)

`time (t, dt, tick, seedF)`, `seedU (seed, jitterSeed, step, 0)`, `res (W, H, aspect, 0)`, `gridF`, `gridC`, `bands`, `onset (strength, beatPhase, beatPulse, flux)`, `spectral (centroid, rms, loudness, harmonic)`, `fogA (k, nearTone, farTone, gamma)`, `fogB (exposure, lightGain, blob, ambient)`, `fogC (light cx, cy, dir cos, sin)`, `fogD (parallax dx, dy, pivot, gradientWidth)`, `depthA (ema a, edgeReset, figDepth, maskMargin)`, `depthB (sigma_s, sigma_r, maskOn, sharpen)`, `haloA (threshold, gain, 0, 0)`, `lookA (T, B, dirThreshold, oriented)`, `lookB (grain, flicker, vignette, rampJitter)`, `lookC (levels, paper, peakMix, dust)`, `lookD (view, gridOverlay, gradLo, gradHi)`, `camA (uv scale x, y, on, stale)`, `flipA (progress, kind, section, 0)`, `palette[4]`.

## Audio to visual mapping (`director.py`, per 60 Hz step, `Envelope` attack/release)

| Feature | Drives | Attack / release | Range |
|---|---|---|---|
| loudness (2 s) | fog density k; exposure E (loud = whiteout) | 500 / 2000 ms | k x0.8..1.5; E x0.85..1.35 |
| sub | slow fog swell; parallax pivot | 60 / 400 | k +0..0.4; pivot 0.5..0.6 |
| bass | depth contrast (near/far spread) | 30 / 250 | d gamma 1.0..0.75 |
| kick (ranked strength >= 0.35) | light flash decaying 150 ms; grain burst +0.08 for one tick | 0 / 150 | flash 0.3..0.9 |
| hats | jitterSeed += 1 (ramp jitter reseed); dust +1 tick | one-shot | - |
| onset | momentary edge sharpening (sigma_r down, gradient weight up) | 0 / 150 | sigma_r 0.08..0.03 |
| centroid | tonal spread (far - near) and coarse threshold T | 300 / 600 | spread 0.75..1.0; T 0.52..0.72 |
| harmonic | halation gain and radius | 200 / 1200 | gain 0.2..0.7 |
| flux | ramp jitter amount | 20 / 200 | 0.3..0.9 |
| high | grain amount; fine-grid bias | 15 / 120 | grain +0..0.05; T +0.1 |
| beat phase | parallax wobble; exposure breath `1 + 0.05 cos` | none | w 0..0.006 |
| section boundary (next downbeat) | polarity flip with p 0.35 (2-beat whiteout/blackout); new light position/direction; new glyph set and grid pair; new jitter | one-shot | - |

## Seed, presets, determinism, tape

- `seed.resolve(preset, seed, aspect)` draws in a fixed order: polarity (p 0.75 high-key); light centre (safe area 0.2-0.8), direction, width, blob radius, gain; fog k and tones (jitter triples); grain, dust; glyph set; grid pair; pure/paper; palette tint family (neutral 0.5 / warm 0.25 / cold 0.25) + strength; parallax drift direction and period; halation; then every jitter triple section by section. `section_layout(seed, k)` uses `default_rng([seed, 7919, k])` as audioviz does.
- Presets: JSON with `[value, lo, hi]` triples, `--set key=value`, `Ctrl+S` freezes to `presets/user_NN.json`. Screenshots and exports write `resolved_params.json` with song sha1, extractor version, depth source (`device` / `synthetic` / `tape:<dir>`), model, EP, tape sha1 and the code hash (depthviz + imported audioviz modules).
- **Depth tape** (makes a camera-driven look exportable and reproducible): `--tape-out [dir]` records per picture tick `depth` uint16 (normalised 0..1), `mask` uint8 and the header, batched 64 ticks per compressed `chunk_NNNN.npz` from a writer thread (est. 100-200 MB per 3-minute song), plus `meta.json` (song sha1, look_fps, tick0, sizes, model, EP, seed). `export song --tape DIR --seed N` re-renders deterministically (missing ticks hold the previous frame; D1 uses the constant tick dt). `play --source tape:DIR` replays live for look work without a camera; `--source synthetic` is the moving SDF figure, usable live, in export and in tests. The sim/director never reads the wall clock.

## CLI and keys

`python -m depthviz play [song] [--seed N] [--preset fog_white] [--aspect 16:9|9:16] [--size WxH] [--source device|synthetic|tape:DIR] [--camera-device NAME] [--model auto|midas_small|dav2_small] [--ep auto|dml|coreml|cpu] [--model-size N] [--mask auto|mp|depth|off] [--cam-exposure N] [--look-fps 12|24|60] [--record [path]] [--tape-out [dir]] [--mute] [--no-hud] [--offset-ms N] [--no-loop] [--debug-view ascii|depth|fog|mask] [--quit-after S] [--auto-keys ...] [--set k=v]`
`export song (--tape DIR | --source synthetic) --seed N [--size] [--fps] [--look-fps] [--out] [--dump-frame N] [--params json]`; `still`; `analyze`; `check`; `fetch-models`; `bench-depth`; `camera-test [--seconds] [--save-frames DIR]`.
Keys mirror audioviz: `R` reseed, `[` `]`, `Space`, `S` screenshot + sidecar, `Ctrl+S` save preset, `A` cycle view (ascii / depth / fog / mask), `B` halation, `G` grid overlay, `H` HUD, `F` fullscreen, `P` look fps 12/24/60, `1-9` presets, `Left/Right` scrub, `,` `.` offset, `E` export (from the current tape when `--tape-out` is on, else synthetic), `Esc`. Re-purposed: `C` camera on/off (off holds the last depth), `V` polarity flip now, `T` pure/paper. HUD adds a camera line: cam fps, model ms, brightness, gain, lo/hi, mask coverage, EP, stalled/dark, tape state.

## Milestones and verification

- **M0 depth benchmark spike** (`python -m depthviz bench-depth --frames DIR` -> `out/bench/depth_<platform>_<renderer>.json`). Inputs: 30 lit and 30 dark (brightness 4-10) frames saved by `camera-test --save-frames` on each machine (synthetic frames if no camera). Matrix: model {midas_small, dav2_small fp32 / fp16 / q8} x input size {256 | 252x196, 336x252, 420x308} x EP {dml id 0 / 1 | coreml | cpu}. Per cell: session build s, warm-up, mean/p95 inference ms (60 runs), end-to-end worker fps and capture-to-shm latency, peak RSS, and the GL frame p95 of a concurrent 1080p moderngl loop (GPU contention on the 1650 Ti). Quality on the dark set after alignment + normalisation: temporal std between consecutive still frames < 0.04; figure/background separation (median inside mask minus outside) >= 0.2 with gain (also recorded without); bounds jumps > 0.2 in < 5% of frames. Accept a profile when worker fps >= 10, latency <= 150 ms, GL p95 < 16 ms, quality met; write winners into `config.DEPTH_PROFILES`. Also: MediaPipe timing at 320x240 and whether it imports on macOS. Deliverable: numbers per machine, pinned model URLs/hashes, dynamic-shape support confirmed or input sizes fixed.
- **M1 worker + plumbing**: `camera-test` prints cam fps, depth ms, gain, mask coverage, clean stop < 1 s, 3 consecutive opens without a wedge; `tests/depthviz_verify_tape.py`: 10 s recorded from `--source synthetic`, read back tick-exact.
- **M2 fog look, headless** (`tests/depthviz_verify_fog.py`): synthetic depth through D1-D2b. Checks: high-key far-region mean v > 0.85 and figure mean < 0.2 (inverted the reverse); stabiliser step response settles within 3 ticks and a still noisy input has output std < 0.3x input std; mask-halo test (6 px blurred silhouette + sharp mask -> output edge width <= 2 texels); polarity flip crossfade monotonic with the exposure pole reached; portrait not stretched. Writes PNGs for the user to judge against the references **before** any ASCII work.
- **M3 ASCII** (`tests/depthviz_verify_ascii.py`): pure vs paper renders of one tape frame; contour orientation (synthetic ramps at 0/45/90/135 degrees -> 100% of oriented votes in the expected row); grain refresh (two ticks differ, two frames within a tick identical by md5); D4 timer about 0.3 ms. Contact sheet of seeds for the user.
- **M4 audio + live** (`tests/depthviz_live.py`, muted, `--auto-keys`): a full song with `--source synthetic` and with the camera: 0 underflows after start, 0 dropped steps, kicks flash, sections flip with p about 0.35, every key works, `--record` output plays with sound; then a lights-off live session by the user with screenshots.
- **M5 export + determinism** (`tests/depthviz_verify_export.py`): export twice from one tape -> frame 300 md5 identical, and identical from `--params`; synthetic export identical across runs; 1080p60 faster than real time; A/V durations equal; portrait export.
- **M6 Mac**: run M0, M1, M4 on the MacBook Neo from Terminal.app (camera permission); fill `DEPTH_PROFILES` for `Apple A18 Pro` (and an M-series renderer when available); confirm CoreML EP and MediaPipe status; record RSS of both processes on the 8 GB box. No Mac patch should be needed: depthviz ships Mac-safe code from the start (no `noise3`, `window_size_for` darwin branch, 720p crop built in).

Build order: M0 first (it decides model/EP and whether the dark room works at all), then M1 -> M2 -> M3 -> M4 -> M5, M6 when a Mac is at hand. Keep `NOTES_depthviz_log.md` as the checkpoint log, and verify after every milestone that `git apply --check macpatch/mac.patch` still passes (it must, since nothing in audioviz changes).

## Risks and mitigations

- **Dark room (brightness 4-10)**: the model may hallucinate on noise. tmix + percentile gain before the model, affine alignment + asymmetric bounds EMA, D1 EMA with edge reset, mask-driven silhouette; M0 measures it on real dark frames before any look work; `--cam-exposure` as a fallback; if separation stays < 0.2, the figure comes from the mask alone (`d = mix(far, fig_depth, m)`), which still produces the look.
- **Model flicker / scale breathing**: affine alignment, bounds EMA, the 12 fps hold, and the deliberate grain masks what remains.
- **Mac EP availability**: CoreML EP is in the official arm64 wheel (checked at M0); CPU with MiDaS 256 is the floor; the A18 Pro has 2 performance cores, so the capture thread must stay light.
- **8 GB on the Neo**: fp16/q8 variants, one ORT session, no librosa in the worker; RSS measured in M0/M6; tape writer batches 64 ticks (about 15 MB).
- **Startup**: camera 3-4.6 s and session build overlap; textures primed before prewarm; HUD shows "opening..." while the held/synthetic frame draws.
- **MediaPipe**: known good on Windows only; macOS unverified. The mask is optional (depth-Otsu fallback), with a selfie-segmentation ONNX through the same onnxruntime as the second fallback.
- **GPU contention** (DML inference + GL in two processes on the 1650 Ti): measured in M0; if GL p95 slips, use a smaller input or the CPU EP (the look only needs 10-12 depth fps).
- **Hybrid graphics `device_id`** for DML: benchmarked and pinned per machine.
- **Determinism**: synthetic and tape runs are exact; live camera runs are reproducible only through their tape (documented).
- **Camera wedge**: never 720p without MJPEG on Windows; graceful `q` + stdout drain copied from audioviz's shutdown.
- **Look risk**: dense bright glyphs everywhere in high-key may read heavy; paper mode, `levels`, soft glyph sets and the grain are the knobs, compared on one tape in M3 before seed weights are fixed.

## Out of scope

Modifying `audioviz/` in any way; generated geometry; particle smoke; Spout/Syphon output; packaging; metric depth.
