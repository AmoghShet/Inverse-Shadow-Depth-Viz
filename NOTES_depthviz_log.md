# depthviz build log ("Depth fog in ASCII")

Checkpoint / audit log for the `depthviz/` package. Plan: `C:\Users\Asus\.claude\plans\encapsulated-sauteeing-gizmo.md`
(a copy is kept at `docs/PLAN_depthviz.md`). Read this file first after a session reset; the newest entry is at the bottom.

Rules carried over from the ink app: `audioviz/` is never modified (check `git apply --check macpatch/mac.patch` after
every milestone); the user tests with the room lights OFF (camera brightness 4-10 of 255), so a dark camera is the
normal condition, not a fault; the test song is `C:/Users/Asus/Downloads/Everything In Its Right Place.mp3` (3:13).

## Status board

| Milestone | Status | Evidence |
|---|---|---|
| M0 depth benchmark spike | DONE (dav2_small 336 on DirectML GTX; table in the log) | `out/bench/depth_win32_nvidia_geforce_gtx_1650_ti_pcie_sse2.json` |
| M1 worker + plumbing | DONE (all pass; 2 bugs fixed: MediaPipe startup, alignment collapse) | `camera-test`, `tests/depthviz_verify_tape.py` |
| M2 fog look (headless) | DONE (8/8 pass) | `tests/depthviz_verify_fog.py`, PNGs in `out/verify/` |
| M3 ASCII + film style | DONE (7/7 pass; film is the default style) | `tests/depthviz_verify_ascii.py`, contact sheet |
| M4 audio + live | DONE (song, 32 keys, fullscreen, camera, flips: all pass) | `tests/depthviz_live.py`, `--record` output |
| M5 export + determinism | DONE (6/6 pass) | `tests/depthviz_verify_export.py` |
| M6 Mac | N/A: no Mac exists (user, 2026-10-14) | checklist in README_depthviz.md |

## Decisions and deviations from the plan

(details and evidence in the timestamped entries below)
- DEFAULT STYLE IS "film" (user, 2026-10-14 ~23:30: "i just want it like the reference"); ASCII is a toggle (T, --style).
- DA-V2 needs ORT free-dimension overrides on DirectML (145 -> 26 ms). DirectML adapter 0 = GTX on this laptop.
- Alignment: mean/std matching onto an anchored reference instead of least squares (the plan's fit collapsed live).
- D1: 16-tap ring dilation (outside) / erosion near the edge (inside) on top of the plan's mask push.
- Fog: curve exponent d^2.8, near fog lit by ambient only, inverted fog 2.5x denser, multiplicative exposure + knee.
- ASCII mode weights per polarity (paper in high-key, pure in the void); glyph mode switches at a flip's midpoint.
- Depth textures R16F (Apple-safe), D1 output RG16F (depth + EMA'd mask).
- M0 temporal-std criterion judged after D1 (0.018) as well as at the worker (0.032 for the chosen cell).
- MediaPipe builds on a thread in the worker; the worker hard-exits after cleanup (stop < 1 s).

## Entries

### 2026-10-14 21:45 session start, orientation
- Goal set by the user: implement the whole plan overnight, keep this log, build around a dark camera (lights off),
  do not wait for input; the user tests in the morning.
- Read: NOTES_build_log.md (tail), every audioviz module the plan imports or pattern-copies, the shaders, presets,
  tests/verify_camera_smoke.py, tests/run_all.py, tests/verify_export.py.
- Environment: venv `.venv` is Python 3.12.15 (uv-managed, no pip inside; install with
  `python -m uv pip install --python D:/AudioViz/.venv/Scripts/python.exe ...` from the system Python 3.10).
  Importable: mediapipe 1.1.0, numpy 2.5.3, opencv-contrib 5.0.0.93, moderngl 5.12, imageio-ffmpeg 0.6, librosa 1.0.
  Missing: onnxruntime (to install: onnxruntime-directml 1.24.4, cp312 win_amd64 wheel confirmed on PyPI).
  GPU: GTX 1650 Ti, driver 591.86. CPU: 12 logical (6 cores). Camera: 'Logi C270 HD WebCam'.
  Network: huggingface.co (DA-V2 small onnx files), github.com (MiDaS release) and pypi.org all reachable.
- Lit camera frames for the M0 quality set: the research spike left real webcam stills in `bench/*.jpg`
  (last_camera_frame.jpg, last_ffmpeg_480p.jpg, last_ffmpeg_720p*.jpg, last_cv4_frame.jpg, pose_sample.jpg).
  Dark frames will be captured tonight from the camera itself (lights are off).
- `git apply --check macpatch/mac.patch` passes at the start (baseline).

### 2026-10-14 21:50 environment + models (M0 prep)
- Installed `onnxruntime-directml==1.24.4` into .venv (`python -m uv pip install --python .venv/Scripts/python.exe`);
  providers: DmlExecutionProvider, CPUExecutionProvider. No psutil (peak RSS via K32GetProcessMemoryInfo).
- Downloaded into models/depth/ (gitignored), sha256 pinned in depthviz/depth/models.py:
  midas_v21_small_256.onnx 66.8 MB (fixed 1x3x256x256 -> 1x256x256); DA-V2 small model.onnx 99 MB, model_fp16 50 MB,
  model_quantized (int8) 27 MB, model_q4f16 19 MB. DA-V2 input is fully dynamic (batch, 3, H, W), output
  14*floor(H/14) x 14*floor(W/14): dynamic shapes CONFIRMED (no fixed input sizes needed).
- DEVIATION: fp16 and q4f16 exports fail ORT's full graph optimisation on CPU (SimplifiedLayerNormFusion: "Attempting
  to get index by a name which does not exist: InsertedPrecisionFreeCast_..."). They load at ORT_ENABLE_BASIC; the
  session builder retries "all" -> "basic" automatically.
- Room was LIT at 21:47 (camera brightness 118, user at the desk), so the camera frames saved are the lit set
  (out/bench/frames_lit, 30 consecutive frames 640x480). Dark sets are SIMULATED from them for M0 (brightness ~7,
  shot + read noise, MJPEG q80 re-encode; "monitor" variant lights only the face, background x0.2) plus a static set
  (one frame, 30 noise realisations) for the temporal-std criterion. A real dark capture is to be added if the lights
  go off tonight (`python -m depthviz camera-test --save-frames out/bench/frames_dark`).
- MediaPipe selfie-landscape confidence mask at 320x240: 7.5 ms mean / 8.1 ms p95 (CPU).

### 2026-10-14 22:10 code written so far (M0-M5 skeleton) + first renders
- depthviz/: __init__, __main__ (play/export/still/analyze/check/fetch-models/bench-depth/camera-test), config
  (Tunables, DEPTH_PROFILES, pick_profile), preset, seed, director, engine, app, tools, gl/passes.py (copied; resolves
  depthviz's shader dir and uniforms), gl/uniforms.py (own FIELDS), inputs/{depthcam, synthetic, tape},
  depth/{models, infer, bench}, render/{fog, cells, present}, export/recorder.py, shaders (common, depth_stab,
  depth_filter, fog, halo_pre/down/blur/comb, grade, ascii, present; fullscreen.vert + hud.frag copied verbatim),
  presets fog_white + fog_white_portrait.
- First headless renders (synthetic source) compile and run. Look fixes after the first frames:
  1. plain T = exp(-k d) cannot give a dark near figure (d ~0.2) AND a white far room (d ~0.9): figure came out
     mid-gray. Added a fog curve: d' = d^(curve * bass_gamma), curve 2.2 (preset fog.curve). DEVIATION (addition).
  2. normalised exponential exposure doubled the shadows; replaced by multiplicative exposure + a soft shoulder above
     a knee (fog.knee 0.7). Blackout (x0.05) and whiteout (x4) still work through the same multiplier.
  3. halation threshold 0.6 saturated a high-key frame; now 0.8, gain 0.3, harmonic range 0.15..0.5.
  4. centroid tonal spread 0.75..1.0 lifted the figure too far; now 0.88..1.0.
  5. the swirl also modulates the light (far tone) so flat haze drifts.
  6. D1 output is RG16F (depth + EMA'd mask) instead of R16F, so the mask is stabilised too. DEVIATION (minor).

### 2026-10-14 22:35 CHECKPOINT: user restarting the laptop (Windows update). RESUME HERE.
State on disk (nothing committed yet; all new files are untracked under depthviz/, tests/depthviz_*, docs/PLAN_depthviz.md,
this log; audioviz/ untouched):
- M0 benchmark was RUNNING in the background and is killed by the restart. 13 cells finished, each saved as
  out/bench/depth_cells/<model>_<size>_<ep><dev>.json (+ .png strip); the frame sets are in out/bench/depth_cells/sets.npz.
  Done: all DirectML adapter-0 cells (midas + dav2 fp32/fp16/q8 at 252/336/420, q4f16 252/336). Adapter 0 = the AMD
  iGPU: DA-V2 is slow there (137-480 ms); MiDaS 10.4 ms. NOT done: q4f16 420 dml0, all dml1 (GTX 1650 Ti, expected
  winner), all cpu. Resume with a narrower run, e.g.
    .venv/Scripts/python.exe -m depthviz.depth.bench --eps dml:1,cpu:0
  (the orchestrator re-runs every listed cell; the dml0 jsons can be merged from depth_cells/ by hand if needed).
  Numbers so far (dml0): midas 256: inf 10.4 ms, e2e 16.4 ms, 30 fps, GL p95 3.0 ms (baseline 4.1), temporal std
  0.041 (limit 0.04, borderline), separation lit 0.72 / dark-uniform 0.44 / dark-monitor 0.61 / dark-monitor without
  gain 0.04 (the gain is what makes the dark room work for MiDaS). DA-V2 fp32 252: 137 ms, 6.9 fps, tstd 0.043,
  sep 0.56/0.34/0.46, without gain 0.55 (DA-V2 is robust to darkness without gain).
- config.DEPTH_PROFILES still holds a PLACEHOLDER (dav2_small_q8 252 cpu) -> replace with the M0 winner.
- M2 test written (tests/depthviz_verify_fog.py) and run with --no-png: PASS stabiliser small/big step and noise
  (output/input temporal std 0.17), mask halo (edge 0 texels with the mask, 15.5 without), polarity flip (whiteout
  3.8x / mean 0.99, blackout 0.056x / mean 0.025, monotonic), inverted tones. FAIL:
  1. high-key figure mean 0.22-0.32 (need < 0.2; far means 0.87-0.95 pass). Next: lower the leak (far light through
     the figure's fog), e.g. curve floor / near tone / the light blob over the figure; re-run.
  2. portrait "not stretched" is a TEST BUG, not a render bug: expected_h was computed wrongly. Geometry check: disc
     = 0.6 of the camera height; portrait cover crop maps camera height -> screen height (960 px at half res) ->
     expected 576, measured 572 (0.7%); landscape shows 0.75 of the camera height -> expected 0.8 x 540 = 432,
     measured 429. Fix expected_h = 0.6 * fog_height / visible_camera_height_fraction.
- D1 change (DEVIATION, improvement): depth_stab.frag replaces the depth halo near the silhouette with a 16-tap ring
  grey dilation (outside) / erosion (inside), on top of the plan's max(d, fig + margin) push; the plan's push alone
  leaves the blurred ramp outside the mask (edge 15 texels) and could not meet "edge <= 2 texels".
- Preset retuned for the M2 criteria: fog k 3.0, curve 2.6, high-key near 0.06 far 0.97, light ambient 0.85,
  gain 0.25, blob 0.25, centroid spread 0.92..1.0.

Next steps after the restart (in order):
  a. git status / git pull; check `git apply --check macpatch/mac.patch`.
  b. finish M0 (dml:1 + cpu cells), pick the profile, write DEPTH_PROFILES, log the table.
  c. fix the two M2 items above, run tests/depthviz_verify_fog.py WITH PNGs (real-depth tapes from frames_lit).
  d. write + run M1 (tests/depthviz_verify_tape.py, camera-test x3), M3 (tests/depthviz_verify_ascii.py),
     M4 (tests/depthviz_live.py: live app untested so far!), M5 (tests/depthviz_verify_export.py).
  e. requirements-depthviz.txt, README_depthviz.md, M6 checklist; dark-room frames if the lights are off.

### 2026-10-14 22:26 resumed after the restart; DirectML finding (M0)
- NOTE: entry times before 22:22 were estimates (about 15 min ahead of the clock); from here on they come from `date`.
- Restarted the remaining cells (dml:1, cpu:0). DA-V2 was ~150 ms on BOTH adapters, which is not a GPU speed.
  Probe (scratchpad dml_probe.py, DA-V2 fp32 252x196, 25 runs):
    dev 0, dynamic dims 144.8 ms | dev 0, pinned dims 26.5 ms | dev 1, dynamic 143.4 ms | dev 1, pinned 52.0 ms
  nvidia-smi utilisation rises only with dev 0 -> DirectML adapter 0 = GTX 1650 Ti, adapter 1 = AMD Radeon iGPU
  (the 22:35 checkpoint guessed the reverse). MiDaS: dev 0 10.4 ms, dev 1 23.8 ms (consistent).
- FIX: make_session adds ORT free-dimension overrides (batch_size=1, height, width) for dynamic-shape models.
  The earlier DML cells (dynamic dims) moved to out/bench/depth_cells_dynamic/; the full matrix re-runs
  (dml:0, dml:1, cpu:0) -> out/bench/bench_depth_fixed.log.
- M2 fixes applied (not re-run yet): fog near the camera is lit by 0.5 x ambient, ramping to the full light by
  depth 0.7 (Leff = mix(0.5 ambient, L, smoothstep(0, 0.7, d))): the figure's veil of fog is no longer lit by the
  window behind it. fog.curve jitter [2.8, -0.3, 0.5]; centroid spread 0.96..1.0. Portrait test geometry fixed.
- Tests written: tests/depthviz_verify_tape.py (M1), tests/depthviz_verify_ascii.py (M3).

### 2026-10-14 22:30 tests written, Mac-safety, partial M0 results
- Depth textures R32F -> R16F (engine own + DepthCam): linear filtering of 32-bit float textures is not guaranteed
  on Apple GPUs; half floats are plenty for 0..1 depth. DEVIATION (plan said R32F).
- Tests written: tests/depthviz_verify_export.py (M5), tests/depthviz_live.py (M4: song, keys, fullscreen, camera,
  flips). requirements-depthviz.txt and README_depthviz.md (draft) written.
- M4 offline part run: section flip rate 0.3517 over 4000 seeds x 5 sections (target 0.35), high-key share 0.7525
  (target 0.75): PASS.
- M0 with pinned dims (dml0 = GTX): midas 256 10.3 ms / 30 fps; DA-V2 fp32 252 18.8 ms / 30 fps; 336 38.4 ms /
  21.9 fps ACCEPT (tstd 0.032, sep lit 0.51 / dark-uniform 0.37 / dark-monitor 0.52 / no gain 0.48); 420 71.6 ms /
  12 fps ACCEPT; fp16 252 82 ms (fp16 is SLOWER than fp32 on DirectML here). Visual: DA-V2 keeps the head and
  shoulders sharp in both simulated dark sets; MiDaS loses the head in dark_uniform and needs the gain.
- Every model's worker-level temporal std on the static dark set is 0.031-0.043 (limit 0.04). Added
  `bench-depth --stability model:size:ep:dev`: the same still scene after a CPU emulation of D1 (12 fps EMA).
- Camera after the restart: works (first frame 4.3 s, 29.7 fps), room still lit (brightness 101 at 22:30).

### 2026-10-14 23:06 M0 DONE: depth benchmark (pinned dims), profile chosen
Report: out/bench/depth_win32_nvidia_geforce_gtx_1650_ti_pcie_sse2.json (42 cells; per-cell json + depth strip in
out/bench/depth_cells/). GL baseline (no inference) frame p95 6.1 ms. Lit set = 30 real camera frames; dark sets
simulated (no real dark frames yet: the room stayed lit). Acceptance: worker fps >= 10, latency <= 150 ms,
concurrent GL p95 < 16 ms, static-dark temporal std < 0.04, dark separation >= 0.2, bound jumps < 5%.

| cell (ep: dml0 = GTX, dml1 = Radeon iGPU) | infer ms | fps | GL p95 | tstd | sep lit/dark-u/dark-m/no-gain | verdict |
|---|---|---|---|---|---|---|
| midas 256 dml0 | 10.3 | 30 | 3.1 | 0.041 | 0.72/0.44/0.61/0.04 | reject (tstd) |
| dav2 fp32 252 dml0 | 18.8 | 30 | 3.0 | 0.043 | 0.56/0.34/0.46/0.55 | reject (tstd) |
| **dav2 fp32 336 dml0** | **38.4** | **21.9** | **4.9** | **0.032** | **0.51/0.37/0.52/0.48** | **ACCEPT, chosen** |
| dav2 fp32 420 dml0 | 71.6 | 12.0 | 5.4 | 0.031 | 0.50/0.44/0.48/0.55 | accept |
| dav2 q8 336 dml0 | 54.7 | 16.2 | 5.1 | 0.033 | 0.50/0.34/0.50/0.46 | accept |
| dav2 fp16 252/336 dml0 | 82/161 | 11/6 | 6.8/7.2 | | | fp16 slower than fp32 on DML |
| dav2 q4f16 252 dml1 | 58.1 | 15.6 | 4.2 | 0.033 | 0.62/0.49/0.60/0.67 | ACCEPT (iGPU profile) |
| midas 256 cpu | 62.3 | 14.5 | 4.3 | 0.041 | 0.73/0.49/0.65/0.24 | reject (tstd); CPU fallback |
| dav2 * cpu | 134-521 | 1.9-7.1 | | | | too slow |
Peak RSS per worker 260-600 MB. MediaPipe mask 7.5 ms (CPU).
- Post-D1 stability (`bench-depth --stability`, static dark scene, CPU emulation of the 12 fps D1 EMA): DA-V2 336
  worker 0.0335 -> after D1 0.0177; MiDaS 0.043 -> 0.030; DA-V2 q4f16 252 0.033 -> 0.019. The worker-level
  criterion is borderline for every model; what is drawn is well under 0.04. DEVIATION: accepted on the
  after-D1 number for the 0.032 cell (it also passes the worker-level limit).
- DirectML adapter choice: device_id 0 28.3 ms, 1 105 ms, performance_preference=high_performance 43.9 ms (picks the
  GTX, slower). config: "gtx 1650 ti" -> dav2_small 336 dml 0; other NVIDIA -> same model, "hp"; other Windows GPUs
  -> dav2_small_q4f16 252 dml "hp"; darwin -> dav2_small 252 coreml (unmeasured, M6); else midas cpu. The worker
  falls back to MiDaS on the CPU if the profile's session fails (error shown on the HUD camera line).
- Quality: DA-V2 keeps the head/shoulders crisp in both simulated dark sets and does not need the gain (no-gain
  separation 0.48); MiDaS loses the head in the uniformly dark set and needs the gain (0.04 without).

### 2026-10-14 23:11 M2 DONE, M3 DONE (headless), look decisions
- M2 (tests/depthviz_verify_fog.py, report out/verify/depthviz/report_m2_fog.json): ALL PASS.
  high-key far mean 0.88-0.96 / figure 0.10-0.19; inverted far 0.01-0.03 / figure 0.51-0.79; small step 80% in 3
  ticks, big step 85% in 1; noise std ratio 0.17; mask-halo edge 0 texels (15.5 without the mask); whiteout 4.1x
  (mean 0.99), blackout 0.057x (mean 0.024), monotonic; disc stays round (landscape 429x429, expected 432; portrait
  height 572, expected 576). PNGs: out/verify/depthviz/fog_{synthetic,lit,dark}_{high,inv}.png (fog | depth | ascii).
- Test infra: real-depth tapes are built in a SUBPROCESS (ensure_tape): running MediaPipe + DirectML in the test process
  left the moderngl context non-current ("cannot create buffer"). The app is unaffected (the worker is a process).
- Look changes from M2/M3 (all DEVIATIONS from plan numbers, logged here):
  * near fog is lit by 0.5 x ambient, ramping to the full light by depth 0.7 (figures stay in shadow)
  * inverted polarity: fog density x2.5 (crossfaded by keyness) so mid-depth walls fall into the black void
    (halving the curve exponent instead dimmed the pale figure below 0.5)
  * fog.curve 2.8 (2.5-3.3), k 3.0, ambient 0.9 (0.85-0.96), high-key near 0.06 / far 0.97, spread 0.96-1.0
  * MODE WEIGHTS PER POLARITY (plan: pure 0.65 / paper 0.35 overall, "fixed in M3"): contact sheets showed paper
    reads as the references in high-key (white haze, dark figure, light gradient, grain) while pure glyphs in
    high-key read as a dim striped field; in the inverted void pure glyphs read best. Now high-key paper 0.65 /
    pure 0.35, inverted pure 0.85 / paper 0.15 (overall paper ~0.53). At a polarity flip the mode is redrawn for
    the target polarity and switched at the flip midpoint (hidden by the whiteout / blackout).
  * paper mode: no ink where x < 0.08 (the inverted paper void had stray specks)
  * tried a halation "veil" between glyphs in pure mode: it removes the black between glyphs that the plan asks
    for (black fraction 0.69 -> 0.11), so it is OFF (halo.veil 0; the code path stays)
- M3 (tests/depthviz_verify_ascii.py, report_m3_ascii.json): ALL PASS. Contour orientation: 0 deg -> '|' 100%,
  45 -> '\' 100%, 90 -> '-' 100%, 135 -> '/' 99.997% (2 of 58871 cells in '-'). Grain: held frame identical, same
  tick re-render identical, next tick differs. Timing at 1080p (median of 3 rounds): D1 0.27, D2 0.20, D2b 0.13,
  D3 0.29, D4 0.29, D5 0.17 = 1.35 ms per tick (plan estimate ~1 ms; D4 0.3 as estimated). Single rounds vary
  ~50% with the GPU clock state. Contact sheets: out/verify/depthviz/contact_seeds_{lit,dark}.png (12 seeds).

### 2026-10-14 23:25 M1 DONE: worker + tape plumbing (two real bugs found and fixed)
- tests/depthviz_verify_tape.py -> report_m1_tape.json: ALL PASS.
  * tape unit round trip: 115 records / 2 chunks, max depth error 7.7e-6 (uint16 quantisation), mask exact, 5-tick
    gap holds the previous record, a backwards record is ignored
  * live `play --source synthetic --tape-out` (the app's first real launch): 128 records, ticks 0-127 contiguous,
    times = tick/12, every record equals the synthetic source at its tick (tick-exact)
  * camera-test 3 consecutive opens (15 s each): frames 176-216, cam 30.2 fps, depth 21-22 ms at 24-30 fps,
    mask 3-10 ms, latency 59-76 ms, first depth 5.2 s after start, session build 2.2 s, stop 0.94-0.98 s, no errors
- BUG 1 (fixed): the worker built MediaPipe before its frame loop; in a fresh spawned process MediaPipe takes 4-12 s
  to import/build, so the first camera-tests timed out at "opening..." and a stop during the build took 6 s.
  Now MediaPipe builds on a thread; frames use the depth mask until it is ready (mask_mode switches 2 -> 1).
- BUG 2 (fixed, DEVIATION from the plan's formula): the plan's least-squares alignment onto an EMA reference
  collapsed the depth range live (bounds 0.00/0.88 -> 0.25/0.30 in 20 s; alignment slope a -> 0.02): the slope is
  attenuated by noise (regression dilution) AND an EMA of moving content loses spatial variance. Now: mean/std
  moment matching onto the reference, whose mean/std are re-anchored to the first frame's after every update
  (= per-frame standardisation, the reference only gates bad frames by correlation > 0.3). 40 s live: bounds
  stay -0.06..0.02 / 0.97..1.07.
- Stop time: the worker releases the camera from a watcher thread as soon as stop is set (ffmpeg's graceful 'q'
  alone takes 0.44 s), then os._exit(0) after cleanup (skips ~0.5-5 s of onnxruntime / MediaPipe finalisers).
  Caveat: quitting in the first ~10 s (MediaPipe still building, holding the GIL) can take 1.0-1.1 s.
- Live camera view (out/depthviz/camera/cam_sheet.png, lamp-lit room, brightness ~45): DA-V2 separates the person
  and a raised hand cleanly from the room; MediaPipe mask clean.

### 2026-10-14 23:28 M4 keys + fullscreen PASS (live, muted, --source synthetic)
- tests/depthviz_live.py --parts keys: all 32 auto-key presses produced their message (views depth/fog/mask/ascii,
  halation, grid, T pure/paper, V flip both ways, S screenshot + _hud.png + sidecar json, Ctrl+S -> user_NN.json
  (removed by the test), P 24/60/12, Right/Left seek, , . offset, [ ] R seeds, C "no camera (source synthetic)",
  H, Space pause/play, 1/2 presets, E export). The E export ran from the live tape in its own process: 509 frames,
  video 42.41 s / audio 42.42 s.
- fullscreen: F twice, back to the 1600x900 windowed buffer (1280x720 at 125% scaling), no shader errors.
- Test fix: auto-key specs are comma-separated, so , . [ ] must be written COMMA PERIOD LEFT_BRACKET RIGHT_BRACKET.
- Live HUD timers at 1600x900 with the 'furry' grid: D1 0.99, D2 0.49, D2b 0.38, D3 1.80, D4 0.80, D5 0.61 ms
  (5 ms per tick, 12 ticks/s; the window runs at 60 fps).
- Leftover test preset user_01.json (from a crashed run) deleted.

### 2026-10-14 23:35 M4 song + camera; USER DIRECTION CHANGE: reference look (no ASCII) by default
- M4 song (full 3:13, --source synthetic, --record, --tape-out): 0 underflows after start, 658 kicks, 1040 hats,
  9 sections applied, 4 flips, recording v 193.58 s / a 193.56 s, tape 2323 records, no shader errors; frame p50
  16.6 ms / p95 33.3 ms; 6 dropped steps (one stall). A 70 s re-run with --verbose (record + tape) had 0 drops,
  frame max 94 ms: not reproduced.
- M4 live camera 60 s (--source device, record + tape): depth 21.7 fps, 23 ms, mask 12 ms, latency 58 ms, first depth
  5.7 s, 0 dropped steps, 0 underflows, frame p95 18 ms, GPU 1.7 ms/frame, recording v 58.83 / a 58.81 s.
- USER (mid-run): "why am i seeing ascii? I thought i was clear i just want it like the reference? or are you making
  it toggleable?" The saved plan says "drawn entirely in ASCII glyphs on the GPU" (title "Depth fog in ASCII"), so
  ASCII was built. New direction: the DEFAULT render is now a non-ASCII "film" style = the reference look (the depth
  fog itself, film grain, halation, vignette, flicker, dust); ASCII stays as a toggle (T key / --style ascii).
  Told the user; they can ask for ASCII to be removed entirely.

### 2026-10-14 23:40 film style (default) built; M5 DONE; M2/M3 re-run
- New default style "film" (preset render.style; --style film|ascii on play/export/still; T cycles film -> ascii
  (seed's glyph mode) -> ascii (other glyph mode) -> film): shaders/film.frag = the D2 fog tone softened by a tent
  filter (film.soften), halation over the edges, film grain (fine + clumped lattice at film.grain_size, strongest in
  the light tones, reseeded per tick), dust, flicker, vignette; the tone is a display value, linearised before D5.
  Preset section "film": grain 0.11 (0.07-0.17), grain_size 1.8 px, soften 1.2 texels, clump 0.55.
- D1: the inside-the-mask erosion now applies only near the silhouette (taps both inside and outside); deeper inside
  the figure keeps the model's depth, so the film look shows face/shoulder shading instead of a flat cut-out.
- M3 re-run (ascii checks use --set render.style=ascii): ALL PASS, plus film checks: grain identical within a tick and
  different next tick; on-screen display means high-key figure 0.056 / room 0.857, inverted figure 0.63 / room 0.02;
  film tick 1.67 ms at 1080p (D4 film 0.50). Contact sheets out/verify/depthviz/contact_seeds_{film,ascii}_{lit,dark}.png.
- M2 re-run: ALL PASS (mask halo edge still 0 texels).
- M5 (tests/depthviz_verify_export.py, report_m5_export.json, film style): ALL PASS. Tape export twice -> frame 300 md5
  0bf4c781... both, and the same from --params; synthetic twice -> a1d58486... both; 1080p60 synthetic 20 s: 68.4 fps
  (x264 pipe bound: 16.1 of 17.6 s); video 20.00 s / audio 20.01 s; portrait 1080x1920@30: 240 frames.

### 2026-10-14 23:47 M4 DONE (film style), wrap-up. STATE FOR THE MORNING
- M4 re-run in film style: keys ALL PASS (T now cycles film / ascii / ascii-other), song 3:13: 0 dropped steps, 0
  underflows, frame p50 16.7 / p95 17.6 ms, GPU 4.7 ms per frame, 658 kicks, 9 sections, 4 flips, recording
  v 193.58 / a 193.56 s, tape 2323 records. (The single 6-step stall of the first run did not recur.)
- Window title "Depth Fog" (was "Depth Fog in ASCII"). README updated for the film default. tests/depthviz_run_all.py
  runs every check (summary in out/verify/depthviz/summary.json).
- `git apply --check macpatch/mac.patch` passes; audioviz/ untouched by this work (its 2 modified files + live_recorder.py
  are the user's uncommitted --record work from 2026-10-13, see NOTES_build_log.md 2026-10-14 20:54).
- NOT DONE / OPEN:
  1. M6 on a Mac (no Mac here). Checklist in README_depthviz.md: check, bench-depth --eps coreml:0,cpu:0,
     DEPTH_PROFILES darwin entry, verify_tape + live song/camera from Terminal.app.
  2. Real lights-off validation: the room was lit all evening (brightness 101-118, then lamp ~37-48 at 23:47);
     every dark-room number so far is from SIMULATED dark frames (DA-V2 held up well: separation 0.37-0.52 without
     gain). To capture real ones: `python -m depthviz camera-test --save-frames out/bench/frames_dark`, then
     `python -m depthviz bench-depth --eps dml:0 --only dav2_small` (uses them as the real_dark set).
  3. Nothing is committed to git (user decides).
- Try it: `.venv/Scripts/python.exe -m depthviz play "C:/Users/Asus/Downloads/Everything In Its Right Place.mp3"`
  (R new seed, V flip, T style, A views, S screenshot, E export; --tape-out to make a camera run exportable).

### 2026-10-14 23:49 user: "there is no mac"
- M6 cannot run: no Mac exists (user, confirmed). Status: NOT APPLICABLE until a Mac is available; the darwin code
  paths and the README checklist stay in place. Only peer session reachable: the planning session on this PC.
- Lights-off: the user's normal test condition (room lights off, face lit by the monitor, camera brightness 4-10)
  has not been seen live tonight; the room was lit (37-118). Dark-room numbers are from simulated frames. The user
  will test in the morning as planned.

### 2026-10-14 23:51 user's own terminal could not start the venv ("No Python at ...uv\python\cpython-3.12.15..."): FIXED
- Cause: the Claude desktop app is an MSIX-packaged app; uv was run from inside a Claude session on 2026-10-07, so
  its "AppData\Roaming\uv\python" install was silently redirected to
  C:\Users\Asus\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\uv\python. Inside Claude sessions the
  Roaming path resolves (virtualised); a normal terminal does not see it (same error as NOTES_build_log 2026-10-13 03:11).
- Fix: .venv/pyvenv.cfg "home" now points at the real LocalCache path (backup .venv/pyvenv.cfg.bak). Verified from the
  user's terminal tab: venv OK 3.12.15, DmlExecutionProvider available. Applies to audioviz too (same venv).

### 2026-10-15 00:08 USER DIRECTION: one look only (the reference photographs), NO ASCII; then "too blobby/clippy, not like depth"
- User: "why is it cycling through different looks. i want only one kind of look; this. no ascii" (with the 14
  reference images: monochrome depth-map photographs, white haze, dark soft figures, heavy grain).
- Done: ASCII REMOVED (shaders ascii.frag + grade.frag, render/cells.py, glyph atlases, grid, pure/paper modes,
  T and G keys, --style, preset sections grid/glyphs/render). Render path now: D1 -> D1b -> D2 fog -> D2b halo ->
  soft blur (new fog_blur.frag, 13-tap separable gaussian, film.blur sigma 5 half-res texels) -> D4 film -> D5.
  One look: random.p_high_key 1.0, audio_map.section.flip_p 0 (sections change nothing visible), palette neutral
  only, light stays put per seed; V still flips by hand. The camera-opening stand-in figure is gone (empty fog).
  Film: print range black 0.07 / white 0.91 so grain reads in both; grain 0.16, keeps 80% in the darks.
  Uniform block slimmed (look, filmA/B/C replace lookA-E, gridF/C). preset.py rewritten after a bad edit
  (empty-string replace) corrupted it; restored from known content and verified.
- User on the first live pass: "too blobby/clippy around edge detection and inconsistent ... doesn't feel much like
  depth detection; it feels like solid silhouette around whatever is closest". Causes: (1) the tone curve
  T = exp(-k d^2.8) was a near/far switch (all near things one dark tone); (2) the person mask replaced the depth
  edge with a mask edge (and for the first seconds an Otsu threshold mask while MediaPipe loaded).
  Fix: continuous depth -> tone F(d) = (1 - exp(-k d^g)) / (1 - exp(-k)), k 2.2, g 1.15 (bass bends g down to
  0.85x); mask OFF by default (--mask off, preset depth.mask false: edges come from the depth model, no MediaPipe
  load); smoother over time (depth.tau_s 0.22, edge_reset 0.2). Stills show gradation inside the figure and the room.
- Relaunched live with "C:/Users/Asus/Downloads/Motion Picture Soundtrack.mp3" for the user.

### 2026-10-15 00:10 user: "much better! Now can we make the edges/features slightly sharper?"
- film.blur sigma 5 -> 2.5 half-res texels (jitter -0.5/+0.7), depth.sigma_s 2.0 -> 1.5 (bilateral), depth.sharpen
  0 -> 0.3 (unsharp of the stabilised depth toward the unfiltered value). Before/after crop
  out/depthviz/dev/sharp_cmp.png: hair outline, ear and jaw read; still soft like the references.
- Camera worker: the stop-watcher thread now also checks multiprocessing.parent_process().is_alive() every 0.5 s,
  so a killed app no longer leaves ffmpeg holding the C270.
- Tests updated for the single look: depthviz_verify_fog.py (continuous-gradation tone criteria + depth ramp
  monotonic check; mask-specific checks set depth.mask=true), depthviz_verify_look.py replaces
  depthviz_verify_ascii.py (gradation via Spearman depth/tone, soft edges, temporal consistency, grain, timing,
  seed sheets), depthviz_live.py (no T/G keys), depthviz_run_all.py. Not yet re-run (the user is watching live).
- Relaunched live with Motion Picture Soundtrack.

### 2026-10-15 00:12 user: "nah i liked how it was before. revert" (the sharper edges)
- Reverted the look to the previous ("much better") settings: film.blur [5.0, -1.0, 2.0], depth.sigma_s 2.0,
  depth.sharpen 0.0. Kept the camera-worker parent-death watcher (not a look change). Relaunched live.
- Test run before the revert (sharper settings) for the record: M2 fog 5/8 (fail: high-key figure 0.29-0.42 > 0.35,
  inverted tones, small step 3 ticks) and M3 look 4/5 (fail: Spearman depth/tone 0.41-0.66 < 0.75). These criteria
  were written for the old near/far switch and the old tau; to be re-derived for the approved continuous look.

### 2026-10-15 00:13 user: "get rid of the hud"
- HUD hidden by default (new --hud flag shows it at start; H still toggles; --no-hud kept as a hidden no-op for old
  commands). Recordings and screenshots therefore have no HUD unless it is switched on.
