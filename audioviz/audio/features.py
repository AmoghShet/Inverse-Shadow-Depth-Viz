"""Feature lookup, attack/release envelopes, and the Director that maps audio to visual parameters.

The Director runs once per fixed simulation step with the song time of that step. It depends only
on (song features, resolved params, step times), so export mode is deterministic.
Mapping (see the plan's table): sub -> curl scale + trail persistence; bass -> emission rate +
advection speed; mid -> tone gamma; high -> octave-3 weight, grain, fine-grid bias; kicks -> burst +
radial impulse; onset -> exposure/bloom kick; hats -> glyph jitter reseed + fine shimmer; beat
phase -> exposure breathing; centroid -> coarse threshold T + cyan; flux -> octave-2 + ramp jitter;
harmonic -> persistence + bloom; percussive -> tone knee; loudness -> emitter count, exposure,
vignette; section boundary -> new emitter layout / palette / grid pair on the next downbeat with a
2-beat crossfade.
"""
import math

import numpy as np

from ..gl.uniforms import MAX_EMITTERS
from ..seed import section_layout

ENV_KEYS = ("sub", "bass", "mid", "high", "onset", "centroid", "flux", "harmonic", "percussive", "loudness")
SILENT_DEFAULTS = {"sub": 0.55, "bass": 0.55, "mid": 0.5, "high": 0.35, "onset": 0.0, "centroid": 0.45,
                   "flux": 0.3, "harmonic": 0.55, "percussive": 0.4, "loudness": 0.65, "rms": 0.5}


def lerp(r, x):
    return r[0] + (r[1] - r[0]) * x


class FeatureTrack:
    """Continuous features sampled by time, plus event lists."""

    def __init__(self, data):
        self.d = data
        self.fps = float(data["fps"])
        self.duration = float(data["duration"])
        self.n = len(data["rms"])

    def sample(self, t):
        x = t * self.fps
        i = int(math.floor(x))
        a = x - i
        i0 = min(max(i, 0), self.n - 1)
        i1 = min(max(i + 1, 0), self.n - 1)
        out = {}
        for k in ENV_KEYS + ("rms",):
            arr = self.d[k]
            out[k] = float(arr[i0] * (1.0 - a) + arr[i1] * a)
        return out


class Envelope:
    def __init__(self, attack, release, value=0.0):
        self.attack = max(attack, 1e-4)
        self.release = max(release, 1e-4)
        self.value = value

    def update(self, target, dt):
        tau = self.attack if target > self.value else self.release
        self.value += (target - self.value) * (1.0 - math.exp(-dt / tau))
        return self.value


class EventCursor:
    """Yields events whose time falls in (t_prev, t]; resyncs on seeks."""

    def __init__(self, times, strengths=None):
        self.t = np.asarray(times, np.float64)
        self.s = np.ones(len(self.t)) if strengths is None else np.asarray(strengths, np.float64)
        self.i = 0

    def resync(self, t):
        self.i = int(np.searchsorted(self.t, t, side="right"))

    def crossed(self, t_prev, t):
        if t < t_prev or t - t_prev > 1.0:   # seek or loop
            self.resync(t)
            return []
        out = []
        while self.i < len(self.t) and self.t[self.i] <= t:
            if self.t[self.i] > t_prev:
                out.append((float(self.t[self.i]), float(self.s[self.i])))
            self.i += 1
        return out


class Director:
    def __init__(self, preset, resolved, n_particles, track=None):
        self.preset = preset
        self.track = track
        self.n_particles = int(n_particles)
        self.set_resolved(resolved)

    # -- setup -----------------------------------------------------------------------------
    def set_resolved(self, resolved):
        self.r = resolved
        self.P = resolved["params"]
        am = self.P["audio_map"]
        self.env = {k: Envelope(am[k]["attack"], am[k]["release"], SILENT_DEFAULTS[k])
                    for k in ENV_KEYS}
        self.env["onset"].value = 0.0
        self.hat_env = Envelope(0.0, am["high"]["release"], 0.0)
        self.impulse = np.zeros(MAX_EMITTERS)
        self.emit_acc = 0.0
        self.burst_pending = 0.0
        self.burst_emitter = 0
        self.jitter_seed = 1
        self.t_prev = None
        self.kick_count = 0
        self.hat_count = 0
        self.section = 0
        self.pending_section = None
        lay = section_layout(resolved, self.preset, 0)
        self.pos_from = np.array(lay["layout"]["positions"])
        self.pos_to = self.pos_from.copy()
        self.layout = lay["layout"]
        self.pal_from = np.array(lay["palette"])
        self.pal_to = self.pal_from.copy()
        self.palette_variant = lay["palette_variant"]
        self.grid_pair = lay["grid_pair"]
        self.xf_start = -1e9
        self.xf_dur = 1.0
        self.last_event = ""
        self.auto = 1.0          # auto-exposure multiplier, adapted in the log domain
        self.meter = None
        tr = self.track
        if tr is not None:
            d = tr.d
            # kick strength as a rank within the song: only the stronger half puffs hard
            ks = np.asarray(d["kick_strength"], np.float64)
            rank = (np.argsort(np.argsort(ks)) + 0.5) / max(len(ks), 1) if len(ks) else ks
            self.kicks = EventCursor(d["kicks"], rank)
            self.hats = EventCursor(d["hats"], d["hat_strength"])
            self.sections = EventCursor(d["sections"])
            self.beats = np.asarray(d["beats"], np.float64)
            self.downbeats = np.asarray(d["downbeats"], np.float64)
        else:
            period = 60.0 / 96.0
            self.kicks = EventCursor([])
            self.hats = EventCursor([])
            self.sections = EventCursor([])
            self.beats = np.arange(0.0, 3600.0, period)
            self.downbeats = self.beats[::4]
        self.state = {}

    def set_particles(self, n):
        self.n_particles = int(n)

    def feed_meter(self, m, dt, snap=False):
        """Exposure meter reading -> slow auto-exposure (target tone for the covered-area mean)."""
        self.meter = m
        ae = self.P["tone"].get("auto_exposure", {})
        if not ae.get("enabled", False) or m is None or m["coverage"] < 0.002 or m["mean_covered"] <= 1e-5:
            return
        E = self.P["tone"]["exposure"]
        want = -math.log(1.0 - ae["target"]) / (E * m["mean_covered"])
        want = float(np.clip(want, ae["min"], ae["max"]))
        if snap:
            self.auto = want
            return
        k = 1.0 - math.exp(-dt / max(ae["tau"], 1e-3))
        self.auto = math.exp(math.log(self.auto) + (math.log(want) - math.log(self.auto)) * k)

    # -- helpers ---------------------------------------------------------------------------
    def _beat(self, t):
        b = self.beats
        if len(b) < 2:
            return 0.0, 0.0, 0.5
        i = int(np.searchsorted(b, t, side="right")) - 1
        if i < 0:
            period = b[1] - b[0]
            return ((t - b[0]) / period) % 1.0, 0.0, period
        if i >= len(b) - 1:
            period = b[-1] - b[-2]
            ph = ((t - b[-1]) / period) % 1.0
            return ph, math.exp(-(t - b[-1]) * 8.0), period
        period = b[i + 1] - b[i]
        return (t - b[i]) / period, math.exp(-(t - b[i]) * 8.0), period

    def _next_downbeat(self, t):
        d = self.downbeats
        j = int(np.searchsorted(d, t, side="left"))
        return float(d[j]) if j < len(d) else t

    # -- per step --------------------------------------------------------------------------
    def step(self, t, dt):
        P = self.P
        am = P["audio_map"]
        sim = P["sim"]
        t_prev = self.t_prev if self.t_prev is not None else t - dt
        self.t_prev = t

        raw = self.track.sample(t + am["lookahead_s"]) if self.track is not None else dict(SILENT_DEFAULTS)
        e = {k: self.env[k].update(raw[k], dt) for k in ENV_KEYS}

        # events
        for (te, s) in self.kicks.crossed(t_prev, t):
            self.kick_count += 1
            self.jitter_seed += 1
            if s >= 0.35:
                s = (s - 0.35) / 0.65
                rng = np.random.default_rng([self.r["seed"], 31337, self.kick_count])
                n_act = max(1, int(self.state.get("num_emitters", 1)))
                k = 0 if rng.random() < 0.6 else int(rng.integers(0, n_act))
                self.burst_emitter = k
                self.impulse[k] = max(self.impulse[k], lerp(am["onset"]["impulse"], s))
                self.burst_pending += self.n_particles * sim["burst_fraction"] * (0.25 + 0.75 * s * s)
                self.last_event = f"kick {s:.2f} -> emitter {k}"
        for (te, s) in self.hats.crossed(t_prev, t):
            self.hat_count += 1
            self.jitter_seed += 1
            self.hat_env.value = max(self.hat_env.value, s)
        for (te, _) in self.sections.crossed(t_prev, t):
            self.pending_section = (self.section + 1, self._next_downbeat(te))
            self.last_event = f"section boundary at {te:.1f}s"
        if self.pending_section is not None and t >= self.pending_section[1]:
            self._apply_section(self.pending_section[0], t)
            self.pending_section = None

        decay_imp = math.exp(-dt / max(am["onset"]["release"], 1e-3))
        self.impulse *= decay_imp
        self.hat_env.update(0.0, dt)

        phase, pulse, period = self._beat(t)
        self.state = self._params(t, dt, e, phase, pulse, period)
        return self.state

    def _apply_section(self, k, t):
        lay = section_layout(self.r, self.preset, k)
        cur = self._positions_base(t)
        self.pos_from = cur
        self.pos_to = np.array(lay["layout"]["positions"])
        self.pal_from = self._palette_base(t)
        self.pal_to = np.array(lay["palette"])
        self.layout = lay["layout"]
        self.palette_variant = lay["palette_variant"]
        self.grid_pair = lay["grid_pair"]
        _, _, period = self._beat(t)
        self.xf_start = t
        self.xf_dur = max(0.25, self.P["audio_map"]["section"]["crossfade_beats"] * period)
        self.section = k
        self.last_event = f"section {k}: {self.palette_variant}, grid {self.grid_pair}"

    def _xf(self, t):
        x = np.clip((t - self.xf_start) / self.xf_dur, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _positions_base(self, t):
        return self.pos_from + (self.pos_to - self.pos_from) * self._xf(t)

    def _palette_base(self, t):
        return self.pal_from + (self.pal_to - self.pal_from) * self._xf(t)

    def _params(self, t, dt, e, phase, pulse, period):
        P = self.P
        am, sim, tone, grid, grain, bloom, pal = (P["audio_map"], P["sim"], P["tone"], P["grid"], P["grain"],
                                                  P["bloom"], P["palette"])
        r = self.r
        aspect = r["aspect"]
        short = min(aspect, 1.0)
        om = r["octave_mul"]
        scales = [s * r["flow_scale_mul"] for s in sim["scales"]]
        sub = e["sub"]
        w = sim["weights"]

        speed = lerp(am["bass"]["speed"], e["bass"])
        decay = min(0.992, lerp(am["sub"]["decay"], sub) + am["harmonic"]["persistence_add"] * e["harmonic"])
        alt_scale = scales[0] * am["sub"]["curl_scale"][1]
        w1 = w[0] * om[0] * (1.0 - sub)
        w_alt = w[0] * om[0] * sub
        w2 = lerp(am["flux"]["octave2"], e["flux"]) * om[1] * (w[1] / 0.5)
        hat = self.hat_env.value
        w3 = (w[2] + lerp(am["high"]["octave3"], e["high"]) + 0.15 * hat) * om[2]

        rate = lerp(am["bass"]["emit_rate"], e["bass"])
        self.emit_acc += self.n_particles / max(sim["life"], 0.5) * rate * dt
        emit = int(self.emit_acc)
        self.emit_acc -= emit
        burst = int(self.burst_pending)
        self.burst_pending = 0.0

        n0 = r["emitter_count"]
        n_cont = float(np.clip(n0 * lerp(am["loudness"]["emitters"], e["loudness"]), 1.0, MAX_EMITTERS))
        num = int(math.ceil(n_cont - 1e-6))
        base = self._positions_base(t)
        lay = self.layout
        emitters = np.zeros((MAX_EMITTERS, 4), np.float32)
        emittersB = np.zeros((MAX_EMITTERS, 4), np.float32)
        wander = sim["wander"] * short
        jet = sim["jet"] * (0.6 + 0.8 * e["bass"])
        for i in range(MAX_EMITTERS):
            ph = lay["wander_phase"][i]
            fq = lay["wander_freq"][i]
            x = base[i][0] + wander * math.sin(2 * math.pi * fq[0] * t + ph[0])
            y = base[i][1] + wander * math.sin(2 * math.pi * fq[1] * t + ph[1])
            weight = float(np.clip(n_cont - i, 0.0, 1.0))
            emitters[i] = (x, y, weight, sim["emitter_radius"] * short * lay["radius_mul"][i])
            ang = lay["jet_angles"][i] + 0.35 * math.sin(2 * math.pi * 0.03 * t + ph[0])
            emittersB[i] = (math.cos(ang), math.sin(ang), jet * lay["jet_mul"][i] * weight,
                            self.impulse[i] * sim["impulse_gain"])

        exposure = (tone["exposure"] * self.auto * lerp(am["loudness"]["exposure"], e["loudness"])
                    * (1.0 + am["beat"]["exposure_breath"] * math.cos(2 * math.pi * phase))
                    + am["onset"]["exposure_add"] * e["onset"] * self.auto)
        knee = lerp(am["percussive"]["knee"], e["percussive"])
        gamma = lerp(am["mid"]["gamma"], e["mid"]) * (tone["gamma"] / 0.75)
        T = lerp(am["centroid"]["T"], e["centroid"]) + am["high"]["T_add"] * e["high"] + (grid["T"] - 0.62)
        grain_v = (grain["grain"] + am["high"]["grain_add"] * e["high"]
                   + lerp(am["loudness"]["grain_add"], e["loudness"]))
        vignette = grain["vignette"] + lerp(am["loudness"]["vignette_add"], e["loudness"])
        cyan = pal["cyan_tint"] + am["centroid"]["cyan_tint_add"] * e["centroid"]
        bloom_gain = lerp(am["harmonic"]["bloom"], e["harmonic"]) * (bloom["gain"] / 0.5) + am["onset"]["bloom_add"] * e["onset"]
        ramp_jitter = lerp(am["flux"]["jitter"], e["flux"]) * (tone["ramp_jitter"] / 0.4)

        palette = self._palette_base(t).copy()
        cy = np.array(pal["cyan"])
        palette[3, :3] = palette[3, :3] * (1 - cyan) + cy * cyan
        palette[2, :3] = palette[2, :3] * (1 - 0.5 * cyan) + cy * palette[2, :3].mean() * 0.5 * cyan

        drift = sim["drift"]
        da = r["drift_angle"]
        return {
            "t": t, "dt": dt, "env": e, "hat": hat, "phase": phase, "pulse": pulse,
            "speed": speed, "decay": decay, "scales": scales, "alt_scale": alt_scale,
            "w": (w1, w2, w3), "w_alt": w_alt,
            "emit": emit, "burst": burst, "burst_emitter": self.burst_emitter,
            "num_emitters": num, "emitters": emitters, "emittersB": emittersB,
            "exposure": exposure, "knee": knee, "gamma": gamma, "T": T,
            "grain": grain_v, "vignette": vignette, "cyan": cyan, "bloom_gain": bloom_gain,
            "ramp_jitter": ramp_jitter, "palette": palette, "drift": (drift * math.cos(da), drift * math.sin(da)),
            "jitter_seed": self.jitter_seed, "auto": self.auto,
        }

    def fill_ubo(self, d, s, N):
        """Write the step's parameters into the Frame UBO record `d` (numpy structured row)."""
        P = self.P
        sim, tone, grid, grain, bloom, glyphs = P["sim"], P["tone"], P["grid"], P["grain"], P["bloom"], P["glyphs"]
        r = self.r
        short = min(r["aspect"], 1.0)
        d["seedU"][1] = s["jitter_seed"]
        e = s["env"]
        d["bands"] = (e["sub"], e["bass"], e["mid"], e["high"])
        d["onset"] = (e["onset"], s["phase"], s["pulse"], e["flux"])
        d["spectral"] = (e["centroid"], 0.0, e["loudness"], e["harmonic"])
        d["simB"] = (s["scales"][0], s["scales"][1], s["scales"][2], sim["flow_gain"])
        d["simC"] = (s["w"][0], s["w"][1], s["w"][2], sim["fur"])
        d["simD"] = (sim["time_rates"][0], sim["time_rates"][1], sim["time_rates"][2], sim["diffusion"] * short)
        d["simE"] = (s["drift"][0], s["drift"][1], 0.0, sim["life"])   # splat set by the engine
        d["simF"] = (r["noise"]["offset"][0], r["noise"]["offset"][1],
                     math.cos(r["noise"]["rotation"]), math.sin(r["noise"]["rotation"]))
        d["simG"] = (sim["bias"], 1.0, sim["edge_margin"], sim["splat_sharpness"])
        d["simH"] = (s["alt_scale"], s["w_alt"], s["hat"], 0.0)
        d["emitB"] = (s["num_emitters"], 1.0, sim["impulse_radius"] * short, sim["burst_radius_scale"])
        d["look"] = (s["exposure"], s["knee"], s["gamma"], tone["glow"])
        d["lookB"] = (s["T"], grid["B"], grid["dir_threshold"], grid["dir_max_ramp"])
        d["lookC"] = (s["grain"], grain["flicker"], s["vignette"], s["cyan"])
        d["lookD"] = (s["bloom_gain"], bloom["threshold"], s["ramp_jitter"], grid["peak_mix"])
        d["lookF"] = (bloom["tint"][0], bloom["tint"][1], bloom["tint"][2], glyphs["levels"])
        d["emitters"] = s["emitters"]
        d["emittersB"] = s["emittersB"]
        d["palette"] = s["palette"]
        # simA: speed, decay, trailAdvect, pointSize (pointSize set by the engine)
        d["simA"][0] = s["speed"]
        d["simA"][1] = s["decay"]
        d["simA"][2] = sim["trail_advect"]
