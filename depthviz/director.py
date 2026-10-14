"""Audio -> fog parameters, once per fixed 60 Hz step (deterministic: song features, resolved params
and step times only; never the wall clock).

One look throughout (the reference photographs); the song moves it, it never switches it:
loudness -> fog density k and exposure (loud = brighter haze); sub -> slow fog swell and the parallax
pivot; bass -> depth contrast; kicks (ranked strength >= 0.35) -> a light flash (150 ms) + a one-tick
grain burst; hats -> grain reseed + a one-tick dust burst; onset -> momentary edge sharpening;
centroid -> tonal spread; harmonic -> halation gain and radius; flux and high -> grain amount;
beat phase -> parallax wobble + exposure breath. Section boundaries change nothing visible (preset
audio_map.section.flip_p 0); the V key flips polarity by hand (2-beat whiteout / blackout).
"""
import math

import numpy as np

from audioviz.audio.features import ENV_KEYS, Envelope, EventCursor, lerp

from .seed import section_layout

SILENT = {"sub": 0.5, "bass": 0.5, "mid": 0.5, "high": 0.35, "onset": 0.0, "centroid": 0.45, "flux": 0.3,
          "harmonic": 0.55, "percussive": 0.4, "loudness": 0.6, "rms": 0.5}
_AR_DEFAULT = {"mid": (0.04, 0.2), "percussive": (0.01, 0.1)}


def smooth01(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class Director:
    def __init__(self, preset, resolved, track=None):
        self.preset = preset
        self.track = track
        self.set_resolved(resolved)

    # -- setup -----------------------------------------------------------------------------
    def set_resolved(self, resolved):
        self.r = resolved
        self.P = resolved["params"]
        am = self.P["audio_map"]
        self.env = {}
        for k in ENV_KEYS:
            a, rl = (am[k]["attack"], am[k]["release"]) if (k in am and "attack" in am[k]) else _AR_DEFAULT.get(k, (0.05, 0.3))
            self.env[k] = Envelope(a, rl, SILENT[k])
        self.env["onset"].value = 0.0
        self.flash = 0.0
        self.burst_grain = 0.0
        self.burst_dust = 0.0
        self.jitter_seed = 1
        self.t_prev = None
        self.kick_count = 0
        self.hat_count = 0
        self.flip_count = 0
        self.section = 0
        self.pending_section = None
        self.last_event = ""
        fog = self.P["fog"]
        self.tones = {"high": (fog["high_key"]["near"], fog["high_key"]["far"]),
                      "inv": (fog["inverted"]["near"], fog["inverted"]["far"])}
        self.polarity = resolved["polarity"]
        self.pair_from = np.array(self.tones[self.polarity], np.float64)
        self.pair_to = self.pair_from.copy()
        self.key_from = self.key_to = 1.0 if self.polarity == "high" else 0.0
        self.flip_start, self.flip_dur, self.flip_kind = -1e9, 1.0, 0
        self.light = dict(resolved["light"])
        tr = self.track
        if tr is not None:
            d = tr.d
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
            self.beats = np.arange(0.0, 7200.0, period)
            self.downbeats = self.beats[::4]
        self.state = {}

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
            return ((t - b[-1]) / period) % 1.0, math.exp(-(t - b[-1]) * 8.0), period
        period = b[i + 1] - b[i]
        return (t - b[i]) / period, math.exp(-(t - b[i]) * 8.0), period

    def _next_downbeat(self, t):
        d = self.downbeats
        j = int(np.searchsorted(d, t, side="left"))
        return float(d[j]) if j < len(d) else t

    def _flip_x(self, t):
        return min(max((t - self.flip_start) / self.flip_dur, 0.0), 1.0)

    # -- events ----------------------------------------------------------------------------
    def flip_now(self, t, beats=None):
        """Polarity flip starting at t over `flip_beats` beats (V key)."""
        _, _, period = self._beat(t)
        s = smooth01(self._flip_x(t))
        cur_pair = self.pair_from + (self.pair_to - self.pair_from) * s
        cur_key = self.key_from + (self.key_to - self.key_from) * s
        target = "inv" if self.polarity == "high" else "high"
        self.pair_from, self.pair_to = cur_pair, np.array(self.tones[target], np.float64)
        self.key_from, self.key_to = cur_key, (1.0 if target == "high" else 0.0)
        self.flip_kind = 1 if target == "inv" else -1      # +1 whiteout (to inverted), -1 blackout
        self.flip_start = t
        nb = beats if beats is not None else self.P["audio_map"]["section"].get("flip_beats", 2)
        self.flip_dur = max(0.25, nb * period)
        self.polarity = target
        self.flip_count += 1
        self.last_event = f"flip -> {'inverted (whiteout)' if target == 'inv' else 'high-key (blackout)'}"

    def _apply_section(self, k, t):
        lay = section_layout(self.r, self.preset, k)
        self.jitter_seed += lay["jitter"]
        self.section = k
        self.last_event = f"section {k}"
        if lay["flip"]:
            self.flip_now(t)

    def take_tick_bursts(self):
        """One-tick bursts (kick grain, hat dust), consumed by the next picture tick."""
        g, d = self.burst_grain, self.burst_dust
        self.burst_grain = 0.0
        self.burst_dust = 0.0
        return g, d

    # -- per step --------------------------------------------------------------------------
    def step(self, t, dt):
        am = self.P["audio_map"]
        t_prev = self.t_prev if self.t_prev is not None else t - dt
        self.t_prev = t
        raw = self.track.sample(t + am["lookahead_s"]) if self.track is not None else dict(SILENT)
        e = {k: self.env[k].update(raw[k], dt) for k in ENV_KEYS}
        kc = am["kick"]
        for (_, s) in self.kicks.crossed(t_prev, t):
            self.kick_count += 1
            if s >= kc["min_rank"]:
                s = (s - kc["min_rank"]) / max(1.0 - kc["min_rank"], 1e-6)
                self.flash = max(self.flash, lerp(kc["flash"], s))
                self.burst_grain = max(self.burst_grain, kc["grain_burst"])
                self.last_event = f"kick {s:.2f} -> flash {self.flash:.2f}"
        for _ in self.hats.crossed(t_prev, t):
            self.hat_count += 1
            self.jitter_seed += 1
            self.burst_dust = max(self.burst_dust, am["hats"]["dust_burst"])
        for (te, _) in self.sections.crossed(t_prev, t):
            self.pending_section = (self.section + 1, self._next_downbeat(te))
        if self.pending_section is not None and t >= self.pending_section[1]:
            self._apply_section(self.pending_section[0], t)
            self.pending_section = None
        self.flash *= math.exp(-dt / max(kc["flash_release"], 1e-3))
        phase, pulse, period = self._beat(t)
        self.state = self._params(t, e, phase, pulse)
        return self.state

    def _params(self, t, e, phase, pulse):
        P = self.P
        am, fog, depth, halo, film = P["audio_map"], P["fog"], P["depth"], P["halo"], P["film"]
        x = self._flip_x(t)
        s = smooth01(x)
        near, far = self.pair_from + (self.pair_to - self.pair_from) * s
        key = self.key_from + (self.key_to - self.key_from) * s
        if 0.0 < x < 1.0 and self.flip_kind != 0:
            emul = 1.0 + 3.0 * math.sin(math.pi * x) if self.flip_kind > 0 else 1.0 - 0.95 * math.sin(math.pi * x)
        else:
            emul = 1.0
        spread = lerp(am["centroid"]["spread"], e["centroid"])
        near = far + (near - far) * spread
        k = fog["k"] * lerp(am["loudness"]["k"], e["loudness"]) + lerp(am["sub"]["k_add"], e["sub"])
        exposure = (fog["exposure"] * lerp(am["loudness"]["exposure"], e["loudness"])
                    * (1.0 + am["beat"]["exposure_breath"] * math.cos(2 * math.pi * phase)) * emul)
        pivot = lerp(am["sub"]["pivot"], e["sub"]) + (fog["pivot"] - 0.5)
        dgamma = lerp(am["bass"]["depth_gamma"], e["bass"])
        da = self.r["drift_angle"]
        period_s = max(fog["parallax_period_s"], 1.0)
        amp = fog["parallax"] * math.sin(2 * math.pi * t / period_s + self.r["drift_phase"])
        wob = lerp(am["beat"]["wobble"], e["bass"]) * math.sin(2 * math.pi * phase)
        par = (amp * math.cos(da) - wob * math.sin(da), amp * math.sin(da) + wob * math.cos(da))
        sigma_r = lerp(am["onset"]["sigma_r"], e["onset"])
        sharpen = depth["sharpen"] + lerp(am["onset"]["sharpen"], e["onset"])
        halo_gain = lerp(am["harmonic"]["halo_gain"], e["harmonic"]) * (halo["gain"] / 0.3)
        halo_spread = lerp(am["harmonic"]["halo_spread"], e["harmonic"]) * halo.get("spread", 1.0)
        grain_v = film["grain"] + lerp(am["high"]["grain_add"], e["high"]) + lerp(am["flux"]["grain_add"], e["flux"])
        return {
            "t": t, "env": e, "phase": phase, "pulse": pulse, "near": near, "far": far, "key": key,
            "flip_x": x if 0.0 < x < 1.0 else (1.0 if x >= 1.0 and self.flip_kind else 0.0),
            "flip_kind": self.flip_kind if 0.0 < x < 1.0 else 0, "exposure": exposure, "k": k, "pivot": pivot,
            "depth_gamma": dgamma, "parallax": par, "light": self.light, "flash": self.flash, "sigma_r": sigma_r,
            "sharpen": sharpen, "halo_gain": halo_gain, "halo_spread": halo_spread, "grain": grain_v,
            "jitter_seed": self.jitter_seed,
        }

    def fill_ubo(self, d, s, look_fps, bursts=(0.0, 0.0)):
        """Write the step's parameters into the Frame UBO record `d` (numpy structured row)."""
        P = self.P
        fog, light, grain, depth, halo, film = P["fog"], P["light"], P["grain"], P["depth"], P["halo"], P["film"]
        e = s["env"]
        lt = s["light"]
        d["seedU"][1] = s["jitter_seed"] & 0xFFFFFFFF
        d["bands"] = (e["sub"], e["bass"], e["mid"], e["high"])
        d["onset"] = (e["onset"], s["phase"], s["pulse"], e["flux"])
        d["spectral"] = (e["centroid"], 0.0, e["loudness"], e["harmonic"])
        d["fogA"] = (s["k"], s["near"], s["far"], fog["gamma"])
        d["fogB"] = (s["exposure"], light["gain"] * lt["gain_mul"], light["blob"] * lt["blob_mul"], light["ambient"])
        d["fogC"] = (lt["centre"][0], lt["centre"][1], lt["dir"][0], lt["dir"][1])
        d["fogD"] = (s["parallax"][0], s["parallax"][1], s["pivot"], light["width"] * lt["width_mul"])
        d["fogE"] = (light["blob_radius"], s["depth_gamma"], s["flash"], fog["swirl"])
        d["fogF"] = (fog["knee"], fog["swirl_scale"], fog["swirl_speed"], fog["curve"])
        a = 1.0 - math.exp(-(1.0 / max(look_fps, 1)) / max(depth["tau_s"], 1e-3))
        d["depthA"][0] = a
        d["depthA"][1] = depth["edge_reset"]
        d["depthA"][3] = depth["mask_margin"]
        d["depthB"][0] = depth["sigma_s"]
        d["depthB"][1] = s["sigma_r"]
        d["depthB"][3] = s["sharpen"]
        d["haloA"] = (halo["threshold"], s["halo_gain"], 0.0, 0.0)
        d["look"] = (s["key"], 0.0, 0.0, 0.0)
        d["filmA"] = (s["grain"] + bursts[0], film["grain_size"], film["clump"], film.get("dark_grain", 0.8))
        d["filmB"] = (grain["flicker"], grain["vignette"], max(0.0, grain["dust"] + bursts[1]), film["blur"])
        d["filmC"] = (film.get("black", 0.07), film.get("white", 0.91), 0.0, 0.0)
        d["flipA"] = (s["flip_x"], float(s["flip_kind"]), float(self.section), 0.0)
        d["palette"] = np.array(self.r["palette"], np.float32)
