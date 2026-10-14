"""Seed -> resolved parameters for depthviz.

One look (the reference photographs): every seed is high-key, neutral gray, film grain. A seed only varies
things inside that look: where the soft light behind the figure sits, fog density and tones within their
ranges, grain amount and size, blur, parallax drift. `numpy.random.default_rng(seed)` draws in a fixed order:
polarity (preset random.p_high_key, 1.0 = always high-key); light centre (safe area) and direction; fog k and
tones (both polarities, so the V key can still flip by hand); grain; palette family (neutral only) +
strength; parallax drift direction and phase; halation; then every remaining jitter triple, section by section.
"""
import copy
import math

import numpy as np

from . import preset as presetmod


def _weighted_choice(rng, weights):
    names = sorted(weights.keys())
    w = np.array([max(0.0, float(weights[n])) for n in names])
    w = w / w.sum()
    return names[int(rng.choice(len(names), p=w))]


def light_layout(rng, rnd, aspect):
    """Light centre in p-space (inside the safe area) and direction of its bright side."""
    lo, hi = rnd["safe_area"]
    u = rng.uniform(lo, hi, size=2)
    centre = [float((u[0] - 0.5) * aspect), float(u[1] - 0.5)]
    a0, a1 = rnd.get("light_dir_deg", [0.0, 360.0])
    ang = math.radians(float(rng.uniform(a0, a1)))
    mul = rng.uniform(0.75, 1.3, size=3)
    return {"centre": centre, "dir": [math.cos(ang), math.sin(ang)], "angle": ang,
            "gain_mul": float(mul[0]), "width_mul": float(mul[1]), "blob_mul": float(mul[2])}


def palette_stops(pal, family, strength):
    neutral = np.array(pal["families"]["neutral"], np.float64)
    fam = np.array(pal["families"][family], np.float64)
    stops = neutral + (fam - neutral) * float(np.clip(strength, 0.0, 1.0))
    stops[:, 3] = fam[:, 3]
    return stops


def resolve(preset, seed, aspect):
    seed = int(seed) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    rnd = preset["random"]
    # 1. polarity (1.0 = always the high-key reference look)
    polarity = "high" if rng.random() < float(rnd.get("p_high_key", 1.0)) else "inv"
    # 2. light
    light = light_layout(rng, rnd, aspect)
    # 3. fog k and tones (both polarities), 4. grain
    fog = presetmod.resolve_jitter(preset["fog"], rng)
    grain = presetmod.resolve_jitter(preset["grain"], rng)
    # 5. palette family + strength
    pal = preset["palette"]
    family = _weighted_choice(rng, pal["family_weights"])
    strength = presetmod.resolve_jitter({"s": pal["strength"]}, rng)["s"]
    # 6. parallax drift direction and phase
    drift_angle = float(rng.uniform(0.0, 2.0 * math.pi))
    drift_phase = float(rng.uniform(0.0, 2.0 * math.pi))
    # 7. halation
    halo = presetmod.resolve_jitter(preset["halo"], rng)
    # 8. the rest, section by section
    params = {"fog": fog, "grain": grain, "halo": halo}
    for sec in presetmod.SECTIONS:
        if sec not in params:
            params[sec] = presetmod.resolve_jitter(preset[sec], rng)
    return {
        "seed": seed,
        "preset": preset["meta"]["name"],
        "aspect": aspect,
        "polarity": polarity,
        "light": light,
        "palette_family": family,
        "palette_strength": strength,
        "palette": palette_stops(pal, family, strength).tolist(),
        "drift_angle": drift_angle,
        "drift_phase": drift_phase,
        "params": params,
    }


def section_layout(resolved, preset, k):
    """Section k: one look, so nothing visible changes (no flip, same light); only the jitter seed moves.
    A flip still happens if the preset sets audio_map.section.flip_p > 0."""
    if k <= 0:
        return {"flip": False, "light": resolved["light"], "jitter": 0}
    rng = np.random.default_rng([resolved["seed"], 7919, k])
    flip = bool(rng.random() < float(preset["audio_map"]["section"].get("flip_p", 0.0)))
    return {"flip": flip, "light": resolved["light"], "jitter": int(rng.integers(1, 1 << 20))}


def frozen_preset(resolved, preset):
    """Resolved params with the seed's draws pinned (Ctrl+S)."""
    p = copy.deepcopy(resolved["params"])
    p["palette"]["family_weights"] = {resolved["palette_family"]: 1.0}
    p["palette"]["strength"] = resolved["palette_strength"]
    p["random"]["p_high_key"] = 1.0 if resolved["polarity"] == "high" else 0.0
    p["meta"]["aspect"] = preset["meta"].get("aspect", "16:9")
    return p
