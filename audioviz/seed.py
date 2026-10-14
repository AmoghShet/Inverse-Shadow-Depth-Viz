"""Seed -> resolved parameters.

`numpy.random.default_rng(seed)` draws sub-seeds in a fixed order: noise domain offset and
rotation, drift direction, emitter count and positions, flow scale, octave weight jitter, palette
variant (+ tint jitter), glyph set, grid pair, then every preset jitter triple. Invariants (black
background, cold palette family, tone curve shape) are never randomised; only bounded values are.
Section boundaries get their own layout from `section_layout(seed, k)`.
"""
import math

import numpy as np

from . import preset as presetmod
from .gl.uniforms import MAX_EMITTERS


def _weighted_choice(rng, weights):
    names = sorted(weights.keys())
    w = np.array([max(0.0, float(weights[n])) for n in names])
    w = w / w.sum()
    return names[int(rng.choice(len(names), p=w))]


def _short(aspect):
    return min(aspect, 1.0)


def emitter_layout(rng, rnd, sim, aspect, drift_angle):
    """Positions in p-space (height 1, centred). Rejection-sample a min spacing in short-side units."""
    lo, hi = rnd["safe_area"]
    spacing = rnd["min_spacing"] * _short(aspect)
    pts = []
    tries = 0
    while len(pts) < MAX_EMITTERS and tries < 4000:
        tries += 1
        u = rng.uniform(lo, hi, size=2)
        p = np.array([(u[0] - 0.5) * aspect, u[1] - 0.5])
        sp = spacing if tries < 2000 else spacing * 0.5
        if all(np.linalg.norm(p - q) >= sp for q in pts):
            pts.append(p)
    while len(pts) < MAX_EMITTERS:  # degenerate tiny frames
        pts.append(np.array([0.0, 0.0]))
    # shift the layout upstream of the drift so plumes cross the frame instead of leaving it
    up = -np.array([math.cos(drift_angle), math.sin(drift_angle)]) * sim.get("upstream_shift", 0.15) * _short(aspect)
    half = np.array([aspect * 0.5, 0.5])
    pts = [np.clip(p + up, -half * 0.8, half * 0.8) for p in pts]
    spread = math.radians(sim["jet_spread_deg"])
    jet_angles = [drift_angle + rng.uniform(-spread, spread) for _ in range(MAX_EMITTERS)]
    jet_mul = rng.uniform(0.5, 1.5, size=MAX_EMITTERS)
    wander_phase = rng.uniform(0, 2 * math.pi, size=(MAX_EMITTERS, 2))
    wander_freq = rng.uniform(0.015, 0.06, size=(MAX_EMITTERS, 2))
    radius_mul = rng.uniform(0.7, 1.4, size=MAX_EMITTERS)
    return {
        "positions": [p.tolist() for p in pts],
        "jet_angles": [float(a) for a in jet_angles],
        "jet_mul": jet_mul.tolist(),
        "wander_phase": wander_phase.tolist(),
        "wander_freq": wander_freq.tolist(),
        "radius_mul": radius_mul.tolist(),
    }


def palette_stops(pal, variant, tint, tint_jitter_rgb):
    stops = np.array(pal["variants"][variant], dtype=np.float64)
    stops[:, :3] = np.clip(stops[:, :3] + tint_jitter_rgb[None, :] * (stops[:, 3:4] > 0), 0.0, 1.0)
    return stops


def resolve(preset, seed, aspect):
    """Return the full resolved-parameter dict for (preset, seed, aspect)."""
    seed = int(seed) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    rnd = preset["random"]

    # 1. noise domain offset and rotation
    noise_offset = rng.uniform(-100.0, 100.0, size=2)
    noise_rot = float(rng.uniform(0.0, 2.0 * math.pi))
    # 2. drift direction (mostly upward like rising smoke, sometimes sideways or sinking ink)
    drift_angle = float(math.pi / 2 + rng.normal(0.0, 0.9))
    # 3. emitter count and layout
    n_lo, n_hi = rnd["emitters"]
    emitter_count = int(rng.integers(n_lo, n_hi + 1))
    sim_raw = preset["sim"]
    layout = emitter_layout(rng, rnd, sim_raw, aspect, drift_angle)
    # 4. flow scale
    flow_scale_mul = float(rng.uniform(*rnd["flow_scale"]))
    # 5. octave weight jitter
    oj = rnd["octave_jitter"]
    octave_mul = (1.0 + rng.uniform(-oj, oj, size=3)).tolist()
    # 6. palette variant + tint jitter
    pal = preset["palette"]
    palette_variant = _weighted_choice(rng, pal["variant_weights"])
    tint_jitter = rng.uniform(-pal["tint_jitter"], pal["tint_jitter"], size=3)
    # 7. glyph set
    glyph_set = _weighted_choice(rng, preset["glyphs"]["set_weights"])
    # 8. grid pair
    grid_pair = _weighted_choice(rng, preset["grid"]["pair_weights"])
    # 9. every jitter triple, section by section in a fixed order
    params = {}
    for sec in presetmod.SECTIONS:
        params[sec] = presetmod.resolve_jitter(preset[sec], rng)

    return {
        "seed": seed,
        "preset": preset["meta"]["name"],
        "aspect": aspect,
        "noise": {"offset": noise_offset.tolist(), "rotation": noise_rot},
        "drift_angle": drift_angle,
        "emitter_count": emitter_count,
        "layout": layout,
        "flow_scale_mul": flow_scale_mul,
        "octave_mul": octave_mul,
        "palette_variant": palette_variant,
        "tint_jitter": tint_jitter.tolist(),
        "palette": palette_stops(pal, palette_variant, None, tint_jitter).tolist(),
        "glyph_set": glyph_set,
        "grid_pair": grid_pair,
        "params": params,
    }


def section_layout(resolved, preset, k):
    """New emitter layout / palette variant / grid pair for section k (k >= 1)."""
    if k <= 0:
        return {
            "layout": resolved["layout"],
            "palette": resolved["palette"],
            "palette_variant": resolved["palette_variant"],
            "grid_pair": resolved["grid_pair"],
        }
    rng = np.random.default_rng([resolved["seed"], 7919, k])
    lay = emitter_layout(rng, preset["random"], preset["sim"], resolved["aspect"], resolved["drift_angle"])
    pal = preset["palette"]
    variant = _weighted_choice(rng, pal["variant_weights"])
    pair = _weighted_choice(rng, preset["grid"]["pair_weights"])
    return {
        "layout": lay,
        "palette": palette_stops(pal, variant, None, np.array(resolved["tint_jitter"])).tolist(),
        "palette_variant": variant,
        "grid_pair": pair,
    }


def frozen_preset(resolved):
    """Resolved params with the seed's discrete choices pinned, for Ctrl+S."""
    import copy

    p = copy.deepcopy(resolved["params"])
    p["palette"]["variant_weights"] = {resolved["palette_variant"]: 1.0}
    p["glyphs"]["set_weights"] = {resolved["glyph_set"]: 1.0}
    p["grid"]["pair_weights"] = {resolved["grid_pair"]: 1.0}
    return p
