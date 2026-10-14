"""Preset JSON: load (merged over the base preset), resolve jitter, write resolved_params.json.

A preset value written `[value, jitter_lo, jitter_hi]` (three numbers with lo <= 0 <= hi) resolves
to `value + uniform(lo, hi)` drawn from the run's seed. Keys listed in VECTOR_KEYS are plain vectors.
"""
import copy
import hashlib
import json
import os

from . import PKG_DIR, PRESET_DIR

BASE_PRESET = "ink_cold"
SECTIONS = ("meta", "grid", "glyphs", "tone", "palette", "bloom", "grain", "sim", "audio_map", "random")
VECTOR_KEYS = {"scales", "weights", "time_rates", "tint", "cyan", "safe_area", "emitters", "flow_scale"}


def preset_path(name_or_path):
    if os.path.isfile(name_or_path):
        return name_or_path
    p = os.path.join(PRESET_DIR, name_or_path + ".json")
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"preset not found: {name_or_path}")


def list_presets():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(PRESET_DIR) if f.endswith(".json"))


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(name_or_path=BASE_PRESET):
    with open(preset_path(BASE_PRESET), "r", encoding="utf-8") as fh:
        base = json.load(fh)
    path = preset_path(name_or_path)
    if os.path.abspath(path) == os.path.abspath(preset_path(BASE_PRESET)):
        data = base
    else:
        with open(path, "r", encoding="utf-8") as fh:
            data = deep_merge(base, json.load(fh))
    missing = [s for s in SECTIONS if s not in data]
    if missing:
        raise ValueError(f"preset {path} is missing sections {missing}")
    data["meta"]["path"] = path
    return data


def is_jitter(key, v):
    return (
        key not in VECTOR_KEYS
        and isinstance(v, list)
        and len(v) == 3
        and all(isinstance(x, (int, float)) for x in v)
        and v[1] <= 0.0 <= v[2]
    )


def resolve_jitter(section, rng):
    """Resolve every jitter triple in a section dict, walking keys in sorted order (stable draws)."""
    out = {}
    for k in sorted(section.keys()):
        v = section[k]
        if is_jitter(k, v):
            out[k] = float(v[0] + rng.uniform(v[1], v[2]))
        elif isinstance(v, dict):
            out[k] = resolve_jitter(v, rng)
        else:
            out[k] = copy.deepcopy(v)
    return out


def code_hash():
    h = hashlib.sha1()
    for root, _, files in os.walk(PKG_DIR):
        if "__pycache__" in root:
            continue
        for f in sorted(files):
            if f.endswith((".py", ".glsl", ".frag", ".vert", ".json")):
                with open(os.path.join(root, f), "rb") as fh:
                    h.update(f.encode())
                    h.update(fh.read())
    return h.hexdigest()[:12]


def write_resolved(path, resolved, extra=None):
    data = dict(resolved)
    if extra:
        data.update(extra)
    data["code_hash"] = code_hash()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    return path


def save_user_preset(frozen_params, origin, name=None):
    """Ctrl+S: write frozen resolved values (no jitter, choices pinned) as a new preset file."""
    n = 1
    while name is None:
        cand = f"user_{n:02d}"
        if not os.path.exists(os.path.join(PRESET_DIR, cand + ".json")):
            name = cand
        n += 1
    params = copy.deepcopy(frozen_params)
    params["meta"] = dict(params.get("meta", {}))
    params["meta"]["name"] = name
    params["meta"].pop("path", None)
    params["meta"]["frozen_from"] = origin
    path = os.path.join(PRESET_DIR, name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, default=_json_default)
    return path


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
    except Exception:
        pass
    raise TypeError(f"not JSON serialisable: {type(o)}")
