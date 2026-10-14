"""Preset JSON for depthviz: load (merged over fog_white), resolve jitter, write resolved_params.json.

A preset value written `[value, jitter_lo, jitter_hi]` (three numbers, lo <= 0 <= hi) resolves to
`value + uniform(lo, hi)` drawn from the run's seed. Keys in VECTOR_KEYS are plain vectors.
`is_jitter` / `resolve_jitter` are copied from audioviz/preset.py (that version reads audioviz's own
module-global VECTOR_KEYS); `deep_merge` is imported unchanged.
"""
import copy
import hashlib
import json
import os

from audioviz.preset import _json_default, deep_merge

from . import PKG_DIR, PRESET_DIR, ROOT_DIR

BASE_PRESET = "fog_white"
SECTIONS = ("meta", "depth", "fog", "light", "halo", "film", "palette", "grain", "audio_map", "random")
VECTOR_KEYS = {"tint", "safe_area", "light_dir"}

# audioviz modules depthviz imports (they are part of what produced a frame)
IMPORTED = [
    "audioviz/gl/context.py", "audioviz/gl/pingpong.py", "audioviz/gl/timing.py",
    "audioviz/audio/analysis.py", "audioviz/audio/player.py", "audioviz/audio/features.py",
    "audioviz/render/ascii.py", "audioviz/render/present.py",
    "audioviz/export/live_recorder.py", "audioviz/preset.py", "audioviz/inputs/camera.py",
]


def preset_path(name_or_path):
    if os.path.isfile(name_or_path):
        return name_or_path
    p = os.path.join(PRESET_DIR, name_or_path + ".json")
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"preset not found: {name_or_path}")


def list_presets():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(PRESET_DIR) if f.endswith(".json"))


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
    """sha1 over depthviz's code/shaders/presets and the audioviz modules it imports."""
    h = hashlib.sha1()
    files = []
    for root, _, names in os.walk(PKG_DIR):
        if "__pycache__" in root:
            continue
        for f in sorted(names):
            if f.endswith((".py", ".glsl", ".frag", ".vert", ".json")):
                files.append(os.path.join(root, f))
    files = sorted(files) + [os.path.join(ROOT_DIR, p) for p in IMPORTED]
    for p in files:
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                h.update(os.path.relpath(p, ROOT_DIR).replace("\\", "/").encode())
                h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def write_resolved(path, resolved, extra=None):
    data = dict(resolved)
    if extra:
        data.update(extra)
    data["code_hash"] = code_hash()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
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


def apply_overrides(preset, sets):
    """--set fog.k=2.5 (value parsed as JSON when possible)."""
    for item in sets or []:
        if "=" not in item:
            raise SystemExit(f"--set needs key=value, got {item!r}")
        key, val = item.split("=", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        d = preset
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in d:
                raise SystemExit(f"--set: unknown key {key}")
            d = d[p]
        if parts[-1] not in d:
            raise SystemExit(f"--set: unknown key {key}")
        d[parts[-1]] = val
    return preset
