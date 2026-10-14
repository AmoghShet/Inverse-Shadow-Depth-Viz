"""The per-tick `Frame` uniform block (std140) for depthviz, defined once here and shared by every program.

Same code as audioviz/gl/uniforms.py (`_dtype`, `glsl_block`, `FrameUBO`), with depthviz's own FIELDS.
Every field is a vec4 / ivec4 / uvec4 (or an array of vec4), so std140 offsets are 16-byte steps and
the numpy layout matches the GLSL one. `#include "frame_ubo"` injects the GLSL declaration.
"""
import numpy as np

BINDING = 0

# (name, glsl type, array length or 1, comment)
FIELDS = [
    ("time", "vec4", 1, "t (song s), dt (tick s), tick index, seedF"),
    ("seedU", "uvec4", 1, "seed, jitterSeed, tick, frame"),
    ("res", "vec4", 1, "fbW, fbH, aspect, 0"),
    ("bands", "vec4", 1, "sub, bass, mid, high"),
    ("onset", "vec4", 1, "strength, beatPhase, beatPulse, flux"),
    ("spectral", "vec4", 1, "centroid, rms, loudness, harmonic"),
    ("fogA", "vec4", 1, "k (density), nearTone, farTone, gamma"),
    ("fogB", "vec4", 1, "exposure multiplier, lightGain, blob, ambient"),
    ("fogC", "vec4", 1, "light centre x, y (p-space), dir cos, dir sin"),
    ("fogD", "vec4", 1, "parallax dx, dy (uv), pivot, light edge width"),
    ("fogE", "vec4", 1, "blob radius, depth gamma, flash, swirl"),
    ("fogF", "vec4", 1, "shoulder knee, swirl scale, swirl speed, fog curve exponent"),
    ("depthA", "vec4", 1, "ema a (per tick), edge reset threshold, figDepth, mask margin"),
    ("depthB", "vec4", 1, "sigma_s (texels), sigma_r, maskOn, sharpen"),
    ("haloA", "vec4", 1, "threshold, gain, 0, 0"),
    ("look", "vec4", 1, "keyness (1 high-key, 0 inverted), 0, 0, 0"),
    ("filmA", "vec4", 1, "grain, grain clump size (px at 1080p), clump mix, grain share in the darks"),
    ("filmB", "vec4", 1, "flicker, vignette, dust, blur sigma (half-res texels)"),
    ("filmC", "vec4", 1, "print black point, white point, 0, 0"),
    ("camA", "vec4", 1, "camera uv scale x, y (cover crop), depth on, stale"),
    ("flipA", "vec4", 1, "flip progress 0..1, kind (+1 to inverted / -1 to high-key / 0), section, 0"),
    ("palette", "vec4", 4, "rgb, stop position"),
]

_NP = {"vec4": ("<f4", 4), "ivec4": ("<i4", 4), "uvec4": ("<u4", 4)}


def _dtype():
    fields = []
    for name, gtype, n, _ in FIELDS:
        base, comps = _NP[gtype]
        shape = (n, comps) if n > 1 else (comps,)
        fields.append((name, base, shape))
    dt = np.dtype(fields)
    assert dt.itemsize % 16 == 0
    return dt


FRAME_DTYPE = _dtype()


def glsl_block():
    lines = ["layout(std140) uniform Frame {"]
    for name, gtype, n, comment in FIELDS:
        arr = f"[{n}]" if n > 1 else ""
        lines.append(f"    {gtype} u_{name}{arr};  // {comment}")
    lines.append("};")
    return "\n".join(lines) + "\n"


class FrameUBO:
    def __init__(self, ctx):
        self.ctx = ctx
        self.data = np.zeros(1, FRAME_DTYPE)[0]
        self.buffer = ctx.buffer(reserve=FRAME_DTYPE.itemsize, dynamic=True)

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, k, v):
        self.data[k] = v

    def upload(self):
        self.buffer.write(self.data.tobytes())
        self.buffer.bind_to_uniform_block(BINDING)

    def attach(self, program):
        """Point a program's Frame block at our binding. Programs that do not use it are skipped."""
        try:
            program["Frame"].binding = BINDING
        except KeyError:
            pass

    def release(self):
        self.buffer.release()
