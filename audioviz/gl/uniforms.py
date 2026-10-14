"""The per-frame `Frame` uniform block (std140), defined once here and shared by every program.

Every field is a vec4 / ivec4 / uvec4 (or an array of vec4), so std140 offsets are just 16-byte
steps and the numpy layout matches the GLSL layout with no padding rules to get wrong. The GLSL
declaration is generated from the same table and injected wherever a shader says
`#include "frame_ubo"`.

GLSL 4.1 has no `layout(binding=N)` for blocks, so the binding is set from Python per program.
"""
import numpy as np

BINDING = 0
MAX_EMITTERS = 8

# (name, glsl type, array length or 1, comment)
FIELDS = [
    ("time", "vec4", 1, "t, dt, frameIndex, seedF"),
    ("seedU", "uvec4", 1, "seed, jitterSeed, stepIndex, numParticles"),
    ("res", "vec4", 1, "fbW, fbH, aspect, densityScale"),
    ("gridF", "vec4", 1, "fine cell w, h (px), cols, rows"),
    ("gridC", "vec4", 1, "coarse cell w, h (px), cols, rows"),
    ("bands", "vec4", 1, "sub, low, mid, high"),
    ("onset", "vec4", 1, "strength, beatPhase, beatPulse, flux"),
    ("spectral", "vec4", 1, "centroid, rms, loudness, harmonic"),
    ("simA", "vec4", 1, "speed, decay(per 60Hz step), trailAdvect, pointSize(px)"),
    ("simB", "vec4", 1, "curl scale 1, 2, 3, flowGain"),
    ("simC", "vec4", 1, "octave weight 1, 2, 3, fur (per-particle speed spread)"),
    ("simD", "vec4", 1, "time rate 1, 2, 3, diffusion"),
    ("simE", "vec4", 1, "drift x, drift y, splat, life (s)"),
    ("simF", "vec4", 1, "noise offset x, y, rot cos, rot sin"),
    ("simG", "vec4", 1, "bias, emitter radius scale, edge kill margin, splat sharpness"),
    ("simH", "vec4", 1, "octave-1 alt scale (sub crossfade), alt weight, 0, 0"),
    ("emit", "ivec4", 1, "emitStart, emitCount, burstCount, burstEmitter"),
    ("emitB", "vec4", 1, "numEmitters, impulse strength, impulse radius, burst radius scale"),
    ("look", "vec4", 1, "exposure, knee, gamma, glow"),
    ("lookB", "vec4", 1, "coarse threshold T, band B, dirThreshold, dirMaxRamp"),
    ("lookC", "vec4", 1, "grain, flicker, vignette, cyanTint"),
    ("lookD", "vec4", 1, "bloomGain, bloomThreshold, rampJitter, peakMix"),
    ("lookE", "vec4", 1, "glyphGlow, extFlowGain, view, gridOverlay"),
    ("lookF", "vec4", 1, "bloom tint rgb, levels"),
    ("camA", "vec4", 1, "camera background: level (max ramp), black point, white point, on"),
    ("camB", "vec4", 1, "camera uv scale x, y (cover crop), gamma, brightness"),
    ("camC", "vec4", 1, "camera blur (mip lod), contrast crush 0..1, palette tint 0..1, fine-grid bias where the camera dominates"),
    ("camD", "vec4", 1, "camera-as-smoke: density gain, relax per 60 Hz step, noise breakup 0..1, flow warp (s)"),
    ("camE", "vec4", 1, "camera ghosts: fade half-life (s), drift with the flow (0..1), trail brightness (0..1), 0"),
    ("hazeA", "vec4", 1, "near-black haze: coverage 0..1, brightness (linear), noise scale, drift speed"),
    ("emitters", "vec4", MAX_EMITTERS, "x, y (p-space), weight, radius"),
    ("emittersB", "vec4", MAX_EMITTERS, "jet dir x, y, jet strength, impulse"),
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
