"""Exposure meter: 64x36 summary of the density buffer, read back asynchronously through PBOs.

The CPU gets (mean density over covered area, coverage, peak) one or two frames late, without
stalling the GPU. The Director turns it into a slow auto-exposure so the tone curve sits in the
same range whatever the seed, tier, resolution or emitter count.
"""
import numpy as np

from ..gl.passes import FullscreenPass
from ..gl.pingpong import Target

SIZE = (64, 36)


class Meter:
    def __init__(self, ctx, lib, geom):
        self.ctx = ctx
        self.pass_ = FullscreenPass(lib, geom, "meter", "meter.frag")
        self.target = Target(ctx, SIZE, 4, "f4", "nearest")
        nbytes = SIZE[0] * SIZE[1] * 4 * 4
        self.pbos = [ctx.buffer(reserve=nbytes, dynamic=True) for _ in range(2)]
        self.pending = [False, False]
        self.i = 0
        self.last = None

    def measure(self, dens_tex):
        """Queue a measurement; returns the result of the one queued two calls ago (or None)."""
        self.pass_.run(self.target.fbo, [("u_dens", dens_tex)])
        result = None
        j = self.i
        if self.pending[j]:
            result = self._decode(self.pbos[j].read())
        self.target.tex.read_into(self.pbos[j])
        self.pending[j] = True
        self.i = 1 - j
        if result is not None:
            self.last = result
        return result

    def measure_now(self, dens_tex):
        self.pass_.run(self.target.fbo, [("u_dens", dens_tex)])
        self.last = self._decode(self.target.tex.read())
        return self.last

    @staticmethod
    def _decode(raw):
        a = np.frombuffer(raw, np.float32).reshape(SIZE[1], SIZE[0], 4)
        total = float(a[..., 0].sum())
        cov = float(a[..., 1].mean())
        covered_cells = float(a[..., 1].sum())
        mean_cov = total / max(covered_cells, 1e-3)
        return {"mean_covered": mean_cov, "coverage": cov, "peak": float(a[..., 2].max())}

    def release(self):
        self.target.release()
        for b in self.pbos:
            b.release()
