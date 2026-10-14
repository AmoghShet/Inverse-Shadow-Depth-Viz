"""P4: threshold + 3-level downsample + separable blur + combine, output at 1/4 resolution.

Used twice: on density (feeds the cell grading, mode "density") and on the linear ASCII frame
(glyph glow, mode "color").
"""
from ..gl.passes import FullscreenPass
from ..gl.pingpong import Target


class Bloom:
    def __init__(self, ctx, lib, geom, src_size, mode="density"):
        comps = 1 if mode == "density" else 4
        define = {"MODE_DENSITY": 1} if mode == "density" else {"MODE_COLOR": 1}
        self.pre = FullscreenPass(lib, geom, f"bloom_pre_{mode}", "bloom_prefilter.frag", define)
        self.down = FullscreenPass(lib, geom, "bloom_down", "bloom_down.frag")
        self.blur = FullscreenPass(lib, geom, "bloom_blur", "bloom_blur.frag")
        self.comb = FullscreenPass(lib, geom, "bloom_combine", "bloom_combine.frag")
        w, h = src_size
        self.sizes = [(max(2, w // d), max(2, h // d)) for d in (4, 8, 16)]
        self.levels = [Target(ctx, s, comps, "f2") for s in self.sizes]
        self.temps = [Target(ctx, s, comps, "f2") for s in self.sizes]
        self.out = Target(ctx, self.sizes[0], comps, "f2")
        self.weights = (0.5, 0.35, 0.3)

    @property
    def tex(self):
        return self.out.tex

    def run(self, src_tex, threshold):
        self.pre.run(self.levels[0].fbo, [("u_src", src_tex)], u_threshold=float(threshold))
        for i in (1, 2):
            self.down.run(self.levels[i].fbo, [("u_src", self.levels[i - 1].tex)])
        for lv, tmp in zip(self.levels, self.temps):
            self.blur.run(tmp.fbo, [("u_src", lv.tex)], u_dir=(1.0, 0.0))
            self.blur.run(lv.fbo, [("u_src", tmp.tex)], u_dir=(0.0, 1.0))
        self.comb.run(
            self.out.fbo,
            [("u_l0", self.levels[0].tex), ("u_l1", self.levels[1].tex), ("u_l2", self.levels[2].tex)],
            u_weights=self.weights,
        )
        return self.out.tex

    def release(self):
        for t in self.levels + self.temps + [self.out]:
            t.release()
