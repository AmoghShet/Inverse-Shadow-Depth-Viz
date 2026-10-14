"""D1 stabilise, D1b filter, D2 fog and D2b halation passes."""
from audioviz.gl.pingpong import PingPong, Target

from ..gl.passes import FullscreenPass


class DepthStage:
    """D1 (RG16F ping-pong, EMA + mask push) and D1b (RGBA16F: depth, gradient, mask), camera space."""

    def __init__(self, ctx, lib, geom, size=(640, 480)):
        self.size = tuple(size)
        self.stab = PingPong(ctx, self.size, 2, "f2", "linear")
        self.stab.clear((1.0, 0.0, 0.0, 1.0))
        self.filt = Target(ctx, self.size, 4, "f2", "linear")
        self.filt.fbo.clear(1.0, 0.0, 0.0, 0.0)
        self.p_stab = FullscreenPass(lib, geom, "depth_stab", "depth_stab.frag")
        self.p_filt = FullscreenPass(lib, geom, "depth_filter", "depth_filter.frag")

    @property
    def tex(self):
        return self.filt.tex

    def run(self, depth_tex, mask_tex, reset=False):
        self.p_stab.run(self.stab.dst, [("u_depth", depth_tex), ("u_mask", mask_tex), ("u_prev", self.stab.src)],
                        u_reset=1.0 if reset else 0.0)
        self.stab.swap()
        self.p_filt.run(self.filt.fbo, [("u_stab", self.stab.src)])
        return self.filt.tex

    def release(self):
        self.stab.release()
        self.filt.release()


class FogStage:
    """D2 at half the framebuffer resolution."""

    def __init__(self, ctx, lib, geom, fb_size):
        w, h = fb_size
        self.size = (max(2, w // 2), max(2, h // 2))
        self.out = Target(ctx, self.size, 4, "f2", "linear")
        self.p = FullscreenPass(lib, geom, "fog", "fog.frag")

    @property
    def tex(self):
        return self.out.tex

    def run(self, filt_tex):
        self.p.run(self.out.fbo, [("u_filt", filt_tex)])
        return self.out.tex

    def release(self):
        self.out.release()


class Halo:
    """D2b: threshold + 1/4, 1/8, 1/16 levels, separable blur, combine at 1/4 (R16F)."""

    def __init__(self, ctx, lib, geom, fb_size):
        self.pre = FullscreenPass(lib, geom, "halo_pre", "halo_pre.frag")
        self.down = FullscreenPass(lib, geom, "halo_down", "halo_down.frag")
        self.blur = FullscreenPass(lib, geom, "halo_blur", "halo_blur.frag")
        self.comb = FullscreenPass(lib, geom, "halo_comb", "halo_comb.frag")
        w, h = fb_size
        self.sizes = [(max(2, w // d), max(2, h // d)) for d in (4, 8, 16)]
        self.levels = [Target(ctx, s, 1, "f2") for s in self.sizes]
        self.temps = [Target(ctx, s, 1, "f2") for s in self.sizes]
        self.out = Target(ctx, self.sizes[0], 1, "f2")
        self.out.fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.weights = (0.45, 0.35, 0.35)

    @property
    def tex(self):
        return self.out.tex

    def run(self, fog_tex, spread=1.0):
        self.pre.run(self.levels[0].fbo, [("u_src", fog_tex)])
        for i in (1, 2):
            self.down.run(self.levels[i].fbo, [("u_src", self.levels[i - 1].tex)])
        for lv, tmp in zip(self.levels, self.temps):
            self.blur.run(tmp.fbo, [("u_src", lv.tex)], u_dir=(1.0, 0.0), u_spread=float(spread))
            self.blur.run(lv.fbo, [("u_src", tmp.tex)], u_dir=(0.0, 1.0), u_spread=float(spread))
        self.comb.run(self.out.fbo, [("u_l0", self.levels[0].tex), ("u_l1", self.levels[1].tex),
                                     ("u_l2", self.levels[2].tex)], u_weights=self.weights)
        return self.out.tex

    def clear(self):
        self.out.fbo.clear(0.0, 0.0, 0.0, 0.0)

    def release(self):
        for t in self.levels + self.temps + [self.out]:
            t.release()
