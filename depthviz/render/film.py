"""Soft blur of the fog tone and D4 film (the one look: the depth fog as a grainy photograph)."""
from audioviz.gl.pingpong import Target

from ..gl.passes import FullscreenPass


class FogBlur:
    """Separable gaussian on the fog tone at half resolution (sigma from filmB.w): soft blurred shapes."""

    def __init__(self, ctx, lib, geom, fog_size):
        self.size = tuple(fog_size)
        self.p = FullscreenPass(lib, geom, "fog_blur", "fog_blur.frag")
        self.tmp = Target(ctx, self.size, 1, "f2", "linear")
        self.out = Target(ctx, self.size, 1, "f2", "linear")
        self.out.fbo.clear(1.0, 0.0, 0.0, 0.0)

    @property
    def tex(self):
        return self.out.tex

    def run(self, fog_tex):
        self.p.run(self.tmp.fbo, [("u_src", fog_tex)], u_dir=(1.0, 0.0))
        self.p.run(self.out.fbo, [("u_src", self.tmp.tex)], u_dir=(0.0, 1.0))
        return self.out.tex

    def release(self):
        self.tmp.release()
        self.out.release()


class FilmPass:
    def __init__(self, ctx, lib, geom):
        self.pass_ = FullscreenPass(lib, geom, "film", "film.frag")

    def run(self, fbo, soft_tex, halo_tex):
        self.pass_.run(fbo, [("u_fogSoft", soft_tex), ("u_halo", halo_tex)])
