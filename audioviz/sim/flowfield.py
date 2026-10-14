"""P0: curl-noise flow field at 1/4 resolution (RG16F, p-space units per second)."""
import numpy as np

from ..gl.passes import FullscreenPass
from ..gl.pingpong import Target, make_texture


class FlowField:
    def __init__(self, ctx, lib, geom, fb_size, div=4):
        self.ctx = ctx
        self.pass_ = FullscreenPass(lib, geom, "flow", "flow.frag")
        self.size = (max(8, fb_size[0] // div), max(8, fb_size[1] // div))
        self.target = Target(ctx, self.size, 2, "f2", "linear")
        self.zero_ext = make_texture(ctx, (1, 1), 2, "f4", "linear", data=np.zeros(2, "f4").tobytes())
        self.ext = self.zero_ext
        self.ext_scale = (1.0, 1.0)

    @property
    def tex(self):
        return self.target.tex

    def set_external(self, tex, scale=(1.0, 1.0)):
        """Camera stir velocity texture (or None for the 1x1 zero texture)."""
        self.ext = tex if tex is not None else self.zero_ext
        self.ext_scale = scale

    def run(self):
        self.pass_.run(self.target.fbo, [("u_ext", self.ext)], u_extScale=self.ext_scale)

    def release(self):
        self.target.release()
        self.zero_ext.release()
