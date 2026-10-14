"""D5 present (the film frame, or the depth / fog / mask debug views)."""
from ..gl.passes import FullscreenPass

VIEWS = ("frame", "depth", "fog", "mask")


class Present:
    def __init__(self, lib, geom):
        self.pass_ = FullscreenPass(lib, geom, "present", "present.frag")

    def run(self, fbo, frame, filt, fog, halo, view=0, viewport=None):
        self.pass_.run(fbo, [("u_frame", frame), ("u_filt", filt), ("u_fog", fog), ("u_halo", halo)],
                       viewport=viewport, u_view=int(view))
