"""P3: density trail buffer (R16F ping-pong), advected by the flow and decayed each step."""
from ..gl.passes import FullscreenPass
from ..gl.pingpong import PingPong


class Ghost:
    """Camera ghost buffer (RG16F, 1/4 res): R live camera level, G peak-hold with a fade."""

    def __init__(self, ctx, lib, geom, fb_size):
        self.size = (max(8, fb_size[0] // 4), max(8, fb_size[1] // 4))
        self.pass_ = FullscreenPass(lib, geom, "ghost", "ghost.frag")
        self.buf = PingPong(ctx, self.size, 2, "f2", "linear")
        self.buf.clear()

    @property
    def tex(self):
        return self.buf.src

    def step(self, cam_tex, vel_tex):
        self.pass_.run(self.buf.dst, [("u_ghost", self.buf.src), ("u_cam", cam_tex), ("u_vel", vel_tex)])
        self.buf.swap()

    def clear(self):
        self.buf.clear()

    def release(self):
        self.buf.release()


class Trails:
    def __init__(self, ctx, lib, geom, fb_size, density_scale=1.0):
        self.size = (max(8, int(round(fb_size[0] * density_scale))), max(8, int(round(fb_size[1] * density_scale))))
        self.pass_ = FullscreenPass(lib, geom, "trails", "trails.frag")
        self.dens = PingPong(ctx, self.size, 1, "f2", "linear")
        self.dens.clear()

    @property
    def tex(self):
        """Current density (after the latest advect + splat)."""
        return self.dens.src

    def advect(self, vel_tex, ghost_tex):
        """Write advected/decayed density (pulled up toward the camera ghost image) into the back
        buffer and return its FBO for splatting."""
        self.pass_.run(self.dens.dst, [("u_dens", self.dens.src), ("u_vel", vel_tex), ("u_ghost", ghost_tex)])
        return self.dens.dst

    def swap(self):
        self.dens.swap()

    def clear(self):
        self.dens.clear()

    def release(self):
        self.dens.release()
