"""P5: per-cell grading for the fine and coarse grids."""
import math

from ..gl.passes import FullscreenPass
from ..gl.pingpong import Target


def grid_geometry(fb_size, cell_1080, ref=1080):
    """Cell size scales with the short side; columns/rows are derived, never fixed."""
    w, h = fb_size
    s = min(w, h) / float(ref)
    cw = max(2, int(round(cell_1080[0] * s)))
    ch = max(4, int(round(cell_1080[1] * s)))
    return (cw, ch), (int(math.ceil(w / cw)), int(math.ceil(h / ch)))


class Grade:
    def __init__(self, ctx, lib, geom):
        self.ctx = ctx
        self.pass_ = FullscreenPass(lib, geom, "grade", "grade.frag")
        self.targets = {}

    def target(self, key, grid):
        t = self.targets.get(key)
        if t is None or t.size != tuple(grid):
            if t is not None:
                t.release()
            t = Target(self.ctx, grid, 4, "f2", "nearest")
            self.targets[key] = t
        return t

    def run(self, key, cell, grid, dens_tex, bloom_tex, vel_tex, cam_tex, ghost_tex):
        t = self.target(key, grid)
        self.pass_.run(
            t.fbo,
            [("u_dens", dens_tex), ("u_bloom", bloom_tex), ("u_vel", vel_tex), ("u_cam", cam_tex),
             ("u_ghost", ghost_tex)],
            u_grid=(float(cell[0]), float(cell[1]), float(grid[0]), float(grid[1])),
        )
        return t.tex

    def release(self):
        for t in self.targets.values():
            t.release()
