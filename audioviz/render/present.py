"""P7 present (ASCII + glyph glow, raw density view, flow arrows view) and the HUD overlay.

The HUD is text drawn with the same glyph rendering as the ASCII pass: a small grid of character
codes uploaded when the text changes, drawn over a dim backdrop.
"""
import moderngl
import numpy as np

from ..gl.passes import FullscreenPass
from ..gl.pingpong import make_texture
from .ascii import build_hud_atlas

VIEWS = ("ascii", "density", "flow")


class Present:
    def __init__(self, lib, geom):
        self.pass_ = FullscreenPass(lib, geom, "present", "present.frag")

    def run(self, fbo, frame, glow, dens, bloom, vel, view=0, glow_gain=1.0, viewport=None):
        self.pass_.run(
            fbo,
            [("u_frame", frame), ("u_glow", glow), ("u_dens", dens), ("u_bloom", bloom), ("u_vel", vel)],
            viewport=viewport,
            u_view=int(view),
            u_glowGain=float(glow_gain),
        )


class Hud:
    def __init__(self, ctx, lib, geom, out_height, font="mono"):
        self.ctx = ctx
        self.pass_ = FullscreenPass(lib, geom, "hud", "hud.frag")
        s = max(1.0, out_height / 1080.0)
        self.cell = (int(round(8 * s)), int(round(16 * s)))
        img = build_hud_atlas(self.cell, font)
        self.atlas = make_texture(ctx, (img.shape[1], img.shape[0]), 1, "f1", "nearest", data=img.tobytes())
        self.text_tex = None
        self.size = (0, 0)
        self._last = None

    def set_lines(self, lines):
        """lines: list of (text, colour index) with colour 0 = grey, 1 = red, 2 = cyan."""
        key = tuple(lines)
        if key == self._last:
            return
        self._last = key
        cols = max([len(t) for t, _ in lines] + [1])
        rows = max(1, len(lines))
        grid = np.zeros((rows, cols, 2), np.uint8)
        for r, (text, ci) in enumerate(lines):
            for c, chr_ in enumerate(text):
                o = ord(chr_)
                grid[r, c, 0] = o if 32 <= o < 127 else 63
                grid[r, c, 1] = ci
            grid[r, len(text):, 0] = 0
            if len(text) < cols:
                grid[r, len(text):cols, 0] = 0
        if self.text_tex is None or self.size != (cols, rows):
            if self.text_tex is not None:
                self.text_tex.release()
            self.text_tex = make_texture(self.ctx, (cols, rows), 2, "f1", "nearest")
            self.size = (cols, rows)
        self.text_tex.write(grid.tobytes())

    def draw(self, fbo, out_size, origin=(12, 12)):
        if self.text_tex is None:
            return
        ctx = self.ctx
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        prog = self.pass_.h.prog
        self.text_tex.use(0)
        self.atlas.use(1)
        for name, val in (("u_text", 0), ("u_hudAtlas", 1)):
            try:
                prog[name].value = val
            except KeyError:
                pass
        self.pass_.set("u_hudCell", (float(self.cell[0]), float(self.cell[1])))
        self.pass_.set("u_hudOrigin", (float(origin[0]), float(origin[1])))
        self.pass_.set("u_outSize", (float(out_size[0]), float(out_size[1])))
        fbo.use()
        fbo.viewport = (0, 0, out_size[0], out_size[1])
        self.pass_.geom.fullscreen_vao(self.pass_.h).render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

    def release(self):
        self.atlas.release()
        if self.text_tex is not None:
            self.text_tex.release()
