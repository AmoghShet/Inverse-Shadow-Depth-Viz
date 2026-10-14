"""P6: directional glyph atlas builder and the ASCII pass.

Atlas: 5 rows (isotropic, horizontal, '/', vertical, '\\') x 10 columns, R8, one per grid size.
Each glyph is rendered at 4x the cell size, area-downsampled to the exact cell (no atlas
aliasing), centred on its ink, peak-normalised so its brightest pixel is 1.0 (thin strokes no
longer cap brightness near 40%), and each row is re-sorted by measured ink. Directional rows hold
7 glyphs; columns 7..9 repeat the brightest isotropic glyphs (highlights read as blobs).
"""
import os
import sys

import cv2
import numpy as np

from ..gl.passes import FullscreenPass
from ..gl.pingpong import make_texture

SUPERSAMPLE = 4
ROW_KEYS = ("iso", "h", "d1", "v", "d2")

_MONO_FONTS = [
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\lucon.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/Library/Fonts/Courier New.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def _mono_font_path():
    for p in _MONO_FONTS:
        if os.path.isfile(p):
            return p
    return None


_GLYPH_CACHE = {}
_FONT_CACHE = {}


class GlyphRenderer:
    """Renders single characters at SUPERSAMPLE x the cell size, centred on their ink."""

    def __init__(self, cell_w, cell_h, font="hershey", weight=1.0):
        self.cw, self.ch = int(cell_w), int(cell_h)
        self.weight = weight
        self.W, self.H = self.cw * SUPERSAMPLE, self.ch * SUPERSAMPLE
        self.font = font
        self.pil_font = None
        if font == "mono":
            path = _mono_font_path()
            if path is None:
                self.font = "hershey"
            else:
                from PIL import ImageFont

                key = (path, int(self.H * 0.78))
                if key not in _FONT_CACHE:
                    _FONT_CACHE[key] = ImageFont.truetype(path, size=key[1])
                self.pil_font = _FONT_CACHE[key]
        if self.font == "hershey":
            self.face = cv2.FONT_HERSHEY_SIMPLEX
            self.thick = max(1, int(round(self.H / 15.0 * weight)))
            (mw, mh), _ = cv2.getTextSize("M", self.face, 1.0, self.thick)
            widest = max(cv2.getTextSize(c, self.face, 1.0, self.thick)[0][0] for c in "M@#W%&")
            self.scale = min(0.6 * self.H / max(mh, 1), 0.9 * self.W / max(widest, 1))

    def raw(self, c, centre=True):
        img = np.zeros((self.H, self.W), np.uint8)
        if c == " ":
            return img
        if self.font == "hershey":
            (tw, th), base = cv2.getTextSize(c, self.face, self.scale, self.thick)
            (_, mh), _ = cv2.getTextSize("M", self.face, self.scale, self.thick)
            org = ((self.W - tw) // 2, (self.H + mh) // 2)   # common baseline
            cv2.putText(img, c, org, self.face, self.scale, 255, self.thick, cv2.LINE_AA)
        else:
            from PIL import Image, ImageDraw

            im = Image.new("L", (self.W, self.H), 0)
            # "ms" = middle/baseline: every glyph shares the baseline, like text
            asc, desc = self.pil_font.getmetrics()
            baseline = self.H // 2 + (asc - desc) // 2
            ImageDraw.Draw(im).text((self.W // 2, baseline), c, fill=255, font=self.pil_font, anchor="ms")
            img = np.array(im)
        return _centre_ink(img) if centre else img

    def glyph(self, c, peak_normalise=True, centre=True):
        """Float32 [ch, cw] in 0..1. centre=True centres each glyph on its ink (smoke); False keeps
        the font baseline (HUD text)."""
        key = (self.font, self.cw, self.ch, self.weight, c, peak_normalise, centre)
        hit = _GLYPH_CACHE.get(key)
        if hit is not None:
            return hit
        big = self.raw(c, centre)
        small = cv2.resize(big, (self.cw, self.ch), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if peak_normalise and small.max() > 1e-6:
            small /= small.max()
        _GLYPH_CACHE[key] = small
        return small


def _centre_ink(img):
    ys, xs = np.nonzero(img > 8)
    if len(xs) == 0:
        return img
    H, W = img.shape
    dx = int(round((W - 1) / 2.0 - (xs.min() + xs.max()) / 2.0))
    dy = int(round((H - 1) / 2.0 - (ys.min() + ys.max()) / 2.0))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)


def _sorted_row(chars, rnd):
    gl = [(float(rnd.glyph(c).mean()) if c != " " else -1.0, c) for c in chars]
    gl.sort(key=lambda g: g[0])
    return [c for _, c in gl]


def atlas_layout(glyph_sets, set_name, dir_rows, cell, font="hershey"):
    """Return (rows of chars, renderer). rows[0] = 10 iso glyphs, rows[1..4] = 10 each."""
    rnd = GlyphRenderer(cell[0], cell[1], font)
    iso = list(glyph_sets[set_name]["iso"])
    assert len(iso) == 10, f"glyph set {set_name}: iso row needs 10 glyphs"
    iso = _sorted_row(iso, rnd)
    rows = [iso]
    for k in ROW_KEYS[1:]:
        d = list(dir_rows[k])
        assert len(d) == 7, f"directional row {k} needs 7 glyphs"
        rows.append(_sorted_row(d, rnd) + iso[7:])
    return rows, rnd


def build_atlas(glyph_sets, set_name, dir_rows, cell, font="hershey"):
    """uint8 atlas [5*ch, 10*cw] (row 0 at the top) and the char layout."""
    rows, rnd = atlas_layout(glyph_sets, set_name, dir_rows, cell, font)
    cw, ch = cell
    atlas = np.zeros((5 * ch, 10 * cw), np.float32)
    for r, chars in enumerate(rows):
        for c, g in enumerate(chars):
            atlas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = rnd.glyph(g)
    return (np.clip(atlas, 0, 1) * 255.0 + 0.5).astype(np.uint8), rows


def build_hud_atlas(cell, font="hershey"):
    """Printable ASCII 32..126 in one row, not peak-normalised (plain text)."""
    rnd = GlyphRenderer(cell[0], cell[1], font, weight=0.9)
    cw, ch = cell
    atlas = np.zeros((ch, 95 * cw), np.float32)
    for i in range(95):
        g = rnd.glyph(chr(32 + i), peak_normalise=False, centre=False)
        if g.max() > 0:
            g = np.clip(g * 1.6, 0, 1)
        atlas[:, i * cw:(i + 1) * cw] = g
    return (atlas * 255.0 + 0.5).astype(np.uint8)


class AsciiPass:
    def __init__(self, ctx, lib, geom, preset_glyphs):
        self.ctx = ctx
        self.pass_ = FullscreenPass(lib, geom, "ascii", "ascii.frag")
        self.glyphs = preset_glyphs
        self._atlases = {}

    def atlas(self, set_name, cell, font=None):
        font = font or self.glyphs.get("font", "hershey")
        key = (set_name, tuple(cell), font)
        tex = self._atlases.get(key)
        if tex is None:
            img, _ = build_atlas(self.glyphs["sets"], set_name, self.glyphs["dir_rows"], cell, font)
            tex = make_texture(self.ctx, (img.shape[1], img.shape[0]), 1, "f1", "nearest", data=img.tobytes())
            self._atlases[key] = tex
        return tex

    def warm(self, set_name, cells):
        """Build atlases ahead of time (a build takes ~50-100 ms; doing it mid-song drops frames)."""
        for c in cells:
            self.atlas(set_name, c)

    def run(self, fbo, cellF_tex, cellC_tex, atlasF, atlasC):
        self.pass_.run(
            fbo,
            [("u_cellF", cellF_tex), ("u_cellC", cellC_tex), ("u_atlasF", atlasF), ("u_atlasC", atlasC)],
        )

    def release(self):
        for t in self._atlases.values():
            t.release()
        self._atlases.clear()


def _dump_atlases(out_dir):
    """Debug: write the atlases for every set at the default cell pair, scaled up 4x."""
    import json

    from .. import PRESET_DIR

    with open(os.path.join(PRESET_DIR, "ink_cold.json"), "r", encoding="utf-8") as fh:
        p = json.load(fh)
    g = p["glyphs"]
    os.makedirs(out_dir, exist_ok=True)
    for font in ("hershey", "mono"):
        for s in g["sets"]:
            for cell in ((6, 12), (12, 24), (4, 8), (8, 16), (16, 32)):
                img, rows = build_atlas(g["sets"], s, g["dir_rows"], cell, font)
                big = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(out_dir, f"atlas_{font}_{s}_{cell[0]}x{cell[1]}.png"), big)
            print(font, s, ["".join(r) for r in rows])


if __name__ == "__main__":
    _dump_atlases(sys.argv[1] if len(sys.argv) > 1 else "out/atlas")
