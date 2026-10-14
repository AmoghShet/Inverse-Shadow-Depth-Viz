"""Deterministic synthetic depth source: a moving SDF figure (capsule torso, head, a waving arm) in
front of a pillar, a back wall with a doorway, and a floor. `frame(t) -> (depth, mask)`, both
320x240 in GL row order (bottom row first) and mirrored camera space like the worker's output:
depth float32 0 near .. 1 far, softened like a monocular model's output (a halo around edges);
mask uint8, the sharp figure. Usable live (`--source synthetic`), in export and in tests.
"""
import math

import cv2
import numpy as np

W, H = 320, 240


def _capsule(px, py, ax, ay, bx, by, r):
    abx, aby = bx - ax, by - ay
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / max(abx * abx + aby * aby, 1e-9), 0.0, 1.0)
    dx, dy = px - (ax + t * abx), py - (ay + t * aby)
    return np.sqrt(dx * dx + dy * dy) - r


class Synthetic:
    def __init__(self, size=(W, H), soften=3.0, flicker=0.0, seed=0):
        self.w, self.h = size
        self.soften = soften
        self.flicker = flicker
        self.seed = seed
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.x = xx / self.h            # height-normalised, x in [0, 4/3)
        self.y = yy / self.h            # 0 top .. 1 bottom
        self.aspect = self.w / self.h
        # static room: wall 0.92, a doorway (far, 1.0), floor rising toward the camera, a pillar
        x, y = self.x, self.y
        room = np.full((self.h, self.w), 0.92, np.float32)
        door = (x > 0.86) & (x < 1.12) & (y > 0.18) & (y < 0.78)
        room[door] = 1.0
        floor = y > 0.78
        room[floor] = (0.92 - (y[floor] - 0.78) / 0.22 * 0.35).astype(np.float32)
        pillar = (np.abs(x - 0.2) < 0.055)
        room[pillar] = np.minimum(room[pillar], 0.55)
        self.room = room

    def frame(self, t):
        x, y = self.x, self.y
        cx = self.aspect * (0.5 + 0.12 * math.sin(2 * math.pi * 0.07 * t))
        lean = 0.04 * math.sin(2 * math.pi * 0.05 * t + 1.0)
        torso = _capsule(x, y, cx + lean, 0.58, cx, 1.15, 0.17)
        head = np.sqrt((x - cx - 1.3 * lean) ** 2 + (y - 0.36) ** 2) - 0.105
        ang = -0.6 + 0.9 * math.sin(2 * math.pi * 0.23 * t)
        sx, sy = cx + 0.15, 0.62
        ex, ey = sx + 0.2 * math.cos(ang), sy - 0.2 * math.sin(ang)
        arm = _capsule(x, y, sx, sy, ex, ey, 0.045)
        sdf = np.minimum(np.minimum(torso, head), arm)
        fig = sdf < 0.0
        round_ = np.clip(-sdf / 0.12, 0.0, 1.0)
        d = self.room.copy()
        d[fig] = (0.24 - 0.06 * round_[fig] + 0.05 * (arm[fig] < 0.0)).astype(np.float32)
        if self.soften > 0:
            d = cv2.GaussianBlur(d, (0, 0), self.soften)
        if self.flicker > 0:
            rng = np.random.default_rng([self.seed, int(round(t * 1000))])
            d = d + rng.normal(0.0, self.flicker, d.shape).astype(np.float32)
        m = cv2.GaussianBlur(fig.astype(np.float32), (0, 0), 0.8)
        depth = np.ascontiguousarray(np.clip(d, 0.0, 1.0)[::-1]).astype(np.float32)
        mask = np.ascontiguousarray((m * 255.0 + 0.5).astype(np.uint8)[::-1])
        return depth, mask

    def header(self, depth, mask):
        m = mask > 127
        fig = float(np.median(depth[m])) if m.mean() > 0.01 else 0.5
        return {"fig_depth": fig, "mask_cov": float(m.mean()), "stale": 0.0}
