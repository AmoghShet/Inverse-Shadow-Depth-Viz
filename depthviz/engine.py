"""GPU pipeline shared by live, export and tests. One look: the depth fog as a grainy photograph.

Per 60 Hz step:      director only (audio -> parameters; no GPU work).
Per picture tick:    D1 stabilise -> D1b filter (camera space) -> D2 fog (1/2 res) -> D2b halation
                     -> soft blur (1/2 res) -> D4 film -> D5 present.
Between ticks:       D5 only (`present_only`), so a 12 fps look holds the frame.
Depth comes from textures set by the caller: the camera worker's (`set_depth_textures`) or the
engine's own, filled from numpy arrays (`upload_depth`: synthetic source, tape).
"""
import numpy as np

from audioviz.gl.pingpong import Target, make_texture
from audioviz.gl.timing import GpuTimers

from . import seed as seedmod
from .config import Tunables
from .director import Director
from .gl.passes import Geometry, ShaderLib
from .gl.uniforms import FrameUBO
from .render.film import FilmPass, FogBlur
from .render.fog import DepthStage, FogStage, Halo
from .render.present import VIEWS, Present

DEPTH_SIZE = (320, 240)


def cover_scale(screen_aspect, cam_aspect=4.0 / 3.0):
    """uv scale so the 4:3 camera covers the screen (crop, no stretch)."""
    if screen_aspect >= cam_aspect:
        return (1.0, cam_aspect / screen_aspect)
    return (screen_aspect / cam_aspect, 1.0)


class Engine:
    def __init__(self, ctx, fb_size, preset, seed, tunables=None, track=None, look_fps=12):
        self.ctx = ctx
        self.tun = tunables or Tunables()
        self.preset = preset
        self.track = track
        self.look_fps = int(look_fps)
        self.ubo = FrameUBO(ctx)
        self.lib = ShaderLib(ctx, self.ubo)
        self.geom = Geometry(ctx)
        self.timers = GpuTimers(ctx, enabled=self.tun.timers)
        self.fb_size = tuple(int(v) for v in fb_size)
        self.aspect = self.fb_size[0] / self.fb_size[1]
        self.seed = int(seed) & 0x7FFFFFFF
        self.resolved = seedmod.resolve(preset, self.seed, self.aspect)
        self.director = Director(preset, self.resolved, track)
        self.view = 0
        self.halo_on = True
        self.last_state = None
        self.tick = 0
        self.frame_index = 0
        # depth input. R16F, not R32F: linear filtering of 32-bit float textures is not guaranteed on Apple GPUs
        self.own_depth = make_texture(ctx, DEPTH_SIZE, 1, "f2", "linear",
                                      data=np.ones(DEPTH_SIZE[0] * DEPTH_SIZE[1], np.float16).tobytes())
        self.own_mask = make_texture(ctx, DEPTH_SIZE, 1, "f1", "linear", data=bytes(DEPTH_SIZE[0] * DEPTH_SIZE[1]))
        self.depth_tex, self.mask_tex = self.own_depth, self.own_mask
        self.depth_on = False
        self.depth_new = False
        self.stale = 0.0
        self.fig_depth = 0.5
        self.mask_cov = 0.0
        self.reset_stab = True
        self.depth = DepthStage(ctx, self.lib, self.geom, self.tun.stab_size)
        self._build_sized()
        self.film = FilmPass(ctx, self.lib, self.geom)
        self.present = Present(self.lib, self.geom)

    # -- resources ---------------------------------------------------------------------------
    def _build_sized(self):
        self.fog = FogStage(self.ctx, self.lib, self.geom, self.fb_size)
        self.blur = FogBlur(self.ctx, self.lib, self.geom, self.fog.size)
        self.halo = Halo(self.ctx, self.lib, self.geom, self.fb_size)
        self.frame = Target(self.ctx, self.fb_size, 4, "f2", "linear")
        self.frame.fbo.clear(0.0, 0.0, 0.0, 1.0)

    def _release_sized(self):
        for o in (self.fog, self.blur, self.halo, self.frame):
            o.release()

    def resize(self, fb_size):
        fb_size = tuple(int(v) for v in fb_size)
        if fb_size == self.fb_size or min(fb_size) < 16:
            return False
        self._release_sized()
        self.fb_size = fb_size
        new_aspect = fb_size[0] / fb_size[1]
        if abs(new_aspect - self.aspect) > 1e-3:
            self.aspect = new_aspect
            self.set_seed(self.seed)
        self._build_sized()
        return True

    def set_seed(self, seed):
        self.seed = int(seed) & 0x7FFFFFFF
        self.resolved = seedmod.resolve(self.preset, self.seed, self.aspect)
        t_prev = self.director.t_prev
        self.director.set_resolved(self.resolved)
        self.director.t_prev = t_prev
        if t_prev is not None:   # resync event cursors to now (a reseed mid-song must not replay events)
            for c in (self.director.kicks, self.director.hats, self.director.sections):
                c.resync(t_prev)

    def set_preset(self, preset):
        self.preset = preset
        self.director.preset = preset
        self.set_seed(self.seed)

    # -- depth input -------------------------------------------------------------------------
    def set_depth_textures(self, depth_tex, mask_tex):
        if depth_tex is None:
            self.depth_tex, self.mask_tex = self.own_depth, self.own_mask
        else:
            self.depth_tex, self.mask_tex = depth_tex, mask_tex

    def upload_depth(self, depth, mask):
        """numpy depth (float, GL rows, 240x320) and mask (uint8) into the engine's own textures."""
        self.own_depth.write(np.ascontiguousarray(depth, np.float16).tobytes())
        self.own_mask.write(np.ascontiguousarray(mask, np.uint8).tobytes())
        self.depth_tex, self.mask_tex = self.own_depth, self.own_mask

    def set_depth_info(self, fig_depth=None, mask_cov=None, stale=0.0, on=True, new=True):
        if fig_depth is not None:
            self.fig_depth = float(fig_depth)
        if mask_cov is not None:
            self.mask_cov = float(mask_cov)
        self.stale = float(stale)
        if on and not self.depth_on:
            self.reset_stab = True
        self.depth_on = bool(on)
        self.depth_new = bool(new)

    # -- per step / per tick -----------------------------------------------------------------
    def sim_step(self, t, dt):
        self.last_state = self.director.step(t, dt)
        return self.last_state

    def _ubo_static(self, t):
        d = self.ubo.data
        W, H = self.fb_size
        d["res"] = (W, H, self.aspect, 0.0)
        d["seedU"][0] = self.seed
        d["seedU"][2] = self.tick & 0xFFFFFFFF
        d["seedU"][3] = self.frame_index & 0xFFFFFFFF
        d["time"] = (t, 1.0 / max(self.look_fps, 1), float(self.tick), float(self.seed % 65536))
        sc = cover_scale(self.aspect)
        d["camA"] = (sc[0], sc[1], 1.0 if self.depth_on else 0.0, self.stale)
        dp = self.resolved["params"]["depth"]
        d["depthA"][2] = self.fig_depth
        d["depthB"][2] = 1.0 if (dp.get("mask", True) and self.mask_cov > 0.005) else 0.0
        if not self.halo_on:
            d["haloA"][1] = 0.0

    def render(self, out_fbo, tick=None, t=None, viewport=None):
        """One picture tick: D1..D5 into out_fbo (viewport defaults to the whole fbo)."""
        if self.last_state is None:
            self.sim_step(0.0 if t is None else t, 1.0 / self.tun.sim_hz)
        if tick is not None:
            self.tick = int(tick)
        s = self.last_state
        d = self.ubo.data
        bursts = self.director.take_tick_bursts()
        self.director.fill_ubo(d, s, self.look_fps, bursts)
        self._ubo_static(s["t"] if t is None else t)
        self.ubo.upload()
        tm = self.timers
        with tm.scope("D1 depth"):
            filt = self.depth.run(self.depth_tex, self.mask_tex, reset=self.reset_stab)
            self.reset_stab = False
        with tm.scope("D2 fog"):
            fog = self.fog.run(filt)
        with tm.scope("D2b halo"):
            if self.halo_on:
                halo = self.halo.run(fog, s["halo_spread"])
            else:
                self.halo.clear()
                halo = self.halo.tex
        with tm.scope("D3 blur"):
            soft = self.blur.run(fog)
        with tm.scope("D4 film"):
            self.film.run(self.frame.fbo, soft, halo)
        with tm.scope("D5 present"):
            self.present.run(out_fbo, self.frame.tex, filt, fog, halo, view=self.view, viewport=viewport)
        self.frame_index += 1
        self.depth_new = False

    def present_only(self, out_fbo, viewport=None):
        """Frame hold: show the last tick again without re-rendering it."""
        self.present.run(out_fbo, self.frame.tex, self.depth.tex, self.fog.tex, self.halo.tex, view=self.view,
                         viewport=viewport)

    def read_tex(self, tex, comps, dtype="f2"):
        w, h = tex.size
        npd = {"f2": np.float16, "f4": np.float32, "f1": np.uint8}[dtype]
        return np.frombuffer(tex.read(), npd).astype(np.float32).reshape(h, w, comps)

    def poll_shaders(self):
        return self.lib.poll()

    def info(self):
        return {"fb_size": self.fb_size, "fog_size": self.fog.size, "stab_size": self.depth.size}

    def release(self):
        self._release_sized()
        self.depth.release()
        self.own_depth.release()
        self.own_mask.release()
        self.ubo.release()


__all__ = ["Engine", "VIEWS", "cover_scale"]
