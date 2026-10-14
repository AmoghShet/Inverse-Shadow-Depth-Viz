"""GPU pipeline shared by live and export modes.

Per simulation step:  P0 flow -> P1 particles -> P3 trail advect/decay -> P2 splat (into P3's output)
Per rendered frame:   P4 bloom(density) -> P5 grade (fine + coarse cells) -> P6 ASCII -> glyph glow -> P7 present
"""
import numpy as np

from . import seed as seedmod
from .audio.features import Director
from .config import TIER_BY_NAME, TIERS, Tunables
from .gl.context import point_size_range
from .gl.passes import Geometry, ShaderLib
from .gl.pingpong import Target, make_texture
from .gl.timing import GpuTimers
from .gl.uniforms import FrameUBO
from .render.ascii import AsciiPass
from .render.bloom import Bloom
from .render.grade import Grade, grid_geometry
from .render.meter import Meter
from .render.present import Present
from .sim.flowfield import FlowField
from .sim.particles import Particles
from .sim.trails import Ghost, Trails

N_REF = 1024 * 1024
A_REF = 1920 * 1080


class Engine:
    def __init__(self, ctx, fb_size, preset, seed, tier="high", tunables=None, track=None):
        self.ctx = ctx
        self.tun = tunables or Tunables()
        self.preset = preset
        self.track = track
        self.ubo = FrameUBO(ctx)
        self.lib = ShaderLib(ctx, self.ubo)
        self.geom = Geometry(ctx)
        self.timers = GpuTimers(ctx, enabled=self.tun.timers)
        self.tier = TIER_BY_NAME[tier]
        self.fb_size = tuple(int(v) for v in fb_size)
        self.aspect = self.fb_size[0] / self.fb_size[1]
        self.seed = int(seed)
        self.resolved = seedmod.resolve(preset, self.seed, self.aspect)
        self.director = Director(preset, self.resolved, self.tier.particles, track)
        self.view = 0
        self.bloom_on = True
        self.grid_overlay = False
        self.ext_gain = 0.0
        self.cam_bg = None            # (texture, uv scale, black, white) or None
        self.cam_bg_on = True
        self.step_index = 0
        self.frame_index = 0
        self.sim_time = 0.0
        self.last_state = None
        self.ps_range = point_size_range(ctx)
        self.zero_tex = make_texture(ctx, (1, 1), 4, "f2", "linear", data=np.zeros(4, "f2").tobytes())
        self.zero_r8 = make_texture(ctx, (1, 1), 1, "f1", "linear", data=bytes(1))
        self.particles = Particles(ctx, self.lib, self.geom, self.tier.state_size, self.seed, self.tun.dummy_attrib)
        self.flow = None
        self._build_sized()
        self.grade = Grade(ctx, self.lib, self.geom)
        self.ascii = AsciiPass(ctx, self.lib, self.geom, self.resolved["params"]["glyphs"])
        self.present = Present(self.lib, self.geom)
        self.meter = Meter(ctx, self.lib, self.geom)
        self._meter_t = None

    # -- resources ---------------------------------------------------------------------------
    def _build_sized(self):
        ctx, lib, geom = self.ctx, self.lib, self.geom
        ext, ext_scale = None, (1.0, 1.0)
        if self.flow is not None and self.flow.ext is not self.flow.zero_ext:
            ext, ext_scale = self.flow.ext, self.flow.ext_scale   # camera texture survives a rebuild
        self.flow = FlowField(ctx, lib, geom, self.fb_size, self.tun.flow_div)
        if ext is not None:
            self.flow.set_external(ext, ext_scale)
        self.trails = Trails(ctx, lib, geom, self.fb_size, self.tier.density_scale)
        self.ghost = Ghost(ctx, lib, geom, self.fb_size)
        self.bloom = Bloom(ctx, lib, geom, self.trails.size, "density")
        self.frame = Target(ctx, self.fb_size, 4, "f2", "linear")
        self.glow = Bloom(ctx, lib, geom, self.fb_size, "color")

    def _release_sized(self):
        for o in (self.flow, self.trails, self.ghost, self.bloom, self.frame, self.glow):
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
            self.set_seed(self.seed, keep_particles=True)
        self._build_sized()
        return True

    def set_tier(self, name):
        tier = TIER_BY_NAME[name]
        if tier == self.tier:
            return
        self.tier = tier
        self.particles.release()
        self.particles = Particles(self.ctx, self.lib, self.geom, tier.state_size, self.seed, self.tun.dummy_attrib)
        self.director.set_particles(tier.particles)
        self._release_sized()
        self._build_sized()

    def lower_tier(self):
        names = [t.name for t in TIERS]
        i = names.index(self.tier.name)
        if i + 1 < len(names):
            self.set_tier(names[i + 1])
            return True
        return False

    def set_seed(self, seed, keep_particles=False):
        self.seed = int(seed) & 0x7FFFFFFF
        self.resolved = seedmod.resolve(self.preset, self.seed, self.aspect)
        self.director.set_resolved(self.resolved)
        if hasattr(self, "ascii"):
            self.ascii.glyphs = self.resolved["params"]["glyphs"]
        if not keep_particles:
            self.particles.reset(self.seed)
            if hasattr(self, "trails"):
                self.trails.clear()

    def set_preset(self, preset):
        self.preset = preset
        self.director.preset = preset
        self.set_seed(self.seed)

    def set_camera_background(self, tex, scale=(1.0, 1.0), lo=0.0, hi=1.0):
        self.cam_bg = (tex, scale, lo, hi) if tex is not None else None

    def set_external_flow(self, tex, scale=(1.0, 1.0), gain=0.0):
        self.flow.set_external(tex, scale)
        self.ext_gain = float(gain)

    # -- per step ----------------------------------------------------------------------------
    def warm_atlases(self, all_sets=False):
        """Atlases for every grid pair the preset can switch to (section changes), for the current
        glyph set or all of them (startup, so a reseed never builds one mid-song)."""
        cells = []
        for f1080, c1080 in self.resolved["params"]["grid"]["pairs"].values():
            cF, _ = grid_geometry(self.fb_size, f1080, self.tun.ref_height)
            cells += [cF, (cF[0] * 2, cF[1] * 2), grid_geometry(self.fb_size, c1080, self.tun.ref_height)[0]]
        cells = list(dict.fromkeys(tuple(c) for c in cells))
        sets = list(self.resolved["params"]["glyphs"]["sets"]) if all_sets else [self.resolved["glyph_set"]]
        for gs in sets:
            self.ascii.warm(gs, cells)

    def grids(self):
        pairs = self.resolved["params"]["grid"]["pairs"]
        f1080, c1080 = pairs[self.director.grid_pair]
        cellF, gridF = grid_geometry(self.fb_size, f1080, self.tun.ref_height)
        cellC = (cellF[0] * 2, cellF[1] * 2) if tuple(c1080) == (f1080[0] * 2, f1080[1] * 2) else None
        if cellC is None:
            cellC, gridC = grid_geometry(self.fb_size, c1080, self.tun.ref_height)
        else:
            gridC = (-(-self.fb_size[0] // cellC[0]), -(-self.fb_size[1] // cellC[1]))
        return cellF, gridF, cellC, gridC

    def _ubo_static(self):
        d = self.ubo.data
        W, H = self.fb_size
        d["res"] = (W, H, self.aspect, self.tier.density_scale)
        d["seedU"][0] = self.seed
        d["seedU"][3] = self.particles.N
        cellF, gridF, cellC, gridC = self.grids()
        d["gridF"] = (cellF[0], cellF[1], gridF[0], gridF[1])
        d["gridC"] = (cellC[0], cellC[1], gridC[0], gridC[1])
        d["lookE"] = (self.resolved["params"]["bloom"]["glyph_glow"], self.ext_gain, self.view,
                      1.0 if self.grid_overlay else 0.0)
        hz = self.resolved["params"]["grain"].get("haze", {})
        d["hazeA"] = (hz.get("coverage", 0.0) if hz.get("enabled", True) else 0.0, hz.get("brightness", 0.02),
                      hz.get("scale", 2.2), hz.get("speed", 0.05))
        cb = self.resolved["params"]["tone"].get("camera_bg", {})
        if self.cam_bg is not None and self.cam_bg_on and cb.get("enabled", True):
            _, sc, lo, hi = self.cam_bg
            d["camA"] = (cb.get("level", 0.33), lo, hi, 1.0)
            d["camB"] = (sc[0], sc[1], cb.get("gamma", 1.2), cb.get("brightness", 1.0))
            d["camC"] = (cb.get("blur", 1.5), cb.get("contrast", 0.55), cb.get("tint", 0.7), cb.get("fine_bias", 0.25))
            d["camD"] = (cb.get("gain", 0.0), cb.get("relax", 0.12), cb.get("breakup", 0.35), cb.get("warp", 0.06))
            d["camE"] = (cb.get("ghost_s", 1.5), cb.get("ghost_drift", 0.35), cb.get("ghost_gray", 0.45), 0.0)
        else:
            d["camA"] = (0.0, 0.0, 1.0, 0.0)
        return cellF, gridF, cellC, gridC

    def point_size_and_splat(self):
        sim = self.resolved["params"]["sim"]
        dw, dh = self.trails.size
        scale = min(dw, dh) / 1080.0
        ps = float(np.clip(sim["point_size"] * scale, max(1.0, self.ps_range[0]), self.ps_range[1]))
        area = dw * dh
        splat = sim["splat"] * (N_REF / self.particles.N) * (area / A_REF) * (sim["point_size"] ** 2 / ps ** 2)
        return ps, splat

    def sim_step(self, t, dt):
        s = self.director.step(t, dt)
        self.last_state = s
        d = self.ubo.data
        self.director.fill_ubo(d, s, self.particles.N)
        self._ubo_static()
        start, count, burst = self.particles.emit_window(s["emit"], s["burst"])
        d["emit"] = (start, count, burst, s["burst_emitter"])
        d["time"] = (t, dt, self.frame_index, float(self.seed % 65536))
        d["seedU"][2] = self.step_index & 0xFFFFFFFF
        ps, splat = self.point_size_and_splat()
        d["simA"][3] = ps
        d["simE"][2] = splat
        self.ubo.upload()
        tm = self.timers
        with tm.scope("P0 flow"):
            self.flow.run()
        with tm.scope("P1 particles"):
            self.particles.step(self.flow.tex)
        with tm.scope("P3 trails"):
            if self.cam_bg is not None and self.cam_bg_on:
                self.ghost.step(self.cam_bg[0], self.flow.tex)
            dst = self.trails.advect(self.flow.tex, self.ghost.tex)
        with tm.scope("P2 splat"):
            self.particles.draw_splats(dst)
        self.trails.swap()
        self.step_index += 1
        self.sim_time = t

    def prewarm(self, seconds, t0=0.0, dt=1.0 / 60.0):
        """Simulate `seconds` before t0 (features clamp to the song start)."""
        self.warm_atlases(all_sets=True)
        n = int(round(seconds / dt))
        for i in range(n):
            self.sim_step(t0 - (n - i) * dt, dt)
        self.director.t_prev = None
        self.director.feed_meter(self.meter.measure_now(self.trails.tex), 0.0, snap=True)

    # -- per frame ---------------------------------------------------------------------------
    def render(self, out_fbo, out_size=None, viewport=None):
        d = self.ubo.data
        cellF, gridF, cellC, gridC = self._ubo_static()
        d["time"][2] = self.frame_index
        self.ubo.upload()
        p = self.resolved["params"]
        tm = self.timers
        m = self.meter.measure(self.trails.tex)
        if m is not None:
            dtm = 0.0 if self._meter_t is None else max(0.0, self.sim_time - self._meter_t)
            self._meter_t = self.sim_time
            self.director.feed_meter(m, dtm)
        with tm.scope("P4 bloom"):
            bloom_tex = self.bloom.run(self.trails.tex, p["bloom"]["threshold"]) if self.bloom_on else self.zero_tex
        with tm.scope("P5 grade"):
            cam = self.cam_bg[0] if (self.cam_bg is not None and self.cam_bg_on) else self.zero_r8
            cF = self.grade.run("F", cellF, gridF, self.trails.tex, bloom_tex, self.flow.tex, cam, self.ghost.tex)
            cC = self.grade.run("C", cellC, gridC, self.trails.tex, bloom_tex, self.flow.tex, cam, self.ghost.tex)
        gset = self.resolved["glyph_set"]
        aF = self.ascii.atlas(gset, cellF)
        aC = self.ascii.atlas(gset, cellC)
        with tm.scope("P6 ascii"):
            self.ascii.run(self.frame.fbo, cF, cC, aF, aC)
        with tm.scope("P7 present"):
            gg = p["bloom"]["glyph_glow"] if self.bloom_on else 0.0
            glow_tex = self.glow.run(self.frame.tex, p["bloom"]["glyph_glow_threshold"]) if gg > 0 else self.zero_tex
            self.present.run(out_fbo, self.frame.tex, glow_tex, self.trails.tex, bloom_tex, self.flow.tex,
                             view=self.view, glow_gain=gg, viewport=viewport)
        self.frame_index += 1

    def present_only(self, out_fbo, viewport=None):
        """Frame hold (12/24 fps look): show the last ASCII frame again without re-rendering it."""
        p = self.resolved["params"]
        gg = p["bloom"]["glyph_glow"] if self.bloom_on else 0.0
        glow_tex = self.glow.tex if gg > 0 else self.zero_tex
        bloom_tex = self.bloom.tex if self.bloom_on else self.zero_tex
        self.present.run(out_fbo, self.frame.tex, glow_tex, self.trails.tex, bloom_tex, self.flow.tex,
                         view=self.view, glow_gain=gg, viewport=viewport)

    def poll_shaders(self):
        return self.lib.poll()

    def info(self):
        cellF, gridF, cellC, gridC = self.grids()
        return {
            "fb_size": self.fb_size, "tier": self.tier.name, "particles": self.particles.N,
            "density_size": self.trails.size, "flow_size": self.flow.size,
            "cell_fine": cellF, "grid_fine": gridF, "cell_coarse": cellC, "grid_coarse": gridC,
            "point_size_splat": self.point_size_and_splat(),
        }

    def release(self):
        self._release_sized()
        self.particles.release()
        self.grade.release()
        self.meter.release()
        self.ascii.release()
        self.zero_tex.release()
        self.zero_r8.release()
        self.ubo.release()
