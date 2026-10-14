"""P1 particle update (ping-pong RGBA32F state) and P2 splat (GL_POINTS from gl_VertexID).

State texel: (u, v, age, hash). age >= 1 means dead; the emission ring respawns particles in index
order, so the CPU only decides how many to emit per step and where the ring starts.
"""
import moderngl
import numpy as np

from ..gl.passes import FullscreenPass
from ..gl.pingpong import PingPong


class Particles:
    def __init__(self, ctx, lib, geom, state_size, seed, dummy_attrib=False):
        self.ctx = ctx
        self.S = int(state_size)
        self.N = self.S * self.S
        self.update = FullscreenPass(lib, geom, "particles", "particles.frag")
        defines = {"DUMMY_ATTRIB": 1} if dummy_attrib else {}
        self.splat = lib.program("splat", "splat.vert", "splat.frag", defines)
        self.dummy_attrib = dummy_attrib
        self._dummy_buf = None
        self._vao = None
        self._vao_gen = -1
        self.state = PingPong(ctx, (self.S, self.S), 4, "f4", "nearest")
        self.cursor = 0   # emission ring position
        self.reset(seed)

    def reset(self, seed):
        rng = np.random.default_rng([int(seed), 4242])
        data = np.zeros((self.S, self.S, 4), np.float32)
        data[..., 2] = 2.0                         # all dead; the ring fills the frame
        data[..., 3] = rng.random((self.S, self.S), dtype=np.float32)
        for t in self.state.tex:
            t.write(data.tobytes())
        self.cursor = 0

    def emit_window(self, count, burst):
        """Advance the ring. Returns (start, count, burst) for the UBO `emit` field."""
        count = int(max(0, min(self.N, count)))
        burst = int(max(0, min(self.N - count, burst)))
        start = self.cursor
        self.cursor = (self.cursor + count + burst) % self.N
        return start, count, burst

    def step(self, vel_tex):
        self.update.run(self.state.dst, [("u_state", self.state.src), ("u_vel", vel_tex)])
        self.state.swap()

    def _splat_vao(self):
        h = self.splat
        if self._vao is None or self._vao_gen != h.generation:
            if self._vao is not None:
                self._vao.release()
            if self.dummy_attrib:
                if self._dummy_buf is None:
                    self._dummy_buf = self.ctx.buffer(np.zeros(self.N, "f4").tobytes())
                self._vao = self.ctx.vertex_array(h.prog, [(self._dummy_buf, "f", "a_dummy")])
            else:
                self._vao = self.ctx.vertex_array(h.prog, [])
            self._vao_gen = h.generation
        return self._vao

    def draw_splats(self, fbo):
        """Additively splat every particle into `fbo` (no clear)."""
        self.state.src.use(0)
        self.splat.set("u_state", 0)
        fbo.use()
        fbo.viewport = (0, 0) + tuple(fbo.size)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self._splat_vao().render(moderngl.POINTS, vertices=self.N)
        self.ctx.disable(moderngl.BLEND)

    def read_state(self):
        return np.frombuffer(self.state.src.read(), np.float32).reshape(self.S, self.S, 4)

    def release(self):
        self.state.release()
        if self._vao is not None:
            self._vao.release()
        if self._dummy_buf is not None:
            self._dummy_buf.release()
