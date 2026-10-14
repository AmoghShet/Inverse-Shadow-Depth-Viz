"""Ping-pong texture/FBO pairs: read one, write the other, swap."""
import moderngl


def make_texture(ctx, size, components, dtype, filt="linear", data=None):
    tex = ctx.texture(size, components, data=data, dtype=dtype)
    f = moderngl.LINEAR if filt == "linear" else moderngl.NEAREST
    tex.filter = (f, f)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


class PingPong:
    def __init__(self, ctx, size, components, dtype, filt="linear"):
        self.ctx = ctx
        self.size = tuple(size)
        self.tex = [make_texture(ctx, size, components, dtype, filt) for _ in range(2)]
        self.fbo = [ctx.framebuffer(color_attachments=[t]) for t in self.tex]
        self.i = 0

    @property
    def src(self):
        return self.tex[self.i]

    @property
    def dst_tex(self):
        return self.tex[1 - self.i]

    @property
    def dst(self):
        return self.fbo[1 - self.i]

    @property
    def src_fbo(self):
        return self.fbo[self.i]

    def swap(self):
        self.i = 1 - self.i

    def clear(self, value=(0.0, 0.0, 0.0, 0.0)):
        for f in self.fbo:
            f.clear(*value)

    def release(self):
        for f in self.fbo:
            f.release()
        for t in self.tex:
            t.release()


class Target:
    """A single texture with its FBO."""

    def __init__(self, ctx, size, components, dtype, filt="linear"):
        self.size = tuple(size)
        self.tex = make_texture(ctx, size, components, dtype, filt)
        self.fbo = ctx.framebuffer(color_attachments=[self.tex])

    def release(self):
        self.fbo.release()
        self.tex.release()
