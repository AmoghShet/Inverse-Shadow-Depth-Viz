"""Shader library (with `#include` and hot reload) and the fullscreen-pass helper.

`#include "name.glsl"` pulls a file from the shader directory; `#include "frame_ubo"` pulls the
generated `Frame` block from gl/uniforms.py. On reload, a program that fails to compile keeps its
old version and the error text is kept in `ShaderLib.errors` so the HUD can show it.
"""
import os
import re
import time

import moderngl
import numpy as np

from .. import SHADER_DIR
from . import uniforms

_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.M)


class ShaderError(RuntimeError):
    pass


class ProgramHandle:
    """Stable reference to a program that may be swapped by hot reload."""

    def __init__(self, key, vs, fs, defines):
        self.key = key
        self.vs = vs
        self.fs = fs
        self.defines = dict(defines or {})
        self.prog = None
        self.files = set()
        self.generation = 0

    def __getitem__(self, name):
        return self.prog[name]

    def set(self, name, value):
        """Set a uniform if the program still has it (the compiler drops unused ones)."""
        try:
            self.prog[name].value = value
        except KeyError:
            pass


class ShaderLib:
    def __init__(self, ctx, ubo, shader_dir=SHADER_DIR, poll_interval=0.25):
        self.ctx = ctx
        self.ubo = ubo
        self.dir = shader_dir
        self.handles = {}
        self.errors = {}
        self.poll_interval = poll_interval
        self._last_poll = 0.0
        self._mtimes = {}

    # -- source assembly -------------------------------------------------------------------
    def _load(self, name, files, depth=0):
        if depth > 8:
            raise ShaderError(f"include depth exceeded at {name}")
        if name == "frame_ubo":
            return uniforms.glsl_block()
        path = os.path.join(self.dir, name)
        files.add(path)
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()

        def repl(m):
            return f"// begin {m.group(1)}\n" + self._load(m.group(1), files, depth + 1) + f"// end {m.group(1)}\n"

        return _INCLUDE.sub(repl, src)

    def _assemble(self, name, defines, files):
        src = self._load(name, files)
        lines = src.split("\n")
        if not lines[0].startswith("#version"):
            raise ShaderError(f"{name}: first line must be #version")
        defs = [f"#define {k} {v}" for k, v in defines.items()]
        return "\n".join([lines[0]] + defs + lines[1:])

    def _compile(self, h):
        files = set()
        vs = self._assemble(h.vs, h.defines, files)
        fs = self._assemble(h.fs, h.defines, files)
        prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.ubo.attach(prog)
        return prog, files

    # -- public ----------------------------------------------------------------------------
    def program(self, key, vs, fs, defines=None):
        """Compile (or reuse, when the same key/sources/defines were already compiled)."""
        old = self.handles.get(key)
        if old is not None and (old.vs, old.fs, old.defines) == (vs, fs, dict(defines or {})):
            return old
        h = ProgramHandle(key, vs, fs, defines)
        prog, files = self._compile(h)  # first compile must succeed
        h.prog, h.files = prog, files
        for f in files:
            self._mtimes[f] = os.path.getmtime(f)
        self.handles[key] = h
        return h

    def poll(self, force=False):
        """Recompile programs whose sources changed. Returns True if anything was reloaded."""
        now = time.perf_counter()
        if not force and now - self._last_poll < self.poll_interval:
            return False
        self._last_poll = now
        changed = set()
        for f, mt in list(self._mtimes.items()):
            try:
                m = os.path.getmtime(f)
            except OSError:
                continue
            if m != mt:
                self._mtimes[f] = m
                changed.add(f)
        if not changed and not force:
            return False
        reloaded = False
        for key, h in self.handles.items():
            if not force and not (h.files & changed):
                continue
            try:
                prog, files = self._compile(h)
            except Exception as e:  # keep the old program
                self.errors[key] = _short_error(e)
                for f in h.files:   # keep watching this program's files so a fix is picked up
                    self._mtimes.setdefault(f, os.path.getmtime(f))
                continue
            old = h.prog
            h.prog, h.files = prog, files
            h.generation += 1
            self.errors.pop(key, None)
            for f in files:
                self._mtimes.setdefault(f, os.path.getmtime(f))
            if old is not None:
                old.release()
            reloaded = True
        return reloaded


def _short_error(e):
    text = str(e).strip().splitlines()
    keep = [ln for ln in text if ln.strip()][:6]
    return "\n".join(keep)


class Geometry:
    """Shared fullscreen quad buffer (a real buffer, since empty VAOs are unreliable on Apple)."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._vaos = {}

    def fullscreen_vao(self, handle):
        k = (id(handle), handle.generation)
        vao = self._vaos.get(k)
        if vao is None:
            for old in [kk for kk in self._vaos if kk[0] == id(handle)]:
                self._vaos.pop(old).release()
            vao = self.ctx.vertex_array(handle.prog, [(self.quad, "2f", "in_pos")])
            self._vaos[k] = vao
        return vao


class FullscreenPass:
    def __init__(self, lib, geom, key, fs, defines=None, vs="fullscreen.vert"):
        self.lib = lib
        self.geom = geom
        self.h = lib.program(key, vs, fs, defines)

    def set(self, name, value):
        self.h.set(name, value)

    def run(self, fbo, textures=(), viewport=None, blend=False, **uniforms):
        prog = self.h.prog
        for unit, (name, tex) in enumerate(textures):
            tex.use(unit)
            try:
                prog[name].value = unit
            except KeyError:
                pass
        for k, v in uniforms.items():
            try:
                prog[k].value = v
            except KeyError:
                pass
        fbo.use()
        if viewport is not None:
            fbo.viewport = viewport
        else:
            fbo.viewport = (0, 0) + tuple(fbo.size)
        ctx = self.lib.ctx
        if blend:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
        else:
            ctx.disable(moderngl.BLEND)
        self.geom.fullscreen_vao(self.h).render(moderngl.TRIANGLE_STRIP)
        if blend:
            ctx.disable(moderngl.BLEND)
