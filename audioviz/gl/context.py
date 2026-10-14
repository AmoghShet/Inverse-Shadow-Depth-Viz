"""OpenGL context creation: a GLFW window at 4.1 core for live mode, a standalone context for export.

4.1 core is the macOS ceiling, so everything is written against it: no compute shaders, no SSBOs,
no `layout(binding=)` (that is 4.2), only vertex and fragment shaders.

Run `python -m audioviz.gl.context` for the M0 check (window opens, prints GL info, point sizes).
"""
import platform
import sys

import moderngl

GL_VERSION = (4, 1)
GL_POINT_SIZE_RANGE = 0x0B12
GL_ALIASED_POINT_SIZE_RANGE = 0x846D


def is_darwin():
    return sys.platform == "darwin"


def create_window(title="ASCII Ink Visualizer", size=(1280, 720), fullscreen=False, vsync=True, visible=True):
    """Create a GLFW window with a 4.1 core context. Returns the moderngl-window window object."""
    import moderngl_window as mglw

    cls = mglw.get_local_window_cls("glfw")
    window = cls(
        title=title,
        gl_version=GL_VERSION,
        size=size,
        resizable=True,
        visible=visible,
        fullscreen=fullscreen,
        vsync=vsync,
        samples=0,
        cursor=True,
    )
    window.exit_key = None  # Esc is handled by the app
    mglw.activate_context(window=window)
    setup_context(window.ctx)
    return window


def create_headless():
    """Standalone context for export and tests (no window)."""
    ctx = moderngl.create_standalone_context(require=410)
    setup_context(ctx)
    return ctx


def setup_context(ctx):
    # Without this, core profile ignores gl_PointSize (always 1 px) on macOS.
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)


def point_size_range(ctx):
    """(min, max) supported point size, used to clamp gl_PointSize."""
    try:
        lo, hi = ctx.info["GL_POINT_SIZE_RANGE"]
        return float(lo), float(hi)
    except Exception:
        return 1.0, 64.0


def gl_info(ctx):
    info = ctx.info
    return {
        "GL_VENDOR": info.get("GL_VENDOR"),
        "GL_RENDERER": info.get("GL_RENDERER"),
        "GL_VERSION": info.get("GL_VERSION"),
        "version_code": ctx.version_code,
        "point_size_range": point_size_range(ctx),
        "max_texture_size": info.get("GL_MAX_TEXTURE_SIZE"),
        "max_viewport_dims": info.get("GL_MAX_VIEWPORT_DIMS"),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def renderer_name(ctx):
    return str(ctx.info.get("GL_RENDERER", "")).strip()


def _m0_check():
    import time

    window = create_window(title="ASCII Ink Visualizer M0 check", size=(640, 360))
    ctx = window.ctx
    info = gl_info(ctx)
    for k, v in info.items():
        print(f"{k:>18}: {v}")
    print(f"{'buffer_size':>18}: {window.buffer_size} (window size {window.size})")
    ok_version = ctx.version_code >= 410
    print("M0:", "PASS" if ok_version else "FAIL", "- 4.1+ context, PROGRAM_POINT_SIZE enabled without error")
    t0 = time.perf_counter()
    while not window.is_closing and time.perf_counter() - t0 < 1.5:
        ctx.clear(0.02, 0.03, 0.04)
        window.swap_buffers()
    window.destroy()
    return 0 if ok_version else 1


if __name__ == "__main__":
    sys.exit(_m0_check())
