"""Tunables and per-platform depth profiles (model, input size, execution provider, device id).

DEPTH_PROFILES is filled from the M0 benchmark (`python -m depthviz bench-depth`, results in
out/bench/depth_<platform>_<renderer>.json). `pick_profile` matches the platform and GL renderer.
"""
import platform
import sys
from dataclasses import dataclass


@dataclass
class Tunables:
    sim_hz: int = 60                 # director step rate (fixed)
    max_steps_per_frame: int = 3     # live catch-up cap
    ref_height: int = 1080           # cell sizes are defined at 1080p-equivalent
    depth_size: tuple = (320, 240)   # what the worker publishes (camera space, 4:3)
    stab_size: tuple = (640, 480)    # D1 / D1b resolution
    stall_s: float = 2.0             # no new depth for this long -> hold the last one
    timers: bool = True
    hud: bool = True

    @staticmethod
    def platform_summary():
        return f"{platform.system()} {platform.machine()} py{sys.version.split()[0]}"


# Ordered: the first profile whose platform matches and whose renderer substring is in the GL
# renderer string wins. From the M0 benchmark (2026-10-14, out/bench/depth_win32_nvidia_geforce_gtx_1650_ti_pcie_sse2.json;
# numbers in NOTES_depthviz_log.md). device_id "hp" = DirectML's high-performance adapter preference
# (for machines whose adapter order is unknown); an integer pins the DXGI adapter index.
DEPTH_PROFILES = [
    # GTX 1650 Ti (DirectML adapter 0 on the dev laptop): 38 ms, 22 fps, GL p95 4.9 ms, after-D1 std 0.018
    {"platform": "win32", "renderer": "gtx 1650 ti", "model": "dav2_small", "size": 336, "ep": "dml", "device_id": 0,
     "note": "M0: DA-V2 fp32 336x252 on the GTX 1650 Ti"},
    # other NVIDIA machines: same model, adapter chosen by DirectML's high-performance preference (on the dev
    # laptop that preference picked the GTX but ran 44 ms instead of 28 ms, hence the pinned index above)
    {"platform": "win32", "renderer": "nvidia", "model": "dav2_small", "size": 336, "ep": "dml", "device_id": "hp",
     "note": "DA-V2 fp32 336x252, DirectML high-performance adapter"},
    # other Windows GPUs (AMD / Intel integrated): q4f16 252 ran 58 ms / 15.6 fps on the Radeon iGPU
    {"platform": "win32", "renderer": "", "model": "dav2_small_q4f16", "size": 252, "ep": "dml", "device_id": "hp",
     "note": "M0: DA-V2 q4f16 252x196 on an integrated GPU"},
    # Apple Silicon: unmeasured until M6 (CoreML EP)
    {"platform": "darwin", "renderer": "", "model": "dav2_small", "size": 252, "ep": "coreml", "device_id": 0,
     "note": "unmeasured (M6)"},
    # anything else: MiDaS small on the CPU (62 ms / 14.5 fps on the Ryzen 4600HS)
    {"platform": "", "renderer": "", "model": "midas_small", "size": 256, "ep": "cpu", "device_id": 0,
     "note": "generic CPU fallback"},
]
CPU_FALLBACK = {"model": "midas_small", "size": 256, "ep": "cpu", "device_id": 0}


def gl_renderer_headless():
    """GL renderer string from a throwaway headless context (for tools without a window)."""
    try:
        from audioviz.gl.context import create_headless, renderer_name

        ctx = create_headless()
        r = renderer_name(ctx)
        ctx.release()
        return r
    except Exception:
        return ""


def pick_profile(renderer="", plat=None):
    plat = plat or sys.platform
    r = (renderer or "").lower()
    for p in DEPTH_PROFILES:
        if p["platform"] and not plat.startswith(p["platform"]):
            continue
        if p["renderer"] and p["renderer"] not in r:
            continue
        return dict(p)
    return dict(DEPTH_PROFILES[-1])
