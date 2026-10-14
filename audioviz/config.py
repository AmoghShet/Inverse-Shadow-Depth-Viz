"""Tunables and per-platform particle tiers."""
import platform
import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tier:
    name: str
    state_size: int        # particle state texture is state_size x state_size
    density_scale: float   # density/trail buffer resolution relative to the framebuffer

    @property
    def particles(self):
        return self.state_size * self.state_size


# Ordered high -> low; the runtime controller only ever steps down.
TIERS = [
    Tier("high", 1024, 1.0),   # GTX 1650 Ti class: 1.05M
    Tier("mid", 724, 1.0),     # M1 and later: 524k
    Tier("low", 512, 0.5),     # A18 Pro (MacBook Neo): 262k, half-res density
    Tier("min", 362, 0.5),     # emergency floor: 131k
]
TIER_BY_NAME = {t.name: t for t in TIERS}


def detect_tier(renderer: str) -> str:
    """Pick the starting tier from the GL renderer string and platform."""
    r = (renderer or "").lower()
    if sys.platform == "darwin":
        if "a18" in r or "a17" in r or "a16" in r:
            return "low"
        return "mid"
    if any(k in r for k in ("nvidia", "geforce", "quadro", "rtx", "radeon rx", "radeon pro")):
        return "high"
    if "intel" in r or "radeon(tm) graphics" in r or "vega" in r:
        return "mid"
    return "mid"


def is_darwin():
    return sys.platform == "darwin"


@dataclass
class Tunables:
    sim_hz: int = 60                 # fixed simulation step rate
    max_steps_per_frame: int = 3     # live mode catch-up cap
    flow_div: int = 4                # flow field at 1/4 res
    tier_drop_ms: float = 14.0       # rolling p95 GPU frame time that triggers a tier drop
    tier_drop_frames: int = 60       # ...sustained for this many frames
    dummy_attrib: bool = field(default_factory=is_darwin)   # Apple: empty VAOs may draw nothing
    timers: bool = True
    hud: bool = True
    ref_height: int = 1080           # cell sizes and point sizes are defined at 1080p-equivalent

    @staticmethod
    def platform_summary():
        return f"{platform.system()} {platform.machine()} py{sys.version.split()[0]}"
