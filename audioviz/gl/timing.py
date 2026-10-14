"""Per-pass GPU timers with `ctx.query(time=True)`.

Reading a query result blocks until the GPU has finished it, so each pass keeps a small ring of
queries and the value reported is the one from RING-1 frames ago, which is already done.
"""
import collections

import numpy as np

RING = 4


class _Slot:
    def __init__(self, ctx):
        self.q = ctx.query(time=True)
        self.used = False


class GpuTimers:
    def __init__(self, ctx, enabled=True, history=120):
        self.ctx = ctx
        self.enabled = enabled
        self.frame = 0
        self.rings = {}
        self.order = []
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=history))
        self.history = history
        self._active = None

    def scope(self, name):
        return _Scope(self, name)

    def _slot(self, name):
        ring = self.rings.get(name)
        if ring is None:
            ring = [_Slot(self.ctx) for _ in range(RING)]
            self.rings[name] = ring
            self.order.append(name)
        return ring[self.frame % RING]

    def end_frame(self):
        """Collect results from the oldest slot of each ring, then advance."""
        if self.enabled:
            nxt = (self.frame + 1) % RING
            for name, ring in self.rings.items():
                s = ring[nxt]
                if s.used:
                    try:
                        self.hist[name].append(s.q.elapsed / 1e6)
                    except Exception:
                        pass
                    s.used = False
        self.frame += 1

    def mean(self, name):
        h = self.hist.get(name)
        return float(np.mean(h)) if h else 0.0

    def p95(self, name):
        h = self.hist.get(name)
        return float(np.percentile(h, 95)) if h else 0.0

    def summary(self):
        return {n: round(self.mean(n), 3) for n in self.order}

    def total(self):
        return sum(self.mean(n) for n in self.order if n != "frame")


class _Scope:
    def __init__(self, timers, name):
        self.t = timers
        self.name = name
        self.slot = None

    def __enter__(self):
        if self.t.enabled:
            slot = self.t._slot(self.name)
            if slot.used:  # pass ran more than once this frame (several sim steps): time the first
                return self
            self.slot = slot
            self.slot.q.__enter__()
        return self

    def __exit__(self, *exc):
        if self.slot is not None:
            self.slot.q.__exit__(*exc)
            self.slot.used = True
        return False
