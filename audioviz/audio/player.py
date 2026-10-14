"""Song playback with sounddevice; the audio callback is the master clock.

Song position = (index of the first sample of the last delivered block) / sr
              + (time since that callback) - (output latency) + user offset.
The latency comes from the callback's DAC timestamp when the host API provides one, otherwise
from `stream.latency`. If no output device can be opened, a wall-clock fallback keeps the visuals
running (silent).
"""
import threading
import time

import numpy as np


class Player:
    def __init__(self, pcm, sr, loop=True, offset_s=0.0, device=None, blocksize=0, mute=False, latency="high"):
        self.pcm = pcm                 # float32 [n, ch]
        self.sr = int(sr)
        self.n = pcm.shape[0]
        self.loop = loop
        self.offset = float(offset_s)
        self.lock = threading.Lock()
        self.pos = 0                   # next sample to deliver
        self.paused = False
        self.underflows = 0
        self.underflow_at = []         # callback indices (the first one or two are stream priming)
        self.callbacks = 0
        self.ended = False
        self._cb_pos = 0
        self._cb_perf = None
        self._cb_lat = 0.0
        self._last_t = 0.0
        self.stream = None
        self.silent = False
        self._wall0 = None
        self._wall_pos0 = 0.0
        self.device = device
        self.blocksize = blocksize
        self.error = None
        self.mute = mute               # stream runs (same clock/latency) but outputs silence
        self.latency = latency         # PortAudio latency hint; the clock compensates for it

    @property
    def duration(self):
        return self.n / self.sr

    def start(self):
        try:
            import sounddevice as sd

            self.stream = sd.OutputStream(
                samplerate=self.sr,
                channels=self.pcm.shape[1],
                dtype="float32",
                callback=self._callback,
                device=self.device,
                blocksize=self.blocksize,
                latency=self.latency,
            )
            self.stream.start()
        except Exception as e:  # no device: run on the wall clock, silently
            self.error = str(e)
            self.silent = True
            self._wall0 = time.perf_counter()
            self._wall_pos0 = self.pos / self.sr

    def _callback(self, outdata, frames, time_info, status):
        if status.output_underflow:
            self.underflows += 1
            self.underflow_at.append(self.callbacks)
        with self.lock:
            self.callbacks += 1
            if self.paused:
                outdata.fill(0.0)
                self._cb_perf = None
                return
            start = self.pos
            end = start + frames
            if end <= self.n:
                outdata[:] = self.pcm[start:end]
                self.pos = end
            else:
                k = max(0, self.n - start)
                outdata[:k] = self.pcm[start:self.n]
                if self.loop:
                    rest = frames - k
                    outdata[k:] = self.pcm[:rest]
                    self.pos = rest
                else:
                    outdata[k:] = 0.0
                    self.pos = self.n
                    self.ended = True
            if self.mute:
                outdata.fill(0.0)
            self._cb_pos = start
            self._cb_perf = time.perf_counter()
            try:
                lat = float(time_info.outputBufferDacTime - time_info.currentTime)
            except Exception:
                lat = -1.0
            if not (0.0 < lat < 1.0):
                lat = float(self.stream.latency) if self.stream is not None else 0.0
            self._cb_lat = lat

    def time(self):
        """Current song position in seconds (what is being heard now), including the offset."""
        if self.silent:
            if self.paused or self._wall0 is None:
                return self._wall_pos0 + self.offset
            t = self._wall_pos0 + (time.perf_counter() - self._wall0)
            if self.loop:
                t %= self.duration
            return t + self.offset
        with self.lock:
            if self._cb_perf is None:
                return self._last_t
            t = self._cb_pos / self.sr + (time.perf_counter() - self._cb_perf) - self._cb_lat + self.offset
        # the callback position jumps at a loop; otherwise keep the clock monotonic
        if self._last_t - 0.5 < t < self._last_t:
            t = self._last_t
        self._last_t = t
        return t

    def set_paused(self, paused):
        if self.silent:
            if paused and not self.paused:
                self._wall_pos0 = self.time() - self.offset
            elif not paused and self.paused:
                self._wall0 = time.perf_counter()
            self.paused = paused
            return
        with self.lock:
            self.paused = paused

    def seek(self, t):
        t = float(np.clip(t, 0.0, max(0.0, self.duration - 0.05)))
        if self.silent:
            self._wall_pos0 = t
            self._wall0 = time.perf_counter()
            return
        with self.lock:
            self.pos = int(t * self.sr)
            self._cb_perf = None
            self._last_t = t + self.offset
            self.ended = False

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def info(self):
        lat = self._cb_lat
        return {"underflows": self.underflows, "underflows_after_start": sum(1 for i in self.underflow_at if i > 4),
                "underflow_callbacks": self.underflow_at[:10], "latency_ms": round(lat * 1000.0, 1), "silent": self.silent,
                "offset_ms": round(self.offset * 1000.0, 1)}


class WallClock:
    """Clock for running without a song."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.paused_at = None
        self.offset = 0.0
        self.duration = float("inf")
        self.underflows = 0
        self.silent = True
        self.ended = False

    def start(self):
        self.t0 = time.perf_counter()

    def time(self):
        if self.paused_at is not None:
            return self.paused_at
        return time.perf_counter() - self.t0

    def set_paused(self, paused):
        if paused and self.paused_at is None:
            self.paused_at = self.time()
        elif not paused and self.paused_at is not None:
            self.t0 = time.perf_counter() - self.paused_at
            self.paused_at = None

    @property
    def paused(self):
        return self.paused_at is not None

    def seek(self, t):
        t = max(0.0, t)
        if self.paused_at is not None:
            self.paused_at = t
        else:
            self.t0 = time.perf_counter() - t

    def stop(self):
        pass

    def info(self):
        return {"underflows": 0, "latency_ms": 0.0, "silent": True, "offset_ms": 0.0}
