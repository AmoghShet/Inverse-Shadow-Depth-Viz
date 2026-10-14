"""Depth tape: what the depth source delivered at each picture tick, so a camera-driven look can be
exported and reproduced exactly.

Layout of a tape directory:
  meta.json          song sha1, look_fps, first tick, sizes, model, EP, seed, record count
  chunk_NNNN.npz     up to 64 records each (compressed): ticks int64[n], times float64[n],
                     depth uint16[n, 240, 320] (0..1 -> 0..65535, GL rows), mask uint8[n, 240, 320],
                     hdr float64[n, 32] (the worker header at that tick)
Records are written only on ticks where a new depth arrived; the reader holds the previous record
over missing ticks. Times must increase: a record at or before the last time (song loop, scrub
back) is ignored, so a tape is one forward pass through the song.
"""
import glob
import json
import os
import queue
import threading
import time

import numpy as np

CHUNK = 64


class TapeWriter:
    def __init__(self, out_dir, meta):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        for old in glob.glob(os.path.join(out_dir, "chunk_*.npz")):
            os.remove(old)
        self.meta = dict(meta)
        self.meta.update({"format": 1, "chunk": CHUNK, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "records": 0, "chunks": 0, "first_time": None, "last_time": None})
        self.buf = []
        self.chunk_i = 0
        self.records = 0
        self.last_time = None
        self.skipped_back = 0
        self.q = queue.Queue(maxsize=8)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.bytes = 0
        self.closed = False
        self._write_meta()

    def _write_meta(self):
        self.meta.update({"records": self.records, "chunks": self.chunk_i, "last_time": self.last_time,
                          "skipped_backwards": self.skipped_back, "bytes": self.bytes})
        tmp = os.path.join(self.dir, "meta.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=1)
        os.replace(tmp, os.path.join(self.dir, "meta.json"))

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            path, arrays = item
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)
            self.bytes += os.path.getsize(path)

    def write(self, tick, t, depth, mask, hdr=None):
        if self.closed:
            return False
        if self.last_time is not None and t <= self.last_time:
            self.skipped_back += 1
            return False
        if self.meta["first_time"] is None:
            self.meta["first_time"] = float(t)
            self.meta["first_tick"] = int(tick)
        d16 = (np.clip(depth, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        h = np.zeros(32, np.float64) if hdr is None else np.asarray(hdr, np.float64)[:32]
        self.buf.append((int(tick), float(t), d16, np.asarray(mask, np.uint8).copy(), h.copy()))
        self.last_time = float(t)
        self.records += 1
        if len(self.buf) >= CHUNK:
            self._emit()
        return True

    def _emit(self):
        if not self.buf:
            return
        b = self.buf
        self.buf = []
        arrays = {"ticks": np.array([r[0] for r in b], np.int64), "times": np.array([r[1] for r in b], np.float64),
                  "depth": np.stack([r[2] for r in b]), "mask": np.stack([r[3] for r in b]),
                  "hdr": np.stack([r[4] for r in b])}
        path = os.path.join(self.dir, f"chunk_{self.chunk_i:04d}.npz")
        self.chunk_i += 1
        self.q.put((path, arrays))
        self._write_meta()

    def flush(self):
        """Write what is buffered (an export started mid-run sees everything recorded so far)."""
        self._emit()
        deadline = time.time() + 10
        while not self.q.empty() and time.time() < deadline:
            time.sleep(0.01)
        self._write_meta()

    def close(self):
        if self.closed:
            return
        self._emit()
        self.q.put(None)
        self.thread.join(timeout=30)
        self.closed = True
        self._write_meta()


class TapeReader:
    def __init__(self, tape_dir):
        self.dir = tape_dir
        with open(os.path.join(tape_dir, "meta.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.chunks = sorted(glob.glob(os.path.join(tape_dir, "chunk_*.npz")))
        if not self.chunks:
            raise FileNotFoundError(f"tape {tape_dir} has no chunks")
        self.index_t = []     # per chunk: times array
        for p in self.chunks:
            with np.load(p) as z:
                self.index_t.append(z["times"].copy())
        self.starts = np.array([t[0] for t in self.index_t])
        self._cur = None      # (chunk index, loaded arrays)
        self.records = int(sum(len(t) for t in self.index_t))

    @property
    def t_first(self):
        return float(self.index_t[0][0])

    @property
    def t_last(self):
        return float(self.index_t[-1][-1])

    @property
    def look_fps(self):
        return int(self.meta.get("look_fps", 12))

    def locate(self, t):
        """(chunk, row) of the latest record with time <= t (+ a tiny epsilon), or None."""
        t = float(t) + 1e-6
        ci = int(np.searchsorted(self.starts, t, side="right")) - 1
        if ci < 0:
            return None
        ri = int(np.searchsorted(self.index_t[ci], t, side="right")) - 1
        return (ci, ri) if ri >= 0 else None

    def _chunk(self, ci):
        if self._cur is None or self._cur[0] != ci:
            with np.load(self.chunks[ci]) as z:
                self._cur = (ci, {k: z[k] for k in ("ticks", "times", "depth", "mask", "hdr")})
        return self._cur[1]

    def at_time(self, t):
        """(depth float32 GL rows, mask uint8, hdr float64[32], record id) or None before the tape."""
        loc = self.locate(t)
        if loc is None:
            return None
        c = self._chunk(loc[0])
        r = loc[1]
        depth = c["depth"][r].astype(np.float32) * (1.0 / 65535.0)
        return depth, c["mask"][r], c["hdr"][r], (loc[0], r)

    def at_tick(self, tick, look_fps=None):
        return self.at_time(tick / float(look_fps or self.look_fps))

    def sha1(self):
        import hashlib

        h = hashlib.sha1()
        for p in self.chunks:
            with open(p, "rb") as fh:
                h.update(fh.read())
        return h.hexdigest()
