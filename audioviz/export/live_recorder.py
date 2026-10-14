"""Record the live window (what is on screen, HUD included) to mp4 with the song muxed in sync.

Frames are read back from the window's back buffer through two PBOs once per picture tick
(look frame rate), so frame k is song time (tick0 + k) / fps. Missed ticks repeat the previous
frame. A writer thread feeds ffmpeg; the song is added from the first recorded tick.
"""
import os
import queue
import subprocess
import threading

import imageio_ffmpeg


class LiveRecorder:
    def __init__(self, ctx, size, fps, out, song=None):
        self.ctx = ctx
        self.size = tuple(size)
        self.fps = int(fps)
        self.out = out
        self.song = song
        self.nbytes = self.size[0] * self.size[1] * 3
        self.pbos = [ctx.buffer(reserve=self.nbytes) for _ in range(2)]
        self.i = 0
        self.pending = None          # (pbo index, tick) waiting to be mapped
        self.tick0 = None
        self.next_tick = None
        self.last = None
        self.proc = None
        self.q = queue.Queue(maxsize=24)
        self.thread = None
        self.frames = 0
        self.stopped = False

    def _start(self, tick0):
        self.tick0 = tick0
        self.next_tick = tick0
        W, H = self.size
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(self.fps), "-i", "-"]
        if self.song:
            cmd += ["-ss", f"{tick0 / self.fps:.6f}", "-i", self.song, "-map", "0:v", "-map", "1:a",
                    "-c:a", "aac", "-b:a", "320k", "-shortest"]
        cmd += ["-vf", "vflip,scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.out]
        os.makedirs(os.path.dirname(os.path.abspath(self.out)), exist_ok=True)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()

    def _writer(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            try:
                self.proc.stdin.write(item)
            except (BrokenPipeError, OSError):
                break

    def _emit_until(self, tick, raw):
        """Write `raw` for `tick`, repeating the previous frame over any skipped ticks."""
        while self.next_tick < tick and self.last is not None:
            self.q.put(self.last)
            self.frames += 1
            self.next_tick += 1
        if tick >= self.next_tick:
            self.q.put(raw)
            self.frames += 1
            self.next_tick = tick + 1
        self.last = raw

    def capture(self, fbo, tick):
        """Call after the frame (and HUD) is drawn, before swap, once per new picture tick."""
        if self.stopped or tick is None:
            return
        if self.proc is None:
            self._start(tick)
        if self.pending is not None:
            j, t = self.pending
            self._emit_until(t, self.pbos[j].read())
        fbo.read_into(self.pbos[self.i], viewport=(0, 0) + self.size, components=3, alignment=1)
        self.pending = (self.i, tick)
        self.i = 1 - self.i

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if self.proc is None:
            return
        if self.pending is not None:
            j, t = self.pending
            self._emit_until(t, self.pbos[j].read())
        self.q.put(None)
        self.thread.join()
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait()
        for b in self.pbos:
            b.release()
