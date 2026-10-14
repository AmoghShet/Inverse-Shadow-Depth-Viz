"""Offline song analysis with librosa, cached as .npz keyed by file hash + extractor version.

Continuous features (one value per hop of 512 samples at 44.1 kHz, about 86 per second) are
normalised to the song's own percentiles (p5 -> 0, p95 -> 1, clipped). Events are times in seconds.

Run directly to analyse a song and print a summary:
    python -m audioviz.audio.analysis "D:/Downloads/Myxomatosis.mp3"
"""
import hashlib
import os
import sys
import time

import numpy as np

from .. import CACHE_DIR

EXTRACTOR_VERSION = 5
SR = 44100
HOP = 512
NFFT = 2048

CONTINUOUS = (
    "sub", "bass", "mid", "high", "onset", "onset_low", "onset_high",
    "centroid", "flux", "harmonic", "percussive", "rms", "loudness",
)

BANDS = {"sub": (20, 60), "bass": (60, 200), "mid": (200, 2500), "high": (2500, 16000)}


def file_hash(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_audio(path, sr=SR):
    """Decode with soundfile (MP3/WAV/FLAC/OGG), falling back to the bundled ffmpeg for others.

    Returns (float32 [n, channels], sr)."""
    try:
        import soundfile as sf

        y, file_sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        y, file_sr = _ffmpeg_decode(path, sr)
    if file_sr != sr:
        import soxr

        y = soxr.resample(y, file_sr, sr, quality="HQ").astype(np.float32)
    return np.ascontiguousarray(y), sr


def _ffmpeg_decode(path, sr):
    import subprocess

    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy(), sr


def _norm(x, lo_p=5, hi_p=95):
    lo, hi = np.percentile(x, lo_p), np.percentile(x, hi_p)
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _db(x):
    return 20.0 * np.log10(np.maximum(x, 1e-6))


def stft_mag(y, n_fft=NFFT, hop=HOP):
    """Magnitude STFT (bins x frames), centred frames, periodic Hann. ~30x faster than librosa.stft here."""
    import scipy.fft
    from numpy.lib.stride_tricks import sliding_window_view

    yp = np.pad(y, (n_fft // 2, n_fft // 2), mode="constant")
    frames = sliding_window_view(yp, n_fft)[::hop]
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    S = np.empty((n_fft // 2 + 1, frames.shape[0]), np.float32)
    step = 4096
    for i in range(0, frames.shape[0], step):
        S[:, i:i + step] = np.abs(scipy.fft.rfft(frames[i:i + step] * win, axis=1, workers=-1)).T
    return S


def _band_rms(S, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return np.sqrt(np.mean(S[m] ** 2, axis=0) + 1e-12)


def _flux(Sl):
    d = np.diff(Sl, axis=1, prepend=Sl[:, :1])
    return np.maximum(d, 0.0).sum(axis=0)


def _moving_avg(x, n):
    n = max(1, int(n))
    k = np.ones(n) / n
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def _detect_events(env_norm, fps, delta, wait_s):
    import librosa

    frames = librosa.util.peak_pick(
        env_norm, pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=delta, wait=max(1, int(wait_s * fps))
    )
    return frames.astype(np.int64)


def _sections(S, sr, beat_frames, duration, fps):
    """Novelty boundaries on beat-synchronous MFCC + chroma (checkerboard kernel)."""
    import librosa

    if len(beat_frames) < 32:
        step = 30.0
        return np.arange(step, max(step, duration - 10.0), step)
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=64)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=sr)
    z = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)
    feat = np.vstack([z, chroma * 2.0])
    sync = librosa.util.sync(feat, beat_frames, aggregate=np.median)
    sync = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-9)
    ssm = sync.T @ sync
    B = ssm.shape[0]
    K = 16
    g = np.exp(-0.5 * (np.linspace(-1.5, 1.5, 2 * K)) ** 2)
    ker = np.outer(g, g) * np.outer(np.r_[-np.ones(K), np.ones(K)], np.r_[-np.ones(K), np.ones(K)])
    pad = np.pad(ssm, K, mode="edge")
    nov = np.array([np.sum(pad[i:i + 2 * K, i:i + 2 * K] * ker) for i in range(B)])
    nov = np.maximum(nov, 0.0)
    if nov.max() <= 0:
        return np.array([])
    nov /= nov.max()
    peaks = librosa.util.peak_pick(nov, pre_max=8, post_max=8, pre_avg=12, post_avg=12, delta=0.08, wait=16)
    peaks = peaks[(peaks > 4) & (peaks < B - 4)]
    # keep the strongest, at most one per 20 s
    max_n = int(duration / 20.0)
    if len(peaks) > max_n:
        peaks = np.sort(peaks[np.argsort(nov[peaks])[::-1][:max_n]])
    bt = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    idx = np.clip(peaks, 0, len(bt) - 1)
    return bt[idx]


def analyze(path, verbose=True):
    import librosa

    t0 = time.perf_counter()
    y2, sr = load_audio(path)
    y = y2.mean(axis=1)
    duration = len(y) / sr
    S = stft_mag(y)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    fps = sr / HOP
    T = S.shape[1]
    out = {}

    for name, (lo, hi) in BANDS.items():
        out[name] = _norm(_db(_band_rms(S, freqs, lo, hi)))

    # HPSS on a 4x frequency-pooled spectrogram (7 s instead of 45 s; energies only, so fine)
    Sp = S[:1024].reshape(256, 4, -1).mean(axis=1)
    fp = freqs[:1024].reshape(256, 4).mean(axis=1)
    H, P = librosa.decompose.hpss(Sp, kernel_size=(31, 9), margin=1.0)
    out["harmonic"] = _norm(_db(np.sqrt(np.mean(H ** 2, axis=0))))
    out["percussive"] = _norm(_db(np.sqrt(np.mean(P ** 2, axis=0))))

    melP = librosa.feature.melspectrogram(S=P ** 2, sr=sr, n_fft=510, n_mels=48)
    out["onset"] = _norm(librosa.onset.onset_strength(S=librosa.power_to_db(melP), sr=sr, hop_length=HOP))

    Pl = np.log1p(100.0 * P)
    low = (fp >= 20) & (fp < 200)
    high = (fp >= 3000) & (fp < 16000)
    onset_low = _flux(Pl[low])
    onset_high = _flux(Pl[high])
    out["onset_low"] = _norm(onset_low, 5, 99)
    out["onset_high"] = _norm(onset_high, 5, 99)

    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    out["centroid"] = _norm(np.log2(np.maximum(cent, 20.0)))
    out["flux"] = _norm(_flux(np.log1p(100.0 * S)))
    rms = np.sqrt(np.mean(S ** 2, axis=0))
    rms_db = _db(rms)
    out["rms"] = _norm(rms_db)
    out["loudness"] = _norm(_moving_avg(rms_db, 2.0 * fps))

    melS = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=128)
    onset_full = librosa.onset.onset_strength(S=librosa.power_to_db(melS), sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_full, sr=sr, hop_length=HOP, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beat_frames) < 8:
        period = 60.0 / (tempo if tempo > 40 else 120.0)
        beat_times = np.arange(0.0, duration, period)
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    # downbeat: the beat index offset (mod 4) with the most low-end energy
    lowe = out["sub"] + out["bass"]
    bf = np.clip(beat_frames, 0, T - 1)
    scores = [lowe[bf[k::4]].mean() if len(bf[k::4]) else 0.0 for k in range(4)]
    db_off = int(np.argmax(scores))
    downbeats = beat_times[db_off::4]

    kick_frames = _detect_events(out["onset_low"], fps, delta=0.22, wait_s=0.2)
    hat_frames = _detect_events(out["onset_high"], fps, delta=0.15, wait_s=0.07)
    sections = _sections(S, sr, beat_frames, duration, fps)

    feats = {k: out[k].astype(np.float32) for k in CONTINUOUS}
    result = {
        "version": EXTRACTOR_VERSION,
        "fps": fps,
        "duration": duration,
        "tempo": tempo,
        "beats": beat_times.astype(np.float64),
        "downbeats": downbeats.astype(np.float64),
        "kicks": librosa.frames_to_time(kick_frames, sr=sr, hop_length=HOP),
        "kick_strength": out["onset_low"][kick_frames] if len(kick_frames) else np.zeros(0, np.float32),
        "hats": librosa.frames_to_time(hat_frames, sr=sr, hop_length=HOP),
        "hat_strength": out["onset_high"][hat_frames] if len(hat_frames) else np.zeros(0, np.float32),
        "sections": np.asarray(sections, np.float64),
        **feats,
    }
    if verbose:
        print(
            f"analysed {os.path.basename(path)}: {duration:.1f} s, tempo {tempo:.1f}, {len(beat_times)} beats, "
            f"{len(result['kicks'])} kicks, {len(result['hats'])} hats, {len(sections)} section boundaries "
            f"in {time.perf_counter() - t0:.1f} s"
        )
    return result


def cache_path(path):
    os.makedirs(os.path.join(CACHE_DIR, "features"), exist_ok=True)
    return os.path.join(CACHE_DIR, "features", f"{file_hash(path)}_v{EXTRACTOR_VERSION}.npz")


def load_or_analyze(path, verbose=True):
    cp = cache_path(path)
    if os.path.isfile(cp):
        with np.load(cp, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        for k in ("version", "fps", "duration", "tempo"):
            d[k] = float(d[k])
        d["cache"] = cp
        if verbose:
            print(f"features from cache {os.path.basename(cp)}")
        return d
    d = analyze(path, verbose=verbose)
    np.savez_compressed(cp, **d)
    d["cache"] = cp
    return d


if __name__ == "__main__":
    d = load_or_analyze(sys.argv[1])
    for k in CONTINUOUS:
        v = d[k]
        print(f"{k:>11}: mean {v.mean():.2f}  p50 {np.median(v):.2f}")
    print("sections at", np.round(d["sections"], 1))
    print("first kicks", np.round(d["kicks"][:8], 2))
