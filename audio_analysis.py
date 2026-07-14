"""
audio_analysis.py

Lightweight WAV analysis for the Y2K Visualizer — no librosa dependency.
Extracts, per output video frame:
    - rms:      overall loudness (0..1, normalized)
    - bass:     low-band energy (20-250 Hz)
    - mid:      mid-band energy (250-4000 Hz)
    - treble:   high-band energy (4000-16000 Hz)
    - onset:    spectral-flux onset strength (0..1)
    - is_beat:  bool, True on detected onset peaks
    - is_drop:  bool, True when energy spikes far above its rolling average
    - t:        timestamp in seconds

Everything is precomputed into flat numpy arrays aligned 1:1 with video
frames, so the renderer can just index into `features[i]` per frame.
"""
from __future__ import annotations

import dataclasses
import wave
import numpy as np
from scipy.io import wavfile


@dataclasses.dataclass
class Features:
    fps: float
    n_frames: int
    t: np.ndarray
    rms: np.ndarray
    bass: np.ndarray
    mid: np.ndarray
    treble: np.ndarray
    onset: np.ndarray
    is_beat: np.ndarray
    is_drop: np.ndarray
    duration: float

    def __getitem__(self, i: int) -> dict:
        i = min(max(i, 0), self.n_frames - 1)
        return dict(
            t=float(self.t[i]),
            rms=float(self.rms[i]),
            bass=float(self.bass[i]),
            mid=float(self.mid[i]),
            treble=float(self.treble[i]),
            onset=float(self.onset[i]),
            is_beat=bool(self.is_beat[i]),
            is_drop=bool(self.is_drop[i]),
        )


def _load_wav_mono(path: str) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), info.max)
        data = data.astype(np.float64) / scale
    else:
        data = data.astype(np.float64)
        peak = np.max(np.abs(data)) or 1.0
        if peak > 1.0:
            data = data / peak

    return sr, data


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    # cap the window to the array length -- np.convolve(..., mode="same")
    # returns an array of length max(len(x), len(kernel)), so an
    # uncapped window on a short clip (e.g. a few seconds at 20-30fps)
    # silently produces an output LONGER than x and breaks every
    # downstream elementwise comparison against x.
    win = max(1, min(int(win), len(x)))
    if win <= 1:
        return x.copy()
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x - np.min(x)
    m = np.max(x)
    if m > 1e-9:
        x = x / m
    return np.clip(x, 0.0, 1.0)


def analyze(path: str, fps: float = 30.0) -> Features:
    """Analyze a WAV file and return per-video-frame audio features."""
    sr, samples = _load_wav_mono(path)
    duration = len(samples) / sr

    # STFT parameters: window sized for decent frequency resolution while
    # still landing close to one analysis frame per video frame.
    hop = max(1, int(round(sr / fps)))
    win_size = int(2 ** np.ceil(np.log2(hop * 4)))
    win_size = max(win_size, 1024)
    window = np.hanning(win_size)

    n_frames = int(np.floor((len(samples) - win_size) / hop)) + 1
    n_frames = max(n_frames, 1)

    freqs = np.fft.rfftfreq(win_size, d=1.0 / sr)

    def band_mask(lo, hi):
        return (freqs >= lo) & (freqs < hi)

    bass_mask = band_mask(20, 250)
    mid_mask = band_mask(250, 4000)
    treble_mask = band_mask(4000, min(16000, sr / 2))

    rms = np.zeros(n_frames)
    bass = np.zeros(n_frames)
    mid = np.zeros(n_frames)
    treble = np.zeros(n_frames)
    flux = np.zeros(n_frames)

    prev_mag = None
    # pad so the last window doesn't run off the end
    padded = np.pad(samples, (0, win_size))

    for i in range(n_frames):
        start = i * hop
        frame = padded[start:start + win_size] * window
        rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)

        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)

        bass[i] = mag[bass_mask].mean() if bass_mask.any() else 0.0
        mid[i] = mag[mid_mask].mean() if mid_mask.any() else 0.0
        treble[i] = mag[treble_mask].mean() if treble_mask.any() else 0.0

        if prev_mag is not None:
            diff = mag - prev_mag
            flux[i] = np.sum(diff[diff > 0])
        prev_mag = mag

    rms = _normalize(rms)
    bass = _normalize(np.sqrt(bass))
    mid = _normalize(np.sqrt(mid))
    treble = _normalize(np.sqrt(treble))
    onset = _normalize(flux)

    # Peak-pick onset envelope for beat markers: local max above an
    # adaptive (rolling-median + margin) threshold, with a minimum spacing
    # so we don't fire multiple "beats" within the same transient.
    win_local = max(3, int(fps * 0.35))
    local_med = _moving_average(onset, win_local)
    threshold = local_med + 0.12
    is_beat = np.zeros(n_frames, dtype=bool)
    min_gap = max(1, int(fps * 0.12))
    last_beat = -min_gap
    for i in range(1, n_frames - 1):
        if (
            onset[i] > threshold[i]
            and onset[i] >= onset[i - 1]
            and onset[i] >= onset[i + 1]
            and (i - last_beat) >= min_gap
        ):
            is_beat[i] = True
            last_beat = i

    # "Drop" detection: rms spikes well above its slow rolling average AND
    # is loud in absolute terms, with a minimum spacing so drops stay rare,
    # dramatic moments rather than firing on every kick hit.
    slow_win = max(3, int(fps * 4))
    slow_avg = _moving_average(rms, slow_win)
    drop_candidate = (rms > (slow_avg + 0.3)) & (rms > 0.55)
    is_drop = np.zeros(n_frames, dtype=bool)
    min_drop_gap = max(1, int(fps * 1.5))
    last_drop = -min_drop_gap
    for i in range(n_frames):
        if drop_candidate[i] and (i - last_drop) >= min_drop_gap:
            is_drop[i] = True
            last_drop = i

    t = np.arange(n_frames) / fps

    return Features(
        fps=fps,
        n_frames=n_frames,
        t=t,
        rms=rms,
        bass=bass,
        mid=mid,
        treble=treble,
        onset=onset,
        is_beat=is_beat,
        is_drop=is_drop,
        duration=duration,
    )


def wav_duration(path: str) -> float:
    """Fast duration lookup (seconds) without decoding the whole file."""
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def load_waveform_preview(path: str, n_points: int = 900) -> np.ndarray:
    """A coarse, downsampled amplitude envelope (0..1, length n_points) for
    drawing a waveform strip in the GUI's snippet picker -- cheap enough to
    compute on the main thread response but still called from a background
    thread since even this is a bit of I/O for a long file."""
    sr, samples = _load_wav_mono(path)
    n = len(samples)
    if n == 0:
        return np.zeros(n_points)
    edges = np.linspace(0, n, n_points + 1).astype(int)
    env = np.zeros(n_points)
    for i in range(n_points):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        env[i] = np.abs(samples[a:b]).max() if b > a else 0.0
    peak = env.max() or 1.0
    return env / peak


def trim_wav(src_path: str, start: float, end: float | None, dst_path: str) -> None:
    """Losslessly slice [start, end) seconds of src_path (by raw PCM frames,
    no resampling/decoding round-trip) and write it out as dst_path."""
    with wave.open(src_path, "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        start_f = max(0, min(nframes, int(round(start * framerate))))
        end_f = nframes if end is None else max(0, min(nframes, int(round(end * framerate))))
        if end_f <= start_f:
            end_f = min(nframes, start_f + 1)
        wf.setpos(start_f)
        frames = wf.readframes(end_f - start_f)
    with wave.open(dst_path, "wb") as out:
        out.setnchannels(nchannels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.writeframes(frames)


if __name__ == "__main__":
    import sys
    f = analyze(sys.argv[1])
    print(f"duration={f.duration:.2f}s frames={f.n_frames} fps={f.fps}")
    print(f"beats detected: {int(f.is_beat.sum())}")
    print(f"drops detected: {int(f.is_drop.sum())}")
