"""Generate a more musically varied synthetic demo track: quiet intro ->
build-up -> big drop -> breakdown -> second drop -> outro. Used only to
produce a demo video; swap in a real song for actual use."""
import numpy as np
from scipy.io import wavfile

sr = 44100
bpm = 126
beat_dur = 60.0 / bpm


def section(duration, intensity, kick=True, hats=True, noise_sweep=False, chord=True):
    n = int(sr * duration)
    t = np.arange(n) / sr
    audio = np.zeros(n)
    if chord:
        for f in [220, 277.18, 329.63, 440]:
            audio += 0.035 * intensity * np.sin(2 * np.pi * f * t)
    if kick:
        for bt in np.arange(0, duration, beat_dur):
            i0 = int(bt * sr)
            dur = min(int(sr * 0.25), n - i0)
            if dur <= 0:
                continue
            tt = np.arange(dur) / sr
            env = np.exp(-tt * 18)
            audio[i0:i0 + dur] += 0.95 * intensity * np.sin(2 * np.pi * 60 * tt) * env
    if hats:
        for bt in np.arange(beat_dur / 2, duration, beat_dur):
            i0 = int(bt * sr)
            dur = min(int(sr * 0.05), n - i0)
            if dur <= 0:
                continue
            nz = np.random.uniform(-1, 1, dur) * np.exp(-np.arange(dur) / sr * 60)
            audio[i0:i0 + dur] += 0.22 * intensity * nz
    if noise_sweep:
        sweep = np.linspace(0, 1, n) ** 1.5
        audio += sweep * 0.4 * np.sin(2 * np.pi * (80 + sweep * 400) * t)
    return audio


parts = [
    section(6, 0.35, kick=False, hats=False, chord=True),          # intro pad
    section(6, 0.55, kick=True, hats=False, chord=True),           # kick enters
    section(4, 0.7, kick=True, hats=True, noise_sweep=True),       # build-up sweep
    section(8, 1.0, kick=True, hats=True, chord=True),             # drop 1
    section(5, 0.4, kick=False, hats=True, chord=True),            # breakdown
    section(3, 0.75, kick=True, hats=True, noise_sweep=True),      # build-up 2
    section(8, 1.0, kick=True, hats=True, chord=True),             # drop 2
    section(4, 0.3, kick=True, hats=False, chord=True),            # outro
]
audio = np.concatenate(parts)
audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.92
wavfile.write("demo_song.wav", sr, (audio * 32767).astype(np.int16))
print(f"wrote demo_song.wav: {len(audio)/sr:.1f}s")
