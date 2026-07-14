"""Generate a synthetic rhythmic WAV for testing the visualizer pipeline
(kick-ish thumps on a beat grid + a chord pad + a rising 'drop' section)."""
import numpy as np
from scipy.io import wavfile
import sys

sr = 44100
bpm = 128
beat_dur = 60.0 / bpm
duration = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
out_path = sys.argv[2] if len(sys.argv) > 2 else "test_song.wav"

n = int(sr * duration)
t = np.arange(n) / sr
audio = np.zeros(n)

# chord pad (mid/treble content)
for f in [220, 277.18, 329.63, 440]:
    audio += 0.05 * np.sin(2 * np.pi * f * t)

# kick drum on every beat: short low-freq thump with fast decay
beat_times = np.arange(0, duration, beat_dur)
for bt in beat_times:
    i0 = int(bt * sr)
    dur = int(sr * 0.25)
    if i0 + dur > n:
        dur = n - i0
    if dur <= 0:
        continue
    tt = np.arange(dur) / sr
    env = np.exp(-tt * 18)
    kick = np.sin(2 * np.pi * 60 * tt) * env
    audio[i0:i0 + dur] += 0.9 * kick

# hi-hat on off-beats (treble content) every half beat
for bt in np.arange(beat_dur / 2, duration, beat_dur):
    i0 = int(bt * sr)
    dur = int(sr * 0.05)
    if i0 + dur > n:
        dur = n - i0
    if dur <= 0:
        continue
    noise = np.random.uniform(-1, 1, dur) * np.exp(-np.arange(dur) / sr * 60)
    audio[i0:i0 + dur] += 0.25 * noise

# a "drop": loud noisy swell in the second half
drop_start = duration * 0.55
drop_end = min(duration, drop_start + 2.0)
if drop_end > drop_start:
    i0, i1 = int(drop_start * sr), int(drop_end * sr)
    ramp = np.linspace(0, 1, i1 - i0) ** 2
    audio[i0:i1] += ramp * 0.6 * np.sin(2 * np.pi * 90 * t[i0:i1])

audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.9
audio_i16 = (audio * 32767).astype(np.int16)
wavfile.write(out_path, sr, audio_i16)
print(f"wrote {out_path}: {duration:.1f}s @ {sr}Hz, {len(beat_times)} beats, bpm={bpm}")
