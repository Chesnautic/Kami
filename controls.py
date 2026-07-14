"""
controls.py — the "manipulate generated features" knobs.

A Controls bundle carries the global reactivity/intensity settings that
apply across every pattern:

    chaos             0..1   overall randomness/aggressiveness (segment
                              length, glitch frequency, wobble amount)
    bass_gain         multiplier on the bass-band feature before it
    mid_gain          multiplier on the mid-band feature reaches any
    treble_gain       multiplier on the treble-band feature pattern
    onset_gain        multiplier on the onset/transient strength
    glow_strength     multiplier on neon/glow blur intensity
    particle_density  multiplier on particle/spark/dot spawn counts
    switch_speed      multiplier on how fast the director cuts between
                       patterns (>1 = faster cuts, <1 = slower/longer)

These are the "global reactivity knobs" — they don't touch any single
pattern's internals, they scale the audio-feature inputs and a couple of
universal rendering knobs (glow, particle counts) that every pattern reads.
"""
from __future__ import annotations

import dataclasses
import json


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


@dataclasses.dataclass
class Controls:
    chaos: float = 0.65
    bass_gain: float = 1.0
    mid_gain: float = 1.0
    treble_gain: float = 1.0
    onset_gain: float = 1.0
    glow_strength: float = 1.0
    particle_density: float = 1.0
    switch_speed: float = 1.0

    def clamp(self) -> "Controls":
        self.chaos = _clip(self.chaos, 0.0, 1.0)
        self.bass_gain = _clip(self.bass_gain, 0.0, 3.0)
        self.mid_gain = _clip(self.mid_gain, 0.0, 3.0)
        self.treble_gain = _clip(self.treble_gain, 0.0, 3.0)
        self.onset_gain = _clip(self.onset_gain, 0.0, 3.0)
        self.glow_strength = _clip(self.glow_strength, 0.0, 2.5)
        self.particle_density = _clip(self.particle_density, 0.0, 3.0)
        self.switch_speed = _clip(self.switch_speed, 0.25, 3.0)
        return self

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Controls":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields and v is not None}).clamp()

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.as_dict(), fh, indent=2)

    def apply_to_feature(self, feat: dict) -> dict:
        """Return a copy of a per-frame audio-feature dict with gains applied
        and clipped back into a sane 0..1.5 range so patterns don't choke on
        huge multiplied values."""
        f = dict(feat)
        f["bass"] = _clip(f["bass"] * self.bass_gain, 0.0, 1.5)
        f["mid"] = _clip(f["mid"] * self.mid_gain, 0.0, 1.5)
        f["treble"] = _clip(f["treble"] * self.treble_gain, 0.0, 1.5)
        f["onset"] = _clip(f["onset"] * self.onset_gain, 0.0, 1.5)
        return f


DEFAULT_CONTROLS = Controls()
