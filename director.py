"""
director.py — decides which pattern is active on every video frame.

Auto mode picks a new pattern every ~2-6 seconds (shorter as chaos/energy
rises), weighted by each pattern's "energy affinity" and biased away from
immediate repeats, snapping segment boundaries to nearby detected beats for
clean cuts. Manual overrides let you force specific patterns for specific
time ranges — auto-fills whatever you didn't specify.
"""
from __future__ import annotations

import numpy as np

from audio_analysis import Features
from patterns import PATTERN_NAMES

HIGH_ENERGY = {
    # pack 1: waveforms
    "particle_burst", "glitch_vhs", "starburst_pop",
    # pack 2: cars
    "pixel_cars", "pixel_rain", "cars_burnout", "cars_drag_race",
    "cars_taillight_trails", "cars_headlights",
    # pack 3: space & sunsets
    "meteor_shower",
    # pack 4: retro y2k
    "chrome_bubble_text",
}
LOW_ENERGY = {
    # pack 1: waveforms
    "kaleidoscope", "chrome_tunnel", "oscilloscope_wave",
    # pack 2: cars
    "cars_showroom_spin",
    # pack 3: space & sunsets
    "sunset_pixel", "shooting_stars", "comet_flyby", "purple_asteroid",
    "constellations", "aurora_borealis", "galaxy_swirl",
    # pack 4: retro y2k
    "pixel_globe", "virtual_pet", "holo_sticker", "crt_boot",
}


class Director:
    def __init__(
        self,
        features: Features,
        pool: list[str] | None = None,
        weights: dict[str, float] | None = None,
        forced_segments: list[dict] | None = None,
        chaos: float = 0.6,
        seed: int = 0,
        switch_speed: float = 1.0,
    ):
        self.features = features
        self.pool = pool or list(PATTERN_NAMES)
        for p in self.pool:
            if p not in PATTERN_NAMES:
                raise ValueError(f"Unknown pattern '{p}'. Available: {PATTERN_NAMES}")
        self.weights = weights or {}
        self.forced = forced_segments or []
        self.chaos = float(np.clip(chaos, 0.0, 1.0))
        self.seed = seed
        # >1 = faster cuts (shorter segments), <1 = slower/longer segments
        self.switch_speed = float(np.clip(switch_speed, 0.25, 3.0))
        self.schedule: list[str] = []

    def _affinity(self, pattern: str, energy: float) -> float:
        if pattern in HIGH_ENERGY:
            a = 1.0 + (energy - 0.5) * 1.4
        elif pattern in LOW_ENERGY:
            a = 1.0 - (energy - 0.5) * 1.4
        else:
            a = 1.0
        return float(np.clip(a, 0.15, 3.0))

    def _segment_seconds(self, energy: float, rng: np.random.Generator) -> float:
        lo, hi = 2.2, 6.5
        shrink = 0.5 * self.chaos + 0.3 * energy
        lo = max(0.7, lo - shrink * 2.5)
        hi = max(lo + 0.5, hi - shrink * 3.0)
        return rng.uniform(lo, hi) / self.switch_speed

    def _pick_pattern(self, rng: np.random.Generator, recent: list[str], energy: float) -> str:
        last = recent[-1] if recent else None
        pool = [p for p in self.pool if p != last] or self.pool
        w = []
        for p in pool:
            affinity = self._affinity(p, energy)
            # penalize patterns used recently (last up to 3 segments) so the
            # auto-cycle doesn't loop back to the same couple of patterns
            recency_penalty = 1.0
            for k, rp in enumerate(reversed(recent[-3:])):
                if rp == p:
                    recency_penalty *= 0.35 / (k + 1)
            w.append(self.weights.get(p, 1.0) * affinity * recency_penalty)
        w = np.clip(np.array(w), 1e-4, None)
        w = w / w.sum()
        return rng.choice(pool, p=w)

    def build_schedule(self) -> list[str]:
        n = self.features.n_frames
        fps = self.features.fps
        schedule: list[str | None] = [None] * n

        sorted_forced = sorted(self.forced, key=lambda f: f["start"])
        for idx, f in enumerate(sorted_forced):
            start_f = int(round(f["start"] * fps))
            if f.get("end") is not None:
                end_f = int(round(f["end"] * fps))
            elif idx + 1 < len(sorted_forced):
                end_f = int(round(sorted_forced[idx + 1]["start"] * fps))
            else:
                end_f = n
            start_f = max(0, min(start_f, n))
            end_f = max(start_f, min(end_f, n))
            for i in range(start_f, end_f):
                schedule[i] = f["pattern"]

        rng = np.random.default_rng(self.seed)
        recent_patterns: list[str] = []
        i = 0
        while i < n:
            if schedule[i] is not None:
                if not recent_patterns or recent_patterns[-1] != schedule[i]:
                    recent_patterns.append(schedule[i])
                i += 1
                continue
            j = i
            while j < n and schedule[j] is None:
                j += 1
            k = i
            while k < j:
                energy = float(self.features.rms[k])
                seg_len = max(1, int(self._segment_seconds(energy, rng) * fps))
                seg_end = min(j, k + seg_len)
                if seg_end < j:
                    snap = int(fps * 0.3)
                    lo, hi = max(k + 1, seg_end - snap), min(j, seg_end + snap)
                    beats = [b for b in range(lo, hi) if self.features.is_beat[b]]
                    if beats:
                        seg_end = min(beats, key=lambda b: abs(b - seg_end))
                pattern = self._pick_pattern(rng, recent_patterns, energy)
                for fi in range(k, seg_end):
                    schedule[fi] = pattern
                recent_patterns.append(pattern)
                k = seg_end
            i = j

        # extra chaos: brief glitch flash-cuts on big drops
        if self.chaos > 0.35 and "glitch_vhs" in self.pool:
            flash_len = max(2, int(fps * 0.15))
            for idx in range(n):
                if self.features.is_drop[idx] and rng.random() < self.chaos * 0.4:
                    end = min(n, idx + flash_len)
                    for fi in range(idx, end):
                        schedule[fi] = "glitch_vhs"

        self.schedule = schedule  # type: ignore
        return self.schedule  # type: ignore

    def summary(self) -> str:
        from collections import Counter
        counts = Counter(self.schedule)
        total = len(self.schedule) or 1
        lines = [f"{p}: {c/total*100:.1f}%" for p, c in counts.most_common()]
        return "\n".join(lines)


def parse_sequence_arg(seq: str | None) -> list[dict]:
    """Parse '--sequence' CLI syntax: 'start:pattern,start:pattern,...'
    e.g. '0:chrome_tunnel,15.5:particle_burst,40:glitch_vhs'
    Each forced segment runs until the next one's start (or end of track)."""
    if not seq:
        return []
    out = []
    for part in seq.split(","):
        part = part.strip()
        if not part:
            continue
        start_str, pattern = part.split(":", 1)
        out.append(dict(start=float(start_str), pattern=pattern.strip(), end=None))
    return out


def parse_weights_arg(w: str | None) -> dict[str, float]:
    """Parse '--weights' CLI syntax: 'pattern=1.5,pattern2=0.5'"""
    if not w:
        return {}
    out = {}
    for part in w.split(","):
        part = part.strip()
        if not part:
            continue
        name, val = part.split("=", 1)
        out[name.strip()] = float(val)
    return out
