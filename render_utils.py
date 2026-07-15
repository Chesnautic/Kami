"""Shared rendering helpers used across Y2K visualizer patterns."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def pick_color(colors, t):
    """Sample a smooth cycle across a palette color list. t can be any float."""
    n = len(colors)
    t = t % n
    i0 = int(t) % n
    i1 = (i0 + 1) % n
    return lerp_color(colors[i0], colors[i1], t - int(t))


def add_glow(base: Image.Image, glow_layer: Image.Image, radius: int, strength: float = 1.0) -> Image.Image:
    """Screen-blend a blurred copy of glow_layer onto base for a neon glow look."""
    blurred = glow_layer.filter(ImageFilter.GaussianBlur(radius))
    if strength != 1.0:
        arr = np.asarray(blurred).astype(np.float32) * strength
        blurred = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return ImageChops.screen(base, blurred)


def make_polar_grids(w: int, h: int):
    """Precompute per-pixel radius/angle grids centered on the canvas, cached by size.

    float32 rather than numpy's default float64: these grids feed every
    polar-pattern's per-frame elementwise math (chrome_tunnel, checker_tunnel,
    etc), so halving their memory footprint roughly halves the memory
    bandwidth (and therefore wall-clock cost) of all of that downstream math,
    for no visible difference at 8-bit output precision.
    """
    cx, cy = w / 2.0, h / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    return r, theta


_GRID_CACHE = {}


def polar_grids(w: int, h: int):
    key = (w, h)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = make_polar_grids(w, h)
    return _GRID_CACHE[key]


def star_points(cx, cy, outer_r, inner_r, spikes, rotation=0.0):
    pts = []
    for i in range(spikes * 2):
        ang = rotation + i * np.pi / spikes
        rad = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    return pts


def blob_points(cx, cy, base_r, n, wobble, phase, seedvals):
    """A wobbly organic blob outline (tribal/flame silhouette base shape)."""
    pts = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        wob = 1.0 + wobble * np.sin(ang * seedvals[0] + phase) * 0.5 \
            + wobble * 0.5 * np.sin(ang * seedvals[1] - phase * 1.7)
        rad = base_r * wob
        pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    return pts


def hsv_wave_color(phase):
    """Cheap HSV->RGB rainbow cycle for holographic/iridescent effects."""
    import colorsys
    h = (phase % 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


_CART_CACHE = {}


def cartesian_grids(w: int, h: int):
    key = (w, h)
    if key not in _CART_CACHE:
        ys, xs = np.mgrid[0:h, 0:w]
        _CART_CACHE[key] = (xs.astype(np.float32), ys.astype(np.float32))
    return _CART_CACHE[key]


def build_palette_lut(colors, size: int = 512) -> np.ndarray:
    """Vectorized LUT: size x 3 uint8 array cycling smoothly through `colors`."""
    n = len(colors)
    arr = np.array(colors, dtype=np.float32)
    positions = np.linspace(0, n, size, endpoint=False)
    i0 = np.floor(positions).astype(int) % n
    i1 = (i0 + 1) % n
    frac = (positions - np.floor(positions))[:, None]
    lut = arr[i0] * (1 - frac) + arr[i1] * frac
    return np.clip(lut, 0, 255).astype(np.uint8)


def sample_lut(lut: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Sample a palette LUT (size,3) at fractional positions t (any shape, any range) -> (...,3) uint8.

    Uses np.take(..., mode='wrap') to fuse the wraparound + gather into a
    single pass instead of separate mod/multiply/astype/clip passes -- this
    is a hot path (called every frame on a full-resolution array by several
    patterns) and the two implementations are numerically identical for all
    inputs (verified against the previous np.mod-based version across large
    negative/positive ranges, fractional values, and boundary cases).
    """
    size = lut.shape[0]
    idx = np.floor(t * size).astype(np.int64)
    return lut.take(idx, axis=0, mode='wrap')


# --------------------------------------------------------------------------
# pixel-art rendering: draw everything onto a small internal canvas, then
# nearest-neighbor upscale — a cheap, consistent way to get a chunky 8-bit
# "retro game" look regardless of the pattern's actual math/complexity.
# --------------------------------------------------------------------------
TARGET_PIXEL_WIDTH = 160  # internal canvas width target, in "big pixels"


def pixel_canvas(w: int, h: int, bg_color, target_width: int = TARGET_PIXEL_WIDTH):
    """Create a small low-res RGB canvas sized so that upscaling it back to
    (w, h) with nearest-neighbor gives roughly `target_width`-wide chunky
    pixels, however big the real output resolution is. Returns
    (small_image, draw, iw, ih, scale)."""
    scale = max(1, round(w / target_width))
    iw, ih = max(1, w // scale), max(1, h // scale)
    img = Image.new("RGB", (iw, ih), bg_color)
    return img, ImageDraw.Draw(img), iw, ih, scale


def upscale_pixelated(img: Image.Image, w: int, h: int) -> Image.Image:
    return img.resize((w, h), Image.NEAREST)
