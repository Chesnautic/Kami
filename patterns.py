"""
patterns.py — the Y2K chaotic pattern generators.

Every pattern is a function:

    render_X(w, h, feat, local_t, rng, pal, ctrl, state) -> PIL.Image (RGB)

    w, h      : output frame size
    feat      : dict of current audio features (rms, bass, mid, treble,
                onset, is_beat, is_drop) — see audio_analysis.py. Gains from
                `ctrl` have already been applied to these by the time a
                pattern sees them (see controls.py).
    local_t   : seconds this pattern has been on-screen (its own clock,
                only advances while it is the active pattern, so its
                animation resumes smoothly if the director cuts back to it)
    rng       : a numpy Generator seeded deterministically per pattern
    pal       : palette dict (see palettes.py) — may be a user-customized
                palette with arbitrary hex colors, not just a preset
    ctrl      : dict form of a Controls bundle (see controls.py) — at
                minimum has 'chaos' (0..1); patterns also read
                'glow_strength' and 'particle_density' where relevant
    state     : a persistent dict this pattern can stash things in across
                frames (particle lists, phase offsets, etc). Empty dict on
                first call.

Register new patterns in PATTERN_REGISTRY at the bottom of this file.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageOps, ImageFont

from render_utils import (
    polar_grids, cartesian_grids, build_palette_lut, sample_lut,
    add_glow, star_points, blob_points, clamp, pixel_canvas, upscale_pixelated,
    hsv_wave_color,
)

_DEFAULT_FONT = ImageFont.load_default()


# --------------------------------------------------------------------------
# 1. CHROME TUNNEL — warping concentric ring tunnel, chrome/neon colored
# --------------------------------------------------------------------------
def render_chrome_tunnel(w, h, feat, local_t, rng, pal, ctrl, state):
    chaos = ctrl["chaos"]
    density = ctrl.get("particle_density", 1.0)

    r, theta = polar_grids(w, h)
    lut = state.setdefault("lut", build_palette_lut(pal["colors"]))

    scroll = state.get("scroll", 0.0)
    speed = 0.35 + feat["mid"] * 1.4 + feat["rms"] * 0.6
    scroll += speed * (1 / 30.0) * (1.0 + chaos)
    state["scroll"] = scroll

    ring_scale = 0.028 + chaos * 0.01
    wobble = chaos * (0.5 + feat["onset"] * 2.0) * np.sin(theta * 5 + local_t * 3)
    ring_phase = r * ring_scale - scroll + wobble * 0.15

    hue_shift = local_t * 0.05
    color = sample_lut(lut, ring_phase * 0.5 + hue_shift).astype(np.float32)

    brightness = 0.35 + 0.65 * (0.5 + 0.5 * np.cos(ring_phase * 2 * np.pi))
    pulse = 1.0 + feat["bass"] * 0.9 * np.exp(-r / (0.5 * max(w, h)))
    brightness = np.clip(brightness * pulse, 0, 1.6)

    img_arr = np.clip(color * brightness[..., None], 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr, "RGB")

    # chrome highlight sparkles, more on beats
    draw = ImageDraw.Draw(img)
    n_sparkle = int((4 + feat["onset"] * 20 + (14 if feat["is_beat"] else 0)) * density)
    for _ in range(n_sparkle):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, min(w, h) * 0.5)
        cx, cy = w / 2 + rad * np.cos(ang), h / 2 + rad * np.sin(ang)
        s = rng.uniform(2, 6) * (1.5 if feat["is_beat"] else 1.0)
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=(255, 255, 255))

    return img


# --------------------------------------------------------------------------
# 2. EQUALIZER BARS — chrome mirrored spectrum bars, winamp-esque
# --------------------------------------------------------------------------
def render_equalizer_bars(w, h, feat, local_t, rng, pal, ctrl, state):
    chaos = ctrl["chaos"]

    n_bars = state.setdefault("n_bars", 42)
    phases = state.setdefault("phases", rng.uniform(0, 10, n_bars))
    img = Image.new("RGB", (w, h), pal["bg"])
    draw = ImageDraw.Draw(img)

    cy = h / 2
    bar_w = w / n_bars
    max_h = h * 0.42

    colors = pal["colors"]
    for j in range(n_bars):
        pos = j / (n_bars - 1)
        if pos < 0.4:
            band = feat["bass"] * (1 - pos / 0.4) + feat["mid"] * (pos / 0.4)
        elif pos < 0.7:
            local = (pos - 0.4) / 0.3
            band = feat["mid"] * (1 - local) + feat["treble"] * local * 0.6
        else:
            local = (pos - 0.7) / 0.3
            band = feat["treble"] * (0.6 + 0.4 * local)

        wiggle = 0.55 + 0.45 * np.sin(local_t * 2.2 + phases[j])
        spike = feat["onset"] * (1.2 if feat["is_beat"] else 0.4)
        bar_h = max_h * clamp(band * wiggle + spike * 0.5, 0.03, 1.3) * (0.6 + 0.4 * chaos)

        x0 = j * bar_w + 1
        x1 = x0 + bar_w - 2
        col = colors[j % len(colors)]
        top_col = tuple(min(255, c + 90) for c in col)

        # top bar
        draw.rectangle([x0, cy - bar_h, x1, cy], fill=col, outline=(10, 10, 10))
        draw.rectangle([x0, cy - bar_h, x1, cy - bar_h + max(2, bar_h * 0.08)], fill=top_col)
        # mirrored bottom bar (shorter, hazier)
        bh2 = bar_h * 0.55
        draw.rectangle([x0, cy, x1, cy + bh2], fill=tuple(c // 2 for c in col))

    return img


# --------------------------------------------------------------------------
# 3. PARTICLE BURST — exploding Y2K confetti/sparkle particles
# --------------------------------------------------------------------------
_SHAPES = ["star", "circle", "cross", "diamond"]


def _draw_particle(draw, p, glow_draw):
    x, y, size, col, shape, alpha_life = p["x"], p["y"], p["size"], p["color"], p["shape"], p["life_frac"]
    s = size * clamp(alpha_life * 1.4, 0.15, 1.0)
    if shape == "star":
        pts = star_points(x, y, s, s * 0.42, 5, rotation=p.get("rot", 0))
        draw.polygon(pts, fill=col)
    elif shape == "circle":
        draw.ellipse([x - s, y - s, x + s, y + s], fill=col)
    elif shape == "cross":
        draw.line([x - s, y, x + s, y], fill=col, width=max(1, int(s * 0.4)))
        draw.line([x, y - s, x, y + s], fill=col, width=max(1, int(s * 0.4)))
    else:  # diamond
        draw.polygon([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], fill=col)
    glow_draw.ellipse([x - s * 1.3, y - s * 1.3, x + s * 1.3, y + s * 1.3], fill=col)


def render_particle_burst(w, h, feat, local_t, rng, pal, ctrl, state):
    chaos = ctrl["chaos"]
    density = ctrl.get("particle_density", 1.0)
    glow_strength = ctrl.get("glow_strength", 1.0)

    particles = state.setdefault("particles", [])
    dt = 1 / 30.0

    # (kept as an /8 of the original counts -- full brightness was too harsh)
    n_spawn = int((2 + feat["rms"] * 6 * (1 + chaos)) * density / 8)
    if feat["is_beat"]:
        n_spawn += max(1, int((18 + 40 * chaos) * density / 8))
    if feat["is_drop"]:
        n_spawn += max(1, int((30 + 60 * chaos) * density / 8))

    cx, cy = w / 2, h / 2
    for _ in range(n_spawn):
        ang = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(80, 420) * (1 + feat["rms"])
        particles.append(dict(
            x=cx + rng.uniform(-30, 30), y=cy + rng.uniform(-30, 30),
            vx=np.cos(ang) * speed, vy=np.sin(ang) * speed,
            size=rng.uniform(6, 22) * (1 + chaos * 0.6),
            color=pal["colors"][rng.integers(0, len(pal["colors"]))],
            shape=_SHAPES[rng.integers(0, len(_SHAPES))],
            rot=rng.uniform(0, 2 * np.pi),
            age=0.0, maxlife=rng.uniform(0.7, 1.8),
        ))

    img = Image.new("RGB", (w, h), pal["bg"])
    glow_layer = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    glow_draw = ImageDraw.Draw(glow_layer)

    alive = []
    for p in particles:
        p["age"] += dt
        if p["age"] >= p["maxlife"]:
            continue
        p["x"] += p["vx"] * dt
        p["y"] += p["vy"] * dt
        p["vx"] *= 0.985
        p["vy"] = p["vy"] * 0.985 + 40 * dt  # slight gravity drift
        p["life_frac"] = 1.0 - p["age"] / p["maxlife"]
        if -50 < p["x"] < w + 50 and -50 < p["y"] < h + 50:
            _draw_particle(draw, p, glow_draw)
            alive.append(p)
    state["particles"] = alive[-1800:]  # safety cap

    return add_glow(img, glow_layer, radius=9, strength=0.9 * glow_strength)


# --------------------------------------------------------------------------
# 4. KALEIDOSCOPE — N-fold mirrored mandala of chaotic blobs/lines
# --------------------------------------------------------------------------
def render_kaleidoscope(w, h, feat, local_t, rng, pal, ctrl, state):
    chaos = ctrl["chaos"]
    glow_strength = ctrl.get("glow_strength", 1.0)

    # This pattern's cost is dominated by five full-frame PIL rotations plus
    # several full-frame composites and a gaussian blur (profiled: rotation
    # alone is ~57% of the pattern's runtime, blur ~17%, composites ~12%).
    # All of that scales with pixel count, and the blob shapes are soft and
    # get blurred by add_glow anyway, so we build the whole mandala at a
    # reduced internal resolution and upscale the finished frame with a
    # single bilinear resize -- ~4x fewer pixels through the expensive part
    # of the pipeline for output that's visually very close to full-res.
    scale = 0.5
    rw, rh = max(2, round(w * scale)), max(2, round(h * scale))

    blobs = state.setdefault("blobs", [
        dict(ang=rng.uniform(0, 2 * np.pi), rad=rng.uniform(0.35, 0.85),
             speed=rng.uniform(-0.6, 0.6), size=rng.uniform(0.035, 0.09),
             color_t=rng.uniform(0, 1), seed=rng.uniform(1, 4, size=2))
        for _ in range(6)
    ])
    lut = state.setdefault("lut", build_palette_lut(pal["colors"]))

    base = Image.new("RGB", (rw, rh), (0, 0, 0))
    draw = ImageDraw.Draw(base)
    R = min(rw, rh) * 0.5
    cx, cy = rw * 0.32, rh * 0.32  # off-center so rotation makes a mandala, not a static circle

    for b in blobs:
        b["ang"] += b["speed"] * (1 / 30.0) * (1 + feat["mid"] * 1.5)
        rad = b["rad"] * R * (0.8 + 0.3 * feat["bass"])
        bx = cx + rad * np.cos(b["ang"])
        by = cy + rad * np.sin(b["ang"])
        size = b["size"] * R * (0.7 + feat["treble"] * 0.8 + (0.4 if feat["is_beat"] else 0))
        col = tuple(int(c) for c in sample_lut(lut, np.array([b["color_t"] + local_t * 0.05]))[0])
        pts = blob_points(bx, by, size, 14, 0.5 + chaos * 0.5, local_t * 2, b["seed"])
        draw.polygon(pts, fill=col)

    n_fold = 6
    result = base
    for k in range(1, n_fold):
        rotated = base.rotate(k * 360 / n_fold, resample=Image.BILINEAR, center=(rw / 2, rh / 2))
        result = ImageChops.lighter(result, rotated)
    mirrored = result.transpose(Image.FLIP_LEFT_RIGHT)
    result = ImageChops.lighter(result, mirrored)

    bg = Image.new("RGB", (rw, rh), pal["bg"])
    result = ImageChops.lighter(bg, result)
    glowed = add_glow(result, result, radius=max(1, round(5 * scale)), strength=0.35 * glow_strength)

    if (rw, rh) != (w, h):
        glowed = glowed.resize((w, h), Image.BILINEAR)
    return glowed


# --------------------------------------------------------------------------
# 5. CHECKER TUNNEL — Tron-style perspective floor/ceiling grid + sun
# --------------------------------------------------------------------------
def render_checker_tunnel(w, h, feat, local_t, rng, pal, ctrl, state):
    glow_strength = ctrl.get("glow_strength", 1.0)

    xs, ys = cartesian_grids(w, h)
    cx, cy = w / 2.0, h * 0.5

    scroll = state.get("scroll", 0.0)
    scroll += (0.6 + feat["rms"] * 2.5 + feat["bass"] * 1.5) * (1 / 30.0)
    state["scroll"] = scroll

    dy = ys - cy
    sign = np.sign(dy)
    sign[sign == 0] = 1
    depth = 1.0 / (np.abs(dy) / h + 0.035)
    u = (xs - cx) * depth * 0.015
    v = depth * 0.26 * sign + scroll

    tile = 1.0
    checker = (np.floor(u / tile) + np.floor(v / tile)).astype(np.int64) % 2

    colors = pal["colors"]
    col_a = np.array(colors[0], dtype=np.float32)
    col_b = np.array(pal["bg"], dtype=np.float32)
    base = np.where(checker[..., None] == 0, col_a, col_b)

    fog = np.clip(1.0 - np.abs(dy) / (h * 0.75), 0.0, 1.0)
    fade = np.clip(1.0 / (1.0 + (depth * 0.01)), 0, 1)
    brightness = (0.15 + 0.85 * fade) * (1 - fog * 0.15 + 0.15)
    img_arr = np.clip(base * brightness[..., None], 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr, "RGB")

    # horizon glow "sun"
    glow_layer = Image.new("RGB", (w, h), (0, 0, 0))
    gdraw = ImageDraw.Draw(glow_layer)
    sun_r = min(w, h) * (0.09 + feat["bass"] * 0.08 + (0.05 if feat["is_beat"] else 0))
    accent = pal["accent"]
    gdraw.ellipse([cx - sun_r, cy - sun_r, cx + sun_r, cy + sun_r], fill=accent)
    img = add_glow(img, glow_layer, radius=25, strength=1.3 * glow_strength)
    draw = ImageDraw.Draw(img)
    draw.line([(0, cy), (w, cy)], fill=(255, 255, 255), width=2)

    return img


# --------------------------------------------------------------------------
# 6. GLITCH / VHS — plasma field torn apart by datamosh + scanline glitches
# --------------------------------------------------------------------------
def render_glitch_vhs(w, h, feat, local_t, rng, pal, ctrl, state):
    chaos = ctrl["chaos"]
    density = ctrl.get("particle_density", 1.0)

    xs, ys = cartesian_grids(w, h)
    lut = state.setdefault("lut", build_palette_lut(pal["colors"]))

    t = local_t
    val = (
        np.sin(xs * 0.018 + t * 1.3)
        + np.sin(ys * 0.024 - t * 1.1)
        + np.sin((xs + ys) * 0.012 + t * 0.7)
        + np.sin(np.sqrt((xs - w / 2) ** 2 + (ys - h / 2) ** 2) * 0.02 - t * 2)
    ) / 4.0
    val = (val + 1) / 2.0
    val = val * (0.6 + feat["mid"] * 0.8) + feat["bass"] * 0.15

    color = sample_lut(lut, val + t * 0.03).astype(np.uint8)
    arr = color.copy()

    # scanlines
    arr[::2] = (arr[::2].astype(np.float32) * 0.72).astype(np.uint8)

    # RGB channel split (VHS chromatic aberration), amount grows with chaos/onset
    shift = int(2 + chaos * 6 + feat["onset"] * 10)
    if shift > 0:
        arr[:, :, 0] = np.roll(arr[:, :, 0], shift, axis=1)
        arr[:, :, 2] = np.roll(arr[:, :, 2], -shift, axis=1)

    # random horizontal glitch bands (row-shift blocks) — more on beats, scaled by density
    n_bands = int((2 + chaos * 6 + (10 if feat["is_beat"] else 0) + feat["onset"] * 8) * density)
    for _ in range(n_bands):
        y0 = rng.integers(0, h)
        band_h = rng.integers(2, max(3, int(h * 0.04)))
        y1 = min(h, y0 + band_h)
        offset = rng.integers(-int(w * 0.12) - 1, int(w * 0.12) + 1)
        arr[y0:y1] = np.roll(arr[y0:y1], offset, axis=1)

    # occasional big block glitch (copy a chunk elsewhere) on drops
    if feat["is_drop"] or rng.random() < 0.03 * (1 + chaos):
        bw, bh = rng.integers(w // 8, w // 3), rng.integers(h // 10, h // 5)
        sx, sy = rng.integers(0, max(1, w - bw)), rng.integers(0, max(1, h - bh))
        dx, dy = rng.integers(0, max(1, w - bw)), rng.integers(0, max(1, h - bh))
        arr[dy:dy + bh, dx:dx + bw] = arr[sy:sy + bh, sx:sx + bw]

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return img


# --------------------------------------------------------------------------
# 7. SUNSET PIXEL — purple/pink synthwave sunset with a striped retro sun
# --------------------------------------------------------------------------
def render_sunset_pixel(w, h, feat, local_t, rng, pal, ctrl, state):
    img, draw, iw, ih, scale = pixel_canvas(w, h, (20, 8, 35))

    top_col = np.array((35, 8, 60), dtype=np.float32)
    horizon_col = np.array(pal["accent"], dtype=np.float32) * 0.75 + np.array((255, 130, 70)) * 0.25
    horizon_y = ih * 0.60

    bands = 14
    for by in range(bands):
        t = by / (bands - 1)
        y0 = int(t * horizon_y)
        y1 = int((t + 1.0 / bands) * horizon_y) + 1
        col = tuple(int(c) for c in (top_col * (1 - t) + horizon_col * t))
        draw.rectangle([0, y0, iw, y1], fill=col)

    # water below the horizon -- a darkening gradient that the sun/sky
    # reflect into, so the scene reads as being over the sea
    water_top = horizon_col * 0.55
    water_bot = horizon_col * 0.18
    water_h = max(1, ih - horizon_y)
    for yy in range(int(horizon_y), ih):
        t = (yy - horizon_y) / water_h
        col = tuple(int(c) for c in (water_top * (1 - t) + water_bot * t))
        draw.line([(0, yy), (iw, yy)], fill=col)

    # twinkling stars above the horizon
    star_state = state.setdefault("stars", [
        (rng.uniform(0, iw), rng.uniform(0, horizon_y * 0.85), rng.uniform(0, 6.28))
        for _ in range(26)
    ])
    for sx, sy, ph in star_state:
        tw = 0.5 + 0.5 * np.sin(local_t * 3 + ph + feat["treble"] * 4)
        if tw > 0.35:
            b = int(180 + 75 * tw)
            draw.point((sx, sy), fill=(b, b, min(255, b + 20)))

    # the retro striped sun, half-set into the horizon
    sun_r = ih * (0.16 + feat["bass"] * 0.05 + (0.02 if feat["is_beat"] else 0))
    scx, scy = iw / 2, horizon_y
    sun_top = np.array((255, 232, 120))
    sun_bot = np.array((255, 40, 130))
    stripe_offset = (local_t * (1.2 + feat["mid"] * 2)) % 1.0
    for yy in range(int(scy - sun_r), int(scy) + 1):
        dy = scy - yy
        dx = np.sqrt(max(0, sun_r * sun_r - dy * dy))
        frac = 1.0 - (dy / sun_r)
        col = tuple(int(c) for c in (sun_top * (1 - frac) + sun_bot * frac))
        row_pos = (frac * 5 + stripe_offset * 2) % 1.0
        if frac > 0.42 and row_pos < 0.38:
            continue  # stripe cutout gap
        draw.line([(scx - dx, yy), (scx + dx, yy)], fill=col)

    # mirrored, rippling, fading reflection of the sun in the water
    refl_r = sun_r
    for dy in range(0, int(refl_r) + 1):
        yy = int(scy + dy)
        if yy >= ih:
            break
        dx = np.sqrt(max(0, refl_r * refl_r - dy * dy))
        frac = 1.0 - (dy / refl_r)
        sun_col = tuple(int(c) for c in (sun_top * (1 - frac) + sun_bot * frac))
        row_pos = (frac * 5 + stripe_offset * 2) % 1.0
        if frac > 0.42 and row_pos < 0.38:
            continue  # matching stripe cutout
        depth_t = dy / max(1, refl_r)
        wt = (yy - horizon_y) / water_h
        water_col = tuple(int(c) for c in (water_top * (1 - wt) + water_bot * wt))
        fade = max(0.0, 1.0 - depth_t) * 0.7
        blended = tuple(int(sun_col[k] * fade + water_col[k] * (1 - fade)) for k in range(3))
        wobble = int(round(np.sin(yy * 0.7 + local_t * 1.8) * 2 * (1 - depth_t * 0.5)))
        draw.line([(scx - dx + wobble, yy), (scx + dx + wobble, yy)], fill=blended)

    # a few horizontal shimmer streaks further out in the water
    shimmer = state.setdefault("shimmer", [
        (rng.uniform(0, iw), rng.uniform(0.15, 1.0), rng.uniform(0, 6.28), rng.uniform(2, 6))
        for _ in range(10)
    ])
    for sx, sd, ph, slen in shimmer:
        yy = int(horizon_y + water_h * sd)
        if yy >= ih:
            continue
        tw = 0.5 + 0.5 * np.sin(local_t * 2.2 + ph)
        if tw < 0.5:
            continue
        wt = (yy - horizon_y) / water_h
        base = tuple(int(c) for c in (water_top * (1 - wt) + water_bot * wt))
        col = tuple(min(255, int(c + 60 * tw)) for c in base)
        draw.line([(sx, yy), (sx + slen, yy)], fill=col)

    upscaled = upscale_pixelated(img, w, h)
    return upscaled


# --------------------------------------------------------------------------
# 8. PIXEL CARS — blocky 8-bit traffic driving across a night highway
# --------------------------------------------------------------------------
def render_pixel_cars(w, h, feat, local_t, rng, pal, ctrl, state):
    density = ctrl.get("particle_density", 1.0)
    horizon = h * 0.42
    img = Image.new("RGB", (w, h), (10, 8, 18))
    draw = ImageDraw.Draw(img)
    _night_gradient(draw, w, h, horizon, top=(10, 8, 22), mid=(35, 15, 42), bottom=(20, 18, 26))
    _city_skyline(draw, w, horizon, rng, state, n=16)
    scroll = state.get("scroll", 0.0) + (0.06 + feat["rms"] * 0.22) * (1 / 30.0)
    state["scroll"] = scroll % 1.0
    _road(draw, w, h, horizon, state["scroll"], lane_count=3)

    n_lanes = 3
    cars = state.setdefault("cars", [])
    n_spawn_target = int(3 + density * 3)
    if len(cars) < n_spawn_target and (rng.random() < 0.05 + feat["onset"] * 0.4):
        lane = int(rng.integers(0, n_lanes))
        direction = 1 if lane % 2 == 0 else -1
        cars.append(dict(
            lane=lane, t=(0.0 if direction > 0 else 1.0),
            speed=direction * rng.uniform(0.10, 0.2) * (1 + feat["rms"]),
            color=pal["colors"][rng.integers(0, len(pal["colors"]))],
        ))

    alive = []
    for c in cars:
        c["t"] += c["speed"] * (1 / 30.0)
        # lane position across the road's perspective trapezoid, near-side (t~1) is bigger/lower
        tt = np.clip(0.15 + 0.8 * ((c["lane"] + 0.5) / n_lanes), 0, 1)
        depth = np.clip(c["t"], 0.02, 1.0)
        cy = horizon + (h - horizon) * (depth ** 1.5)
        vp = w / 2
        half = (w * 0.025 + (w * 0.62 - w * 0.025) * (depth ** 1.5)) * (tt - 0.5) * 2
        cx = vp + half
        s = h * 0.02 + h * 0.075 * (depth ** 1.5)
        # cars in near-side lanes (t rising toward the viewer) are driving
        # TOWARD the camera -- show their front/headlights; far-side lanes
        # (t falling toward the horizon) are driving AWAY -- show their
        # rear/taillights. A straight-ahead road only ever shows a car head
        # on or from behind, never in side profile.
        view = "front" if c["speed"] > 0 else "rear"
        _paste_car_body(img, s, c["color"], cx, cy, cabin_color=(150, 210, 235), view=view)
        if -0.15 < c["t"] < 1.15:
            alive.append(c)
    state["cars"] = alive
    return img


# --------------------------------------------------------------------------
# 9. SHOOTING STARS — night sky with streaking pixel shooting stars
# --------------------------------------------------------------------------
def render_shooting_stars(w, h, feat, local_t, rng, pal, ctrl, state):
    density = ctrl.get("particle_density", 1.0)
    img, draw, iw, ih, scale = pixel_canvas(w, h, (10, 6, 24))

    star_state = state.setdefault("bg_stars", [
        (rng.uniform(0, iw), rng.uniform(0, ih), rng.uniform(0, 6.28), rng.uniform(0.5, 1.0))
        for _ in range(50)
    ])
    for sx, sy, ph, base in star_state:
        tw = base * (0.6 + 0.4 * np.sin(local_t * 2.5 + ph))
        b = int(120 * tw)
        if b > 15:
            draw.point((sx, sy), fill=(b, b, min(255, b + 30)))

    mcx, mcy, mr = iw * 0.85, ih * 0.16, ih * 0.09
    draw.ellipse([mcx - mr, mcy - mr, mcx + mr, mcy + mr], fill=(235, 230, 210))
    draw.ellipse([mcx - mr * 0.55, mcy - mr, mcx + mr * 1.3, mcy + mr], fill=(10, 6, 24))

    streaks = state.setdefault("streaks", [])
    spawn_p = 0.05 + feat["treble"] * 0.25 + feat["onset"] * 0.4
    if rng.random() < spawn_p * density:
        sx = rng.uniform(0, iw)
        sy = rng.uniform(0, ih * 0.4)
        ang = rng.uniform(0.3, 0.9)
        speed = rng.uniform(60, 110) * (1 + feat["rms"])
        streaks.append(dict(x=sx, y=sy, vx=np.cos(ang) * speed, vy=np.sin(ang) * speed,
                             age=0.0, maxlife=rng.uniform(0.5, 0.9),
                             color=pal["colors"][rng.integers(0, len(pal["colors"]))]))

    alive = []
    dt = 1 / 30.0
    for s in streaks:
        s["age"] += dt
        if s["age"] >= s["maxlife"]:
            continue
        s["x"] += s["vx"] * dt
        s["y"] += s["vy"] * dt
        trail_len = 5 + feat["bass"] * 4
        tx = s["x"] - s["vx"] * dt * trail_len
        ty = s["y"] - s["vy"] * dt * trail_len
        draw.line([(tx, ty), (s["x"], s["y"])], fill=s["color"])
        draw.point((s["x"], s["y"]), fill=(255, 255, 255))
        if -10 < s["x"] < iw + 10 and -10 < s["y"] < ih + 10:
            alive.append(s)
    state["streaks"] = alive

    return upscale_pixelated(img, w, h)


# --------------------------------------------------------------------------
# 10. PIXEL GLOBE — spinning blocky "under construction" Y2K globe
# --------------------------------------------------------------------------
def render_pixel_globe(w, h, feat, local_t, rng, pal, ctrl, state):
    img, draw, iw, ih, scale = pixel_canvas(w, h, pal["bg"])

    cx, cy = iw / 2, ih / 2
    R = min(iw, ih) * 0.36 * (1 + feat["bass"] * 0.08)
    rot = state.get("rot", 0.0)
    rot += (0.6 + feat["mid"] * 2.0) * (1 / 30.0)
    state["rot"] = rot

    ocean = pal["colors"][0]
    land = (60, 200, 120)

    # base sphere with a smooth left-lit shading gradient (column by column,
    # blending toward a shadow tone rather than a hard black cutoff)
    shadow = tuple(int(c * 0.35) for c in ocean)
    for xx in range(int(cx - R), int(cx + R) + 1):
        f = (xx - cx) / max(R, 1)
        if abs(f) > 1:
            continue
        dy = R * np.sqrt(max(0.0, 1 - f * f))
        shade_t = clamp((f + 0.15) / 1.15, 0.0, 1.0)  # 0 = lit rim, 1 = dark rim
        col = tuple(int(ocean[i] * (1 - shade_t) + shadow[i] * shade_t) for i in range(3))
        draw.line([(xx, cy - dy), (xx, cy + dy)], fill=col)

    # rotating "continents" — points on a sphere projected with the classic
    # flat-circle rotation illusion (x = cos(phase), squashed by latitude)
    continents = state.setdefault("continents", [
        dict(lat=rng.uniform(-0.85, 0.85), lon=rng.uniform(0, 6.28), size=rng.uniform(4, 8))
        for _ in range(20)
    ])
    for c in continents:
        phase = c["lon"] + rot
        depth = np.cos(phase)
        if depth < -0.1:
            continue
        lat_r = np.cos(c["lat"])
        px = cx + R * lat_r * np.sin(phase)
        py = cy - R * np.sin(c["lat"]) * 0.92
        s = c["size"] * max(0.35, depth) * 0.5
        shade_t = clamp((np.sin(phase) + 0.15) / 1.15, 0.0, 1.0)
        col = tuple(int(land[i] * (1 - shade_t * 0.6) + shadow[i] * (shade_t * 0.6)) for i in range(3))
        draw.ellipse([px - s, py - s * 0.65, px + s, py + s * 0.65], fill=col)

    # latitude wireframe rings, subtle grid lines curving with the sphere
    grid_col = tuple(min(255, c + 25) for c in shadow)
    for i in range(1, 4):
        f = i / 4
        ry = R * np.sqrt(max(0.0, 1 - f * f)) * 0.94
        draw.arc([cx - R * 0.98, cy - ry, cx + R * 0.98, cy + ry], start=0, end=360, fill=grid_col)
    # one meridian arc that rotates with the globe, for a spinning-wireframe cue
    mphase = rot % (2 * np.pi)
    mx = R * 0.98 * np.sin(mphase)
    if abs(mx) < R * 0.9:
        draw.arc([cx - abs(mx), cy - R, cx + abs(mx), cy + R], start=0, end=360, fill=grid_col)

    # outer rim highlight
    draw.ellipse([cx - R, cy - R, cx + R, cy + R], outline=tuple(min(255, c + 60) for c in ocean))

    # small orbiting satellite/sparkle, blinking on beats
    orbit_r = R * 1.35
    ang = local_t * 1.4
    ox, oy = cx + orbit_r * np.cos(ang), cy + orbit_r * np.sin(ang) * 0.4
    if np.cos(ang) > -0.6:
        blink = (255, 255, 255) if feat["is_beat"] else pal["glow"]
        draw.point((ox, oy), fill=blink)
        draw.point((ox + 1, oy), fill=blink)

    return upscale_pixelated(img, w, h)


# --------------------------------------------------------------------------
# 11. PIXEL BOUNCE — DVD-logo-style bouncing pixel sprites
# --------------------------------------------------------------------------
def _pixel_shape_points(cx, cy, s, shape, rot=0.0):
    if shape == "heart":
        pts = []
        for i in range(24):
            t = i / 24 * 2 * np.pi
            hx = 16 * np.sin(t) ** 3
            hy = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
            pts.append((cx + hx * s / 16, cy + hy * s / 16))
        return pts
    if shape == "diamond":
        return [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
    return star_points(cx, cy, s, s * 0.42, 5, rotation=rot)


def render_pixel_bounce(w, h, feat, local_t, rng, pal, ctrl, state):
    density = ctrl.get("particle_density", 1.0)
    img, draw, iw, ih, scale = pixel_canvas(w, h, pal["bg"])

    n_sprites = max(1, int(1 + density * 2))
    shapes = state.setdefault("sprites", [
        dict(x=rng.uniform(10, iw - 10), y=rng.uniform(10, ih - 10),
             vx=rng.choice([-1, 1]) * rng.uniform(16, 26), vy=rng.choice([-1, 1]) * rng.uniform(16, 26),
             size=rng.uniform(6, 9), shape=rng.choice(["heart", "star", "diamond"]),
             color=pal["colors"][rng.integers(0, len(pal["colors"]))], rot=0.0)
        for _ in range(n_sprites)
    ])
    if len(shapes) < n_sprites:
        for _ in range(n_sprites - len(shapes)):
            shapes.append(dict(x=rng.uniform(10, iw - 10), y=rng.uniform(10, ih - 10),
                                vx=rng.choice([-1, 1]) * 20, vy=rng.choice([-1, 1]) * 20,
                                size=7, shape=rng.choice(["heart", "star", "diamond"]),
                                color=pal["colors"][rng.integers(0, len(pal["colors"]))], rot=0.0))
    shapes = shapes[:n_sprites]

    dt = 1 / 30.0
    pulse = 1.0 + feat["bass"] * 0.3 + (0.2 if feat["is_beat"] else 0)
    for sp in shapes:
        sp["x"] += sp["vx"] * dt * (1 + feat["rms"] * 0.5)
        sp["y"] += sp["vy"] * dt * (1 + feat["rms"] * 0.5)
        sp["rot"] += dt * 0.6
        bounced = False
        if sp["x"] < sp["size"] or sp["x"] > iw - sp["size"]:
            sp["vx"] *= -1
            sp["x"] = clamp(sp["x"], sp["size"], iw - sp["size"])
            bounced = True
        if sp["y"] < sp["size"] or sp["y"] > ih - sp["size"]:
            sp["vy"] *= -1
            sp["y"] = clamp(sp["y"], sp["size"], ih - sp["size"])
            bounced = True
        if bounced:
            sp["color"] = pal["colors"][rng.integers(0, len(pal["colors"]))]
        pts = _pixel_shape_points(sp["x"], sp["y"], sp["size"] * pulse, sp["shape"], sp["rot"])
        draw.polygon(pts, fill=sp["color"])

    state["sprites"] = shapes
    return upscale_pixelated(img, w, h)


# --------------------------------------------------------------------------
# 12. PIXEL RAIN — falling 8-bit hearts/stars/diamonds shower
# --------------------------------------------------------------------------
def render_pixel_rain(w, h, feat, local_t, rng, pal, ctrl, state):
    density = ctrl.get("particle_density", 1.0)
    img, draw, iw, ih, scale = pixel_canvas(w, h, pal["bg"])

    drops = state.setdefault("drops", [])
    n_spawn = int((0.6 + feat["treble"] * 2 + (2 if feat["is_beat"] else 0)) * density)
    for _ in range(n_spawn):
        drops.append(dict(
            x=rng.uniform(0, iw), y=-4.0,
            speed=rng.uniform(18, 40) * (1 + feat["bass"] * 0.6),
            size=rng.uniform(2.5, 4.5),
            shape=rng.choice(["heart", "star", "diamond"]),
            color=pal["colors"][rng.integers(0, len(pal["colors"]))],
        ))

    dt = 1 / 30.0
    alive = []
    for d in drops:
        d["y"] += d["speed"] * dt
        pts = _pixel_shape_points(d["x"], d["y"], d["size"], d["shape"])
        draw.polygon(pts, fill=d["color"])
        if d["y"] < ih + 6:
            alive.append(d)
    state["drops"] = alive[-500:]

    return upscale_pixelated(img, w, h)


# --------------------------------------------------------------------------
# 8. STARBURST POP — halftone dots + comic starburst shapes, CD-cover chaos
# --------------------------------------------------------------------------
def render_starburst_pop(w, h, feat, local_t, rng, pal, ctrl, state):
    density = ctrl.get("particle_density", 1.0)

    xs, ys = cartesian_grids(w, h)
    cx, cy = w / 2, h / 2

    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    dot_size = 10 + feat["bass"] * 6
    grid_x = np.mod(xs, dot_size)
    grid_y = np.mod(ys, dot_size)
    d = np.sqrt((grid_x - dot_size / 2) ** 2 + (grid_y - dot_size / 2) ** 2)
    radius_falloff = np.clip(1.0 - dist / (max(w, h) * 0.75), 0.15, 1.0)
    dot_r = (dot_size * 0.5 * 0.85) * radius_falloff
    mask = d < dot_r

    bg = np.array(pal["bg"], dtype=np.uint8)
    dot_col = np.array(pal["colors"][0], dtype=np.uint8)
    arr = np.tile(bg, (h, w, 1))
    arr[mask] = dot_col
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)

    bursts = state.setdefault("bursts", [
        dict(x=rng.uniform(0.15, 0.85) * w, y=rng.uniform(0.15, 0.85) * h,
             base_r=rng.uniform(40, 90), rot=rng.uniform(0, 6.28), speed=rng.uniform(-1, 1),
             color=pal["colors"][rng.integers(0, len(pal["colors"]))])
        for _ in range(4)
    ])
    for b in bursts:
        b["rot"] += b["speed"] * (1 / 30.0) * (1 + feat["mid"])
        r_out = b["base_r"] * (1 + feat["rms"] * 0.6 + (0.4 if feat["is_beat"] else 0))
        pts = star_points(b["x"], b["y"], r_out, r_out * 0.42, 10, rotation=b["rot"])
        draw.polygon(pts, fill=b["color"], outline=(15, 15, 15), width=4)

    n_sparkle = int((6 + feat["treble"] * 20 + feat["onset"] * 15) * density)
    for _ in range(n_sparkle):
        sx, sy = rng.uniform(0, w), rng.uniform(0, h)
        s = rng.uniform(3, 10)
        pts = star_points(sx, sy, s, s * 0.4, 4, rotation=rng.uniform(0, 6.28))
        draw.polygon(pts, fill=(255, 255, 255))

    return img


# ==========================================================================
# PACK 1 (extra) — OSCILLOSCOPE WAVE — literal scrolling neon waveform
# ==========================================================================
def render_oscilloscope_wave(w, h, feat, local_t, rng, pal, ctrl, state):
    chaos = ctrl["chaos"]
    glow_strength = ctrl.get("glow_strength", 1.0)

    hist = state.setdefault("hist", [])
    hist.append(feat["rms"] * 0.6 + feat["bass"] * 0.4)
    maxlen = 140
    if len(hist) > maxlen:
        hist.pop(0)

    img = Image.new("RGB", (w, h), pal["bg"])
    draw = ImageDraw.Draw(img)
    glow_layer = Image.new("RGB", (w, h), (0, 0, 0))
    gdraw = ImageDraw.Draw(glow_layer)

    grid_col = tuple(c // 6 for c in pal["accent"])
    for gx in range(0, w, max(20, w // 24)):
        draw.line([(gx, 0), (gx, h)], fill=grid_col)
    cy = h / 2
    draw.line([(0, cy), (w, cy)], fill=tuple(c // 4 for c in pal["accent"]))

    if len(hist) > 1:
        col = pal["colors"][int(local_t * 2) % len(pal["colors"])]
        pts = [(w * i / (maxlen - 1), cy - v * h * 0.4 * (1 + chaos * 0.3)) for i, v in enumerate(hist)]
        draw.line(pts, fill=col, width=3)
        gdraw.line(pts, fill=col, width=3)
        pts2 = [(x, h - y) for x, y in pts]
        draw.line(pts2, fill=col, width=2)
        gdraw.line(pts2, fill=col, width=2)

    return add_glow(img, glow_layer, radius=8, strength=0.8 * glow_strength)


# ==========================================================================
# PACK 2 — CARS — shared sprite helper + 8 scenes
#
# Rendered at full resolution with smooth anti-aliased shapes (no chunky
# nearest-neighbor pixel upscaling like the rest of the app) so the cars
# read as actual detailed cars rather than tiny 8-bit blobs, with a night
# street/neon backdrop for a "Tokyo drift" mood.
# ==========================================================================
def _lighten(c, amt):
    return tuple(min(255, int(v + amt)) for v in c)


def _darken(c, amt):
    return tuple(max(0, int(v - amt)) for v in c)


def _car_sprite(s, body_color, cabin_color=None, spoiler=True):
    """Build a detailed car sprite (facing right / +x = front) on its own
    transparent RGBA layer, sized off `s` (half-body scale), so callers can
    flip/rotate/paste it for any facing or drift angle."""
    bl = s * 3.4   # body length
    bh = s * 1.3   # body height
    pad = int(bh * 1.6)
    size_x, size_y = int(bl + pad * 2), int(bh * 3.4 + pad)
    im = Image.new("RGBA", (size_x, size_y), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = size_x / 2.0, size_y * 0.52

    dark = _darken(body_color, 50)
    light = _lighten(body_color, 30)
    cabin_color = cabin_color or (150, 210, 235)

    # ground shadow
    d.ellipse([cx - bl * 0.5, cy + bh * 0.62, cx + bl * 0.48, cy + bh * 0.95], fill=(0, 0, 0, 90))

    # wheels (drawn before body so the body overlaps their top halves)
    wr = bh * 0.5
    wheel_xs = (cx - bl * 0.3, cx + bl * 0.27)
    for wx in wheel_xs:
        d.ellipse([wx - wr, cy + bh * 0.28 - wr, wx + wr, cy + bh * 0.28 + wr], fill=(12, 12, 15, 255))
        d.ellipse([wx - wr * 0.42, cy + bh * 0.28 - wr * 0.42, wx + wr * 0.42, cy + bh * 0.28 + wr * 0.42],
                   fill=(120, 120, 130, 255))

    # lower body capsule (rounded rectangle via rect + end ellipses)
    lb_top, lb_bot = cy - bh * 0.08, cy + bh * 0.42
    d.rectangle([cx - bl * 0.48, lb_top, cx + bl * 0.46, lb_bot], fill=dark)
    d.ellipse([cx - bl * 0.48 - bh * 0.35, lb_top, cx - bl * 0.48 + bh * 0.35, lb_bot], fill=dark)
    d.ellipse([cx + bl * 0.46 - bh * 0.35, lb_top, cx + bl * 0.46 + bh * 0.35, lb_bot], fill=dark)

    # main body color band above the dark lower body
    mb_top, mb_bot = cy - bh * 0.32, cy + bh * 0.02
    d.rectangle([cx - bl * 0.46, mb_top, cx + bl * 0.5, mb_bot], fill=body_color)
    d.ellipse([cx - bl * 0.46 - bh * 0.2, mb_top, cx - bl * 0.46 + bh * 0.2, mb_bot], fill=body_color)

    # front nose taper (sportier pointed front)
    d.polygon([(cx + bl * 0.46, mb_top), (cx + bl * 0.5, cy - bh * 0.15), (cx + bl * 0.46, mb_bot)], fill=body_color)

    # cabin / windshield wedge, offset toward the rear for a fastback rake
    roof_top = cy - bh * 0.95
    cab = [
        (cx - bl * 0.32, mb_top), (cx - bl * 0.14, roof_top),
        (cx + bl * 0.16, roof_top), (cx + bl * 0.30, mb_top),
    ]
    d.polygon(cab, fill=light)
    windshield = [
        (cx + bl * 0.02, mb_top - 1), (cx + bl * 0.10, roof_top + bh * 0.12),
        (cx + bl * 0.28, roof_top + bh * 0.12), (cx + bl * 0.30, mb_top - 1),
    ]
    d.polygon(windshield, fill=cabin_color)
    d.line([(cx + bl * 0.06, roof_top + bh * 0.3), (cx + bl * 0.22, mb_top - 2)],
           fill=(255, 255, 255, 200), width=max(1, int(s * 0.07)))
    rear_glass = [
        (cx - bl * 0.30, mb_top - 1), (cx - bl * 0.14, roof_top + bh * 0.12),
        (cx + bl * 0.00, roof_top + bh * 0.12), (cx - bl * 0.02, mb_top - 1),
    ]
    d.polygon(rear_glass, fill=_darken(cabin_color, 40))

    # side accent stripe
    d.line([(cx - bl * 0.42, cy + bh * 0.1), (cx + bl * 0.44, cy + bh * 0.1)],
           fill=(*_lighten(body_color, 60)[:3], 220), width=max(1, int(s * 0.05)))

    # headlight (front) / taillight (rear)
    hl_x = cx + bl * 0.48
    d.ellipse([hl_x - s * 0.16, cy - bh * 0.05, hl_x + s * 0.05, cy + bh * 0.18], fill=(255, 250, 215, 255))
    tl_x = cx - bl * 0.47
    d.ellipse([tl_x - s * 0.05, cy - bh * 0.05, tl_x + s * 0.16, cy + bh * 0.18], fill=(255, 45, 60, 255))

    if spoiler:
        sp_y = roof_top + bh * 0.05
        d.line([(cx - bl * 0.38, mb_top), (cx - bl * 0.38, sp_y)], fill=dark, width=max(1, int(s * 0.1)))
        d.line([(cx - bl * 0.48, sp_y), (cx - bl * 0.28, sp_y)], fill=dark, width=max(1, int(s * 0.16)))

    return im, dict(front=(hl_x, cy), rear=(tl_x, cy), ground=(cx, cy + bh * 0.75), size=(size_x, size_y))


def _paste_car(base, s, body_color, cx, cy, facing=1, angle=0.0, cabin_color=None, spoiler=True, shadow=True):
    """Draw a car sprite onto `base` (RGB image) centered at (cx, cy),
    optionally mirrored (facing=-1) and rotated by `angle` degrees (positive
    = counter-clockwise) for drift/jump poses. Returns the sprite's
    on-canvas headlight/taillight positions (post-transform, approximate)
    for callers that want to add glow there."""
    sprite, anchors = _car_sprite(s, body_color, cabin_color, spoiler)
    if facing < 0:
        sprite = ImageOps.mirror(sprite)
    if shadow:
        sw, sh = sprite.size
        shadow_im = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow_im).ellipse(
            [sw * 0.18, sh * 0.62, sw * 0.86, sh * 0.92], fill=(0, 0, 0, 70))
    if angle:
        sprite = sprite.rotate(angle, resample=Image.BICUBIC, expand=True)
    sw, sh = sprite.size
    pos = (int(cx - sw / 2), int(cy - sh / 2))
    base.paste(sprite, pos, sprite)
    return dict(front=(cx + facing * s * 1.6, cy), rear=(cx - facing * s * 1.6, cy))


def _night_gradient(draw, w, h, horizon_y, top=(10, 6, 24), mid=(60, 20, 55), bottom=(10, 8, 16)):
    for y in range(0, int(horizon_y)):
        t = y / max(1, horizon_y)
        col = tuple(int(top[k] * (1 - t) + mid[k] * t) for k in range(3))
        draw.line([(0, y), (w, y)], fill=col)
    draw.rectangle([0, horizon_y, w, h], fill=bottom)


def _city_skyline(draw, w, horizon_y, rng, state, key="skyline", n=14, color=(18, 12, 30)):
    bld = state.setdefault(key, [
        dict(x=i * w / n, bw=w / n * rng.uniform(0.55, 0.95), bh=rng.uniform(0.08, 0.32),
             lit=[(rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.9)) for _ in range(rng.integers(2, 6))])
        for i in range(n)
    ])
    for b in bld:
        top = horizon_y - horizon_y * b["bh"] * 2.1
        draw.rectangle([b["x"], top, b["x"] + b["bw"], horizon_y + 2], fill=color)
        for lx, ly in b["lit"]:
            wx, wy = b["x"] + b["bw"] * lx, top + (horizon_y - top) * ly
            draw.point((wx, wy), fill=(255, 210, 120))


def _neon_bokeh(draw_img, w, horizon_y, rng, state, pal, key="bokeh", n=8):
    dots = state.setdefault(key, [
        dict(x=rng.uniform(0, w), y=rng.uniform(horizon_y * 0.1, horizon_y * 0.75),
             r=rng.uniform(6, 18), col=pal["colors"][rng.integers(0, len(pal["colors"]))],
             ph=rng.uniform(0, 6.28))
        for _ in range(n)
    ])
    glow_layer = Image.new("RGB", draw_img.size, (0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    for dot in dots:
        gd.ellipse([dot["x"] - dot["r"], dot["y"] - dot["r"], dot["x"] + dot["r"], dot["y"] + dot["r"]],
                   fill=dot["col"])
    return add_glow(draw_img, glow_layer, radius=int(max(6, dots[0]["r"] * 1.2)) if dots else 8, strength=0.9)


def _road(draw, w, h, horizon_y, scroll, lane_count=3):
    """A converging-perspective asphalt road: smoothly shaded surface,
    curbed edges (so the road reads as a distinct object against the night
    background instead of a flat dark blob), a soft continuous specular
    sheen, and scrolling dashed lane markers."""
    vp = w / 2.0
    road_half_top = w * 0.025
    road_half_bot = w * 0.62

    def half_at(t):
        return road_half_top + (road_half_bot - road_half_top) * (t ** 1.5)

    def edge(t, side):
        return vp + side * half_at(t)

    steps = 20
    for i in range(steps):
        t0, t1 = i / steps, (i + 1) / steps
        y0 = horizon_y + (h - horizon_y) * t0
        y1 = horizon_y + (h - horizon_y) * t1
        shade = 22 + int(9 * (1 - t0))
        draw.polygon([(edge(t0, -1), y0), (edge(t0, 1), y0), (edge(t1, 1), y1), (edge(t1, -1), y1)],
                     fill=(shade, shade - 2, shade + 7))

    # soft specular sheen: one smooth, continuous band down the middle
    # (previously drawn as separate stacked blocks, which read as odd
    # floating patches on the road surface)
    for i in range(steps):
        t0, t1 = i / steps, (i + 1) / steps
        y0 = horizon_y + (h - horizon_y) * t0
        y1 = horizon_y + (h - horizon_y) * t1
        a = max(0, 14 - int(22 * abs(t0 - 0.3)))
        if a <= 0:
            continue
        band = 0.16
        draw.polygon([(edge(t0, -band), y0), (edge(t0, band), y0),
                      (edge(t1, band), y1), (edge(t1, -band), y1)],
                     fill=(30 + a, 28 + a, 38 + a))

    # curb edges along both sides -- gives the road a visible boundary
    # against the dark background instead of bleeding into it
    for i in range(steps):
        t0, t1 = i / steps, (i + 1) / steps
        y0 = horizon_y + (h - horizon_y) * t0
        y1 = horizon_y + (h - horizon_y) * t1
        lw = max(1, int((y1 - y0) * 0.5 + 1))
        for side in (-1, 1):
            draw.line([(edge(t0, side), y0), (edge(t1, side), y1)], fill=(195, 185, 175), width=lw)

    # scrolling dashed lane markers
    for lane in range(1, lane_count):
        f = (lane / lane_count) * 2 - 1
        for seg in range(12):
            t0 = (seg / 12 + scroll) % 1.0
            t1 = t0 + 0.035
            if t1 > 1.0:
                continue
            y0 = horizon_y + (h - horizon_y) * t0
            y1 = horizon_y + (h - horizon_y) * t1
            lw = max(1, int((y1 - y0) * 0.3))
            draw.line([(edge(t0, f), y0), (edge(t1, f), y1)], fill=(165, 160, 170), width=lw)

    return vp


def _soft_blobs(base_img, blobs, radius=6, strength=1.0):
    """Draw a list of (x, y, r, color) blobs pre-blurred for soft smoke/dust."""
    glow_layer = Image.new("RGB", base_img.size, (0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    for x, y, r, col in blobs:
        gd.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return add_glow(base_img, glow_layer, radius=radius, strength=strength)


def _car_body_sprite(s, body_color, cabin_color=None, view="rear"):
    """Build a car viewed nearly straight-on from behind (view='rear') or
    in front (view='front') -- roughly mirror-symmetric left/right -- so it
    reads correctly on a road that recedes straight into the distance
    toward/away from the camera. (The side-profile `_car_sprite` above is
    only correct when the car is genuinely being viewed from the side --
    e.g. a jump or a showroom turntable -- using it on a straight-ahead
    road makes the car look like it's driving perpendicular to the road,
    i.e. "sideways".)"""
    bw = s * 2.6   # width at the bumper (widest point)
    bh = s * 2.3   # height, bumper to roof
    pad = s * 0.8
    size_x = int(bw + pad * 2)
    size_y = int(bh + pad * 2.2)
    im = Image.new("RGBA", (size_x, size_y), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = size_x / 2.0
    bumper_y = size_y - pad * 1.1
    roof_y = bumper_y - bh

    dark = _darken(body_color, 55)
    light = _lighten(body_color, 35)
    cabin_color = cabin_color or (150, 210, 235)

    # ground shadow
    d.ellipse([cx - bw * 0.58, bumper_y - bh * 0.02, cx + bw * 0.58, bumper_y + bh * 0.16], fill=(0, 0, 0, 95))

    # tires peeking out at the bottom corners (only the inner sidewall
    # shows from directly behind/in front)
    wr = s * 0.5
    for wx in (cx - bw * 0.47, cx + bw * 0.47):
        d.ellipse([wx - wr * 0.5, bumper_y - wr * 1.15, wx + wr * 0.5, bumper_y + wr * 0.15], fill=(8, 8, 10, 255))
        d.ellipse([wx - wr * 0.22, bumper_y - wr * 0.85, wx + wr * 0.22, bumper_y - wr * 0.15], fill=(70, 70, 78, 255))

    # bumper band + diffuser/vent detail
    bump_top = bumper_y - bh * 0.18
    d.rounded_rectangle([cx - bw * 0.5, bump_top, cx + bw * 0.5, bumper_y], radius=s * 0.2, fill=dark)
    d.rectangle([cx - bw * 0.14, bumper_y - bh * 0.06, cx + bw * 0.14, bumper_y - bh * 0.01], fill=(15, 15, 18, 255))

    # main body panel, tapering inward toward the shoulder line
    shoulder_y = bumper_y - bh * 0.55
    d.polygon([(cx - bw * 0.5, bump_top), (cx + bw * 0.5, bump_top),
               (cx + bw * 0.38, shoulder_y), (cx - bw * 0.38, shoulder_y)], fill=body_color)
    d.polygon([(cx - bw * 0.06, bump_top), (cx + bw * 0.06, bump_top),
               (cx + bw * 0.045, shoulder_y), (cx - bw * 0.045, shoulder_y)], fill=light)

    # glass band (rear windshield / windshield), tapering to the roof
    glass_top = roof_y + bh * 0.08
    d.polygon([(cx - bw * 0.36, shoulder_y), (cx + bw * 0.36, shoulder_y),
               (cx + bw * 0.25, glass_top), (cx - bw * 0.25, glass_top)],
              fill=cabin_color if view == "front" else _darken(cabin_color, 35))
    d.line([(cx - bw * 0.36, shoulder_y), (cx - bw * 0.25, glass_top)], fill=dark, width=max(1, int(s * 0.09)))
    d.line([(cx + bw * 0.36, shoulder_y), (cx + bw * 0.25, glass_top)], fill=dark, width=max(1, int(s * 0.09)))

    # roof cap
    d.polygon([(cx - bw * 0.25, glass_top), (cx + bw * 0.25, glass_top),
               (cx + bw * 0.19, roof_y), (cx - bw * 0.19, roof_y)], fill=light, outline=dark)

    # side mirrors poking out at the shoulder line
    for mside in (-1, 1):
        mx = cx + mside * bw * 0.42
        d.ellipse([mx - s * 0.12, shoulder_y - s * 0.1, mx + s * 0.12, shoulder_y + s * 0.14], fill=dark)

    # lights at the bumper corners
    light_col = (255, 40, 55, 255) if view == "rear" else (255, 250, 215, 255)
    lw, lh = bw * 0.15, bh * 0.14
    for lxo in (-1, 1):
        lx = cx + lxo * bw * 0.4
        d.rounded_rectangle([lx - lw / 2, bump_top + bh * 0.02, lx + lw / 2, bump_top + bh * 0.02 + lh],
                             radius=s * 0.06, fill=light_col)

    if view == "rear":
        d.rectangle([cx - bw * 0.1, bumper_y - bh * 0.1, cx + bw * 0.1, bumper_y - bh * 0.02],
                     fill=(225, 225, 215, 255))
        d.rectangle([cx - bw * 0.28, shoulder_y - bh * 0.015, cx + bw * 0.28, shoulder_y + bh * 0.02],
                     fill=(200, 30, 45, 210))
    else:
        for gy in range(3):
            yy = bump_top + bh * 0.03 + gy * bh * 0.035
            d.line([(cx - bw * 0.26, yy), (cx + bw * 0.26, yy)], fill=(18, 18, 22, 255), width=max(1, int(s * 0.035)))
        d.ellipse([cx - s * 0.14, shoulder_y - bh * 0.02, cx + s * 0.14, shoulder_y + bh * 0.06], fill=(210, 210, 215, 220))

    return im, dict(size=(size_x, size_y), bumper_y=bumper_y, cx=cx, bw=bw, bh=bh,
                     left_light=(cx - bw * 0.4, bump_top + bh * 0.09),
                     right_light=(cx + bw * 0.4, bump_top + bh * 0.09))


def _paste_car_body(base, s, body_color, cx, cy, cabin_color=None, view="rear", wag=0.0):
    """Paste a rear/front-view car centered laterally at cx with its bumper
    (ground contact point) at cy. `wag` is a small horizontal shear (in
    pixels, top relative to bottom) used to suggest a drift/lean without
    rotating this symmetric silhouette into something that reads as
    sideways."""
    sprite, meta = _car_body_sprite(s, body_color, cabin_color, view)
    sw, sh = sprite.size
    if wag:
        coeffs = (1, wag / max(1, sh), -wag, 0, 1, 0)
        sprite = sprite.transform((sw, sh), Image.AFFINE, coeffs, resample=Image.BICUBIC)
    pos = (int(cx - meta["cx"]), int(cy - meta["bumper_y"]))
    base.paste(sprite, pos, sprite)
    lx, ly = meta["left_light"]
    rx, ry = meta["right_light"]
    return dict(left_light=(pos[0] + lx, pos[1] + ly), right_light=(pos[0] + rx, pos[1] + ry))


def render_cars_headlights(w, h, feat, local_t, rng, pal, ctrl, state):
    """A car approaching head-on down the road, growing from a distant
    pair of headlights near the horizon to a full car passing close by --
    the headlight beams are cast by an actual car, not free-floating light
    shapes with nothing behind them."""
    horizon = h * 0.6
    img = Image.new("RGB", (w, h), (5, 5, 10))
    draw = ImageDraw.Draw(img)
    _night_gradient(draw, w, h, horizon, top=(6, 6, 14), mid=(18, 14, 26), bottom=(10, 9, 14))
    _city_skyline(draw, w, horizon, rng, state, n=16)
    scroll = state.get("scroll", 0.0) + (0.05 + feat["rms"] * 0.12) * (1 / 30.0)
    state["scroll"] = scroll % 1.0
    _road(draw, w, h, horizon, state["scroll"])

    t = state.get("approach_t", 0.0)
    t += (0.1 + feat["rms"] * 0.22) * (1 / 30.0)
    if t >= 1.0:
        t = 0.0
        state["color"] = pal["colors"][rng.integers(0, len(pal["colors"]))]
    state["approach_t"] = t
    color = state.setdefault("color", pal["colors"][0])
    depth = np.clip(t, 0.02, 1.0)
    cy = horizon + (h - horizon) * (depth ** 1.5)
    s = h * 0.025 + h * 0.135 * (depth ** 1.5)
    cx = w / 2

    pulse = 1 + feat["bass"] * 0.5 + (0.35 if feat["is_beat"] else 0)
    anchors = _paste_car_body(img, s, color, cx, cy, cabin_color=(150, 210, 235), view="front")
    glow_layer = Image.new("RGB", (w, h), (0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    for lx, ly in (anchors["left_light"], anchors["right_light"]):
        side = -1 if lx < cx else 1
        beam_len = h * 0.55 * pulse * (0.35 + 0.75 * depth)
        gd.polygon([(lx - s * 0.1, ly), (lx + s * 0.1, ly),
                    (lx + side * beam_len * 0.22, ly - beam_len)], fill=(255, 250, 210))
        gd.ellipse([lx - s * 0.09, ly - s * 0.09, lx + s * 0.09, ly + s * 0.09], fill=(255, 255, 255))
    img = add_glow(img, glow_layer, radius=int(w * 0.025), strength=1.1 * ctrl.get("glow_strength", 1.0))
    draw = ImageDraw.Draw(img)

    fog = [(rng.uniform(0, w), rng.uniform(horizon, cy), rng.uniform(2, 5) + feat["treble"] * 3,
            (90, 90, 110)) for _ in range(int(5 + feat["treble"] * 8))]
    img = _soft_blobs(img, fog, radius=8, strength=0.6)
    return img


def render_cars_burnout(w, h, feat, local_t, rng, pal, ctrl, state):
    horizon = h * 0.55
    img = Image.new("RGB", (w, h), (10, 8, 16))
    draw = ImageDraw.Draw(img)
    _night_gradient(draw, w, h, horizon, top=(8, 6, 18), mid=(45, 16, 45), bottom=(16, 12, 18))
    _city_skyline(draw, w, horizon, rng, state, n=12)
    vp = _road(draw, w, h, horizon, 0.5)

    cx, cy = w * 0.5, h * 0.86
    s = h * 0.11
    drift_angle = 10 * np.sin(local_t * 1.3) + feat["bass"] * 4
    wag = drift_angle * s * 0.1
    rear_x, rear_y = cx, cy + s * 0.15  # rear-wheel contact point, under the car

    # tire skid marks curving behind the car (a rear-view car doing a
    # burnout smokes from both rear tires, so lay down two parallel marks)
    skid = state.setdefault("skid", [])
    skid.append((rear_x - s * 0.45, rear_y))
    state["skid"] = skid[-40:]
    if len(state["skid"]) > 1:
        draw.line(state["skid"], fill=(48, 45, 52), width=max(2, int(w * 0.008)), joint="curve")
        draw.line([(x, y) for x, y in [(px + s * 0.9, py) for px, py in state["skid"]]], fill=(48, 45, 52),
                   width=max(2, int(w * 0.008)), joint="curve")

    smoke = state.setdefault("smoke", [])
    for _ in range(int(2 + feat["rms"] * 4)):
        wheel_x = rear_x + rng.choice([-1, 1]) * s * 0.45
        smoke.append(dict(x=wheel_x + rng.uniform(-w * 0.015, w * 0.015), y=rear_y,
                           vx=rng.uniform(-w * 0.012, w * 0.012), vy=rng.uniform(-h * 0.025, -h * 0.006),
                           age=0.0, maxlife=rng.uniform(1.1, 2.1), size=rng.uniform(w * 0.016, w * 0.028)))
    dt = 1 / 30.0
    alive, blobs = [], []
    for sm in smoke:
        sm["age"] += dt
        sm["x"] += sm["vx"] * dt
        sm["y"] += sm["vy"] * dt
        sm["size"] += w * 0.012 * dt
        if sm["age"] < sm["maxlife"]:
            a = 1 - sm["age"] / sm["maxlife"]
            g = int(140 * a + 30)
            blobs.append((sm["x"], sm["y"], sm["size"], (g, g, g)))
            alive.append(sm)
    state["smoke"] = alive[-320:]
    img = _soft_blobs(img, blobs, radius=int(w * 0.016), strength=0.8)
    draw = ImageDraw.Draw(img)

    anchors = _paste_car_body(img, s, pal["colors"][2], cx, cy, cabin_color=(120, 180, 210), view="rear", wag=wag)
    if feat["is_beat"]:
        sparks = []
        for lx, ly in (anchors["left_light"], anchors["right_light"]):
            for _ in range(2):
                sparks.append((lx + rng.uniform(-8, 8), ly + s * 0.5 + rng.uniform(-4, 4),
                               rng.uniform(2, 4), (255, 220, 90)))
        img = _soft_blobs(img, sparks, radius=4, strength=1.0)
    return img


def render_cars_taillight_trails(w, h, feat, local_t, rng, pal, ctrl, state):
    horizon = h * 0.5
    img = Image.new("RGB", (w, h), (6, 5, 12))
    draw = ImageDraw.Draw(img)
    _night_gradient(draw, w, h, horizon, top=(5, 4, 12), mid=(20, 8, 20), bottom=(8, 6, 12))
    _city_skyline(draw, w, horizon, rng, state, n=14)
    scroll = state.get("scroll", 0.0) + (0.08 + feat["rms"] * 0.2) * (1 / 30.0)
    state["scroll"] = scroll % 1.0
    _road(draw, w, h, horizon, state["scroll"])

    cy = h * 0.6
    glow_layer = Image.new("RGB", (w, h), (0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    for i in range(3):
        ty = cy + (i - 1) * h * 0.035
        off = (state["scroll"] * (1 + i * 0.2)) % 1.0
        for k in np.linspace(0, 1, 40):
            x = ((k - off) % 1.0) * w
            fade = 1 - (x / w)
            if fade > 0.05:
                gd.ellipse([x - 2, ty - 2, x + 2, ty + 2], fill=(int(255 * fade), int(15 * fade), int(25 * fade)))
    img = add_glow(img, glow_layer, radius=6, strength=1.0 * ctrl.get("glow_strength", 1.0))
    draw = ImageDraw.Draw(img)
    # a rear-view car driving away up the road, so its actual taillights
    # are what's streaking off into the distance ahead of it
    anchors = _paste_car_body(img, h * 0.1, (18, 18, 22), w * 0.5, cy + h * 0.16, cabin_color=(30, 30, 36), view="rear")
    tl_glow = Image.new("RGB", (w, h), (0, 0, 0))
    gd2 = ImageDraw.Draw(tl_glow)
    for lx, ly in (anchors["left_light"], anchors["right_light"]):
        gd2.ellipse([lx - 7, ly - 7, lx + 7, ly + 7], fill=(255, 40, 55))
    img = add_glow(img, tl_glow, radius=10, strength=0.9 * ctrl.get("glow_strength", 1.0))
    return img


def render_cars_drag_race(w, h, feat, local_t, rng, pal, ctrl, state):
    horizon = h * 0.45
    img = Image.new("RGB", (w, h), (8, 6, 16))
    draw = ImageDraw.Draw(img)
    _night_gradient(draw, w, h, horizon, top=(6, 5, 16), mid=(30, 12, 32), bottom=(10, 8, 14))
    _city_skyline(draw, w, horizon, rng, state, n=14)
    _road(draw, w, h, horizon, 0.0, lane_count=2)

    tx = w * 0.5
    colors_seq = [(255, 60, 60), (255, 60, 60), (255, 210, 60), (255, 210, 60), (80, 255, 100)]
    stage = int((local_t * 0.8) % 5)
    for i, col in enumerate(colors_seq):
        ly = h * 0.1 + i * h * 0.045
        lit = i <= stage
        c = col if lit else tuple(v // 5 for v in col)
        draw.ellipse([tx - w * 0.012, ly - w * 0.012, tx + w * 0.012, ly + w * 0.012], fill=c)

    cy = h * 0.88
    # a launch-line POV looking down the strip: both cars viewed from
    # directly behind, waiting on the lights, matches the straight-ahead
    # perspective road instead of a side profile that reads as sideways
    _paste_car_body(img, h * 0.095, pal["colors"][0], w * 0.34, cy, cabin_color=(150, 210, 235), view="rear")
    _paste_car_body(img, h * 0.095, pal["colors"][3], w * 0.66, cy, cabin_color=(150, 210, 235), view="rear")
    draw = ImageDraw.Draw(img)
    if stage == 4 and feat["is_beat"]:
        blobs = []
        for _ in range(6):
            blobs.append((w * 0.32 + rng.uniform(-w * 0.03, w * 0.03), cy + h * 0.03 + rng.uniform(-6, 6),
                          rng.uniform(3, 7), (255, 255, 255)))
            blobs.append((w * 0.68 + rng.uniform(-w * 0.03, w * 0.03), cy + h * 0.03 + rng.uniform(-6, 6),
                          rng.uniform(3, 7), (255, 255, 255)))
        img = _soft_blobs(img, blobs, radius=6, strength=1.0)
    return img


def render_cars_showroom_spin(w, h, feat, local_t, rng, pal, ctrl, state):
    img = Image.new("RGB", (w, h), (14, 10, 22))
    draw = ImageDraw.Draw(img)
    cx, cy = w / 2, h * 0.62
    for r, a in ((w * 0.34, 60), (w * 0.24, 90), (w * 0.15, 140)):
        draw.ellipse([cx - r, cy + h * 0.04, cx + r, cy + h * 0.09], fill=(a, a // 3, a // 2))
    rot = local_t * (0.6 + feat["mid"] * 0.6)
    s_side = np.sin(rot)   # which way the car is facing along the turntable
    c_side = np.cos(rot)   # how edge-on (side profile) vs face-on (front/rear) it currently is
    # keep the on-screen size roughly constant across the whole rotation --
    # only a gentle taper right at the front/rear<->side handoff so there's
    # no visible size "pop" between the two sprite types
    squash = 0.82 + 0.18 * min(1.0, abs(c_side) / 0.7)
    facing = 1 if s_side > 0 else -1
    glow_layer = Image.new("RGB", (w, h), (0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    for a in (-1, 1):
        gd.line([(cx + a * w * 0.32, 0), (cx, cy)], fill=(200, 200, 220), width=3)
    img = add_glow(img, glow_layer, radius=int(w * 0.02), strength=0.7)
    draw = ImageDraw.Draw(img)
    color = pal["colors"][int(local_t) % len(pal["colors"])]
    # a real rotating turntable shows the car's actual side profile for
    # part of the spin and its front/rear for the rest -- always using the
    # side sprite made it look frozen sideways no matter which way the car
    # was "facing" on the platform
    if abs(s_side) > 0.45:
        _paste_car(img, h * 0.155 * squash, color, cx, cy + h * 0.015, facing=facing, cabin_color=(150, 210, 235))
    else:
        view = "front" if c_side > 0 else "rear"
        _paste_car_body(img, h * 0.17, color, cx, cy + h * 0.05, cabin_color=(150, 210, 235), view=view)
    return img


# ==========================================================================
# PACK 3 — SPACE & SUNSETS — shared starfield helper + 8 scenes
# ==========================================================================
def _twinkle_stars(draw, star_state, local_t, boost=1.0):
    for sx, sy, ph, base in star_state:
        tw = base * (0.6 + 0.4 * np.sin(local_t * 2.5 + ph)) * boost
        b = int(140 * tw)
        if b > 15:
            draw.point((sx, sy), fill=(min(255, b), min(255, b), min(255, b + 30)))


def _make_star_field(rng, iw, ih, n=40):
    return [(rng.uniform(0, iw), rng.uniform(0, ih), rng.uniform(0, 6.28), rng.uniform(0.5, 1.0)) for _ in range(n)]


def _new_sky_flight(rng, iw, ih, y_lo=0.05, y_hi=0.9):
    """A single straight diagonal path crossing the frame -- reads as a
    natural flight, unlike a sine-wobble path. Enters from one side,
    exits through the other."""
    y0 = rng.uniform(ih * y_lo, ih * (y_lo + (y_hi - y_lo) * 0.4))
    y1 = rng.uniform(ih * (y_lo + (y_hi - y_lo) * 0.6), ih * y_hi)
    x0, x1 = -iw * 0.18, iw * 1.18
    if rng.random() < 0.5:
        x0, x1 = x1, x0
    return dict(t=0.0, x0=x0, y0=y0, x1=x1, y1=y1)


def render_comet_flyby(w, h, feat, local_t, rng, pal, ctrl, state):
    img, draw, iw, ih, scale = pixel_canvas(w, h, (8, 6, 20))
    stars = state.setdefault("stars", _make_star_field(rng, iw, ih))
    _twinkle_stars(draw, stars, local_t, 1 + feat["treble"])

    if "comet" not in state:
        state["comet"] = _new_sky_flight(rng, iw, ih, 0.05, 0.55)
    comet = state["comet"]
    comet["t"] += 0.018 + feat["rms"] * 0.045
    if comet["t"] > 1.0:
        state["comet"] = _new_sky_flight(rng, iw, ih, 0.05, 0.55)
        comet = state["comet"]
    t = comet["t"]
    cx = comet["x0"] + (comet["x1"] - comet["x0"]) * t
    cy = comet["y0"] + (comet["y1"] - comet["y0"]) * t
    # unit vector pointing back along the flight path, so the tail always
    # trails directly behind the direction of travel
    dx, dy = comet["x0"] - comet["x1"], comet["y0"] - comet["y1"]
    dlen = max(1e-4, float(np.hypot(dx, dy)))
    ux, uy = dx / dlen, dy / dlen
    col = pal["accent"]
    for i in range(16):
        a = 1 - i / 16
        c = tuple(int(col[k] * a) for k in range(3))
        tx, ty = cx + ux * i * 3, cy + uy * i * 3
        r = max(0.5, 2 - i * 0.1)
        draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=c)
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 255, 255))
    return upscale_pixelated(img, w, h)


def render_purple_asteroid(w, h, feat, local_t, rng, pal, ctrl, state):
    img, draw, iw, ih, scale = pixel_canvas(w, h, (6, 4, 14))
    stars = state.setdefault("stars", _make_star_field(rng, iw, ih, 30))
    _twinkle_stars(draw, stars, local_t)

    if "flight" not in state:
        state["flight"] = _new_sky_flight(rng, iw, ih, 0.15, 0.85)
    flight = state["flight"]
    flight["t"] += 0.009 + feat["rms"] * 0.018
    if flight["t"] > 1.0:
        state["flight"] = _new_sky_flight(rng, iw, ih, 0.15, 0.85)
        flight = state["flight"]
        state["trail"] = []
    t = flight["t"]
    cx = flight["x0"] + (flight["x1"] - flight["x0"]) * t
    cy = flight["y0"] + (flight["y1"] - flight["y0"]) * t

    # dust trail left behind as it tumbles across the sky
    trail = state.setdefault("trail", [])
    trail.append((cx, cy))
    state["trail"] = trail[-16:]
    n_trail = len(state["trail"])
    for i, (tx, ty) in enumerate(state["trail"][:-1]):
        a = (i + 1) / n_trail
        r = 0.6 + a * 1.6
        shade = (int(70 * a) + 15, int(45 * a) + 10, int(95 * a) + 20)
        draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=shade)

    R = ih * 0.13 * (1 + feat["bass"] * 0.08)
    rot = local_t * 1.4
    # bumps are generated once and kept sorted by angle so the polygon
    # outline never self-intersects (that self-intersection was what made
    # the rock look like it was "splitting apart" as it spun)
    bumps = state.setdefault("bumps", sorted(
        [dict(a=rng.uniform(0, 2 * np.pi), r=rng.uniform(0.75, 1.0)) for _ in range(10)],
        key=lambda b: b["a"]))
    pts = [(cx + R * b["r"] * np.cos(b["a"] + rot), cy + R * b["r"] * np.sin(b["a"] + rot) * 0.9) for b in bumps]
    draw.polygon(pts, fill=(120, 80, 160))
    for b in bumps[::2]:
        a = b["a"] + rot
        px, py = cx + R * 0.5 * np.cos(a), cy + R * 0.5 * np.sin(a) * 0.9
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(80, 50, 110))

    tail = state.setdefault("tail", [])
    tail.append((cx - R, cy))
    if len(tail) > 14:
        tail.pop(0)
    for i, (tx, ty) in enumerate(tail):
        a = i / max(1, len(tail))
        draw.point((tx, ty), fill=(int(150 * a), int(100 * a), int(180 * a)))
    return upscale_pixelated(img, w, h)


def render_constellations(w, h, feat, local_t, rng, pal, ctrl, state):
    img, draw, iw, ih, scale = pixel_canvas(w, h, (6, 6, 16))
    stars = state.setdefault("stars", _make_star_field(rng, iw, ih, 60))
    _twinkle_stars(draw, stars, local_t)

    consts = state.setdefault("consts", [
        [(rng.uniform(0.1, 0.9) * iw, rng.uniform(0.1, 0.9) * ih) for _ in range(int(rng.integers(4, 7)))]
        for _ in range(3)
    ])
    for gi, group in enumerate(consts):
        col = pal["colors"][gi % len(pal["colors"])]
        for i in range(len(group) - 1):
            draw.line([group[i], group[i + 1]], fill=tuple(c // 2 for c in col))
        for (px, py) in group:
            s = 1 + (1 if feat["is_beat"] else 0)
            draw.ellipse([px - s, py - s, px + s, py + s], fill=col)
    return upscale_pixelated(img, w, h)


def render_aurora_borealis(w, h, feat, local_t, rng, pal, ctrl, state):
    img, draw, iw, ih, scale = pixel_canvas(w, h, (4, 4, 12))
    stars = state.setdefault("stars", _make_star_field(rng, iw, int(ih * 0.5), 20))
    _twinkle_stars(draw, stars, local_t)

    for b in range(3):
        col = pal["colors"][b % len(pal["colors"])]
        base_y = ih * (0.25 + b * 0.12)
        pts_top = []
        for x in range(0, iw + 4, 4):
            wob = np.sin(x * 0.05 + local_t * 1.5 + b * 2) * ih * 0.06 * (1 + feat["mid"])
            pts_top.append((x, base_y + wob))
        pts_bot = [(x, y + ih * 0.1) for x, y in reversed(pts_top)]
        draw.polygon(pts_top + pts_bot, fill=col)
    draw.rectangle([0, ih * 0.85, iw, ih], fill=(10, 8, 16))
    return upscale_pixelated(img, w, h)


def render_meteor_shower(w, h, feat, local_t, rng, pal, ctrl, state):
    density = ctrl.get("particle_density", 1.0)
    img, draw, iw, ih, scale = pixel_canvas(w, h, (8, 6, 18))
    stars = state.setdefault("stars", _make_star_field(rng, iw, ih))
    _twinkle_stars(draw, stars, local_t)

    meteors = state.setdefault("meteors", [])
    if rng.random() < (0.15 + feat["onset"] * 0.6) * density:
        meteors.append(dict(x=rng.uniform(0, iw), y=-4.0, vx=rng.uniform(-20, -8), vy=rng.uniform(50, 90),
                             age=0.0, maxlife=1.2, color=pal["colors"][rng.integers(0, len(pal["colors"]))]))
    dt = 1 / 30.0
    alive = []
    for m in meteors:
        m["age"] += dt
        m["x"] += m["vx"] * dt
        m["y"] += m["vy"] * dt
        if m["age"] < m["maxlife"] and m["y"] < ih + 5:
            tx, ty = m["x"] - m["vx"] * 0.08, m["y"] - m["vy"] * 0.08
            draw.line([(tx, ty), (m["x"], m["y"])], fill=m["color"])
            draw.point((m["x"], m["y"]), fill=(255, 255, 255))
            alive.append(m)
    state["meteors"] = alive[-60:]
    return upscale_pixelated(img, w, h)


def render_galaxy_swirl(w, h, feat, local_t, rng, pal, ctrl, state):
    img, draw, iw, ih, scale = pixel_canvas(w, h, (6, 4, 16))
    cx, cy = iw / 2, ih / 2
    lut = state.setdefault("lut", build_palette_lut(pal["colors"]))
    rot = local_t * (0.3 + feat["mid"] * 0.6)
    for arm in range(3):
        base_a = arm * 2 * np.pi / 3
        for k in range(30):
            t = k / 30
            r = t * min(iw, ih) * 0.42
            a = base_a + rot + t * 4.5
            x, y = cx + r * np.cos(a), cy + r * np.sin(a) * 0.55
            col = tuple(int(c) for c in sample_lut(lut, np.array([t + arm * 0.3]))[0])
            s = max(1, 3 * (1 - t))
            draw.ellipse([x - s, y - s, x + s, y + s], fill=col)
    core_r = 4 + feat["bass"] * 3
    draw.ellipse([cx - core_r, cy - core_r, cx + core_r, cy + core_r], fill=(255, 255, 240))
    return upscale_pixelated(img, w, h)


# ==========================================================================
# PACK 4 — RETRO Y2K — 5 new scenes (globe/bounce/rain already built above)
# ==========================================================================
def render_cd_burn_spin(w, h, feat, local_t, rng, pal, ctrl, state):
    img, draw, iw, ih, scale = pixel_canvas(w, h, (10, 8, 16))
    cx, cy = iw / 2, ih * 0.42
    R = min(iw, ih) * 0.28
    rot = local_t * (2 + feat["mid"] * 3)
    lut = state.setdefault("lut", build_palette_lut(pal["colors"]))
    for i in range(24):
        a = i / 24 * 2 * np.pi + rot
        col = tuple(int(c) for c in sample_lut(lut, np.array([i / 24]))[0])
        draw.line([(cx, cy), (cx + R * np.cos(a), cy + R * np.sin(a) * 0.9)], fill=col)
    draw.ellipse([cx - R * 0.18, cy - R * 0.16, cx + R * 0.18, cy + R * 0.16], fill=(20, 20, 24))
    draw.ellipse([cx - R, cy - R * 0.9, cx + R, cy + R * 0.9], outline=(230, 230, 240))

    prog = (state.get("prog", 0.0) + (0.15 + feat["rms"] * 0.4) * (1 / 30.0)) % 1.0
    state["prog"] = prog
    bw, bx, by = iw * 0.7, iw * 0.15, ih * 0.85
    draw.rectangle([bx, by, bx + bw, by + 6], outline=(200, 200, 210))
    draw.rectangle([bx, by, bx + bw * prog, by + 6], fill=pal["accent"])
    return upscale_pixelated(img, w, h)


def render_virtual_pet(w, h, feat, local_t, rng, pal, ctrl, state):
    # a chunky translucent-plastic Y2K keychain gadget (iMac-era colored
    # plastic + a Tamagotchi-style pixel creature on the screen), sitting
    # on a moody backdrop rather than filling the whole frame with color
    img, draw, iw, ih, scale = pixel_canvas(w, h, (14, 10, 24))
    dx, dy = iw / 2, ih * 0.52

    shell = pal["colors"][0]
    shell_dark = tuple(max(0, int(c * 0.55)) for c in shell)
    shell_light = tuple(min(255, int(c + 70)) for c in shell)

    # soft ambient glow behind the device
    glow_layer = Image.new("RGB", (iw, ih), (0, 0, 0))
    ImageDraw.Draw(glow_layer).ellipse(
        [dx - iw * 0.4, dy - ih * 0.4, dx + iw * 0.4, dy + ih * 0.4], fill=shell)
    lit = add_glow(img, glow_layer, radius=max(3, int(iw * 0.08)), strength=0.5)
    img.paste(lit, (0, 0))

    # keychain loop
    draw.ellipse([dx - iw * 0.03, dy - ih * 0.5, dx + iw * 0.03, dy - ih * 0.42], outline=(210, 210, 220))
    # translucent egg shell body (dark rim -> mid tone -> glossy highlight)
    draw.ellipse([dx - iw * 0.34, dy - ih * 0.44, dx + iw * 0.34, dy + ih * 0.44], fill=shell_dark)
    draw.ellipse([dx - iw * 0.30, dy - ih * 0.40, dx + iw * 0.30, dy + ih * 0.40], fill=shell)
    draw.ellipse([dx - iw * 0.20, dy - ih * 0.32, dx - iw * 0.03, dy - ih * 0.14], fill=shell_light)

    # LCD screen bezel + display
    screen_r = min(iw, ih) * 0.22
    draw.ellipse([dx - screen_r * 1.2, dy - ih * 0.28, dx + screen_r * 1.2, dy + screen_r * 0.75],
                 fill=(12, 14, 16))
    draw.ellipse([dx - screen_r, dy - ih * 0.24, dx + screen_r, dy + screen_r * 0.55], fill=(150, 195, 110))
    # faint scanline texture on the LCD
    for ly in range(int(dy - ih * 0.22), int(dy + screen_r * 0.5), 2):
        draw.line([(dx - screen_r * 0.95, ly), (dx + screen_r * 0.95, ly)], fill=(140, 185, 100))

    # the pixel creature living on the screen -- small round body, ear
    # nubs, blinking eyes, reacts to the beat with a little hop
    bounce = abs(np.sin(local_t * 3 + feat["bass"] * 2)) * screen_r * 0.22
    px, py = dx, dy - screen_r * 0.05 - bounce
    s = screen_r * 0.42
    creature = (40, 50, 40)
    draw.ellipse([px - s * 0.55, py - s * 1.3, px - s * 0.15, py - s * 0.85], fill=creature)
    draw.ellipse([px + s * 0.15, py - s * 1.3, px + s * 0.55, py - s * 0.85], fill=creature)
    draw.ellipse([px - s, py - s, px + s, py + s], fill=creature)
    blink = (int(local_t * 2) % 5 == 0)
    if blink:
        draw.line([(px - s * 0.45, py - s * 0.1), (px - s * 0.15, py - s * 0.1)], fill=(150, 195, 110), width=2)
        draw.line([(px + s * 0.15, py - s * 0.1), (px + s * 0.45, py - s * 0.1)], fill=(150, 195, 110), width=2)
    else:
        draw.ellipse([px - s * 0.4, py - s * 0.25, px - s * 0.18, py + s * 0.05], fill=(150, 195, 110))
        draw.ellipse([px + s * 0.18, py - s * 0.25, px + s * 0.4, py + s * 0.05], fill=(150, 195, 110))
    if feat["is_beat"]:
        draw.arc([px - s * 0.3, py + s * 0.15, px + s * 0.3, py + s * 0.55], start=0, end=180, fill=(150, 195, 110))

    # three chunky control buttons below the screen, like a keychain toy
    for i, bx in enumerate((-0.16, 0.0, 0.16)):
        cxb = dx + iw * bx
        cyb = dy + ih * 0.3
        lit_b = feat["is_beat"] and i == int(local_t * 2) % 3
        draw.ellipse([cxb - iw * 0.035, cyb - iw * 0.035, cxb + iw * 0.035, cyb + iw * 0.035],
                     fill=shell_light if lit_b else shell_dark)
    return upscale_pixelated(img, w, h)


def render_holo_sticker(w, h, feat, local_t, rng, pal, ctrl, state):
    img, draw, iw, ih, scale = pixel_canvas(w, h, (15, 10, 20))
    cx, cy = iw / 2, ih / 2
    R = min(iw, ih) * 0.3 * (1 + feat["bass"] * 0.1)
    hue = (local_t * 0.15) % 1.0
    for i in range(10):
        f = i / 10
        col = hsv_wave_color(hue + f * 0.5 + feat["mid"] * 0.3)
        r = R * (1 - f * 0.7)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col)
    pts = star_points(cx, cy, R * 0.5, R * 0.2, 6, rotation=local_t)
    draw.polygon(pts, fill=hsv_wave_color(hue + 0.5))
    return upscale_pixelated(img, w, h)


_Y2K_WORDS = ["Y2K", "RADICAL", "EXTREME", "KAMI", "CYBER", "MILLENNIUM"]


def render_chrome_bubble_text(w, h, feat, local_t, rng, pal, ctrl, state):
    # classic "DVD screensaver" style: the word drifts around and bounces
    # off the edges, changing color every time it hits a wall
    img, draw, iw, ih, scale = pixel_canvas(w, h, (10, 8, 18))
    word_state = state.setdefault("word", dict(idx=0, t=0.0))
    word_state["t"] += 1 / 30.0
    if word_state["t"] > 4.0:
        word_state["t"] = 0.0
        word_state["idx"] = (word_state["idx"] + 1) % len(_Y2K_WORDS)
    word = _Y2K_WORDS[word_state["idx"]]

    bbox = draw.textbbox((0, 0), word, font=_DEFAULT_FONT)
    tw_, th_ = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if "pos" not in state:
        state["pos"] = dict(
            x=rng.uniform(tw_, max(tw_ + 1, iw - tw_)), y=rng.uniform(th_, max(th_ + 1, ih - th_)),
            vx=(1 if rng.random() < 0.5 else -1) * rng.uniform(16, 24),
            vy=(1 if rng.random() < 0.5 else -1) * rng.uniform(12, 20),
            color_idx=0,
        )
    pos = state["pos"]
    speed_mult = 1 + feat["rms"] * 0.7 + (0.4 if feat["is_beat"] else 0)
    dt = 1 / 30.0
    pos["x"] += pos["vx"] * speed_mult * dt
    pos["y"] += pos["vy"] * speed_mult * dt

    bounced = False
    if pos["x"] - tw_ / 2 <= 0:
        pos["x"] = tw_ / 2
        pos["vx"] = abs(pos["vx"])
        bounced = True
    elif pos["x"] + tw_ / 2 >= iw:
        pos["x"] = iw - tw_ / 2
        pos["vx"] = -abs(pos["vx"])
        bounced = True
    if pos["y"] - th_ / 2 <= 0:
        pos["y"] = th_ / 2
        pos["vy"] = abs(pos["vy"])
        bounced = True
    elif pos["y"] + th_ / 2 >= ih:
        pos["y"] = ih - th_ / 2
        pos["vy"] = -abs(pos["vy"])
        bounced = True
    if bounced:
        pos["color_idx"] = (pos["color_idx"] + 1) % len(pal["colors"])

    cx, cy = pos["x"], pos["y"]
    draw.text((cx - tw_ / 2 + 1, cy - th_ / 2 + 1), word, fill=(0, 0, 0), font=_DEFAULT_FONT)
    draw.text((cx - tw_ / 2, cy - th_ / 2), word, fill=pal["colors"][pos["color_idx"] % len(pal["colors"])],
               font=_DEFAULT_FONT)
    return upscale_pixelated(img, w, h)


def render_crt_boot(w, h, feat, local_t, rng, pal, ctrl, state):
    img, draw, iw, ih, scale = pixel_canvas(w, h, (4, 4, 4))
    for y in range(0, ih, 2):
        if rng.random() < 0.3:
            draw.line([(0, y), (iw, y)], fill=(20, 20, 20))
    prog = min(1.0, state.get("prog", 0.0) + (0.08 + feat["rms"] * 0.3) * (1 / 30.0))
    if prog >= 1.0 and rng.random() < 0.02:
        prog = 0.0
    state["prog"] = prog
    lines = ["BOOTING KAMI OS...", "LOADING VISUALS", f"{int(prog * 100)}%"]
    y = ih * 0.35
    for line in lines[:2 if prog < 1 else 3]:
        draw.text((iw * 0.1, y), line, fill=(80, 255, 120), font=_DEFAULT_FONT)
        y += 10
    bw = iw * 0.8
    draw.rectangle([iw * 0.1, ih * 0.7, iw * 0.1 + bw, ih * 0.76], outline=(80, 255, 120))
    draw.rectangle([iw * 0.1, ih * 0.7, iw * 0.1 + bw * prog, ih * 0.76], fill=(80, 255, 120))
    return upscale_pixelated(img, w, h)


PATTERN_REGISTRY = {
    # pack 1: waveforms
    "chrome_tunnel": render_chrome_tunnel,
    "equalizer_bars": render_equalizer_bars,
    "particle_burst": render_particle_burst,
    "kaleidoscope": render_kaleidoscope,
    "checker_tunnel": render_checker_tunnel,
    "glitch_vhs": render_glitch_vhs,
    "starburst_pop": render_starburst_pop,
    "oscilloscope_wave": render_oscilloscope_wave,
    # pack 2: cars
    "pixel_cars": render_pixel_cars,
    "cars_headlights": render_cars_headlights,
    "cars_burnout": render_cars_burnout,
    "cars_taillight_trails": render_cars_taillight_trails,
    "cars_drag_race": render_cars_drag_race,
    "cars_showroom_spin": render_cars_showroom_spin,
    # pack 3: space & sunsets
    "sunset_pixel": render_sunset_pixel,
    "shooting_stars": render_shooting_stars,
    "comet_flyby": render_comet_flyby,
    "purple_asteroid": render_purple_asteroid,
    "constellations": render_constellations,
    "aurora_borealis": render_aurora_borealis,
    "meteor_shower": render_meteor_shower,
    "galaxy_swirl": render_galaxy_swirl,
    # pack 4: retro y2k
    "pixel_globe": render_pixel_globe,
    "pixel_bounce": render_pixel_bounce,
    "pixel_rain": render_pixel_rain,
    "cd_burn_spin": render_cd_burn_spin,
    "virtual_pet": render_virtual_pet,
    "holo_sticker": render_holo_sticker,
    "chrome_bubble_text": render_chrome_bubble_text,
    "crt_boot": render_crt_boot,
}

PATTERN_NAMES = list(PATTERN_REGISTRY.keys())

SCENE_PACKS = {
    "waveforms": ["chrome_tunnel", "equalizer_bars", "particle_burst", "kaleidoscope",
                  "checker_tunnel", "glitch_vhs", "starburst_pop", "oscilloscope_wave"],
    "cars": ["pixel_cars", "cars_headlights", "cars_burnout",
             "cars_taillight_trails", "cars_drag_race", "cars_showroom_spin"],
    "space_sunsets": ["sunset_pixel", "shooting_stars", "comet_flyby", "purple_asteroid",
                       "constellations", "aurora_borealis", "meteor_shower", "galaxy_swirl"],
    "retro_y2k": ["pixel_globe", "pixel_bounce", "pixel_rain", "cd_burn_spin",
                  "virtual_pet", "holo_sticker", "chrome_bubble_text", "crt_boot"],
}
