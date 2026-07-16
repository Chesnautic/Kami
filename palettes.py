"""Y2K color palettes. Each is a small set of RGB tuples used across patterns."""
import colorsys
import random

PALETTES = {
    "chrome": {
        "bg": (8, 8, 18),
        "colors": [(200, 230, 255), (255, 255, 255), (120, 200, 255),
                   (255, 0, 170), (0, 255, 220)],
        "accent": (255, 0, 170),
        "glow": (0, 255, 255),
    },
    "millennium": {
        "bg": (10, 4, 20),
        "colors": [(180, 90, 255), (255, 210, 60), (0, 220, 200),
                   (255, 60, 180), (120, 60, 255)],
        "accent": (255, 210, 60),
        "glow": (180, 90, 255),
    },
    "candy": {
        "bg": (18, 4, 24),
        "colors": [(255, 105, 180), (170, 255, 60), (255, 165, 0),
                   (0, 220, 255), (255, 240, 80)],
        "accent": (255, 105, 180),
        "glow": (170, 255, 60),
    },
    "matrix": {
        "bg": (2, 8, 4),
        "colors": [(60, 255, 120), (10, 200, 80), (200, 255, 220),
                   (0, 100, 40), (140, 255, 180)],
        "accent": (60, 255, 120),
        "glow": (10, 255, 90),
    },
    "vapor": {
        "bg": (12, 6, 28),
        "colors": [(255, 113, 206), (1, 205, 254), (5, 255, 161),
                   (185, 103, 255), (255, 250, 180)],
        "accent": (1, 205, 254),
        "glow": (255, 113, 206),
    },
}

DEFAULT_PALETTE = "chrome"


def get_palette(name: str) -> dict:
    return PALETTES.get(name, PALETTES[DEFAULT_PALETTE])


def hex_to_rgb(h: str) -> tuple:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[int(c) for c in rgb])


def build_custom_palette(
    base: str = DEFAULT_PALETTE,
    bg: str | None = None,
    accent: str | None = None,
    glow: str | None = None,
    colors: list | None = None,
) -> dict:
    """Start from a preset (`base`) and override any of bg/accent/glow/colors
    with user-picked hex strings (e.g. from a GUI color picker). `colors` is
    a list of hex strings replacing the full gradient color list."""
    pal = dict(get_palette(base))
    if bg:
        pal["bg"] = hex_to_rgb(bg)
    if accent:
        pal["accent"] = hex_to_rgb(accent)
    if glow:
        pal["glow"] = hex_to_rgb(glow)
    if colors:
        pal["colors"] = [hex_to_rgb(c) for c in colors if c]
    return pal


def random_palette(seed: int | None = None) -> dict:
    """Generate a fresh, good-looking Y2K neon palette: a near-black tinted
    background plus 5 vivid, varied-hue foreground colors -- used to give
    the app a new look every launch, on demand via the Randomize button,
    and anywhere else a one-off scheme is wanted.

    Colors are spaced around the hue wheel using the golden angle
    (~0.618 of a full turn) rather than pure uniform-random hues -- this is
    a standard trick for picking N well-separated colors: it spreads them
    out so consecutive picks never land close together (avoiding a
    muddy/samey look) without imposing a rigid, obviously-computed pattern
    like evenly-spaced slices would. High saturation + high value keeps
    everything reading as bright neon rather than washed out, matching the
    hand-picked presets above.
    """
    rng = random.Random(seed)

    bg_hue = rng.random()
    bg_r, bg_g, bg_b = colorsys.hsv_to_rgb(bg_hue, rng.uniform(0.35, 0.65), rng.uniform(0.03, 0.09))
    bg = (int(bg_r * 255), int(bg_g * 255), int(bg_b * 255))

    n = 5
    golden_angle = 0.6180339887498949
    base_hue = rng.random()
    colors = []
    for i in range(n):
        hue = (base_hue + i * golden_angle + rng.uniform(-0.04, 0.04)) % 1.0
        sat = rng.uniform(0.55, 0.95)
        val = rng.uniform(0.85, 1.0)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    accent = colors[rng.randrange(n)]
    glow_hue = (base_hue + rng.uniform(0.4, 0.6)) % 1.0
    gr, gg, gb = colorsys.hsv_to_rgb(glow_hue, rng.uniform(0.5, 0.9), 1.0)
    glow = (int(gr * 255), int(gg * 255), int(gb * 255))

    return dict(bg=bg, colors=colors, accent=accent, glow=glow)


def palette_to_hex_fields(pal: dict) -> dict:
    """Inverse of build_custom_palette's overrides — used by the GUI to
    populate color-picker swatches from a preset."""
    return dict(
        bg=rgb_to_hex(pal["bg"]),
        accent=rgb_to_hex(pal["accent"]),
        glow=rgb_to_hex(pal["glow"]),
        colors=[rgb_to_hex(c) for c in pal["colors"]],
    )
