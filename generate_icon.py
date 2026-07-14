"""
generate_icon.py — builds Kami's app icon: an original smiling-flower
mascot in a colorful Y2K/kawaii pop-art spirit (puffy, pillow-like petals,
thick black outlines, flat cel-shaded neon palette) — NOT a reproduction of
any existing artist's copyrighted work, just inspired by that general
aesthetic family.

Produces:
    kami_icon_1024.png    — master (transparent background)
    kami.ico              — multi-resolution Windows icon
    kami_icon_preview.png — composited onto a backdrop, for easy viewing
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
CX, CY = SIZE / 2, SIZE / 2

PETAL_COLORS = [
    (255, 47, 176),   # hot pink
    (0, 225, 255),    # electric cyan
    (255, 214, 0),    # sunshine yellow
    (182, 255, 0),    # lime
    (185, 103, 255),  # violet
]

OUTLINE = (20, 10, 28, 255)


def lighten(color, amt=0.5):
    return tuple(int(c + (255 - c) * amt) for c in color[:3])


def darken(color, amt=0.25):
    return tuple(int(c * (1 - amt)) for c in color[:3])


def petal_points(n_samples, width, length, base_offset, tip_bias=0.75):
    """Petal pointing straight up (negative-y) from (CX, CY): pointed tip,
    rounded belly, narrow base (covered later by the center face circle)."""
    left = []
    right = []
    for i in range(n_samples + 1):
        t = i / n_samples
        w = width * (math.sin(math.pi * t) ** tip_bias)
        y = -(base_offset + t * length)
        left.append((CX - w / 2, CY + y))
        right.append((CX + w / 2, CY + y))
    return left + right[::-1]


def draw_petal_layer(angle_deg, color, width, length, base_offset):
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    pts = petal_points(20, width, length, base_offset)
    draw.polygon(pts, fill=(*color, 255))

    # darker edge crescent (opaque, same hue family) for a cel-shaded
    # sense of volume — offset toward the right edge only
    shade_pts = petal_points(18, width * 0.55, length * 0.85, base_offset + length * 0.02, tip_bias=0.9)
    shade_pts = [(x + width * 0.20, y) for x, y in shade_pts]
    draw.polygon(shade_pts, fill=(*darken(color, 0.30), 255))

    # crisp glossy highlight streak (opaque), upper-left of the petal
    hl_pts = petal_points(14, width * 0.30, length * 0.42, base_offset + length * 0.20, tip_bias=0.9)
    hl_pts = [(x - width * 0.16, y) for x, y in hl_pts]
    draw.polygon(hl_pts, fill=(*lighten(color, 0.55), 255))

    # thick cartoon outline on top
    draw.polygon(pts, outline=OUTLINE, width=14)

    return layer.rotate(angle_deg, resample=Image.BICUBIC, center=(CX, CY))


def draw_center_face(canvas: Image.Image, face_r: float):
    draw = ImageDraw.Draw(canvas)

    cream = (255, 248, 225)
    pink_tint = (255, 224, 232)
    white_tint = (255, 253, 245)

    # puffy cream face base
    draw.ellipse([CX - face_r, CY - face_r, CX + face_r, CY + face_r], fill=(*cream, 255))

    # soft accent shading (small, opaque, pre-blended tones — not full alpha
    # overlays, which don't compose the way you'd expect over opaque fills)
    draw.pieslice([CX - face_r, CY - face_r, CX + face_r, CY + face_r],
                   start=20, end=160, fill=(*pink_tint, 255))
    draw.ellipse([CX - face_r * 0.5, CY - face_r * 0.82, CX + face_r * 0.05, CY - face_r * 0.28],
                 fill=(*white_tint, 255))

    # outline
    draw.ellipse([CX - face_r, CY - face_r, CX + face_r, CY + face_r], outline=OUTLINE, width=16)

    # blush
    br = face_r * 0.16
    for sign in (-1, 1):
        bx = CX + sign * face_r * 0.62
        by = CY + face_r * 0.16
        draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(255, 150, 185, 255))

    # eyes: big round happy eyes with sparkle
    eye_dx = face_r * 0.34
    eye_y = CY - face_r * 0.06
    eye_r = face_r * 0.155
    for sign in (-1, 1):
        ex = CX + sign * eye_dx
        draw.ellipse([ex - eye_r, eye_y - eye_r, ex + eye_r, eye_y + eye_r], fill=(25, 15, 30, 255))
        sp_r = eye_r * 0.34
        sx, sy = ex - eye_r * 0.35, eye_y - eye_r * 0.35
        draw.ellipse([sx - sp_r, sy - sp_r, sx + sp_r, sy + sp_r], fill=(255, 255, 255, 255))

    # smiling mouth
    mouth_w = face_r * 0.5
    mouth_y = CY + face_r * 0.30
    draw.arc([CX - mouth_w, mouth_y - mouth_w * 0.6, CX + mouth_w, mouth_y + mouth_w * 0.6],
              start=15, end=165, fill=OUTLINE, width=14)


def draw_sparkle(draw, cx, cy, r, color):
    pts = []
    for i in range(8):
        ang = i * math.pi / 4
        rad = r if i % 2 == 0 else r * 0.35
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    draw.polygon(pts, fill=color)


def build_flower() -> Image.Image:
    rng = np.random.default_rng(42)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    n_petals = 10
    face_r = SIZE * 0.205
    base_offset = face_r * 0.82

    for i in range(n_petals):
        angle = i * (360.0 / n_petals) + rng.uniform(-4, 4)
        color = PETAL_COLORS[i % len(PETAL_COLORS)]
        width = SIZE * rng.uniform(0.205, 0.235)
        length = SIZE * rng.uniform(0.30, 0.335)
        petal_layer = draw_petal_layer(angle, color, width, length, base_offset)
        canvas = Image.alpha_composite(canvas, petal_layer)

    draw_center_face(canvas, face_r)

    draw = ImageDraw.Draw(canvas)
    sparkle_spots = [
        (SIZE * 0.10, SIZE * 0.16, 34, (255, 255, 255, 255)),
        (SIZE * 0.88, SIZE * 0.13, 26, (0, 225, 255, 255)),
        (SIZE * 0.91, SIZE * 0.81, 30, (255, 47, 176, 255)),
        (SIZE * 0.09, SIZE * 0.83, 22, (255, 214, 0, 255)),
    ]
    for sx, sy, sr, scol in sparkle_spots:
        draw_sparkle(draw, sx, sy, sr, scol)

    return canvas


def add_drop_shadow(flower: Image.Image, offset=(14, 22), blur=18) -> Image.Image:
    alpha = flower.split()[3]
    shadow = Image.new("RGBA", flower.size, (0, 0, 0, 0))
    shadow.paste((10, 0, 20, 160), mask=alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))

    out = Image.new("RGBA", flower.size, (0, 0, 0, 0))
    out.paste(shadow, offset, shadow)
    out = Image.alpha_composite(out, flower)
    return out


def main():
    flower = build_flower()
    icon = add_drop_shadow(flower)

    # gentle anti-alias on the outer silhouette
    alpha = icon.split()[3]
    alpha = alpha.filter(ImageFilter.GaussianBlur(0.6))
    icon.putalpha(alpha)

    icon.save("kami_icon_1024.png")

    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon.save("kami.ico", sizes=sizes)

    grad = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for y in range(SIZE):
        f = y / SIZE
        grad[y, :] = [int(18 + 10 * f), int(6 + 4 * f), int(30 + 20 * f)]
    backdrop = Image.fromarray(grad, "RGB")
    backdrop.paste(icon, (0, 0), icon)
    backdrop.save("kami_icon_preview.png")

    print("Wrote kami_icon_1024.png, kami.ico, kami_icon_preview.png")


if __name__ == "__main__":
    main()
