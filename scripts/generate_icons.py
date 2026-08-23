"""Generates the PWA's app icons as real PNG files, using only the standard library (zlib for
DEFLATE + CRC32 -- no Pillow/cairosvg dependency, since this repo's requirements.txt is kept to
what the quant pipeline itself needs). Run once from repo root:

    python scripts/generate_icons.py

Writes icons/icon-192.png, icons/icon-512.png, icons/icon-512-maskable.png, and
icons/apple-touch-icon.png (180x180, iOS's own required size/name for a home-screen icon --
iOS does not accept SVG here, unlike Android's manifest icons).

Design: a simple monogram badge, not a photo/logo asset -- a brand-green rounded square (the
same green family as index.html's own theme-color/#3a6b3f) with a bold white "Q" (FPL Quant)
built from two primitive shapes (a ring + a diagonal tail stroke), both drawn by direct
distance-field math rather than a font/vector library.
"""

import struct
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ICONS_DIR = REPO_ROOT / "icons"

BG = (43, 82, 48)      # dark pitch green, close to index.html's #3a6b3f but a touch deeper
FG = (255, 255, 255)   # white monogram


def _rounded_square_mask(x: float, y: float, size: float, radius: float) -> bool:
    """True if (x, y) falls inside a size x size square with corner radius `radius`, all in
    the same units as size/radius (not necessarily pixels)."""
    half = size / 2
    dx, dy = abs(x - half), abs(y - half)
    inner = half - radius
    if dx <= inner or dy <= inner:
        return True
    return (dx - inner) ** 2 + (dy - inner) ** 2 <= radius ** 2


def _q_glyph_mask(x: float, y: float, size: float, glyph_radius: float) -> bool:
    """The monogram: a ring (annulus) plus a short diagonal tail stroke, both centered on the
    icon -- a hand-built 'Q' silhouette, not a rendered font."""
    cx = cy = size / 2
    dx, dy = x - cx, y - cy
    dist = (dx * dx + dy * dy) ** 0.5
    ring_outer, ring_inner = glyph_radius, glyph_radius * 0.62
    if ring_inner <= dist <= ring_outer:
        return True
    # tail: a thick stroke from the ring's inner-lower-right edge out past the outer radius,
    # at a fixed ~35 degree angle below the horizontal (a classic "Q" tail direction)
    import math
    angle = math.radians(35)
    tail_dir = (math.cos(angle), math.sin(angle))
    tail_start = ring_outer * 0.55
    tail_end = ring_outer * 1.35
    # project (dx, dy) onto the tail direction; check perpendicular distance + range along it
    proj = dx * tail_dir[0] + dy * tail_dir[1]
    if tail_start <= proj <= tail_end:
        perp_x, perp_y = dx - proj * tail_dir[0], dy - proj * tail_dir[1]
        perp_dist = (perp_x * perp_x + perp_y * perp_y) ** 0.5
        if perp_dist <= glyph_radius * 0.16:
            return True
    return False


def render(size: int, *, maskable: bool = False) -> bytes:
    """Returns raw RGBA bytes, row-major, top-to-bottom."""
    corner_radius = 0 if maskable else size * 0.22
    glyph_radius = size * (0.235 if maskable else 0.30)
    pixels = bytearray(size * size * 4)
    for py in range(size):
        for px in range(size):
            idx = (py * size + px) * 4
            in_bg = _rounded_square_mask(px + 0.5, py + 0.5, size, corner_radius)
            if not in_bg:
                # fully transparent outside the rounded square (only relevant for non-maskable)
                pixels[idx:idx + 4] = (0, 0, 0, 0)
                continue
            if _q_glyph_mask(px + 0.5, py + 0.5, size, glyph_radius):
                pixels[idx:idx + 4] = (*FG, 255)
            else:
                pixels[idx:idx + 4] = (*BG, 255)
    return bytes(pixels)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, size: int, *, maskable: bool = False) -> None:
    rgba = render(size, maskable=maskable)
    raw = bytearray()
    stride = size * 4
    for row in range(size):
        raw.append(0)  # filter type: none
        raw.extend(rgba[row * stride:(row + 1) * stride])

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8-bit RGBA, no interlace
    idat = zlib.compress(bytes(raw), level=9)

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", idat)
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def main() -> None:
    ICONS_DIR.mkdir(exist_ok=True)
    write_png(ICONS_DIR / "icon-192.png", 192)
    write_png(ICONS_DIR / "icon-512.png", 512)
    write_png(ICONS_DIR / "icon-512-maskable.png", 512, maskable=True)
    write_png(ICONS_DIR / "apple-touch-icon.png", 180)
    print(f"[icons] wrote 4 PNGs to {ICONS_DIR}")


if __name__ == "__main__":
    main()
