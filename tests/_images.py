"""Shared helpers that build small input images: sideways photos, cut-outs, masks and seeded noise."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

# Stored 40 x 20 with a red left half and a blue right half, tagged "rotate 90 degrees clockwise" to show upright:
# 20 wide and 40 high, red on top and blue below.
SIDEWAYS_STORED_SIZE = (40, 20)
SIDEWAYS_UPRIGHT_SIZE = (20, 40)


def write_sideways_photo(path: Path) -> str:
    """Write a JPEG whose EXIF orientation tag turns it upright, and return its path."""
    photo = Image.new("RGB", SIDEWAYS_STORED_SIZE, (255, 0, 0))
    photo.paste((0, 0, 255), (20, 0, 40, 20))
    exif = Image.Exif()
    exif[0x0112] = 6
    photo.save(path, quality=100, subsampling=0, exif=exif)
    return str(path)


def write_cutout(path: Path, size: tuple[int, int] = (8, 4)) -> str:
    """Write a PNG of *size* with an opaque red left half and a fully transparent right half (blue underneath), and return its path."""
    width, height = size
    cutout = Image.new("RGBA", size, (255, 0, 0, 255))
    cutout.paste((0, 0, 255, 0), (width // 2, 0, width, height))
    cutout.save(path)
    return str(path)


def noise_picture(width: int, height: int, seed: int = 0) -> Image.Image:
    """Return an RGB picture of seeded random pixels, so a pixel-exact comparison can tell any change from none."""
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8))


def painted_box(height: int, width: int, *boxes: tuple[int, int, int, int]) -> np.ndarray:
    """Return a ``height`` x ``width`` boolean paint with each box ``(x0, y0, x1, y1)`` painted."""
    painted = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        painted[y0:y1, x0:x1] = True
    return painted


def painted_block(height: int, width: int) -> np.ndarray:
    """Return a ``height`` x ``width`` paint with one block painted, from a quarter to a half of each side."""
    return painted_box(height, width, (width // 4, height // 4, width // 2, height // 2))


def mask_png(painted: np.ndarray, *, mode: str = "L") -> bytes:
    """Return a mask PNG of *mode* (``L``, ``RGB`` or ``RGBA``) that is white where *painted*; an RGBA mask carries it in alpha, as the browser's does."""
    height, width = painted.shape
    if mode == "RGBA":
        pixels = np.full((height, width, 4), 255, dtype=np.uint8)
        pixels[..., 3] = painted * 255  # colour is ignored when there is alpha
    elif mode == "RGB":
        pixels = np.repeat((painted * 255).astype(np.uint8)[..., None], 3, axis=2)
    elif mode == "L":
        pixels = (painted * 255).astype(np.uint8)
    else:
        raise ValueError(f"unsupported mask mode {mode!r}")
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode).save(buffer, format="PNG")
    return buffer.getvalue()


def write_mask_png(path: Path, painted: np.ndarray, *, mode: str = "L") -> str:
    """Write the mask PNG of *painted* (see :func:`mask_png`) and return its path."""
    path.write_bytes(mask_png(painted, mode=mode))
    return str(path)


def is_reddish(pixel: tuple[int, ...]) -> bool:
    """Return whether *pixel* is red, allowing for JPEG loss."""
    return pixel[0] > 200 and pixel[1] < 60 and pixel[2] < 60


def is_bluish(pixel: tuple[int, ...]) -> bool:
    """Return whether *pixel* is blue, allowing for JPEG loss."""
    return pixel[2] > 200 and pixel[0] < 60 and pixel[1] < 60


def display_p3_profile() -> bytes:
    """Return a minimal Display P3 style matrix/TRC ICC profile (D50-adapted primaries, gamma 2.2), a profile that is not sRGB."""
    import struct

    def xyz(x: float, y: float, z: float) -> bytes:
        return b"XYZ \0\0\0\0" + b"".join(struct.pack(">i", round(value * 65536)) for value in (x, y, z))

    gamma = b"curv\0\0\0\0" + struct.pack(">IH", 1, round(2.2 * 256)) + b"\0\0"
    text = b"text\0\0\0\0test\0"
    description = b"desc\0\0\0\0" + struct.pack(">I", 3) + b"P3\0" + b"\0" * 78
    tags = {
        b"desc": description,
        b"cprt": text,
        b"wtpt": xyz(0.9642, 1.0, 0.8249),
        b"rXYZ": xyz(0.5151, 0.2412, -0.0011),
        b"gXYZ": xyz(0.2920, 0.6922, 0.0419),
        b"bXYZ": xyz(0.1571, 0.0666, 0.7841),
        b"rTRC": gamma,
        b"gTRC": gamma,
        b"bTRC": gamma,
    }
    offset = 128 + 4 + 12 * len(tags)
    table, body = b"", b""
    for signature, data in tags.items():
        padded = data + b"\0" * (-len(data) % 4)
        table += signature + struct.pack(">II", offset + len(body), len(data))
        body += padded
    size = offset + len(body)
    header = struct.pack(">I4sI4s4s4s12s4s4sI4s", size, b"ziv\0", 0x02200000, b"mntr", b"RGB ", b"XYZ ", b"\0" * 12, b"acsp", b"\0\0\0\0", 0, b"\0\0\0\0")
    header += b"\0" * 8 + struct.pack(">I", 0) + struct.pack(">iii", round(0.9642 * 65536), 65536, round(0.8249 * 65536))
    header += b"\0" * (128 - len(header))
    return header + struct.pack(">I", len(tags)) + table + body


def cmyk_profile() -> bytes:
    """Return a minimal CMYK printer ICC profile (one 2-point lookup table to Lab) for testing colour-managed CMYK."""
    import itertools
    import struct

    def tag_text():
        return b"text\0\0\0\0test\0"

    desc = b"desc\0\0\0\0" + struct.pack(">I", 5) + b"CMYK\0" + b"\0" * 78
    ident = b"".join(struct.pack(">i", v) for v in (65536, 0, 0, 0, 65536, 0, 0, 0, 65536))
    ramp = bytes(range(256))
    clut = bytearray()
    for c, m, y, k in itertools.product((0, 1), repeat=4):  # first channel varies slowest
        lightness = (1 - k) * (1 - 0.35 * (c + m + y) / 3 - 0.25 * max(c, m, y))
        a = (m - c) * 40 * (1 - k)
        b = (y - 0.5 * (c + m)) * 40 * (1 - k)
        clut += bytes((round(255 * lightness), round(a + 128), round(b + 128)))
    lut = b"mft1\0\0\0\0" + bytes((4, 3, 2, 0)) + ident + ramp * 4 + bytes(clut) + ramp * 3
    wtpt = b"XYZ \0\0\0\0" + b"".join(struct.pack(">i", round(v * 65536)) for v in (0.9642, 1.0, 0.8249))
    tags = {b"desc": desc, b"cprt": tag_text(), b"wtpt": wtpt, b"A2B0": lut}
    offset = 128 + 4 + 12 * len(tags)
    table = body = b""
    for sig, data in tags.items():
        table += sig + struct.pack(">II", offset + len(body), len(data))
        body += data + b"\0" * (-len(data) % 4)
    header = struct.pack(">I4sI4s4s4s12s4s4sI4s", offset + len(body), b"ziv\0", 0x02200000, b"prtr", b"CMYK", b"Lab ", b"\0" * 12, b"acsp", b"\0\0\0\0", 0, b"\0\0\0\0")
    header += b"\0" * 8 + struct.pack(">I", 0) + struct.pack(">iii", round(0.9642 * 65536), 65536, round(0.8249 * 65536))
    header += b"\0" * (128 - len(header))
    return header + struct.pack(">I", len(tags)) + table + body
