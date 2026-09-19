"""Pure PIL/numpy image helpers (no torch import, so they are easy to test)."""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

# Canvas every word image is fitted onto. 96x256 keeps thin pen strokes readable
# (the old 64x160 canvas squeezed them into a few pixels).
TARGET_HEIGHT = 96
TARGET_WIDTH = 256


def decode_image(image_bytes) -> Image.Image:
    """Decode PNG/JPEG bytes coming out of the Lance table into a PIL image."""
    if isinstance(image_bytes, memoryview):
        image_bytes = image_bytes.tobytes()
    elif not isinstance(image_bytes, (bytes, bytearray)):
        image_bytes = bytes(image_bytes)
    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        return image.copy()


def to_gray(image: Image.Image) -> Image.Image:
    """Convert to 8-bit grayscale. Transparent pixels become WHITE (not black)."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        image = background
    return image.convert("L")


def tight_crop(gray: Image.Image, margin: float = 0.10, ink_threshold: int = 200) -> Image.Image:
    """Crop to the handwriting's bounding box and stretch contrast.

    Handwriting crops come with very different amounts of empty paper around the
    word. Removing it makes every image show the word at a comparable scale, which
    is one of the biggest single accuracy wins for this kind of dataset.
    Falls back to the (contrast-stretched) full image if no ink is found.
    """
    stretched = ImageOps.autocontrast(gray.convert("L"), cutoff=1)
    ink = np.asarray(stretched) < ink_threshold
    rows = np.flatnonzero(ink.sum(axis=1) >= 2)  # >=2 ignores single speckles
    cols = np.flatnonzero(ink.sum(axis=0) >= 2)
    if rows.size == 0 or cols.size == 0:
        return stretched
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    pad = max(2, int(round(margin * (bottom - top))))
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(stretched.width, right + pad)
    bottom = min(stretched.height, bottom + pad)
    return stretched.crop((left, top, right, bottom))


def letterbox(
    image: Image.Image,
    size: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
    margin: float = 0.05,
) -> Image.Image:
    """Resize keeping aspect ratio and centre on a white canvas (mode 'L').

    A small margin is kept so that augmentation (shift/rotate/scale) does not
    push strokes off the edge of the canvas.
    """
    target_w, target_h = size
    image = image.convert("L")
    avail_w = target_w * (1 - 2 * margin)
    avail_h = target_h * (1 - 2 * margin)
    scale = min(avail_w / image.width, avail_h / image.height)
    new_w = max(1, round(image.width * scale))
    new_h = max(1, round(image.height * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (target_w, target_h), 255)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def prepare_word(image: Image.Image) -> Image.Image:
    """Decode-independent 'clean word' step used for train, eval and inference."""
    return tight_crop(to_gray(image))


def prepare_canvas(image: Image.Image, size: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)) -> Image.Image:
    """Full deterministic preprocessing: gray -> tight crop -> letterbox."""
    return letterbox(prepare_word(image), size)
