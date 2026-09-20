"""Non-destructive preprocessing for full prescription pages."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps


def prepare_page(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Return the original RGB page and a contrast-normalized working page."""
    original = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(original)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    working = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    return original, working