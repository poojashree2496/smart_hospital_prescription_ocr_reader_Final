"""Handwriting augmentation using only PIL + numpy.

Why custom instead of torchvision transforms:
  * no lambdas -> picklable, so DataLoader(num_workers>0) works on Windows
  * handwriting-specific effects (slant, stroke thickness, faint ink)
  * no horizontal flips (a mirrored word is a different word)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from src.image_ops import TARGET_HEIGHT, TARGET_WIDTH, letterbox


@dataclass
class AugmentConfig:
    stretch: tuple[float, float] = (0.85, 1.15)      # writers differ in word width
    rotate: float = 4.0                              # degrees
    shear: float = 10.0                              # degrees of slant
    scale: tuple[float, float] = (0.90, 1.08)
    translate: tuple[float, float] = (0.04, 0.08)    # fraction of canvas (x, y)
    perspective_p: float = 0.30
    perspective: float = 0.05                        # corner jitter, fraction of canvas
    thicken_p: float = 0.25                          # bolder pen strokes
    blur_p: float = 0.25
    faint_ink_p: float = 0.40                        # lighter pen / worse scan
    noise_p: float = 0.20
    erase_p: float = 0.15                            # small white patch


def _affine(canvas: Image.Image, cfg: AugmentConfig, rng) -> Image.Image:
    w, h = canvas.size
    theta = math.radians(rng.uniform(-cfg.rotate, cfg.rotate))
    shear = math.tan(math.radians(rng.uniform(-cfg.shear, cfg.shear)))
    scale = rng.uniform(*cfg.scale)
    tx = rng.uniform(-cfg.translate[0], cfg.translate[0]) * w
    ty = rng.uniform(-cfg.translate[1], cfg.translate[1]) * h
    cx, cy = w / 2.0, h / 2.0
    to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=float)
    scale_m = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=float)
    shear_m = np.array([[1, shear, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    rot_m = np.array(
        [[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]],
        dtype=float,
    )
    back = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=float)
    forward = back @ rot_m @ shear_m @ scale_m @ to_origin
    inverse = np.linalg.inv(forward)  # PIL wants the output->input mapping
    coeffs = (inverse[0, 0], inverse[0, 1], inverse[0, 2], inverse[1, 0], inverse[1, 1], inverse[1, 2])
    return canvas.transform(
        (w, h), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BILINEAR, fillcolor=255
    )


def _perspective_coeffs(out_pts, in_pts) -> list[float]:
    rows, rhs = [], []
    for (x, y), (u, v) in zip(out_pts, in_pts):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    return np.linalg.solve(np.array(rows, dtype=float), np.array(rhs, dtype=float)).tolist()


def _perspective(canvas: Image.Image, magnitude: float, rng) -> Image.Image:
    w, h = canvas.size
    dx, dy = magnitude * w, magnitude * h
    in_pts = [(0, 0), (w, 0), (w, h), (0, h)]
    out_pts = [
        (rng.uniform(0, dx), rng.uniform(0, dy)),
        (w - rng.uniform(0, dx), rng.uniform(0, dy)),
        (w - rng.uniform(0, dx), h - rng.uniform(0, dy)),
        (rng.uniform(0, dx), h - rng.uniform(0, dy)),
    ]
    coeffs = _perspective_coeffs(out_pts, in_pts)
    return canvas.transform(
        (w, h), Image.Transform.PERSPECTIVE, coeffs, resample=Image.Resampling.BILINEAR, fillcolor=255
    )


class TrainAugment:
    """word (mode 'L', tight-cropped) -> augmented 'L' canvas of `size`."""

    def __init__(self, config: AugmentConfig | None = None, size: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT), seed: int | None = None):
        self.config = config or AugmentConfig()
        self.size = size
        # Only store a Random instance when a seed is requested (tests).
        # Otherwise use the module-level RNG, which PyTorch re-seeds in each worker.
        self._rng = random.Random(seed) if seed is not None else None

    def __call__(self, word: Image.Image) -> Image.Image:
        rng = self._rng if self._rng is not None else random
        cfg = self.config

        # 1) random word-width change before fitting to the canvas
        factor = rng.uniform(*cfg.stretch)
        word = word.resize((max(1, round(word.width * factor)), word.height), Image.Resampling.BILINEAR)
        canvas = letterbox(word, self.size)

        # 2) stroke thickness (only thickening: thinning would delete 1px strokes)
        if rng.random() < cfg.thicken_p:
            canvas = canvas.filter(ImageFilter.MinFilter(3))

        # 3) geometry
        canvas = _affine(canvas, cfg, rng)
        if rng.random() < cfg.perspective_p:
            canvas = _perspective(canvas, cfg.perspective, rng)

        # 4) scan / pen quality
        if rng.random() < cfg.blur_p:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))
        array = np.array(canvas, dtype=np.float32)
        if rng.random() < cfg.faint_ink_p:
            strength = rng.uniform(0.70, 1.0)                # 1.0 = unchanged, lower = fainter ink
            array = 255.0 - (255.0 - array) * strength
        if rng.random() < cfg.noise_p:
            sigma = rng.uniform(2.0, 6.0)
            noise = np.random.default_rng(rng.getrandbits(32)).normal(0.0, sigma, array.shape)
            array = array + noise.astype(np.float32)
        if rng.random() < cfg.erase_p:
            h, w = array.shape
            ew = max(2, int(w * rng.uniform(0.03, 0.12)))
            eh = max(2, int(h * rng.uniform(0.10, 0.30)))
            x0 = rng.randint(0, max(0, w - ew))
            y0 = rng.randint(0, max(0, h - eh))
            array[y0:y0 + eh, x0:x0 + ew] = 255.0
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")
