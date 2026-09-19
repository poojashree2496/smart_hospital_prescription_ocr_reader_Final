from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def _binary_ink(image: Image.Image) -> np.ndarray:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    # Adaptive threshold with a light median background estimate. This is
    # deliberately tolerant of phone-camera shadows and yellow paper.
    try:
        import cv2
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 12
        )
        # Remove isolated speckles.
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary > 0
    except ImportError:
        return gray < 190


def _runs(mask: np.ndarray, min_run: int = 2, gap: int = 0):
    """Return inclusive-exclusive runs of True values, merging small gaps."""
    if gap:
        kernel = np.ones(gap + 1, dtype=np.uint8)
        mask = np.convolve(mask.astype(np.uint8), kernel, mode="same") > 0
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(~mask & np.r_[mask[:-1], False])
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    out = []
    for s, e in zip(starts, ends):
        if e - s >= min_run:
            out.append((int(s), int(e)))
    return out


def find_handwriting_regions(
    image: Image.Image,
    min_height: int = 10,
    line_gap: int = 6,
    word_gap: int = 18,
    padding: int = 8,
) -> list[tuple[int, int, int, int]]:
    """Find likely handwritten word/line regions on a full prescription.

    This is a heuristic region proposal stage, not a trained detector. It is
    designed to avoid requiring bounding-box annotations while grouping nearby
    handwriting into readable crops.
    """
    if image.width < 2 or image.height < 2:
        raise ValueError("Prescription image is empty or too small to segment")
    ink = _binary_ink(image)
    h, w = ink.shape

    # Horizontal projection -> text lines.
    row_counts = ink.sum(axis=1)
    row_mask = row_counts >= max(2, int(w * 0.002))
    line_runs = _runs(row_mask, min_run=min_height, gap=line_gap)

    regions = []
    for top, bottom in line_runs:
        band = ink[max(0, top-padding):min(h, bottom+padding), :]
        col_counts = band.sum(axis=0)
        # A larger gap separates words while preserving connected cursive letters.
        col_mask = col_counts >= 1
        word_runs = _runs(col_mask, min_run=3, gap=word_gap)

        # If the line is too fragmented, retain the entire line.
        if not word_runs or len(word_runs) > 16:
            word_runs = [(0, w)]

        for left, right in word_runs:
            l = max(0, left - padding)
            r = min(w, right + padding)
            t = max(0, top - padding)
            b = min(h, bottom + padding)
            if r - l >= max(18, int(w * 0.02)) and b - t >= min_height:
                regions.append((l, t, r, b))

    if not regions and ink.any():
        ys, xs = np.where(ink)
        regions.append((
            max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(w, int(xs.max()) + padding + 1),
            min(h, int(ys.max()) + padding + 1),
        ))

    # Deduplicate overlapping boxes, preferring larger boxes when nearly identical.
    regions = sorted(regions, key=lambda box: (box[1], box[0], -(box[2]-box[0])))
    kept = []
    for box in regions:
        x1, y1, x2, y2 = box
        duplicate = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            area = max(1, (x2-x1)*(y2-y1))
            if inter / area > 0.75:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def save_region_crops(image: Image.Image, regions: list[tuple[int, int, int, int]], output_dir: str | Path) -> list[str]:
    """Save candidate crops in reading order and return their paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, box in enumerate(regions, start=1):
        path = directory / f"crop_{index:03d}.png"
        image.crop(box).save(path)
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Find handwriting candidates on a prescription.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        regions = find_handwriting_regions(image)
        save_region_crops(image, regions, output_dir)
    print(f"Found {len(regions)} candidate regions.")
    for index, box in enumerate(regions, start=1):
        print(f"{index}: {box}")


if __name__ == "__main__":
    main()
