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
        # Remove camera speckles and tiny paper artifacts before projection.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        cleaned = np.zeros_like(binary)
        for component in range(1, count):
            area = stats[component, cv2.CC_STAT_AREA]
            if area >= 12:
                cleaned[labels == component] = 255
        binary = cleaned
        return binary > 0
    except ImportError:
        return gray < 190


def _runs(mask: np.ndarray, min_run: int = 2, gap: int = 0):
    """Return inclusive-exclusive runs of True values, merging small gaps."""
    if gap:
        kernel = np.ones(gap + 1, dtype=np.uint8)
        mask = np.convolve(mask.astype(np.uint8), kernel, mode="same") > 0
    out = []
    start = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_run:
                out.append((start, index))
            start = None
    if start is not None and len(mask) - start >= min_run:
        out.append((start, len(mask)))
    return out


def _merge_boxes(boxes: list[tuple[int, int, int, int]], overlap: float = 0.75):
    kept = []
    for box in sorted(boxes, key=lambda item: (item[1], item[0])):
        x1, y1, x2, y2 = box
        if any(
            max(0, min(x2, kx2) - max(x1, kx1))
            * max(0, min(y2, ky2) - max(y1, ky1))
            / max(1, (x2 - x1) * (y2 - y1)) > overlap
            for kx1, ky1, kx2, ky2 in kept
        ):
            continue
        kept.append(box)
    return kept


def find_prescription_lines(
    image: Image.Image,
    min_height: int = 8,
    line_gap: int = 5,
    padding: int = 8,
) -> list[dict]:
    """Find full-page handwriting bands in top-to-bottom reading order."""
    if image.width < 2 or image.height < 2:
        raise ValueError("Prescription image is empty or too small to segment")
    ink = _binary_ink(image)
    height, width = ink.shape
    row_counts = ink.sum(axis=1)
    row_mask = row_counts >= max(1, int(width * 0.0015))
    runs = _runs(row_mask, min_run=max(3, min_height // 2), gap=0)
    lines = []
    for line_id, (top, bottom) in enumerate(runs, start=1):
        ys, xs = np.where(ink[max(0, top - padding):min(height, bottom + padding)])
        if xs.size == 0:
            continue
        left = max(0, int(xs.min()) - padding)
        right = min(width, int(xs.max()) + padding + 1)
        top_box = max(0, top - padding)
        bottom_box = min(height, bottom + padding)
        lines.append({
            "line_id": line_id,
            "bbox": [left, top_box, right, bottom_box],
            "reading_order": line_id,
        })
    if not lines and ink.any():
        ys, xs = np.where(ink)
        lines.append({
            "line_id": 1,
            "bbox": [
                max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding),
                min(width, int(xs.max()) + padding + 1), min(height, int(ys.max()) + padding + 1),
            ],
            "reading_order": 1,
        })
    return lines


def find_prescription_regions(
    image: Image.Image,
    lines: list[dict] | None = None,
    padding: int = 8,
    word_gap: int = 22,
) -> list[dict]:
    """Find conservative word/line regions while preserving page reading order."""
    lines = lines if lines is not None else find_prescription_lines(image, padding=padding)
    ink = _binary_ink(image)
    height, width = ink.shape
    regions = []
    region_id = 1
    for line in lines:
        x1, y1, x2, y2 = line["bbox"]
        band = ink[y1:y2, x1:x2]
        columns = band.sum(axis=0)
        runs = _runs(columns >= 1, min_run=3, gap=word_gap)
        # Short or fragmented handwriting is safer as one line than as letters.
        if not runs or len(runs) > 12:
            runs = [(0, x2 - x1)]
        for left, right in runs:
            left_box = max(0, x1 + left - padding)
            right_box = min(width, x1 + right + padding)
            if right_box - left_box < max(18, int(width * 0.015)):
                continue
            regions.append({
                "region_id": region_id,
                "line_id": line["line_id"],
                "bbox": [left_box, y1, right_box, y2],
                "reading_order": len(regions) + 1,
            })
            region_id += 1
    return regions


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
    lines = find_prescription_lines(
        image, min_height=min_height, line_gap=line_gap, padding=padding
    )
    return [tuple(region["bbox"]) for region in find_prescription_regions(
        image, lines, padding=padding, word_gap=word_gap
    )]


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
