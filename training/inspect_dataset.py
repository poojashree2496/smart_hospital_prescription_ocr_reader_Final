from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.image_ops import decode_image
from src.lance_data import labels_from_rows, load_rows

DEFAULT_DATASET = ROOT / "dataset" / "handwriting_ocr"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    args = parser.parse_args()

    train_rows = load_rows(args.dataset_dir, "train")
    labels = labels_from_rows(train_rows)
    print(f"Train-derived medicine labels: {len(labels)}")
    print("First 15 labels:", labels[:15])
    for split in ("train", "validation", "test"):
        rows = train_rows if split == "train" else load_rows(args.dataset_dir, split)
        counts = Counter(str(r["medicine_name"]).strip() for r in rows)
        unknown = set(counts) - set(labels)
        print(
            f"{split}: {len(rows)} rows, {len(counts)} labels, images/class min={min(counts.values())} "
            f"max={max(counts.values())}; first label={rows[0]['medicine_name']} (rows are sorted by label!)"
            + (f"; UNSEEN LABELS: {sorted(unknown)}" if unknown else "")
        )
    sizes = [decode_image(r["image"]).size for r in train_rows[:: max(1, len(train_rows) // 200)]]
    widths, heights = zip(*sizes)
    print(f"Image size sample: width {min(widths)}-{max(widths)}, height {min(heights)}-{max(heights)}")


if __name__ == "__main__":
    main()
