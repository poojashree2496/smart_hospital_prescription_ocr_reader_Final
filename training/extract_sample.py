from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lance_data import load_rows

DEFAULT_DATASET = ROOT / "dataset" / "handwriting_ocr"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_rows(args.dataset_dir, args.split)
    if not 0 <= args.index < len(rows):
        raise IndexError(f"Index {args.index} is outside {args.split} split (0..{len(rows)-1})")
    row = rows[args.index]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = row["image"]
    if isinstance(image_bytes, memoryview):
        image_bytes = image_bytes.tobytes()
    output.write_bytes(bytes(image_bytes))
    print(f"Expected medicine: {row['medicine_name']}")
    print(f"Generic name: {row['generic_name']}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
