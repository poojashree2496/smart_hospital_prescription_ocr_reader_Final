from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop one medicine/handwriting region from a prescription image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--box", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Image.open(args.image) as image:
        crop = image.convert("RGB").crop(tuple(args.box))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output)
    print("Saved crop:", output)


if __name__ == "__main__":
    main()
