from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.image_ops import decode_image
from src.lance_data import load_rows
from src.ocr import HandwritingOCR, normalize_text


def edit_similarity(a: str, b: str) -> float:
    a, b = normalize_text(a).casefold(), normalize_text(b).casefold()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                cur[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    distance = prev[-1]
    return 1.0 - distance / max(len(a), len(b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate handwritten OCR on Lance test images.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--ocr-model", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output-csv", default=str(ROOT / "outputs" / "ocr_eval.csv"))
    args = parser.parse_args()

    rows = load_rows(args.dataset_dir, "test")
    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    ocr = HandwritingOCR(args.ocr_model)
    exact = 0
    fuzzy80 = 0
    scores = []
    output = []

    for n, row in enumerate(rows, 1):
        image = decode_image(row["image"])
        predicted = ocr.recognize(image)
        expected = str(row["medicine_name"])
        score = edit_similarity(predicted, expected)
        exact += normalize_text(predicted).casefold() == normalize_text(expected).casefold()
        fuzzy80 += score >= 0.80
        scores.append(score)
        output.append({
            "id": row["id"],
            "expected": expected,
            "ocr": predicted,
            "edit_similarity": f"{score:.4f}",
            "exact": normalize_text(predicted).casefold() == normalize_text(expected).casefold(),
        })
        print(f"[{n}/{len(rows)}] expected={expected!r} | ocr={predicted!r} | similarity={score:.3f}")

    mean_score = sum(scores) / max(1, len(scores))
    print("\nOCR evaluation")
    print("==============")
    print(f"Samples: {len(rows)}")
    print(f"Exact match: {exact/len(rows):.3f}")
    print(f">=80% normalized edit similarity: {fuzzy80/len(rows):.3f}")
    print(f"Mean normalized edit similarity: {mean_score:.3f}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys() if output else ["id"])
        writer.writeheader()
        writer.writerows(output)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
