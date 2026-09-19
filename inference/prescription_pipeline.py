from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.segment import find_handwriting_regions
from src.fusion import fuse_predictions
from src.model import load_checkpoint
from src.ocr import DEFAULT_MODEL, HandwritingOCR
from src.preprocess import prepare_pil
from inference.segment import save_region_crops


def classifier_candidates(model, image: Image.Image, labels: list[str], height: int, width: int, device, top_k: int = 5):
    tensor = prepare_pil(image, height=height, width=width).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(top_k, len(labels)))
    return [
        {"medicine": labels[int(i)], "confidence": float(v)}
        for v, i in zip(values.tolist(), indices.tolist())
    ]


def deduplicate(items: list[dict]) -> list[dict]:
    """Keep the strongest occurrence of the same medicine."""
    best = {}
    for item in items:
        name = item["medicine"]
        score = float(item["fused_score"])
        if name not in best or score > best[name]["fused_score"]:
            best[name] = item
    return sorted(best.values(), key=lambda x: (-x["fused_score"], x["region_index"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a full handwritten prescription and output medicine names as text."
    )
    parser.add_argument("--image", required=True, help="Full prescription image")
    parser.add_argument("--model", default=str(ROOT / "models" / "medicine_classifier" / "best.pt"))
    parser.add_argument("--ocr-model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "prescription"))
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--margin-threshold", type=float, default=0.12)
    parser.add_argument("--max-regions", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-ocr", action="store_true", help="Use classifier only")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, checkpoint = load_checkpoint(args.model, device)
    except FileNotFoundError as exc:
        raise SystemExit(f"Classifier checkpoint not found: {args.model}. Train it first.") from exc
    labels = checkpoint["labels"]
    height, width = checkpoint["image_size"]

    with Image.open(image_path) as source:
        page = ImageOps.exif_transpose(source).convert("RGB")

    regions = find_handwriting_regions(page)[:args.max_regions]
    if not regions:
        raise RuntimeError(
            "No handwriting regions were detected. Use a clearer image or inspect "
            "the saved segmentation output."
        )

    crops_dir = output_dir / "crops"
    save_region_crops(page, regions, crops_dir)
    ocr = None if args.no_ocr else HandwritingOCR(args.ocr_model, device="cpu")

    detailed = []
    for region_index, box in enumerate(regions, start=1):
        crop = page.crop(box)
        crop_path = output_dir / f"crop_{region_index:03d}.png"
        crop.save(crop_path)

        candidates = classifier_candidates(
            model, crop, labels, height, width, device, top_k=args.top_k
        )

        if ocr is not None:
            ocr_result = ocr.recognize_and_match(
                crop, labels, minimum_score=0.50
            )
            ocr_text = ocr_result.raw_text
            ocr_match = ocr_result.matched_medicine
            ocr_match_score = ocr_result.match_score
        else:
            ocr_text = ""
            ocr_match = None
            ocr_match_score = 0.0

        fused = fuse_predictions(
            ocr_text,
            candidates,
            labels,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
        )

        if fused is None:
            continue

        # If OCR has a strong vocabulary match, retain it as an explicit signal.
        detailed.append({
            "region_id": region_index,
            "region_index": region_index,
            "bbox": list(box),
            "box": list(box),
            "crop": str(crop_path),
            "ocr_text": ocr_text,
            "ocr_matched_medicine": ocr_match,
            "classifier_prediction": fused.classifier_prediction,
            "top_alternatives": fused.top_alternatives,
            "ocr_match_score": float(ocr_match_score),
            "classifier_candidates": candidates,
            "medicine": fused.medicine,
            "fused_score": fused.score,
            "classifier_confidence": fused.classifier_confidence,
            "ocr_similarity": fused.ocr_similarity,
            "needs_human_review": fused.needs_human_review,
            "reason": fused.reason,
        })

    final_items = deduplicate(detailed)

    # Conservative final text: only high-confidence / corroborated items are
    # emitted as "recognized"; uncertain items remain in the JSON for review.
    recognized = [
        x["medicine"]
        for x in final_items
        if not x["needs_human_review"]
    ]
    review = [
        x["medicine"]
        for x in final_items
        if x["needs_human_review"]
    ]

    medicine_txt_path = output_dir / "medicine_names.txt"
    medicine_txt_path.write_text(
        "\n".join(recognized) + ("\n" if recognized else ""),
        encoding="utf-8",
    )

    # Preserve the OCR reading itself as plain text too. This is useful when the
    # prescription contains dosage/instruction text that is not one of the 78
    # medicine classes.
    full_ocr_lines = [
        item["ocr_text"].strip()
        for item in detailed
        if item["ocr_text"].strip()
    ]
    full_text_path = output_dir / "prescription_text.txt"
    full_text_path.write_text(
        "\n".join(full_ocr_lines) + ("\n" if full_ocr_lines else ""),
        encoding="utf-8",
    )

    result = {
        "input_image": str(image_path),
        "device": str(device),
        "ocr_model": None if args.no_ocr else args.ocr_model,
        "regions_found": len(regions),
        "ocr_text": "\n".join(full_ocr_lines),
        "medicines": detailed,
        "recognized_medicines": recognized,
        "needs_human_review": sorted(set(review)),
        "full_ocr_text": "\n".join(full_ocr_lines),
        "regions": detailed,
        "notes": [
            "Recognition is limited to the 78 medicines in the training vocabulary.",
            "Low-confidence results are intentionally not emitted as confirmed medicine names.",
            "Verify every medicine against the original prescription before clinical use.",
        ],
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nFINAL MEDICINE NAMES")
    print("====================")
    if recognized:
        for name in recognized:
            print(name)
    else:
        print("No medicine reached the confirmation threshold.")

    if review:
        print("\nNEEDS HUMAN REVIEW")
        print("==================")
        for name in sorted(set(review)):
            print(name)

    print(f"\nDetailed result: {output_dir / 'result.json'}")
    print(f"Plain-text output: {full_text_path}")


if __name__ == "__main__":
    main()
