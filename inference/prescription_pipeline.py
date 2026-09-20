from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.segment import find_prescription_lines, find_prescription_regions
from src.fusion import fuse_predictions
from src.model import load_checkpoint
from src.ocr import DEFAULT_MODEL, HandwritingOCR, normalize_text
from src.page_preprocess import prepare_page
from src.image_ops import prepare_word
from src.preprocess import prepare_pil

MEDICINE_SCORE = 0.45
DOSAGE_RE = re.compile(r"\b(?:\d+\s*[-/]\s*){1,3}\d*\b|\b(?:od|bd|tds|sos|hs)\b|\b\d+\s*(?:mg|ml|mcg|g)\b", re.I)
FREQUENCY_RE = re.compile(r"\b(?:od|bd|tds|sos|hs|once|twice|daily|morning|night)\b|\d\s*[-/]\s*\d", re.I)
DURATION_RE = re.compile(r"\b(?:for\s+)?\d+\s*(?:day|days|week|weeks|month|months)\b", re.I)
INSTRUCTION_RE = re.compile(r"\b(?:before|after)\s+food\b|\b(?:morning|afternoon|evening|night|daily)\b", re.I)


def classifier_candidates(model, image, labels, height, width, device, top_k=5):
    tensor = prepare_pil(image, height=height, width=width).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(top_k, len(labels)))
    return [{"medicine": labels[int(i)], "confidence": float(v)} for v, i in zip(values.tolist(), indices.tolist())]


def classify_region_type(text: str, medicine_score: float, classifier_confidence: float) -> str:
    normalized = normalize_text(text)
    if re.search(r"\b(?:patient|name|age|sex|gender)\b", normalized, re.I):
        return "patient_info"
    if re.search(r"\b(?:doctor|dr)\b", normalized, re.I):
        return "doctor_info"
    if re.search(r"\b(?:date|\d{1,2}[/-]\d{1,2})\b", normalized, re.I):
        return "date"
    if DURATION_RE.search(normalized):
        return "duration"
    if FREQUENCY_RE.search(normalized):
        return "frequency"
    if DOSAGE_RE.search(normalized):
        return "dosage"
    if INSTRUCTION_RE.search(normalized):
        return "instruction"
    if medicine_score >= MEDICINE_SCORE or (classifier_confidence >= 0.45 and len(normalized.split()) <= 3):
        return "medicine"
    return "unknown" if not normalized else "other"


def deduplicate(items: list[dict]) -> list[dict]:
    best = {}
    for item in items:
        name = item["medicine"]
        if name not in best or item["fusion_score"] > best[name]["fusion_score"]:
            best[name] = item
    return sorted(best.values(), key=lambda item: item["reading_order"])


def annotate_page(page: Image.Image, regions: list[dict], output: Path) -> None:
    annotated = page.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    colors = {"medicine": "red", "dosage": "blue", "duration": "purple", "instruction": "green"}
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        color = colors.get(region["region_type"], "orange")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{region['region_id']} {region['region_type']}: {region['ocr_text'][:40]}"
        draw.text((x1, max(0, y1 - 12)), label, fill=color, font=font)
    annotated.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and structure a full handwritten prescription page.")
    parser.add_argument("--image", required=True, help="Complete prescription image")
    parser.add_argument("--model", default=str(ROOT / "models" / "best.pt"))
    parser.add_argument("--ocr-model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "prescription"))
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--margin-threshold", type=float, default=0.12)
    parser.add_argument("--max-regions", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    lines_dir = output_dir / "lines"
    crops_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    lines_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, checkpoint = load_checkpoint(args.model, device)
    except FileNotFoundError as exc:
        raise SystemExit(f"Classifier checkpoint not found: {args.model}. Train it first.") from exc
    labels = checkpoint["labels"]
    height, width = checkpoint["image_size"]
    with Image.open(image_path) as source:
        page, working_page = prepare_page(source)

    lines = find_prescription_lines(working_page)
    regions = find_prescription_regions(working_page, lines)[:args.max_regions]
    if not regions:
        raise RuntimeError("No handwriting regions were detected on the prescription page.")
    for line in lines:
        x1, y1, x2, y2 = line["bbox"]
        page.crop((x1, y1, x2, y2)).save(lines_dir / f"line_{line['line_id']:03d}.png")

    ocr = None if args.no_ocr else HandwritingOCR(args.ocr_model, device="cpu")
    line_ocr: dict[int, str] = {}
    line_matches = {}
    line_by_id = {line["line_id"]: line for line in lines}
    if ocr is not None:
        for line in lines:
            line_crop = page.crop(tuple(line["bbox"]))
            # OCR the complete handwritten band so connected cursive words are
            # not destroyed by component-level segmentation.
            prepared_line = prepare_word(line_crop)
            line_ocr[line["line_id"]] = ocr.recognize(prepared_line)
            line_matches[line["line_id"]] = ocr.recognize_and_match(
                prepared_line, labels, minimum_score=0.50
            )
    detailed = []
    for region in regions:
        box = tuple(region["bbox"])
        crop = page.crop(box)
        crop_path = crops_dir / f"crop_{region['region_id']:03d}.png"
        crop.save(crop_path)
        ocr_text = ""
        ocr_match = None
        ocr_match_score = 0.0
        if ocr is not None:
            ocr_result = ocr.recognize_and_match(prepare_word(crop), labels, minimum_score=0.50)
            component_text = ocr_result.raw_text.strip()
            line_text = line_ocr.get(region["line_id"], "").strip()
            ocr_text = component_text or line_text
            if component_text:
                ocr_match, ocr_match_score = ocr_result.matched_medicine, ocr_result.match_score
            else:
                line_match = line_matches[region["line_id"]]
                ocr_match, ocr_match_score = line_match.matched_medicine, line_match.match_score
        candidates = classifier_candidates(model, crop, labels, height, width, device, args.top_k)
        fused = fuse_predictions(ocr_text, candidates, labels, args.confidence_threshold, args.margin_threshold)
        classifier_confidence = candidates[0]["confidence"] if candidates else 0.0
        region_type = classify_region_type(ocr_text, ocr_match_score, classifier_confidence)
        if region_type == "medicine" and fused is not None:
            medicine = fused.medicine
            fusion_score = fused.score
            needs_review = fused.needs_human_review or not ocr_text.strip()
        else:
            medicine = None
            fusion_score = 0.0
            needs_review = not ocr_text.strip()
        detailed.append({
            "region_id": region["region_id"], "line_id": region["line_id"], "bbox": region["bbox"],
            "reading_order": region["reading_order"], "crop": str(crop_path), "ocr_text": ocr_text,
            "region_type": region_type, "ocr_confidence_if_available": None,
            "classifier_prediction": candidates[0]["medicine"] if candidates else None,
            "classifier_confidence": classifier_confidence, "classifier_candidates": candidates,
            "ocr_matched_medicine": ocr_match, "ocr_similarity": ocr_match_score,
            "medicine": medicine, "fusion_score": fusion_score, "fused_score": fusion_score,
            "needs_human_review": needs_review,
        })

    detailed.sort(key=lambda item: item["reading_order"])
    medicine_items = deduplicate([item for item in detailed if item["medicine"]])
    candidate_names = [item["medicine"] for item in medicine_items]
    recognized = [item["medicine"] for item in medicine_items if not item["needs_human_review"]]
    all_text = [item["ocr_text"] for item in detailed if item["ocr_text"]]
    other_text = [item["ocr_text"] for item in detailed if item["region_type"] != "medicine" and item["ocr_text"]]
    full_text = "\n".join(all_text)
    (output_dir / "prescription_text.txt").write_text(full_text + ("\n" if full_text else ""), encoding="utf-8")
    (output_dir / "medicine_names.txt").write_text(
        "\n".join(candidate_names) + ("\n" if candidate_names else ""), encoding="utf-8"
    )
    annotate_page(page, detailed, output_dir / "annotated_prescription.png")

    structured_medicines = []
    for item in medicine_items:
        related = [x["ocr_text"] for x in detailed if x["line_id"] == item["line_id"] and x["region_id"] != item["region_id"] and x["ocr_text"]]
        structured_medicines.append({
            "medicine_name": item["medicine"], "ocr_text": item["ocr_text"],
            "dosage": next((x for x in related if DOSAGE_RE.search(x)), None),
            "frequency": next((x for x in related if FREQUENCY_RE.search(x)), None),
            "duration": next((x for x in related if DURATION_RE.search(x)), None),
            "instructions": next((x for x in related if INSTRUCTION_RE.search(x)), None),
            "confidence": item["fusion_score"], "bbox": item["bbox"],
            "needs_human_review": item["needs_human_review"],
        })

    result = {
        "input_image": str(image_path), "device": str(device),
        "ocr_model": None if args.no_ocr else args.ocr_model,
        "ocr_text": full_text, "full_ocr_text": full_text,
        "lines": lines, "regions": detailed, "medicines": structured_medicines,
        "recognized_medicines": recognized, "medicine_candidates": candidate_names,
        "other_text": other_text,
        "human_review_required": any(item["needs_human_review"] for item in detailed),
        "needs_human_review": sorted({item["medicine"] for item in medicine_items if item["needs_human_review"] and item["medicine"]}),
        "notes": ["Medicine metrics apply to isolated medicine handwriting samples, not full prescriptions.", "Full-page line and region detection is heuristic; verify every result against the original prescription."],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nFINAL MEDICINE NAMES\n====================")
    print("\n".join(recognized) if recognized else "No medicine reached the confirmation threshold.")
    print(f"\nDetailed result: {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
