from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.medicine_matcher import match_medicine
from src.model import load_checkpoint
from src.preprocess import prepare_pil


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a medicine from ONE handwritten medicine-name crop.")
    parser.add_argument("--model", default=str(ROOT / "models" / "medicine_classifier" / "best.pt"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.model, device)
    labels = checkpoint["labels"]
    height, width = checkpoint["image_size"]

    image_path = Path(args.image)
    with Image.open(image_path) as image:
        tensor = prepare_pil(image, height=height, width=width).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(max(args.top_k, 2), len(labels)))

    print("Device:", device)
    print("Image:", image_path)
    print("Top candidates:")
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"{rank}. {labels[index]}  {value:.4f}")

    result = {
        "image": str(image_path),
        "prediction": labels[int(indices[0])],
        "confidence": float(values[0]),
        "candidates": [
            {"medicine": labels[int(i)], "confidence": float(v)}
            for v, i in zip(values.tolist(), indices.tolist())
        ],
    }
    result.update(match_medicine(
        result["prediction"], result["confidence"], labels, args.confidence_threshold,
        runner_up_confidence=float(values[1]),
    ))
    print(f"\nPredicted medicine: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("Needs human verification" if result["needs_human_review"] else "Confident match (still verify clinically)")
    print("\nJSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
