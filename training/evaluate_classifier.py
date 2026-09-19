from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import LanceMedicineDataset
from src.engine import evaluate_loader
from src.lance_data import load_rows
from src.model import load_checkpoint


def evaluate_split(model, checkpoint, dataset_dir: str, split: str, batch_size: int, device):
    rows = load_rows(dataset_dir, split)
    labels = checkpoint["labels"]
    label_to_index = {label: i for i, label in enumerate(labels)}
    height, width = checkpoint["image_size"]
    dataset = LanceMedicineDataset(rows, label_to_index, train=False, size=(width, height))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return evaluate_loader(model, loader, device, topk=5, keep_predictions=True), labels


def write_reports(result: dict, labels: list[str], output_dir: Path, split: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = [item["expected_index"] for item in result["predictions"]]
    y_pred = [item["top_indices"][0] for item in result["predictions"]]
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(
        y_true, y_pred, labels=list(range(len(labels))), target_names=labels,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))
    )
    report_paths = [output_dir / f"classification_report_{split}.csv"]
    if split == "test":
        report_paths.append(output_dir / "classification_report.csv")
    for report_path in report_paths:
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["class", "precision", "recall", "f1-score", "support"])
            writer.writeheader()
            for name in labels:
                row = report[name]
                writer.writerow({"class": name, "precision": row["precision"], "recall": row["recall"], "f1-score": row["f1-score"], "support": row["support"]})
    matrix_paths = [output_dir / f"confusion_matrix_{split}.csv"]
    if split == "test":
        matrix_paths.append(output_dir / "confusion_matrix.csv")
    for matrix_path in matrix_paths:
        with matrix_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["expected\\predicted", *labels])
            for label, row in zip(labels, matrix.tolist()):
                writer.writerow([label, *row])
    return {
        "split": split,
        "samples": result["samples"],
        "top1_accuracy": result["top1"],
        "top3_accuracy": result["top3"],
        "top5_accuracy": result["top5"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a medicine classifier on validation and test splits.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", "--model", dest="checkpoint", default=str(ROOT / "models" / "medicine_classifier_best.pth"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", choices=("validation", "test", "both"), default="both")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "evaluation"))
    args = parser.parse_args()

    device = torch.device("cpu")
    try:
        model, checkpoint = load_checkpoint(args.checkpoint, device)
    except FileNotFoundError as exc:
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Train the classifier first.") from exc
    splits = ("validation", "test") if args.split == "both" else (args.split,)
    output_dir = Path(args.output_dir)
    metrics = {}
    for split in splits:
        result, labels = evaluate_split(model, checkpoint, args.dataset_dir, split, args.batch_size, device)
        metrics[split] = write_reports(result, labels, output_dir, split)
        print(f"{split}: top-1={result['top1']:.4f} top-3={result['top3']:.4f} top-5={result['top5']:.4f}")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved evaluation artifacts:", output_dir)


if __name__ == "__main__":
    main()
