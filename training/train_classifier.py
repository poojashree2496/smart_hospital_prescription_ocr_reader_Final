from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import LanceMedicineDataset
from src.engine import evaluate_loader, train_one_epoch
from src.image_ops import TARGET_HEIGHT, TARGET_WIDTH
from src.lance_data import labels_from_rows, load_rows
from src.model import ARCHITECTURES, build_model, parameter_groups, save_checkpoint, set_backbone_trainable

DEFAULT_DATASET = ROOT / "dataset" / "handwriting_ocr"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subsample(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Random subset. The Lance rows are sorted by label, so taking the first N rows
    (as the old code did) silently drops most classes."""
    if limit is None or limit >= len(rows):
        return rows
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def cosine_with_warmup(warmup_steps: int, total_steps: int, floor: float = 0.01):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return factor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the handwritten-medicine classifier.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(ROOT / "models"))
    parser.add_argument("--arch", choices=ARCHITECTURES, default="resnet18")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--freeze-epochs", type=int, default=2, help="warm-up epochs that train only the new classifier head")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--image-height", type=int, default=TARGET_HEIGHT)
    parser.add_argument("--image-width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--patience", type=int, default=15, help="stop if validation does not improve for N epochs (0 = never)")
    parser.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows; try 2-4 once everything works")
    parser.add_argument("--max-train-samples", type=int, default=None, help="optional subset for development only")
    parser.add_argument("--max-val-samples", type=int, default=None, help="optional subset for development only")
    parser.add_argument("--include-val", action="store_true", help="final fit: also train on the validation split (no model selection; saves the last epoch)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print("Device:", device, f"({torch.cuda.get_device_name(0)})" if use_amp else "")
    if device.type == "cpu":
        print("Training on CPU: expect roughly a minute or two per epoch for resnet18. "
              "Use --arch mobilenet_v3_small for a faster (usually less accurate) run.")

    full_train = load_rows(args.dataset_dir, "train")
    labels = labels_from_rows(full_train)
    label_to_index = {label: index for index, label in enumerate(labels)}
    counts = Counter(str(r["medicine_name"]).strip() for r in full_train)
    print(f"Classes: {len(labels)} | images per class in train: min={min(counts.values())} max={max(counts.values())}")

    val_rows = load_rows(args.dataset_dir, "validation")
    try:
        test_rows = load_rows(args.dataset_dir, "test")
    except FileNotFoundError:
        test_rows = []
    train_rows = subsample(full_train, args.max_train_samples, args.seed)
    val_eval_rows = subsample(val_rows, args.max_val_samples, args.seed)
    if args.include_val:
        train_rows = train_rows + val_rows
        print("NOTE: --include-val is on, so validation numbers are no longer an honest estimate. Use the test score.")
    print(f"Train rows: {len(train_rows)} | validation rows: {len(val_eval_rows)} | test rows: {len(test_rows)}")

    size = (args.image_width, args.image_height)
    t0 = time.time()
    train_ds = LanceMedicineDataset(train_rows, label_to_index, train=True, size=size)
    val_ds = LanceMedicineDataset(val_eval_rows, label_to_index, train=False, size=size)
    print(f"Decoded and cropped images in {time.time() - t0:.1f}s")

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=use_amp, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **loader_kwargs)

    pretrained = not args.no_pretrained
    model = build_model(len(labels), arch=args.arch, pretrained=pretrained, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    freeze_epochs = min(args.freeze_epochs, max(0, args.epochs - 1)) if pretrained else 0
    steps_per_epoch = len(train_loader)
    total_steps = (args.epochs - freeze_epochs) * steps_per_epoch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_key, best_epoch, best_val_acc = (-1.0, 0.0), 0, 0.0
    epochs_without_improvement = 0

    optimizer, scheduler = None, None
    for epoch in range(1, args.epochs + 1):
        head_only = epoch <= freeze_epochs
        if epoch == 1 and head_only:
            set_backbone_trainable(model, False)
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad), lr=args.lr_head, weight_decay=args.weight_decay
            )
            print(f"Phase 1: training only the classifier head for {freeze_epochs} epoch(s)")
        if epoch == freeze_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(
                parameter_groups(model, args.lr_backbone, args.lr_head), weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, cosine_with_warmup(warmup_steps=steps_per_epoch, total_steps=total_steps)
            )
            print("Phase 2: fine-tuning the whole network (cosine LR schedule)")

        started = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler, use_amp, head_only)
        val = evaluate_loader(model, val_loader, device)
        lr_now = optimizer.param_groups[-1]["lr"]
        print(
            f"Epoch {epoch:>2}/{args.epochs} | train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
            f"val_loss={val['loss']:.3f} val_top1={val['top1']:.3f} val_top3={val['top3']:.3f} | "
            f"lr={lr_now:.2e} | {time.time() - started:.0f}s"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val["loss"], "val_top1": val["top1"], "val_top3": val["top3"]})

        key = (val["top1"], -val["loss"])
        improved = key > best_key
        if improved:
            best_key, best_epoch, best_val_acc = key, epoch, val["top1"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.include_val:
            improved = epoch == args.epochs  # final fit: keep the last epoch
        if improved:
            save_checkpoint(
                output_dir / "best.pt", model, labels, args.arch, (args.image_height, args.image_width),
                extra={"pretrained_backbone": pretrained, "epoch": epoch},
            )
            print("  saved checkpoint")
        if args.patience and not args.include_val and epochs_without_improvement >= args.patience and epoch > freeze_epochs:
            print(f"Early stopping: no validation improvement for {args.patience} epochs.")
            break

    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "class_to_idx.json").write_text(json.dumps(label_to_index, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "idx_to_class.json").write_text(json.dumps({str(v): k for k, v in label_to_index.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Final honest number: reload the saved checkpoint and score the untouched test split once.
    from src.model import load_checkpoint
    best_model, _ = load_checkpoint(output_dir / "best.pt", device)
    summary = {"architecture": args.arch, "best_epoch": best_epoch, "best_validation_top1": best_val_acc,
               "seed": args.seed, "image_size": [args.image_height, args.image_width]}
    if test_rows:
        test_ds = LanceMedicineDataset(test_rows, label_to_index, train=False, size=size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, **{**loader_kwargs, "persistent_workers": False})
        test = evaluate_loader(best_model, test_loader, device)
        summary.update({"test_top1": test["top1"], "test_top3": test["top3"]})
        print(f"\nTEST  top-1 = {test['top1']:.4f}   top-3 = {test['top3']:.4f}   ({test['samples']} samples)")
    (output_dir / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import shutil
    shutil.copy2(output_dir / "best.pt", ROOT / "models" / "medicine_classifier_best.pth")
    print(f"Training complete. Best validation top-1: {best_val_acc:.4f} (epoch {best_epoch}). Checkpoint: {ROOT / 'models' / 'medicine_classifier_best.pth'}")


if __name__ == "__main__":
    main()
