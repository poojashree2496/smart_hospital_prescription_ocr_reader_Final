"""Quick end-to-end check that the environment and code work. Takes well under a minute on CPU.

    python tests/smoke_test.py                      # synthetic data only
    python tests/smoke_test.py --dataset-dir PATH   # also checks the real Lance dataset
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import TrainAugment
from src.dataset import LanceMedicineDataset
from src.engine import evaluate_loader, train_one_epoch
from src.image_ops import TARGET_HEIGHT, TARGET_WIDTH, letterbox, prepare_canvas, tight_crop, to_gray
from src.model import ARCHITECTURES, build_model, load_checkpoint, save_checkpoint, set_backbone_trainable
from src.preprocess import canvas_to_tensor, prepare_pil


def synthetic_rows(names: list[str], per_class: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for name in names:
        for k in range(per_class):
            image = Image.new("RGB", (rng.randint(180, 260), rng.randint(50, 90)), "white")
            ImageDraw.Draw(image).text((rng.randint(5, 40), rng.randint(5, 30)), name, fill="black")
            buffer = BytesIO()
            image.save(buffer, "PNG")
            rows.append({"id": f"{name}_{k}", "image": buffer.getvalue(), "medicine_name": name})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "sample.png"))
    args = parser.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())

    # 1) image operations -------------------------------------------------------------------
    page = Image.new("RGB", (240, 100), "white")
    ImageDraw.Draw(page).text((90, 40), "Napa", fill="black")
    cropped = tight_crop(to_gray(page))
    assert cropped.width < page.width and cropped.height < page.height, "tight crop should remove empty paper"
    assert letterbox(cropped).size == (TARGET_WIDTH, TARGET_HEIGHT)
    transparent = Image.new("RGBA", (50, 20), (0, 0, 0, 0))          # fully transparent PNG
    assert to_gray(transparent).getpixel((5, 5)) == 255, "transparent pixels must become white"
    print("[ok] image ops (tight crop, letterbox, transparency)")

    # 2) augmentation --------------------------------------------------------------------------
    augment = TrainAugment(seed=1)
    pickle.dumps(augment)  # required for DataLoader(num_workers>0) on Windows
    word = tight_crop(to_gray(page))
    outputs = [augment(word) for _ in range(5)]
    assert all(o.size == (TARGET_WIDTH, TARGET_HEIGHT) and o.mode == "L" for o in outputs)
    print("[ok] augmentation (shapes, picklable)")

    # 3) tensors + every architecture -----------------------------------------------------------
    tensor = prepare_pil(page)
    assert tuple(tensor.shape) == (3, TARGET_HEIGHT, TARGET_WIDTH)
    batch = torch.stack([tensor, tensor])
    for arch in ARCHITECTURES:
        model = build_model(12, arch=arch, pretrained=False).eval()
        assert tuple(model(batch).shape) == (2, 12), arch
    print("[ok] forward pass for:", ", ".join(ARCHITECTURES))

    # 4) dataset + a real (tiny) training loop on synthetic words ------------------------------
    names = ["Napa", "Ace", "Metro", "Fexo", "Sergel", "Rivotril"]
    label_to_index = {n: i for i, n in enumerate(names)}
    train_rows = synthetic_rows(names, 8, seed=1)
    val_rows = synthetic_rows(names, 3, seed=2)
    train_ds = LanceMedicineDataset(train_rows, label_to_index, train=True)
    val_ds = LanceMedicineDataset(val_rows, label_to_index, train=False)
    pickle.dumps(train_ds)  # Windows DataLoader workers pickle the dataset
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    device = torch.device("cpu")
    model = build_model(len(names), arch="mobilenet_v3_small", pretrained=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    set_backbone_trainable(model, False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    frozen_before = model.features[0][0].weight.detach().clone()
    bn_before = model.features[0][1].running_mean.detach().clone()
    train_one_epoch(model, train_loader, criterion, optimizer, None, device, scaler, False, head_only=True)
    assert torch.equal(frozen_before, model.features[0][0].weight), "frozen backbone weights must not change"
    assert torch.equal(bn_before, model.features[0][1].running_mean), "frozen backbone BatchNorm stats must not change"
    set_backbone_trainable(model, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, None, device, scaler, False, head_only=False)
    result = evaluate_loader(model, val_loader, device, keep_predictions=True)
    assert 0.0 <= result["top1"] <= 1.0 and len(result["predictions"]) == len(val_ds)
    print(f"[ok] training loop runs (train loss {loss:.3f}, synthetic val top-1 {result['top1']:.2f}; accuracy here is not meaningful)")

    # 5) checkpoint round-trip and old-checkpoint rejection --------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.pt"
        save_checkpoint(path, model, names, "mobilenet_v3_small", (TARGET_HEIGHT, TARGET_WIDTH))
        restored, meta = load_checkpoint(path, "cpu")
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(batch), restored(batch), atol=1e-5)
        assert meta["labels"] == names
        old = Path(tmp) / "old.pt"
        torch.save({"model_state": model.state_dict(), "labels": names}, old)
        try:
            load_checkpoint(old, "cpu")
        except RuntimeError:
            pass
        else:
            raise AssertionError("old-format checkpoint should be rejected")
    print("[ok] checkpoint save/load + old-format rejection")

    # 6) optional: the real dataset ---------------------------------------------------------------
    if args.dataset_dir:
        from src.lance_data import labels_from_rows, load_rows
        from src.image_ops import decode_image
        rows = load_rows(args.dataset_dir, "train")
        labels = labels_from_rows(rows)
        assert len(labels) == 78, f"expected 78 medicine labels, found {len(labels)}"
        image = decode_image(rows[0]["image"])
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        canvas_to_tensor(prepare_canvas(image))
        print(f"[ok] real dataset: {len(rows)} train rows, 78 labels; sample '{rows[0]['medicine_name']}' saved to {output}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
