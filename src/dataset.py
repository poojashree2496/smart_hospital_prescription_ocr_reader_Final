from __future__ import annotations

from typing import Sequence

from torch.utils.data import Dataset

from src.augment import AugmentConfig, TrainAugment
from src.image_ops import TARGET_HEIGHT, TARGET_WIDTH, decode_image, letterbox, prepare_word
from src.preprocess import canvas_to_tensor


class LanceMedicineDataset(Dataset):
    """Medicine word images with cached, tight-cropped grayscale words.

    Decoding + cropping happen once here (not every epoch); only the cheap
    letterbox / augmentation runs inside __getitem__.
    """

    def __init__(
        self,
        rows: Sequence[dict],
        label_to_index: dict[str, int],
        train: bool = False,
        size: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
        augment: AugmentConfig | None = None,
    ):
        self.size = size
        self.train = train
        self.items: list[tuple] = []
        for row in rows:
            name = str(row["medicine_name"]).strip()
            if name not in label_to_index:
                raise KeyError(f"Unknown medicine label: {name!r}")
            word = prepare_word(decode_image(row["image"]))
            self.items.append((word, label_to_index[name], name, str(row["id"])))
        self.augment = TrainAugment(augment, size) if train else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        word, label_index, name, sample_id = self.items[index]
        canvas = self.augment(word) if self.train else letterbox(word, self.size)
        return canvas_to_tensor(canvas), label_index, name, sample_id
