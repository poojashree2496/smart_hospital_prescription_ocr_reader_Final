"""Turn a preprocessed 'L' canvas into a normalised tensor, and prepare inference inputs."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.image_ops import TARGET_HEIGHT, TARGET_WIDTH, prepare_canvas

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def canvas_to_tensor(canvas: Image.Image) -> torch.Tensor:
    """'L' canvas (H, W) -> float tensor (3, H, W), ImageNet-normalised.

    The single gray channel is repeated 3x so ImageNet-pretrained weights can be reused.
    """
    array = np.array(canvas.convert("L"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
    return (tensor - _MEAN) / _STD


def prepare_pil(image: Image.Image, height: int = TARGET_HEIGHT, width: int = TARGET_WIDTH) -> torch.Tensor:
    """Full inference preprocessing for one PIL image: gray -> tight crop -> letterbox -> tensor."""
    return canvas_to_tensor(prepare_canvas(image, (width, height)))
