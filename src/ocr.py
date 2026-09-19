from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image, ImageOps

DEFAULT_MODEL = "microsoft/trocr-small-handwritten"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\n", " ")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    # RapidFuzz is optional. Fall back to SequenceMatcher so the OCR stack
    # remains easy to install.
    try:
        from rapidfuzz.fuzz import ratio
        return ratio(a.casefold(), b.casefold()) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a.casefold(), b.casefold()).ratio()


@dataclass
class OCRResult:
    raw_text: str
    normalized_text: str
    matched_medicine: str | None
    match_score: float


class HandwritingOCR:
    """TrOCR wrapper with optional medicine-vocabulary correction.

    The model reads handwriting. The vocabulary correction is intentionally
    restricted to medicines present in the classifier's 78-class dataset.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_new_tokens: int = 32,
    ):
        from transformers import TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel
        from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

        self.device = torch.device(device or "cpu")
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.processor = TrOCRProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def recognize(self, image: Image.Image) -> str:
        image = ImageOps.exif_transpose(image).convert("RGB")
        pixels = self.processor(images=image, return_tensors="pt").pixel_values
        pixels = pixels.to(self.device)
        generated = self.model.generate(
            pixels,
            max_new_tokens=self.max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
        text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]
        return re.sub(r"\s+", " ", text).strip()

    def ocr_pil_image(self, image: Image.Image) -> str:
        return self.recognize(image)

    def ocr_image(self, image: Image.Image) -> str:
        return self.recognize(image)

    def recognize_and_match(
        self,
        image: Image.Image,
        medicine_names: Sequence[str] = (),
        minimum_score: float = 0.58,
    ) -> OCRResult:
        raw = self.recognize(image)
        normalized = normalize_text(raw)
        best_name = None
        best_score = 0.0
        if normalized and medicine_names:
            for name in medicine_names:
                score = _similarity(normalized, normalize_text(name))
                if score > best_score:
                    best_name, best_score = name, score
            if best_score < minimum_score:
                best_name = None
        return OCRResult(raw, normalized, best_name, best_score)
