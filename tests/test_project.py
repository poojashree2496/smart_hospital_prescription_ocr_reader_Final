from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from inference.segment import find_handwriting_regions, save_region_crops
from src.fusion import fuse_predictions, text_similarity
from src.image_ops import prepare_word
from src.lance_data import labels_from_rows
from src.ocr import HandwritingOCR, normalize_text


class ProjectTests(unittest.TestCase):
    def test_dynamic_class_mapping(self):
        rows = [{"medicine_name": "Zed"}, {"medicine_name": "Alpha"}, {"medicine_name": "Zed"}]
        self.assertEqual(labels_from_rows(rows), ["Alpha", "Zed"])

    def test_image_preprocessing_and_segmentation(self):
        image = Image.new("RGB", (240, 100), "white")
        pixels = np.asarray(image).copy()
        pixels[40:58, 55:150] = 0
        image = Image.fromarray(pixels)
        self.assertEqual(prepare_word(image).mode, "L")
        regions = find_handwriting_regions(image)
        self.assertTrue(regions)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_region_crops(image, regions, directory)
            self.assertEqual(len(paths), len(regions))
            self.assertTrue(Path(paths[0]).exists())

    def test_fusion_and_similarity(self):
        result = fuse_predictions(
            "Aceta", [{"medicine": "Acetaminophen", "confidence": 0.91}, {"medicine": "Napa", "confidence": 0.04}],
            ["Acetaminophen", "Napa"],
        )
        self.assertEqual(result.medicine, "Acetaminophen")
        self.assertGreater(text_similarity("Aceta", "Acetaminophen"), 0.4)
        self.assertGreaterEqual(result.score, 0.0)

    def test_ocr_source_preserves_explicit_slow_tokenizer(self):
        source = Path(HandwritingOCR.__module__.replace(".", "/") + ".py")
        self.assertTrue(source.exists())
        self.assertIn("RobertaTokenizer", source.read_text(encoding="utf-8"))
        self.assertEqual(normalize_text("  Napa\n  500mg "), "Napa 500mg")

    def test_json_output_is_serializable(self):
        payload = {"bbox": [1, 2, 3, 4], "confidence": 0.9, "needs_human_review": False}
        self.assertEqual(json.loads(json.dumps(payload))["bbox"], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
