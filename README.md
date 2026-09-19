# Smart Hospital Handwritten Prescription OCR + Medicine Recognition

A CPU-compatible student/research system that reads a full handwritten prescription image, proposes likely handwriting regions, runs TrOCR handwriting OCR, classifies medicine-name crops with a ResNet-18 model, fuses both signals, and writes review-aware JSON and text outputs.

This is not a medical device and is not a substitute for a doctor or pharmacist. OCR and medicine predictions can be wrong, handwriting can be ambiguous, and every result must be verified against the original prescription before medication is dispensed or administered. The system does not generate dosage recommendations.

## Architecture

```text
full prescription -> preprocessing -> heuristic candidate regions
                                      |                     |
                                      v                     v
                              TrOCR handwritten OCR   ResNet-18 classifier
                                      |                     |
                                      +------ fusion -------+
                                             |
                                    JSON + text + review flags
```

The dataset contains medicine handwriting samples, not fully annotated prescription pages. Classifier metrics are meaningful for medicine crops; full-page region detection is heuristic and is not clinically validated.

## Project Structure

- `src/ocr.py`: reusable TrOCR wrapper with the explicit slow Roberta tokenizer and ViT image processor.
- `src/dataset.py`, `src/lance_data.py`: Lance loading and dynamic `medicine_name` class mapping.
- `src/model.py`, `src/preprocess.py`, `src/augment.py`: ResNet-18 and handwriting-focused preprocessing.
- `src/fusion.py`: transparent OCR/classifier evidence combination and review policy.
- `inference/segment.py`: adaptive-threshold candidate region generator and crop writer.
- `inference/prescription_pipeline.py`: full-page inference and output writer.
- `training/train_classifier.py`: complete training run with validation model selection.
- `training/evaluate_classifier.py`: validation/test metrics and reports.
- `tests/`: lightweight tests that do not retrain or require the external OCR model.

## Requirements

Use the existing global interpreter on Windows:

```powershell
$python = "C:\Users\pooja\AppData\Local\Python\pythoncore-3.14-64\python.exe"
& $python -m pip install -r requirements.txt
```

The project uses Python 3.14, CPU PyTorch, torchvision, Pillow, OpenCV, Lance (`import lance`), Transformers, scikit-learn, and RapidFuzz. Do not create a virtual environment for this project. Internet access is needed the first time `microsoft/trocr-small-handwritten` is loaded; Transformers then uses its local cache.

## Dataset

Expected Lance files:

```text
C:\Users\pooja\OneDrive\Desktop\smart_hospital_prescription_ocr\dataset\handwriting_ocr\data\train.lance
C:\Users\pooja\OneDrive\Desktop\smart_hospital_prescription_ocr\dataset\handwriting_ocr\data\validation.lance
C:\Users\pooja\OneDrive\Desktop\smart_hospital_prescription_ocr\dataset\handwriting_ocr\data\test.lance
```

The training script derives its 78 class names from the training `medicine_name` field. It does not hard-code labels or modify the dataset.

## Training

This command uses all training and validation rows by default, selects the best validation checkpoint, and reports the untouched test split:

```powershell
& "C:\Users\pooja\AppData\Local\Python\pythoncore-3.14-64\python.exe" training\train_classifier.py `
  --dataset-dir "C:\Users\pooja\OneDrive\Desktop\smart_hospital_prescription_ocr\dataset\handwriting_ocr" `
  --epochs 40 `
  --batch-size 32
```

The ResNet-18 uses ImageNet initialization when available, 96x256 grayscale-to-RGB inputs, mild handwriting-preserving augmentation, AdamW, cosine scheduling, and early stopping. The best checkpoint is written to `models/medicine_classifier_best.pth`. Class mappings are written to `models/class_to_idx.json` and `models/idx_to_class.json`; history is written to `outputs/training_history.json`.

Optional `--max-train-samples` and `--max-val-samples` are development-only switches. Do not use them for final training.

## Evaluation

```powershell
& "C:\Users\pooja\AppData\Local\Python\pythoncore-3.14-64\python.exe" training\evaluate_classifier.py `
  --dataset-dir "C:\Users\pooja\OneDrive\Desktop\smart_hospital_prescription_ocr\dataset\handwriting_ocr" `
  --checkpoint models\medicine_classifier_best.pth
```

The evaluator measures validation and test top-1, top-3, top-5, macro precision, recall, and F1. It writes `outputs/evaluation/metrics.json`, per-class reports, and confusion matrices. Accuracy is never assumed or fabricated; use the measured values.

## Full Prescription Inference

After training:

```powershell
& "C:\Users\pooja\AppData\Local\Python\pythoncore-3.14-64\python.exe" inference\prescription_pipeline.py `
  --image "C:\path\to\prescription.jpg" `
  --model models\medicine_classifier_best.pth
```

The pipeline loads TrOCR once and the classifier once, processes RGB PIL crops on CPU, and writes:

```text
outputs/prescription/prescription_text.txt
outputs/prescription/medicine_names.txt
outputs/prescription/result.json
outputs/prescription/crops/crop_001.png
```

`result.json` includes each region bounding box, OCR text, classifier prediction, classifier confidence, top alternatives, OCR similarity, fusion score, and `needs_human_review`. Use `--no-ocr` only for classifier-only debugging.

A single crop can be inspected with `inference/predict.py`, and segmentation can be run independently with `inference/segment.py --image ... --output-dir outputs/prescription/crops`.

## Fusion and Review

Fusion compares normalized OCR text against every known medicine name using RapidFuzz when available, combines that similarity with the classifier probability, and considers the gap between the first two classifier candidates. Low classifier confidence, a small fused margin, weak OCR similarity, or disagreement marks a result for human review. These scores are model confidence, not medically meaningful certainty.

## Tests

```powershell
& "C:\Users\pooja\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m unittest discover -s tests -p "test_*.py"
```

The tests cover dynamic mappings, preprocessing, segmentation crop writing, fusion, OCR slow-tokenizer preservation, and JSON serialization without retraining or downloading models.

## Troubleshooting

- Missing Lance files: pass the directory containing `data/train.lance`, `data/validation.lance`, and `data/test.lance` with `--dataset-dir`.
- Missing checkpoint: run the full training command first, or pass an existing checkpoint with `--checkpoint`/`--model`.
- TrOCR download failure: provide internet access for the first model load, then retry after the Hugging Face cache is populated.
- No regions detected: use a clearer, sufficiently illuminated prescription image; segmentation is a heuristic candidate generator.
- The TrOCR tokenizer path is intentionally explicit in `src/ocr.py`: `ViTImageProcessor` plus slow `RobertaTokenizer`. Do not replace it with default fast-tokenizer conversion.
- Windows commands use the explicit global Python path and do not require activation.

## Known Limitations

The classifier is trained on isolated medicine handwriting examples, while a full prescription may contain lines, dosage instructions, stamps, tables, and overlapping text. Region proposals can miss or merge handwriting. TrOCR can hallucinate or misread text. Human verification is mandatory for any real-world use.
