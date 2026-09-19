from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.ocr import normalize_text


def text_similarity(a: str, b: str) -> float:
    try:
        from rapidfuzz.fuzz import ratio
        return ratio(normalize_text(a).casefold(), normalize_text(b).casefold()) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(
            None, normalize_text(a).casefold(), normalize_text(b).casefold()
        ).ratio()


@dataclass
class FusedPrediction:
    medicine: str
    score: float
    classifier_confidence: float
    ocr_similarity: float
    needs_human_review: bool
    reason: str
    classifier_prediction: str
    top_alternatives: list[dict]


def fuse_predictions(
    ocr_text: str,
    classifier_candidates: Sequence[dict],
    known_medicines: Sequence[str],
    confidence_threshold: float = 0.60,
    margin_threshold: float = 0.12,
    ocr_weight: float = 0.55,
    classifier_weight: float = 0.45,
) -> FusedPrediction | None:
    if not known_medicines:
        return None

    ocr_scores = {
        name: text_similarity(ocr_text, name) if ocr_text.strip() else 0.0
        for name in known_medicines
    }
    clf_scores = {name: 0.0 for name in known_medicines}
    for item in classifier_candidates:
        name = str(item["medicine"])
        if name in clf_scores:
            clf_scores[name] = float(item["confidence"])

    # Blend the two independent signals. This is a conservative vocabulary
    # constrained decoder: OCR proposes text, classifier supplies visual evidence.
    ranked = []
    for name in known_medicines:
        score = ocr_weight * ocr_scores[name] + classifier_weight * clf_scores[name]
        ranked.append(
            (score, name, clf_scores[name], ocr_scores[name])
        )
    ranked.sort(reverse=True)

    best_score, best_name, clf_conf, ocr_sim = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = best_score - second_score

    needs_review = (
        clf_conf < confidence_threshold
        or margin < margin_threshold
        or (ocr_sim < 0.55 and clf_conf < 0.80)
    )
    reasons = []
    if clf_conf < confidence_threshold:
        reasons.append("low classifier confidence")
    if margin < margin_threshold:
        reasons.append("small fused margin")
    if ocr_sim < 0.55:
        reasons.append("OCR text does not strongly match the medicine vocabulary")
    reason = "; ".join(reasons) if reasons else "OCR and classifier agree"

    return FusedPrediction(
        medicine=best_name,
        score=float(best_score),
        classifier_confidence=float(clf_conf),
        ocr_similarity=float(ocr_sim),
        needs_human_review=needs_review,
        reason=reason,
        classifier_prediction=str(classifier_candidates[0]["medicine"]) if classifier_candidates else best_name,
        top_alternatives=[
            {"medicine": name, "confidence": float(clf_scores[name]), "ocr_similarity": float(ocr_scores[name])}
            for _, name, _, _ in ranked[1:4]
        ],
    )
