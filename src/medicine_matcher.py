from __future__ import annotations

from difflib import SequenceMatcher
from typing import Sequence


def match_medicine(
    predicted_name: str,
    confidence: float,
    known_names: Sequence[str],
    threshold: float = 0.60,
    minimum_match_score: float = 0.80,
    runner_up_confidence: float | None = None,
    minimum_margin: float = 0.20,
) -> dict[str, object]:
    """Review-aware label match that never hides an uncertain prediction.

    A prediction needs human review when ANY of these hold:
      * top-1 confidence is below `threshold`
      * the gap to the 2nd-best candidate is below `minimum_margin`
        (the model is torn between two medicines - e.g. Ketocon vs Ketoral)
      * the name does not match a known medicine
    Note: the classifier only ever outputs known names, so the fuzzy match is a
    safety net for future use with free-text OCR; the margin check is what
    matters for this model.
    """
    predicted_name = predicted_name.strip()
    candidate = max(
        known_names,
        key=lambda name: SequenceMatcher(None, predicted_name.casefold(), name.casefold()).ratio(),
        default=predicted_name,
    )
    match_score = SequenceMatcher(None, predicted_name.casefold(), candidate.casefold()).ratio()
    verified_candidate = candidate if match_score >= minimum_match_score else None
    margin = None if runner_up_confidence is None else confidence - runner_up_confidence
    too_close = margin is not None and margin < minimum_margin
    needs_human_review = confidence < threshold or too_close or verified_candidate is None
    return {
        "predicted_name": predicted_name,
        "confidence": confidence,
        "matched_candidate": verified_candidate,
        "needs_human_review": needs_human_review,
        "match_score": match_score,
        "margin_to_runner_up": margin,
    }
