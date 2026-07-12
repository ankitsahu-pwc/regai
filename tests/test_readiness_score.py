"""Runnable demo / test suite for :mod:`services.readiness_score`.

Run with:

    python tests/test_readiness_score.py

Each test prints a short PASS / FAIL line so the script can be used as
a quick smoke check without depending on pytest. The important test is
``test_reference_example`` which validates the example from the product
spec (Overall Readiness Index = 76.0).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from services.readiness_score import (
    DORA_AREA_WEIGHTS,
    classify_question_area,
    compute_weighted_readiness,
    demo_result,
    gap_severity,
    normalise_quantitative_answer,
    readiness_rating,
    validate_weights,
)


PASS = "PASS"
FAIL = "FAIL"


def _report(name: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    line = f"  [{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not ok:
        raise SystemExit(1)


def test_weights_sum_to_100() -> None:
    total = sum(DORA_AREA_WEIGHTS.values())
    _report("weights sum to exactly 100", abs(total - 100.0) < 0.01, f"total={total}")
    validate_weights(DORA_AREA_WEIGHTS)


def test_reference_example() -> None:
    """Reference example from the spec (Overall = 76.0)."""
    r = demo_result()
    _report(
        "reference example overall = 76.0",
        abs(r.overall_readiness_score - 76.0) < 0.001,
        f"got={r.overall_readiness_score}",
    )
    _report(
        "reference rating = Largely Ready",
        r.readiness_rating == "Largely Ready",
        f"got={r.readiness_rating!r}",
    )
    expected_scores = {
        "ICT Governance & Risk Management": 90.0,
        "ICT Policies & Standards": 80.0,
        "ICT Processes & Operating Model": 70.0,
        "ICT Controls & Compliance Controls": 65.0,
        "ICT Technology & Architecture": 85.0,
        "Documentation & Evidence": 60.0,
        "Training & Awareness": 75.0,
    }
    for area, expected in expected_scores.items():
        got = r.area_scores.get(area)
        _report(
            f"area score {area!r}",
            got is not None and abs(got - expected) < 0.01,
            f"expected={expected} got={got}",
        )


def test_rating_bands() -> None:
    for score, expected in (
        (100.0, "Highly Ready"),
        (95.0, "Highly Ready"),
        (90.0, "Highly Ready"),
        (89.9, "Largely Ready"),
        (75.0, "Largely Ready"),
        (74.9, "Moderately Ready"),
        (60.0, "Moderately Ready"),
        (59.9, "Needs Significant Improvement"),
        (40.0, "Needs Significant Improvement"),
        (39.9, "Not Ready"),
        (0.0, "Not Ready"),
    ):
        got = readiness_rating(score)
        _report(f"rating({score}) = {expected!r}", got == expected, f"got={got!r}")


def test_gap_severity_bands() -> None:
    for gap, expected in (
        (0.0, "Low"),
        (10.0, "Low"),
        (10.5, "Medium"),
        (25.0, "Medium"),
        (25.5, "High"),
        (40.0, "High"),
        (40.5, "Critical"),
        (100.0, "Critical"),
    ):
        got = gap_severity(gap)
        _report(f"gap_severity({gap}) = {expected!r}", got == expected, f"got={got!r}")


def test_quant_normalisation() -> None:
    assert normalise_quantitative_answer("MFA coverage 95%") == 95.0
    _report("normalise 'MFA coverage 95%' = 95", True)
    assert normalise_quantitative_answer("60%") == 60.0
    _report("normalise '60%' = 60", True)
    assert normalise_quantitative_answer("60") == 60.0
    _report("normalise '60' (bare int in range) = 60", True)
    assert normalise_quantitative_answer("50k budget") is None
    _report("normalise '50k budget' = None", True)
    assert normalise_quantitative_answer(None) is None
    _report("normalise None = None", True)


def test_area_classifier_fallback() -> None:
    # Question with no ``area`` field but obvious keyword should map.
    q = {"question": "Do you have MFA and access controls in place?"}
    got = classify_question_area(q)
    _report(
        "classify by keywords -> Controls",
        got == "ICT Controls & Compliance Controls",
        f"got={got!r}",
    )
    # Existing legacy label maps via alias table.
    q2 = {"area": "IT, Systems & Technology", "question": ""}
    got2 = classify_question_area(q2)
    _report(
        "classify alias 'IT, Systems & Technology' -> Technology",
        got2 == "ICT Technology & Architecture",
        f"got={got2!r}",
    )
    # Unknown wording falls back to the default operating-model bucket.
    q3 = {"area": "Blah Blah Blah", "question": "Random unrelated text."}
    got3 = classify_question_area(q3)
    _report(
        "classify unknown -> default operating model",
        got3 == "ICT Processes & Operating Model",
        f"got={got3!r}",
    )


def test_empty_state_still_returns_result() -> None:
    """An empty questionnaire yields a zero-score result (not an exception)."""
    r = compute_weighted_readiness([], state=None)
    _report("empty inputs -> overall = 0", r.overall_readiness_score == 0.0)
    _report("empty inputs -> rating = Not Ready", r.readiness_rating == "Not Ready")
    _report(
        "empty inputs -> 7 area rows still emitted",
        len(r.area_details) == len(DORA_AREA_WEIGHTS),
    )


def test_completeness_excludes_na() -> None:
    qs = [
        {
            "question_id": "Q1", "question": "A",
            "options": [{"label": "Yes", "score_value": 100},
                        {"label": "No", "score_value": 0}],
            "question_type": "Single Select",
        },
        {
            "question_id": "Q2", "question": "B",
            "options": [{"label": "Yes", "score_value": 100},
                        {"label": "No", "score_value": 0}],
            "question_type": "Single Select",
        },
        {
            "question_id": "Q3", "question": "C",
            "options": [{"label": "Yes", "score_value": 100},
                        {"label": "No", "score_value": 0}],
            "question_type": "Single Select",
        },
    ]
    r = compute_weighted_readiness(
        qs,
        state=None,
        responses={"Q1": "Yes", "Q2": "N/A"},
    )
    # Q3 unanswered, Q2 explicit N/A → applicable = {Q1, Q3}, answered = {Q1}
    _report(
        "completeness excludes N/A + counts only applicable",
        abs(r.completeness_score - 50.0) < 0.01,
        f"got={r.completeness_score}",
    )


def main() -> int:
    print("== services.readiness_score smoke tests ==")
    print()
    print("[weights]")
    test_weights_sum_to_100()
    print()
    print("[reference example]")
    test_reference_example()
    print()
    print("[rating bands]")
    test_rating_bands()
    print()
    print("[gap severity]")
    test_gap_severity_bands()
    print()
    print("[quantitative normalisation]")
    test_quant_normalisation()
    print()
    print("[area classifier]")
    test_area_classifier_fallback()
    print()
    print("[edge cases]")
    test_empty_state_still_returns_result()
    test_completeness_excludes_na()
    print()
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
