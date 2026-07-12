"""Runnable demo / test suite for :mod:`services.impact_score`.

Run with:

    python tests/test_impact_score.py

Each test prints a short PASS / FAIL line so the script can be used as a
quick smoke check without pytest. The critical test is
``test_reference_example`` which validates the example from the product
spec (Overall Impact = 79.75, "High Impact") plus the priority formula
(Impact 80 x Readiness 60 -> Priority 32).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from services.impact_score import (
    DORA_IMPACT_FACTOR_WEIGHTS,
    compute_weighted_impact,
    demo_result,
    impact_rating,
    priority_score,
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
    total = sum(DORA_IMPACT_FACTOR_WEIGHTS.values())
    _report("weights sum to exactly 100", abs(total - 100.0) < 0.01, f"total={total}")
    validate_weights(DORA_IMPACT_FACTOR_WEIGHTS)


def test_reference_example() -> None:
    """Reference example from the spec (Overall = 79.75, rating = High Impact)."""
    r = demo_result()
    _report(
        "reference example overall = 79.75",
        abs(r.overall_impact_score - 79.75) < 0.01,
        f"got={r.overall_impact_score}",
    )
    _report(
        "reference rating = High Impact",
        r.impact_rating == "High Impact",
        f"got={r.impact_rating!r}",
    )
    expected_scores = {
        "Regulatory Obligation Criticality": 90.0,
        "Business Capability Impact": 80.0,
        "Process Change Impact": 70.0,
        "Technology / System Impact": 85.0,
        "Control & Compliance Impact": 75.0,
        "Data / Reporting Impact": 60.0,
        "Third-Party / Vendor Impact": 90.0,
    }
    expected_weighted = {
        "Regulatory Obligation Criticality": 22.5,
        "Business Capability Impact": 16.0,
        "Process Change Impact": 10.5,
        "Technology / System Impact": 12.75,
        "Control & Compliance Impact": 7.5,
        "Data / Reporting Impact": 6.0,
        "Third-Party / Vendor Impact": 4.5,
    }
    for factor, expected in expected_scores.items():
        got = r.factor_scores.get(factor)
        _report(
            f"factor score {factor!r}",
            got is not None and abs(got - expected) < 0.01,
            f"expected={expected} got={got}",
        )
    for factor, expected in expected_weighted.items():
        got = r.weighted_scores.get(factor)
        _report(
            f"weighted score {factor!r}",
            got is not None and abs(got - expected) < 0.01,
            f"expected={expected} got={got}",
        )


def test_rating_bands() -> None:
    for score, expected in (
        (100.0, "Very High Impact"),
        (95.0, "Very High Impact"),
        (90.0, "Very High Impact"),
        (89.9, "High Impact"),
        (75.0, "High Impact"),
        (74.9, "Medium Impact"),
        (60.0, "Medium Impact"),
        (59.9, "Low Impact"),
        (40.0, "Low Impact"),
        (39.9, "Minimal Impact"),
        (0.0, "Minimal Impact"),
    ):
        got = impact_rating(score)
        _report(f"impact_rating({score}) = {expected!r}", got == expected, f"got={got!r}")


def test_priority_formula() -> None:
    for impact, readiness, expected in (
        (80.0, 60.0, 32.0),        # spec example
        (100.0, 0.0, 100.0),       # max impact, zero readiness
        (0.0, 100.0, 0.0),         # zero impact
        (50.0, 50.0, 25.0),        # 50 x 50 / 100
        (76.0, 76.0, 18.24),       # from the readiness demo
    ):
        got = priority_score(impact, readiness)
        _report(
            f"priority({impact}, {readiness}) = {expected}",
            abs(got - expected) < 0.01,
            f"got={got}",
        )


def test_empty_inputs_produce_baseline() -> None:
    """No data -> the weighted computation returns a low, safe baseline."""
    r = compute_weighted_impact()
    _report(
        "empty inputs -> impact score in [0, 100]",
        0.0 <= r.overall_impact_score <= 100.0,
        f"got={r.overall_impact_score}",
    )
    _report(
        "empty inputs -> 7 factor rows still emitted",
        len(r.factor_details) == len(DORA_IMPACT_FACTOR_WEIGHTS),
    )
    _report(
        "empty inputs -> priority_areas is a list",
        isinstance(r.priority_areas, list),
    )


def test_priority_area_ranking() -> None:
    """High-impact / low-readiness areas should rank above low-impact ones."""
    obligations = [
        {"obligation_id": "OBL-1", "impacted_area": "Front Office", "priority": "Must"},
        {"obligation_id": "OBL-2", "impacted_area": "Front Office", "priority": "Must"},
        {"obligation_id": "OBL-3", "impacted_area": "Front Office", "priority": "Should"},
        {"obligation_id": "OBL-4", "impacted_area": "HR", "priority": "Could"},
    ]

    class _AnalysisStub:
        def __init__(self):
            self.obligations = obligations

    r = compute_weighted_impact(
        analysis=_AnalysisStub(),
        area_readiness={"Front Office": 20.0, "HR": 90.0},
    )
    _report(
        "priority_areas returns >= 2 rows",
        len(r.priority_areas) >= 2,
        f"got={len(r.priority_areas)}",
    )
    top = r.priority_areas[0]
    _report(
        "highest-priority area is Front Office (high impact + low readiness)",
        top.area == "Front Office",
        f"got={top.area}",
    )
    _report(
        "Front Office priority > HR priority",
        r.priority_areas[0].priority_score > r.priority_areas[-1].priority_score,
        f"top={r.priority_areas[0].priority_score} tail={r.priority_areas[-1].priority_score}",
    )


def main() -> int:
    print("== services.impact_score smoke tests ==")
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
    print("[priority formula]")
    test_priority_formula()
    print()
    print("[edge cases]")
    test_empty_inputs_produce_baseline()
    test_priority_area_ranking()
    print()
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
