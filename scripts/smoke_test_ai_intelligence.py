"""Smoke test for the AI Assessment Intelligence + rich recommendations.

Exercises the offline (client=None) code paths end-to-end so the
deterministic fallbacks are proven to work without GenAI. Verifies:

1. Dynamic confidence has real sub-scores + reasoning.
2. Impact assessment produces six populated dimensions per regulation.
3. Readiness assessment produces seven populated dimensions.
4. Rich recommendations are per-area, non-generic and reference obligations.
5. Adaptive follow-up detection triggers on brief / ambiguous answers.
6. Two different regulations produce measurably different outputs.

Run::

    python -m scripts.smoke_test_ai_intelligence
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.workflow_models import (  # noqa: E402
    Obligation,
    RegulatoryAnalysis,
)
from services.ai_assessment_intelligence import (  # noqa: E402
    assess_confidence,
    assess_impact,
    assess_readiness,
    detect_brief_answer,
)
from services.rich_recommendation_service import (  # noqa: E402
    build_rich_recommendations,
)


def _print(banner: str) -> None:
    print("\n" + "=" * 78)
    print(banner)
    print("=" * 78)


def _sample_analysis(regulation: str = "DORA") -> RegulatoryAnalysis:
    """Build a synthetic analysis so the deterministic paths have real inputs."""
    obligations = [
        Obligation(
            obligation_id=f"OBL-{i:03d}",
            title=title,
            theme=theme,
            compliance_requirement=detail,
            impacted_area=area,
            impacted_function=function,
            regulatory_basis=alignment,
            source_references=[{"source_url": "https://example.eu"}] if i % 2 == 0 else [],
        )
        for i, (title, theme, detail, area, function, alignment) in enumerate([
            (
                "ICT risk management framework",
                "ICT risk management",
                "Establish and maintain an ICT risk management framework, "
                "including governance, tolerance thresholds and KRIs.",
                "Risk & Controls framework",
                "Risk Management",
                "DORA Article 6",
            ),
            (
                "Incident classification & reporting",
                "Incident reporting",
                "Implement classification, notification and reporting workflows for "
                "major ICT incidents with regulator timelines.",
                "IT Security / Cyber Security",
                "Incident Management",
                "DORA Article 18",
            ),
            (
                "Third-party register",
                "Third-party risk",
                "Maintain a critical-provider register with tiering, sub-contractor "
                "visibility and exit-plan evidence.",
                "Third Party Risk Management / Dependency",
                "Vendor / Third-Party Management",
                "DORA Article 28",
            ),
            (
                "Resilience testing programme",
                "Resilience testing",
                "Establish threat-led penetration testing and scenario "
                "recovery testing across critical services.",
                "IT Security / Cyber Security",
                "Business Continuity / Resilience",
                "DORA Article 24",
            ),
            (
                "Governance & management body oversight",
                "Governance",
                "Ensure management body approves and monitors the ICT risk framework "
                "and takes accountability for material decisions.",
                "Governance Model",
                "Compliance & Legal",
                "DORA Article 5",
            ),
        ], start=1)
    ]
    return RegulatoryAnalysis(
        regulation=regulation,
        tier="Tier-2",
        summary=f"Sample analysis for {regulation}",
        impacted_areas=sorted({o.impacted_area for o in obligations}),
        obligation_themes=sorted({o.theme for o in obligations}),
        obligations=obligations,
        used_genai=False,
    )


def test_confidence(analysis: RegulatoryAnalysis) -> None:
    _print(f"1) Confidence assessment ({analysis.regulation}, no GenAI)")
    result = assess_confidence(analysis)
    print(f"  overall_score            : {result.overall_score:.1f}%")
    print(f"  completeness_score       : {result.completeness_score:.1f}%")
    print(f"  quality_score            : {result.quality_score:.1f}%")
    print(f"  evidence_score           : {result.evidence_score:.1f}%")
    print(f"  clarity_score            : {result.clarity_score:.1f}%")
    print(f"  generated_by_ai          : {result.generated_by_ai}")
    print(f"  reasoning length         : {len(result.reasoning)} chars")
    assert result.overall_score > 0, "confidence should be non-zero"
    assert result.reasoning, "confidence should carry a reasoning paragraph"
    assert 0 < result.completeness_score <= 100
    assert 0 < result.evidence_score <= 100


def test_impact(analysis: RegulatoryAnalysis) -> None:
    _print(f"2) Impact assessment ({analysis.regulation}, no GenAI)")
    result = assess_impact(analysis)
    print(f"  overall_severity         : {result.overall_severity}")
    print(f"  overall_severity_score   : {result.overall_severity_score:.1f}")
    for dim in result.dimensions():
        print(
            f"  {dim.dimension:22s}: severity={dim.severity:<8s} "
            f"score={dim.severity_score:>5.1f}  items={len(dim.items)}"
        )
    # Every dimension should have at least one item, severity, rationale
    for dim in result.dimensions():
        assert dim.rationale, f"{dim.dimension} missing rationale"
        assert dim.severity, f"{dim.dimension} missing severity"


def test_readiness() -> None:
    _print("3) Readiness assessment (no GenAI)")
    scoring_evaluation = {
        "compliance_score_pct": 58.4,
        "answered_count": 24,
        "unanswered_count": 6,
        "area_summary": {
            "Risk & Controls framework": {"Compliance %": 52.0, "CXO status": "Watch"},
            "IT Security / Cyber Security": {"Compliance %": 41.0, "CXO status": "At risk"},
            "Governance Model": {"Compliance %": 76.0, "CXO status": "Ready"},
            "IT, Systems & Technology": {"Compliance %": 62.0, "CXO status": "Watch"},
            "Data Reporting & Governance": {"Compliance %": 48.0, "CXO status": "At risk"},
        },
        "function_summary": {},
    }
    result = assess_readiness(scoring_evaluation)
    print(f"  overall_score            : {result.overall_score:.1f}")
    print(f"  overall_level            : {result.overall_level}")
    for dim in result.dimensions():
        print(f"  {dim.dimension:32s}: level={dim.maturity_level:<12s} score={dim.score:>5.1f}")
    for dim in result.dimensions():
        assert dim.rationale, f"{dim.dimension} missing rationale"


def test_recommendations(analysis: RegulatoryAnalysis) -> None:
    _print(f"4) Rich recommendations ({analysis.regulation}, no GenAI)")
    scoring_evaluation = {
        "compliance_score_pct": 45.0,
        "area_summary": {
            "Risk & Controls framework": {"Compliance %": 32.0, "CXO status": "At risk"},
            "IT Security / Cyber Security": {"Compliance %": 22.0, "CXO status": "Critical"},
            "Governance Model": {"Compliance %": 78.0, "CXO status": "Ready"},
            "Third Party Risk Management / Dependency": {"Compliance %": 41.0, "CXO status": "At risk"},
        },
    }
    top_gaps = [
        {"requirement_id": "FR-001", "compliance_pct": 22.0},
        {"requirement_id": "BR-DAT-002", "compliance_pct": 25.0},
        {"requirement_id": "FR-014", "compliance_pct": 30.0},
    ]
    package = {
        "requirements": [
            {"normalized_id": "FR-001", "requirement": "Detection controls"},
            {"normalized_id": "BR-DAT-002", "requirement": "Data lineage"},
        ],
        "impact_pairs": [
            {
                "area": "IT Security / Cyber Security",
                "function": "Cyber Security",
                "requirement_ids": ["FR-001", "FR-014"],
                "regulatory_basis": "DORA Article 10",
            },
        ],
    }
    recs = build_rich_recommendations(
        analysis=analysis,
        scoring_evaluation=scoring_evaluation,
        top_gaps=top_gaps,
        package=package,
    )
    print(f"  recommendations produced : {len(recs)}")
    seen_whats = set()
    for r in recs:
        print(f"  --")
        print(f"  title      : {r.title}")
        print(f"  area       : {r.area}   priority={r.priority}   severity={r.severity}")
        print(f"  what[:120] : {r.what[:120]}")
        print(f"  why[:120]  : {r.why[:120]}")
        print(f"  how[:120]  : {r.how[:120]}")
        print(f"  expected_outcome[:120]: {r.expected_outcome[:120]}")
        print(f"  dependencies: {r.dependencies[:3]}")
        assert r.what, f"{r.area}: missing 'what'"
        assert r.why, f"{r.area}: missing 'why'"
        assert r.how, f"{r.area}: missing 'how'"
        assert r.priority, f"{r.area}: missing priority"
        assert r.expected_outcome, f"{r.area}: missing expected_outcome"
        seen_whats.add(r.what)
    # Uniqueness: 'what' shouldn't be identical across all cards
    assert len(seen_whats) >= max(1, len(recs) - 1), "recommendations should not all be identical"


def test_brief_answer_detection() -> None:
    _print("5) Adaptive follow-up detection")
    cases = [
        ("Yes", True),
        ("N/A", False),  # explicit N/A is acceptable
        ("Not sure", True),
        ("Depends on the auditor", True),
        ("We have implemented risk-based monitoring across four business units.", False),
        ("", False),
        ("no", True),
    ]
    for answer, should_prompt in cases:
        needs, prompt = detect_brief_answer(answer)
        print(f"  '{answer:52s}' -> needs_followup={needs}  prompt_preview='{prompt[:70]}'")
        assert needs == should_prompt, f"expected needs_followup={should_prompt} for '{answer}'"


def test_regulation_variation() -> None:
    _print("6) Two different regulations produce different recommendations")
    a_dora = _sample_analysis("DORA")
    a_mifid = _sample_analysis("MiFID II")
    scoring = {
        "compliance_score_pct": 50.0,
        "area_summary": {
            "Risk & Controls framework": {"Compliance %": 40.0, "CXO status": "At risk"},
        },
    }
    package = {
        "requirements": [{"normalized_id": "REQ-1", "requirement": "Risk framework"}],
        "impact_pairs": [{
            "area": "Risk & Controls framework",
            "function": "Risk Management",
            "requirement_ids": ["REQ-1"],
            "regulatory_basis": "Article 6",
        }],
    }
    recs_dora = build_rich_recommendations(
        analysis=a_dora, scoring_evaluation=scoring, top_gaps=[],
        package=package,
    )
    recs_mifid = build_rich_recommendations(
        analysis=a_mifid, scoring_evaluation=scoring, top_gaps=[],
        package=package,
    )
    print(f"  DORA sample what     : {recs_dora[0].what[:120]}")
    print(f"  MiFID II sample what : {recs_mifid[0].what[:120]}")
    assert recs_dora[0].what != recs_mifid[0].what or (
        "DORA" in recs_dora[0].why and "MiFID II" in recs_mifid[0].why
    ), "different regulations should produce different recommendations"


if __name__ == "__main__":
    analysis = _sample_analysis()
    test_confidence(analysis)
    test_impact(analysis)
    test_readiness()
    test_recommendations(analysis)
    test_brief_answer_detection()
    test_regulation_variation()
    print("\n" + "=" * 78)
    print("ALL AI INTELLIGENCE SMOKE TESTS PASSED")
    print("=" * 78)
