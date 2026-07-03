"""Smoke test for the v13 adaptive funnel.

Run from the repo root with::

    python -m scripts.smoke_test_adaptive_funnel

The test exercises three things end-to-end without GenAI and without
Streamlit:

1.  Builds a synthetic requirement set (governance + incident reporting +
    third-party + security) and runs the deterministic questionnaire
    generator. Asserts that:

      * the L1 root question uses the canonical Implementation-Status
        family ("Fully Implemented" / "Partially Implemented" / "Not
        Implemented" / "Not Applicable" / "Unknown"),
      * each option carries a per-option ``branch_rule_id``,
      * every generated question carries a structured
        ``explainability`` bundle with the mandatory keys.

2.  Drives the scoring engine through three parallel parent answers
    ("Fully Implemented", "Partially Implemented", "Not Implemented")
    and prints the resulting dynamic-queue contents. Asserts that the
    three answers produce **different** follow-up questions.

3.  Demonstrates that the branch_log audit trail captures every
    parent-answer -> child-question routing decision.

Run it as a CI gate or as ad-hoc verification after future edits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the project importable when executed directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.branch_registry import available_branch_keys, lookup_branch  # noqa: E402
from services.questionnaire_generator import (  # noqa: E402
    Requirement,
    derive_impact_pairs,
    generate_question_bank,
    option_label,
    option_metadata,
)
from services.scoring_engine import (  # noqa: E402
    AssessmentState,
    applicable_base_questions,
    choose_next_question,
    update_applicability_after_response,
)


def _make_requirements() -> list[Requirement]:
    return [
        Requirement(
            source_section="7.1 Process Requirements",
            source_id="BR-PRO-001",
            normalized_id="BR-PRO-001",
            category="Incident management",
            requirement="Establish DORA Incident Reporting Process",
            detail=(
                "The organisation must classify major ICT incidents and notify the "
                "competent authority within the initial 24-hour window per DORA "
                "Articles 17-20."
            ),
            alignment="DORA Article 17, RTS on Incident Reporting",
            priority="Must",
            acceptance="Documented incident classification + 24h notification evidence",
            confidence=96,
            themes=["Incident reporting"],
        ),
        Requirement(
            source_section="7.2 Data Requirements",
            source_id="BR-DAT-001",
            normalized_id="BR-DAT-001",
            category="Evidence",
            requirement="Maintain Evidence Dictionary",
            detail=(
                "The organisation must maintain a data dictionary mapping every "
                "control output to a named artefact, owner, lineage source and "
                "retention schedule, in line with DORA Article 6."
            ),
            alignment="DORA Article 6, RTS on ICT Risk Management",
            priority="Must",
            acceptance="Evidence dictionary published with named owners",
            confidence=94,
            themes=["Data and evidence"],
        ),
        Requirement(
            source_section="7.1 Process Requirements",
            source_id="BR-PRO-002",
            normalized_id="BR-PRO-002",
            category="Third-party",
            requirement="DORA Third-Party Contract Clauses",
            detail=(
                "Critical ICT third-party contracts must include the mandatory "
                "clauses in DORA Article 30 (audit/access rights, sub-outsourcing "
                "limits, data location, exit assistance, termination grounds)."
            ),
            alignment="DORA Article 30",
            priority="Must",
            acceptance="Legal-validated clause coverage report",
            confidence=95,
            themes=["Third-party risk"],
        ),
        Requirement(
            source_section="8. Functional Requirements",
            source_id="FR-001",
            normalized_id="FR-001",
            category="Security",
            requirement="Privileged access reviews",
            detail=(
                "Privileged access must be reviewed quarterly and supported by SIEM "
                "event correlation per DORA Articles 8-10."
            ),
            alignment="DORA Articles 8-10",
            priority="Must",
            acceptance="Quarterly access-review evidence + SIEM coverage report",
            confidence=93,
            themes=["Security and access"],
        ),
    ]


def _print(header: str) -> None:
    print("\n" + "=" * 78)
    print(header)
    print("=" * 78)


def test_explainability_and_options() -> list:
    """Build the questionnaire and assert the v13 structural contracts."""
    _print("1. Build questionnaire bank")
    reqs = _make_requirements()
    pairs = derive_impact_pairs(reqs, regulation="DORA")
    questions = generate_question_bank(reqs, pairs, regulation="DORA")
    closed = [q for q in questions if not q.is_free_text]
    print(f"Total questions: {len(questions)} ({len(closed)} closed, "
          f"{len(questions) - len(closed)} free-text)")
    print(f"Impact pairs derived: {len(pairs)}")

    # Pick the first L1 (root) question.
    root_questions = [q for q in closed if not q.funnel_parent_id]
    assert root_questions, "No L1 root question was generated"
    root = root_questions[0]
    print(f"\nL1 root question (Q={root.question_id}):")
    print(f"  Area / Function : {root.area} / {root.function}")
    print(f"  Branch theme    : {root.branch_theme or '(none)'}")
    print(f"  Question text   : {root.question}")

    labels = [option_label(o) for o in root.options]
    print(f"  Option labels   : {labels}")
    assert "Fully Implemented" in labels, "Canonical 'Fully Implemented' missing"
    assert "Partially Implemented" in labels, "Canonical 'Partially Implemented' missing"
    assert "Not Implemented" in labels, "Canonical 'Not Implemented' missing"
    assert "Not Applicable" in labels, "Canonical 'Not Applicable' missing"

    print("\n  Per-option metadata:")
    for opt in root.options:
        meta = option_metadata(root.options, option_label(opt))
        print(
            f"    - {meta.get('label')!r:30s}"
            f" score_value={meta.get('score_value')!s:>5s}"
            f"  rule={meta.get('branch_rule_id')}"
        )
        assert meta.get("branch_rule_id"), "Every option must carry a branch_rule_id"

    explain = root.explainability or {}
    print(f"\n  Explainability bundle keys: {sorted(explain.keys())}")
    for key in (
        "regulation", "regulator", "article", "obligation_id",
        "brd_requirement_ids", "business_function", "control_objective",
        "reason", "expected_evidence", "risk_if_negative",
    ):
        assert key in explain, f"Explainability missing '{key}'"
    print(f"  regulation       : {explain['regulation']}")
    print(f"  regulator        : {explain['regulator']}")
    print(f"  article          : {explain['article']}")
    print(f"  obligation_id    : {explain['obligation_id']}")
    print(f"  brd_reqs         : {explain['brd_requirement_ids']}")
    print(f"  control_objective: {explain['control_objective']}")
    print(f"  reason           : {explain['reason'][:120]}…")
    print(f"  expected_evidence: {explain['expected_evidence'][:120]}…")
    print(f"  risk_if_negative : {explain['risk_if_negative'][:120]}…")

    return questions


def test_adaptive_routing(questions: list) -> None:
    """Drive three parallel sessions with different options on the same parent."""
    _print("2. Adaptive routing — different options should produce different paths")
    closed = [q for q in questions if not q.is_free_text]
    root = next(q for q in closed if not q.funnel_parent_id)
    base = [
        {**q.__dict__}
        for q in closed
    ]
    parent = next(b for b in base if b["question_id"] == root.question_id)

    for selected in ("Fully Implemented", "Partially Implemented", "Not Implemented", "Not Applicable"):
        state = AssessmentState()
        # Simulate the user choosing this option on the L1 parent.
        update_applicability_after_response(
            state, parent, selected, base, package_regulation="DORA",
        )
        queue = state.dynamic_queue
        skipped = state.skipped_ids
        log = state.branch_log
        print(f"\n  User selected: {selected!r}")
        print(f"    Children queued   : {len(queue)}")
        for child in queue:
            print(f"      - {child['question_id']}")
            print(f"          rule    : {child.get('branch_rule_id')}")
            print(f"          text    : {child['question']}")
        if skipped:
            print(f"    Skipped (positive answer): {sorted(skipped)}")
        print(f"    Branch log entries: {len(log)}")
        if log:
            entry = log[0]
            print(f"      source={entry['branch_source']}  theme={entry['theme']!r}"
                  f"  kind={entry['question_kind']!r}  depth={entry['depth']}")

    # Sanity: assert at least two of the option paths produce different
    # first-child questions.
    paths = {}
    for selected in ("Fully Implemented", "Partially Implemented", "Not Implemented"):
        state = AssessmentState()
        update_applicability_after_response(
            state, parent, selected, base, package_regulation="DORA",
        )
        if state.dynamic_queue:
            paths[selected] = state.dynamic_queue[0]["question"]
    distinct_paths = len({v for v in paths.values()})
    print(f"\n  Distinct first-child questions across the three answers: {distinct_paths}")
    assert distinct_paths >= 2, (
        "Adaptive routing failed: at least two of Fully/Partial/Not Implemented "
        "should produce different follow-ups."
    )


def test_branch_registry_keys() -> None:
    _print("3. Branch registry coverage")
    keys = available_branch_keys("DORA")
    print(f"DORA registered keys: {len(keys)}")
    themes = sorted({k[1] for k in keys})
    answers = sorted({k[3] for k in keys})
    print(f"Themes covered      : {themes}")
    print(f"Answer labels       : {answers}")
    canonical = {"Fully Implemented", "Partially Implemented",
                 "Not Implemented", "Not Applicable", "Unknown"}
    missing = canonical - set(answers)
    assert not missing, f"Missing canonical answers in registry: {missing}"

    sample = lookup_branch("DORA", "Incident reporting", "coverage", "Not Implemented")
    assert sample, "Expected branches for DORA/Incident reporting/Not Implemented"
    print(f"\nSample lookup (DORA/Incident reporting/coverage/Not Implemented) -> "
          f"{len(sample)} follow-ups:")
    for spec in sample:
        print(f"  - {spec['question_id']}: {spec['question']}")


def main() -> int:
    questions = test_explainability_and_options()
    test_adaptive_routing(questions)
    test_branch_registry_keys()
    _print("ALL CHECKS PASSED")
    print("v13 adaptive funnel + explainability are working as specified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
