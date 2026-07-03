"""End-to-end UI-path validation for the v13 adaptive funnel.

Drives the exact same helpers that ``app.py`` calls inside the Page 4
assessment cockpit — so a green run here is a strong signal that the
Streamlit UI will render the adaptive funnel + explainability panel
correctly.

Run::

    python -m scripts.smoke_test_ui_path
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.questionnaire_generator import (  # noqa: E402
    Requirement,
    _build_package,
    option_label,
    option_metadata,
)
from services.scoring_engine import (  # noqa: E402
    AssessmentState,
    choose_next_question,
    rationale_text,
    update_applicability_after_response,
)


def _make_reqs() -> list[Requirement]:
    return [
        Requirement(
            source_section="7.1 Process Requirements",
            source_id="BR-PRO-001",
            normalized_id="BR-PRO-001",
            category="Governance",
            requirement="Establish ICT Risk Management Framework",
            detail=(
                "The management body must approve the ICT risk-management framework "
                "annually per DORA Article 5."
            ),
            alignment="DORA Article 5",
            priority="Must",
            acceptance="Approved governance pack with traceable evidence",
            confidence=96,
            themes=["Governance"],
        ),
        Requirement(
            source_section="7.1 Process Requirements",
            source_id="BR-PRO-002",
            normalized_id="BR-PRO-002",
            category="Incident management",
            requirement="Establish DORA Incident Reporting Process",
            detail=(
                "Classify major ICT incidents and notify the competent authority "
                "within 24h per DORA Articles 17-20."
            ),
            alignment="DORA Article 17, RTS on Incident Reporting",
            priority="Must",
            acceptance="Documented + tested with regulator submission evidence",
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
                "Map every control output to a named artefact, owner, lineage "
                "source and retention schedule."
            ),
            alignment="DORA Article 6",
            priority="Must",
            acceptance="Evidence dictionary published with named owners",
            confidence=94,
            themes=["Data and evidence"],
        ),
    ]


def main() -> int:
    pkg = _build_package(_make_reqs(), "DORA")
    base_questions = [dict(q) for q in pkg["questions"]]
    state = AssessmentState()

    print("=" * 76)
    print("STEP 1 — choose_next_question (same call the UI cockpit makes)")
    print("=" * 76)
    q = choose_next_question(state, base_questions, focus_area="All")
    assert q is not None, "choose_next_question returned None"
    print(f"Picked        : {q['question_id']}")
    print(f"Area / Function: {q.get('area')} / {q.get('function')}")
    print(f"Question text : {q['question']}")

    print("\nOption metadata (the UI reads these via option_label/option_metadata):")
    for o in q["options"]:
        label = option_label(o)
        meta = option_metadata(q["options"], label)
        score = meta.get("score_value", "n/a")
        rule = meta.get("branch_rule_id", "(none)")
        print(f"  - {label!r:28s}  score={str(score):>5s}  rule={rule}")

    print("\nExplainability bundle (drives 'Why am I being asked this?' panel):")
    explain = q.get("explainability") or {}
    fields = [
        "regulation", "regulator", "article", "obligation_id",
        "business_function", "business_area", "control_objective", "theme",
    ]
    for k in fields:
        print(f"  {k:18s}: {explain.get(k)}")
    brd_ids = explain.get("brd_requirement_ids", [])
    rtm_ids = explain.get("rtm_trace_ids", [])
    print(f"  brd_requirement_ids: {brd_ids}")
    print(f"  rtm_trace_ids      : {rtm_ids}")
    print(f"  reason             : {(explain.get('reason') or '')[:140]}...")
    print(f"  expected_evidence  : {explain.get('expected_evidence')}")
    print(f"  risk_if_negative   : {explain.get('risk_if_negative')}")

    print()
    print("=" * 76)
    print("STEP 2 — Submit 'Partially Implemented' via update_applicability_after_response")
    print("=" * 76)
    update_applicability_after_response(
        state, q, "Partially Implemented", base_questions, package_regulation="DORA",
    )
    print(f"Dynamic queue : {len(state.dynamic_queue)} item(s)")
    print(f"Branch log    : {len(state.branch_log)} entry/entries")
    for entry in state.branch_log:
        print(
            f"  source={entry['branch_source']:>9s}"
            f"  rule={entry['branch_rule_id']}"
        )
    print("Next children queued:")
    for c in state.dynamic_queue:
        print(f"  - {c['question_id']}: {c['question']}")

    print()
    print("=" * 76)
    print("STEP 3 — choose_next_question again should return the first child")
    print("=" * 76)
    nxt = choose_next_question(state, base_questions, focus_area="All")
    assert nxt is not None and nxt.get("dynamic"), (
        "Expected the next question to be the dynamic child"
    )
    print(f"Picked          : {nxt['question_id']}")
    print(f"Is dynamic?     : {nxt.get('dynamic')}")
    print(f"branch_rule_id  : {nxt.get('branch_rule_id')}")
    print(f"Inherited explain? : {bool(nxt.get('explainability'))}")
    rat = rationale_text(nxt, state.responses)
    print(f"rationale_text(): {rat[:200]}...")

    print()
    print("=" * 76)
    print("STEP 4 — Verify the three answer paths produce distinct routes")
    print("=" * 76)
    paths: dict[str, str] = {}
    for answer in ("Fully Implemented", "Partially Implemented", "Not Implemented", "Not Applicable"):
        fresh = AssessmentState()
        update_applicability_after_response(
            fresh, q, answer, base_questions, package_regulation="DORA",
        )
        first_child_text = fresh.dynamic_queue[0]["question"] if fresh.dynamic_queue else "(none)"
        paths[answer] = first_child_text
        print(f"  {answer!r:24s} -> {first_child_text}")
    distinct = len({v for v in paths.values()})
    print(f"\nDistinct first-child questions: {distinct} / {len(paths)}")
    assert distinct >= 3, (
        f"Expected at least 3 distinct paths, got {distinct}. paths={paths}"
    )

    print()
    print("=" * 76)
    print("ALL UI-PATH CHECKS PASSED — Streamlit cockpit will render adaptive funnel.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
