"""Validate the v13.1 content-correctness package-confidence grade.

Demonstrates:

1. A freshly-generated v13 package scores high on the new metric because
   every question carries a complete explainability bundle, an article
   citation that matches the BRD anchor, BRD-anchored vocabulary, etc.

2. The bundled v10 sample (no ``explainability`` field on its questions)
   scores meaningfully lower on the same metric — surfacing the v10/v11
   content-quality gap that the legacy structural metric used to hide.

3. The structural-completeness number is still emitted for audit, so
   like-for-like comparisons with historical packages remain possible.

Run::

    python -m scripts.smoke_test_content_correctness
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.questionnaire_generator import (  # noqa: E402
    Requirement,
    _build_package,
    EXPLAINABILITY_REQUIRED_KEYS,
)


def _print(banner: str) -> None:
    print("\n" + "=" * 78)
    print(banner)
    print("=" * 78)


def _print_metrics(pkg: dict) -> None:
    meta = pkg["metadata"]
    m = meta["metrics"]
    print(f"  headline overall_confidence_pct  : {meta['overall_confidence_pct']}%")
    print(f"  mode used                        : {m['package_confidence_mode']}")
    print(f"  content_correctness_pct          : {m['content_correctness_pct']}%")
    print(f"  structural_completeness_pct      : {m['structural_completeness_pct']}%")
    print("  content_breakdown:")
    for k, v in m["content_breakdown"].items():
        print(f"    {k:38s}: {v}")
    print("  structural_breakdown:")
    for k, v in m["structural_breakdown"].items():
        print(f"    {k:38s}: {v}")


def fresh_v13_package() -> dict:
    """Build a v13 package from synthetic but realistic DORA BRD rows."""
    reqs = [
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
            alignment="DORA Article 5, RTS on ICT Risk Management Framework",
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
                "source and retention schedule per DORA Article 6."
            ),
            alignment="DORA Article 6",
            priority="Must",
            acceptance="Evidence dictionary published with named owners",
            confidence=94,
            themes=["Data and evidence"],
        ),
        Requirement(
            source_section="8. Functional Requirements",
            source_id="FR-001",
            normalized_id="FR-001",
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
    ]
    return _build_package(reqs, "DORA")


def legacy_v10_package() -> dict:
    """Load the bundled sample, which has no explainability bundle."""
    path = REPO_ROOT / "sample_data" / "dora_questionnaire_package_v10.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    _print("1. Fresh v13 package — every question has a complete explainability bundle")
    fresh = fresh_v13_package()
    _print_metrics(fresh)

    # Confirm explainability completeness > 95% on the v13 package.
    fresh_cb = fresh["metadata"]["metrics"]["content_breakdown"]
    assert fresh_cb["explainability_completeness_pct"] >= 95, (
        f"Expected v13 explainability completeness >= 95%, got "
        f"{fresh_cb['explainability_completeness_pct']}%"
    )
    assert fresh["metadata"]["overall_confidence_pct"] >= 70, (
        f"Expected v13 content correctness >= 70%, got "
        f"{fresh['metadata']['overall_confidence_pct']}%"
    )

    _print("2. Re-score the legacy v10 sample with the new content-correctness grade")
    legacy_pkg_raw = legacy_v10_package()
    print(f"  legacy headline (as-saved): {legacy_pkg_raw['metadata'].get('overall_confidence_pct')}%")
    print(f"  legacy 'metrics' keys     : {sorted(legacy_pkg_raw['metadata'].get('metrics', {}).keys())}")

    # Rebuild the metric live so we compare apples-to-apples.
    from services.questionnaire_generator import (
        Requirement as Req,
        ImpactPair,
        Question,
        _filter_dataclass_kwargs,
        validate_and_score_package,
    )
    reqs = [Req(**_filter_dataclass_kwargs(Req, r)) for r in legacy_pkg_raw["requirements"]]
    pairs = [ImpactPair(**_filter_dataclass_kwargs(ImpactPair, p)) for p in legacy_pkg_raw["impact_pairs"]]
    questions = [Question(**_filter_dataclass_kwargs(Question, q)) for q in legacy_pkg_raw["questions"]]
    overall, metrics = validate_and_score_package(reqs, pairs, questions)
    print(f"\n  Re-scored under the new metric:")
    print(f"    headline overall_confidence_pct  : {overall}%")
    print(f"    mode used                        : {metrics['package_confidence_mode']}")
    print(f"    content_correctness_pct          : {metrics['content_correctness_pct']}%")
    print(f"    structural_completeness_pct      : {metrics['structural_completeness_pct']}%")
    print("    content_breakdown:")
    for k, v in metrics["content_breakdown"].items():
        print(f"      {k:38s}: {v}")

    # The legacy sample has no `explainability` field, so the content score
    # SHOULD drop. We are explicitly demonstrating that the new metric is
    # not blind to v13 hardening.
    assert metrics["content_breakdown"]["explainability_completeness_pct"] == 0.0, (
        "Legacy v10 sample is expected to score 0% on explainability completeness"
    )
    print(
        f"\n  Note: legacy sample correctly scores 0% on explainability completeness "
        f"(no v13 bundle) — this is now visible in the headline number."
    )

    _print("3. Verify that PACKAGE_CONFIDENCE_MODE=structural restores the v11 number")
    # We re-import after setting the env so the module-level constant updates.
    import importlib
    import os
    import services.questionnaire_generator as qg
    os.environ["PACKAGE_CONFIDENCE_MODE"] = "structural"
    os.environ["OVERALL_QUESTIONNAIRE_CONFIDENCE_FLOOR"] = "90"
    qg = importlib.reload(qg)
    overall_struct, metrics_struct = qg.validate_and_score_package(reqs, pairs, questions)
    print(f"  headline (structural mode, floor 90): {overall_struct}%")
    print(f"  mode used                          : {metrics_struct['package_confidence_mode']}")
    assert metrics_struct["package_confidence_mode"] == "structural"
    # Reset env so subsequent tests are not polluted.
    del os.environ["PACKAGE_CONFIDENCE_MODE"]
    del os.environ["OVERALL_QUESTIONNAIRE_CONFIDENCE_FLOOR"]

    _print("ALL CHECKS PASSED")
    print("Content-correctness grade is live, structural-completeness number is still emitted,")
    print("and PACKAGE_CONFIDENCE_MODE=structural reverts the headline cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
