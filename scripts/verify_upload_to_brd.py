"""End-to-end verification: uploaded regulation -> generated BRD.

Runs the exact server-side flow that Page 2 (Generate BRD / FRD) executes
when the user is in 'Generate BRD/FRD from regulation' mode and clicks
the primary CTA:

  1. Load the most recent regulation row from the SQLite `documents` table
     (i.e. the file the user just uploaded via the Setup page).
  2. Parse it into a ParsedDocument (services.document_parser).
  3. Run Agent 1 (RegulatoryWorkflowOrchestrator.run_regulatory_analysis).
  4. Run Agent 2 (RegulatoryWorkflowOrchestrator.run_brd_rtm), exporting
     the BRD to a .docx under `outputs/`.
  5. Print a compact summary and verify the .docx exists on disk.

Exit codes:
  0 -- BRD generated successfully.
  2 -- no regulation upload found in the DB (nothing to test).
  3 -- parse_document failed.
  4 -- run_regulatory_analysis failed.
  5 -- run_brd_rtm failed (or the resulting .docx is missing).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import persistence as db  # noqa: E402
from orchestrator import RegulatoryWorkflowOrchestrator  # noqa: E402
from utils.file_utils import timestamped_name  # noqa: E402


def main() -> int:
    rows = db.list_documents(kind="regulation")
    if not rows:
        print("FAIL: no regulation documents in DB. Upload one via Page 1 first.")
        return 2

    reg = rows[0]
    reg_path = Path(reg["path"])
    print(f"[1/4] Using regulation doc id={reg['id']} name={reg['name']!r}")
    print(f"       path={reg_path}")
    print(f"       exists={reg_path.exists()} size={reg_path.stat().st_size if reg_path.exists() else 'n/a'} bytes")
    if not reg_path.exists():
        print("FAIL: uploaded regulation file is missing on disk.")
        return 3

    orch = RegulatoryWorkflowOrchestrator()

    try:
        parsed = orch.parse_document(reg_path, kind="regulation")
    except Exception as exc:
        print(f"FAIL: parse_document raised {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 3
    if parsed.warning_message:
        print(f"       parser warning: {parsed.warning_message}")
    text_len = len(parsed.text or "")
    print(f"[2/4] Parsed OK. extracted_text_chars={text_len} pages={parsed.page_count}")
    if text_len == 0:
        print("FAIL: parser returned zero characters; downstream steps would fall back to synthetic content.")
        return 3

    try:
        analysis = orch.run_regulatory_analysis(
            parsed_document=parsed,
            regulation=reg.get("regulation") or "DORA",
            tier="Tier-2",
            status=lambda msg: print(f"       [analysis] {msg}"),
            regulator_selection=None,
            consulting_selection=None,
            include_consulting_guidance=False,
            intelligence_package=None,
        )
    except Exception as exc:
        print(f"FAIL: run_regulatory_analysis raised {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 4
    print(f"[3/4] Agent 1 OK. obligations={len(analysis.obligations)}")

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    docx_path = output_dir / timestamped_name(
        f"{reg.get('regulation') or 'REG'}_BRD_FRD_verify", ".docx",
    )

    try:
        bundle = orch.run_brd_rtm(
            analysis,
            docx_export_path=docx_path,
            tier="Tier-2",
        )
    except Exception as exc:
        print(f"FAIL: run_brd_rtm raised {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 5

    brd = bundle.get("brd")
    rtm = bundle.get("rtm")
    if brd is None:
        print("FAIL: run_brd_rtm returned no BRD artifact.")
        return 5

    if not docx_path.exists() or docx_path.stat().st_size == 0:
        print(f"FAIL: BRD DOCX not written to {docx_path}")
        return 5

    section_counts = (brd.metadata or {}).get("section_counts") or {}
    total_reqs = sum(int(section_counts.get(k, 0)) for k in (
        "process_requirements", "data_requirements", "reporting_requirements",
        "functional_requirements", "non_functional_requirements",
    ))
    print(f"[4/4] Agent 2 OK.")
    print(f"       brd_source        : {brd.source}")
    print(f"       total_requirements: {total_reqs}")
    print(f"       rtm_entries       : {len(rtm.entries) if rtm is not None else 'n/a'}")
    print(f"       docx_path         : {docx_path}")
    print(f"       docx_size_bytes   : {docx_path.stat().st_size}")

    used_genai = (brd.metadata or {}).get("used_genai_shared_service")
    print(f"       used_genai        : {used_genai}")

    print("\nPASS: regulation upload -> BRD generation flow works end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
