# Regulatory Impact & Readiness Cockpit — Comprehensive Documentation

> Functional and technical reference grounded strictly in the source code of this repository (no inferred behaviour). When code does not implement something explicitly, the gap is called out in an "Unimplemented / Out-of-scope" callout.

---

## Table of Contents

1. [Executive Summary (Business)](#1-executive-summary-business)
2. [Solution Architecture](#2-solution-architecture)
3. [Folder & File Inventory](#3-folder--file-inventory)
4. [End-to-End Workflow (User Journey)](#4-end-to-end-workflow-user-journey)
5. [Module-by-Module Reference](#5-module-by-module-reference)
   - 5.1 [Models layer (`models/`)](#51-models-layer-models)
   - 5.2 [Utilities layer (`utils/`)](#52-utilities-layer-utils)
   - 5.3 [Services layer (`services/`)](#53-services-layer-services)
   - 5.4 [Agents layer (`agents/`)](#54-agents-layer-agents)
   - 5.5 [Orchestrator (`orchestrator.py`)](#55-orchestrator-orchestratorpy)
   - 5.6 [Streamlit UI (`app.py`)](#56-streamlit-ui-apppy)
6. [Logic, Algorithms and Calculations](#6-logic-algorithms-and-calculations)
   - 6.1 [Confidence scoring (BRD, questionnaire, question)](#61-confidence-scoring-brd-questionnaire-question)
   - 6.2 [Impact-pair derivation](#62-impact-pair-derivation)
   - 6.3 [Funnel question synthesis](#63-funnel-question-synthesis)
   - 6.4 [Adaptive branching (registry + generic)](#64-adaptive-branching-registry--generic)
   - 6.5 [Live scoring engine (readiness / compliance %)](#65-live-scoring-engine-readiness--compliance-)
   - 6.6 [CXO status thresholds & heatmap](#66-cxo-status-thresholds--heatmap)
   - 6.7 [Recommendation severity & ranking](#67-recommendation-severity--ranking)
   - 6.8 [Stage 1 / Stage 2 source ranking](#68-stage-1--stage-2-source-ranking)
7. [Decision Points (Every `if/else`, threshold, constant)](#7-decision-points-every-ifelse-threshold-constant)
8. [Database schema (SQLite)](#8-database-schema-sqlite)
9. [Configuration (`.env`)](#9-configuration-env)
10. [Export Surface & File Outputs](#10-export-surface--file-outputs)
11. [Known Limitations / Not Implemented](#11-known-limitations--not-implemented)

---

## 1. Executive Summary (Business)

The Regulatory Impact & Readiness Cockpit is a desktop / browser application that converts a piece of financial-services regulation (e.g. **DORA — Regulation (EU) 2022/2554**) into:

1. A **Business Requirements Document (BRD)** and **Functional Requirements Document (FRD)** — Word, JSON and CSV.
2. A **Requirements Traceability Matrix (RTM)** linking every regulatory obligation to a business requirement, a functional requirement, the impacted business area / function, and the evidence required.
3. A **closed-ended + open-ended questionnaire** that asks the right questions to score current readiness for that regulation.
4. A **live, adaptive assessment** that drives follow-up questions based on each answer.
5. A **readiness dashboard** with compliance %, evaluation confidence %, area / function / requirement heatmaps and "top gaps".
6. A list of **CXO-grade recommendations** ranked by severity, with owner suggestions, time horizons and branch-trace evidence.
7. Exportable artefacts (DOCX, JSON, CSV, Excel) for governance, audit and supervisory use.

It is delivered as a **Streamlit single-page app** with six tabs (Setup → BRD/FRD → Questionnaire → Assessment → Dashboard → Export) backed by an **agentic pipeline** of four explicit agents, an optional **PwC GenAI Shared Service** integration, a deterministic **offline fallback**, and a local **SQLite store** for documents, questionnaires, assessments and answers.

**Why this matters (plain English).** Without this tool, the same exercise would take a team of consultants several weeks of workshops, manual spreadsheet maintenance, manual web research, and bespoke evidence collection. The cockpit collapses that into a guided, auditable workflow that any analyst can re-run.

---

## 2. Solution Architecture

### 2.1 Logical architecture (textual diagram)

```
                          .env (config)
                              |
                              v
+-----------------+   +----------------------+   +-------------------------+
| Streamlit UI    |-->| Orchestrator         |-->| Agents (1..4)           |
| app.py (6 pages)|   | orchestrator.py      |   | agents/*.py             |
+-----------------+   +----------------------+   +-------------------------+
        |                       |                          |
        |                       v                          v
        |        +--------------------------+    +-----------------------+
        |        | Services layer           |<-->| Models layer          |
        |        | services/*.py            |    | models/workflow_models|
        |        +--------------------------+    +-----------------------+
        |                       |
        |                       v
        |        +---------------------------------------------+
        |        | Utilities                                   |
        |        | utils/pdf_parser, docx_parser, file_utils,  |
        |        | json_utils                                  |
        |        +---------------------------------------------+
        |                       |
        v                       v
+------------------+    +----------------------+    +-------------------+
| Local uploads/   |    | SQLite (data/app.db) |    | GenAI Shared Svc  |
| outputs/ folders |    | services/database.py |    | (PwC, HTTP+SSL)   |
+------------------+    +----------------------+    +-------------------+
                                                            |
                                                            v
                                       +-----------------------------------+
                                       | Regulatory Intelligence Pipeline  |
                                       | Stage 1: regulator domains        |
                                       | Stage 2: consulting firms (off)   |
                                       +-----------------------------------+
```

### 2.2 Pipeline (canonical sequence)

Defined in `orchestrator.py` and `models/workflow_models.py`:

```
Upload Regulation
   |  ParsedDocument                                            (services.document_parser)
   v
Agent 1 — Regulatory Analysis
   |  RegulatoryAnalysis (Obligation[], BRD report, metadata)   (agents.regulatory_analysis_agent)
   v
Agent 2 — BRD + RTM
   |  BRDArtifact + RTMArtifact                                 (agents.brd_rtm_agent)
   v
Agent 3 — Questionnaire Generation
   |  QuestionnairePackage                                      (agents.questionnaire_agent)
   v
User Responses                                                  (collected by Streamlit Page 4)
   v
Python Rules Engine
   |  ScoringResult                                             (services.scoring_engine)
   v
Agent 4 — Recommendations
   |  RecommendationResult                                      (agents.recommendation_agent)
   v
Dashboard + Export                                              (app.py Pages 5 & 6)
```

### 2.3 Why this architecture was chosen (per source comments)

- `orchestrator.py` docstring explicitly states each stage is **idempotent and independently testable**, and that `app.py` must call orchestrator methods (not reach into agents/services). This keeps UI changes from leaking into business logic.
- `services/genai_service.py` separates HTTP / SSL / LLM plumbing from BRD generation, so the BRD generator (Agent 1) can be unit-tested with mock clients.
- `services/search_config.py` is the **single source of truth for allowed web sources**. Adding/removing a regulator or consulting firm is a one-file change; no fetcher logic needs updating.
- `models/workflow_models.py` uses plain dataclasses with JSON-friendly fields so every artefact can be persisted via `services/persistence.py` or downloaded by the user without lossy schema conversions.

---

## 3. Folder & File Inventory

```
Reg_Impact/
├── .env                                 # Configuration (secrets, feature flags, tunables)
├── .gitignore                           # Excludes .env, outputs/, uploads/, *.db
├── docker-compose.yml                   # Optional sqlite-web container at http://localhost:8080
├── requirements.txt                     # Python dependencies
├── orchestrator.py                      # Single coordination object (RegulatoryWorkflowOrchestrator)
├── app.py                               # Streamlit UI (six pages + sidebar)
├── .streamlit/config.toml               # Streamlit theme / server defaults
│
├── models/
│   ├── __init__.py                      # Re-exports
│   └── workflow_models.py               # ParsedDocument, RegulatoryAnalysis, Obligation,
│                                        # BRDArtifact, RTMArtifact, RTMEntry,
│                                        # QuestionnairePackage, AssessmentResponse,
│                                        # ScoringResult, RecommendationResult
│
├── utils/
│   ├── __init__.py
│   ├── file_utils.py                    # ensure_dirs, safe_filename, save_upload,
│   │                                    # read_bytes, copy_into, timestamped_name, iter_files
│   ├── json_utils.py                    # read_json, write_json, read/write_package_json,
│   │                                    # validate_package_schema
│   ├── docx_parser.py                   # clean_text, normalise_header, iter_body_blocks,
│   │                                    # extract_paragraphs, extract_tables,
│   │                                    # iter_sectioned_tables, extract_full_text
│   └── pdf_parser.py                    # extract_pdf_pages, extract_pdf_text, PdfExtractionResult
│
├── services/
│   ├── __init__.py
│   ├── database.py                      # SQLite schema + CRUD (documents, requirements,
│   │                                    # questionnaires, assessments, responses)
│   ├── persistence.py                   # Thin re-export of database.py
│   ├── document_parser.py               # parse_pdf, parse_docx, parse_document
│   ├── genai_service.py                 # GenAIClient + httpx/SSL plumbing
│   ├── search_config.py                 # APPROVED_REGULATORS, APPROVED_CONSULTING_FIRMS,
│   │                                    # SOURCE_PRIORITY, PUBLICATION_TYPES,
│   │                                    # is_regulator_url, is_consulting_url, env getters
│   ├── native_regulator_search.py       # Direct HTTP search adapters (EBA, ESMA, EIOPA, FCA)
│   ├── official_regulation_fetcher.py   # Stage 1 of Regulatory Intelligence Pipeline
│   ├── consulting_guidance_fetcher.py   # Stage 2 (currently disabled by .env flag)
│   ├── regulatory_intelligence_service.py  # Coordinator (Stage 1 + Stage 2)
│   ├── brd_frd_generator.py             # Pydantic schemas + GenAI pipeline +
│   │                                    # offline fallback + ensure_minimum_detail +
│   │                                    # write_brd_docx + build_brd_frd_report
│   ├── questionnaire_generator.py       # Requirement extraction + impact-pair derivation
│   │                                    # + theme-aware question synthesis + Excel writer
│   ├── branch_registry.py               # Per-option, regulation-aware adaptive branches
│   ├── scoring_engine.py                # AssessmentState, applicable_base_questions,
│   │                                    # choose_next_question, evaluate, cxo_status,
│   │                                    # dynamic_followups, registry_followups
│   └── recommendation_service.py        # generate_recommendations (deterministic) +
│                                        # enrich_recommendations_with_genai (optional)
│
├── agents/
│   ├── __init__.py                      # Re-exports
│   ├── regulatory_analysis_agent.py     # Agent 1
│   ├── brd_rtm_agent.py                 # Agent 2
│   ├── questionnaire_agent.py           # Agent 3
│   └── recommendation_agent.py          # Agent 4
│
├── data/                                # SQLite DB (data/app.db) — gitignored
├── uploads/                             # User-uploaded PDFs/DOCX — gitignored
├── outputs/                             # Generated DOCX / JSON / Excel — gitignored
└── sample_data/
    ├── DORA_Tier2_Detailed_DetailedBRDFRD.docx   # Bundled sample BRD (used by Page 1)
    ├── DORA_Readiness_Questionnaire_v10.xlsx     # Bundled sample questionnaire
    └── dora_questionnaire_package_v10.json       # Bundled sample package
```

---

## 4. End-to-End Workflow (User Journey)

### 4.1 Page-by-page user journey (from `app.py`)

| Page | Title | What the user does | Code |
|------|-------|--------------------|------|
| 1 | Setup | Enter regulation + tier; optionally upload a regulation PDF/DOCX; choose "Use existing BRD/FRD" or "Generate BRD/FRD from regulation"; preview retrieved regulator sources; optionally load an existing questionnaire from SQLite | `render_setup_page` |
| 2 | BRD / FRD | If "Generate" mode: click **Run Agents 1 + 2** to call `orch.run_regulatory_analysis()` then `orch.run_brd_rtm()`. If "Use existing": click **Parse uploaded BRD** to extract requirements. View obligations, RTM, parsed requirements, provenance, and download DOCX/JSON/CSV | `render_brd_page`, `_run_agent1_and_agent2_with_status`, `_run_agent2_for_uploaded_brd` |
| 3 | Questionnaire | Click **Run Agent 3** (delegates to `orch.run_questionnaire_from_report()` or `orch.run_questionnaire_from_docx()`) or upload a saved package JSON | `render_questionnaire_page`, `_run_agent3` |
| 4 | Assessment | Start/continue an assessment row in SQLite, answer one question at a time via `choose_next_question`, see adaptive follow-ups generated, fill free-text questions in the expander | `render_assessment_page`, `_render_question_card`, `_render_free_text_questions` |
| 5 | Dashboard | View compliance %, evaluation confidence, area/function summaries, area×function heatmap, top gaps, and run **Agent 4** for recommendations | `render_dashboard_page` |
| 6 | Export | Download questionnaire JSON, responses JSON, obligations JSON, RTM JSON, Excel report, BRD/FRD DOCX | `render_export_page` |

### 4.2 Sidebar (always visible)

- Navigation radio (bound to `st.session_state["page"]`) — defined in `_render_sidebar`.
- GenAI Shared Service status pill (`Connected` or `Offline / not configured`) plus a `Re-check GenAI` button.
- Per-agent status caption (`Agent 1 ready - N obligations`, etc.).
- Live questionnaire metrics (questions, requirements, confidence).
- `Reset everything` button that wipes non-private session keys.

### 4.3 Session-state contract (`_DEFAULT_STATE` in `app.py`)

Initialised once via `_init_session_state()`. Holds the regulation label, tier, mode, doc IDs, the Stage-1 intelligence package cache, the four agent outputs (`analysis`, `brd_artifact`, `rtm_artifact`, `questionnaire`), the live `AssessmentState`, the current `ScoringResult`, the recommendation list, and the GenAI probe flag.

---

## 5. Module-by-Module Reference

For each module: business explanation, technical explanation, input, processing, output, files involved, related functions, and design rationale.

### 5.1 Models layer (`models/`)

#### 5.1.1 `models/workflow_models.py`

**Business explanation.** Defines the "shapes of information" that flow between every step of the pipeline. Every other layer talks in terms of these named structures — `ParsedDocument`, `Obligation`, `BRDArtifact`, etc. — so business stakeholders can reason about the data contract without reading code.

**Technical explanation.**

- Pure Python `@dataclass` definitions (no external dependency beyond `dataclasses`).
- All fields are JSON-friendly so `services/persistence.py` can store them and the Streamlit UI can serialise them as download payloads.
- The module docstring contains a mini ASCII diagram of the pipeline showing where each model is produced.

| Dataclass | Purpose | Key fields |
|-----------|---------|-----------|
| `ParsedDocument` | Wraps the text extracted from any uploaded file (PDF/DOCX). | `name`, `kind` (`regulation`/`brd`/`frd`/`other`), `text`, `source_path`, `page_count`, `mime`, `warning_message`, `metadata`. Property `is_empty` for guard logic. |
| `Obligation` | One discrete regulatory obligation extracted by Agent 1. | `obligation_id`, `title`, `theme`, `compliance_requirement`, `impacted_area`, `impacted_function`, `deadline`, `control_expectations`, `evidence_needs`, `risk_implication`, `source_requirement_id`, `regulatory_basis`, `priority` (default `"Should"`), `confidence` (default `92`). |
| `RegulatoryAnalysis` | Bundle returned by Agent 1. | `regulation`, `tier`, `summary`, `impacted_areas[]`, `obligation_themes[]`, `obligations[]`, `used_genai`, `metadata`, `brd_report`. |
| `BRDArtifact` | Wraps the BRD/FRD Pydantic model + export metadata. | `report` (`DoraDetailedBRD`), `metadata`, `docx_path`, `source` (`generated`/`uploaded`/`sample`). |
| `RTMEntry` | One row of the Requirements Traceability Matrix. | `traceability_id`, `obligation_id`, `business_requirement_id`, `functional_requirement_id`, `business_requirement`, `functional_requirement`, `impacted_area`, `impacted_function`, `system_process_impact`, `evidence_required`, `regulatory_basis`, `priority`. |
| `RTMArtifact` | Collection of RTM entries + lookup metadata. | `entries[]`, `metadata`. |
| `QuestionnairePackage` | Wrapper around the existing questionnaire-package dict (schema validated by `utils/json_utils`). | `package` (dict), `source` (`generated_brd`/`uploaded_brd`/`uploaded_json`/`db`), `questionnaire_id`, `name`. Properties `question_count`, `requirement_count`. |
| `AssessmentResponse` | Single answer (used by export payloads). | `question_id`, `answer`, `comments`, `display_sequence`. |
| `ScoringResult` | Deterministic Python rules-engine output. | `evaluation` (full dict from `evaluate`), `top_gaps[]`. Properties `compliance_score_pct`, `evaluation_confidence_pct`. |
| `RecommendationResult` | Agent 4 bundle. | `recommendations[]`, `severity_filter`, `top_n_requirements`, `used_genai`. |

**Why this design was chosen.** Plain dataclasses (rather than Pydantic models) avoid forcing every downstream consumer to install Pydantic, and dataclasses serialise trivially via `dataclasses.asdict`. Pydantic is only used inside `services/brd_frd_generator.py` because LangChain's structured output requires a Pydantic schema.

---

### 5.2 Utilities layer (`utils/`)

#### 5.2.1 `utils/file_utils.py`

**Business explanation.** Reusable filesystem plumbing so every page that handles a user upload behaves the same way (safe filenames, no overwrites, predictable output directories).

**Technical explanation.**

| Function | Signature | Behaviour |
|----------|-----------|-----------|
| `ensure_dirs(*paths)` | varargs | `Path(p).mkdir(parents=True, exist_ok=True)` for each. |
| `safe_filename(name, fallback="upload")` | → `str` | Strips path separators, replaces every non-`[A-Za-z0-9._-]` character with `_`, falls back to `fallback` on empty. Regex constant: `_INVALID_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._\-]+")`. |
| `save_upload(uploaded_file, dest_dir, *, prefix=None)` | → `Path` | Accepts any object with `.name` and `.getbuffer()` (Streamlit `UploadedFile`) or `.read()`. Appends an 8-character `uuid4` hex so re-uploads never overwrite previous files. |
| `read_bytes(path)` | → `bytes` | Raises `FileNotFoundError` if absent. |
| `copy_into(src, dest_dir)` | → `Path` | Uses `shutil.copy2`. |
| `timestamped_name(stem, suffix)` | → `str` | Returns `safe_filename(stem) + "_YYYYMMDDTHHMMSS" + suffix`. UTC timestamp. |
| `iter_files(directory, patterns)` | generator | Deduplicates files matching any of the supplied glob patterns. |

#### 5.2.2 `utils/json_utils.py`

**Business explanation.** Guards every JSON read/write of a questionnaire package against schema drift. If a user uploads a JSON file that does not match the contract produced by the questionnaire generator, the loader rejects it with a list of human-readable problems.

**Technical explanation.**

- `REQUIRED_TOP_KEYS = ("metadata", "requirements", "impact_pairs", "questions")`.
- Required keys per object are spelled out as `REQUIRED_REQUIREMENT_KEYS`, `REQUIRED_PAIR_KEYS`, `REQUIRED_QUESTION_KEYS` (see lines 19-30).
- `validate_package_schema(data)` iterates each top-level key and checks the **first 5 items** of each list (line 98, 110, 122) — a deliberate fast-path, not a deep validation.
- `read_package_json` / `write_package_json` wrap `read_json` / `write_json` with that validation.

**Why first 5 only?** Performance — the file can be hundreds of MB once the question bank is large. The check is meant to catch obviously-broken files, not lint every record.

#### 5.2.3 `utils/docx_parser.py`

**Business explanation.** Generic, domain-agnostic DOCX reading helpers. The BRD/FRD ingest path uses these to walk paragraphs and tables in the order they appear in the document, so section headings (`7.1`, `7.2`, …) are paired with the requirement tables that follow them.

**Technical explanation.**

- `clean_text(value)` collapses whitespace and strips non-breaking spaces (`\u00a0`).
- `normalise_header(value)` lower-cases and strips everything except `[a-z0-9]`. Example: `"DORA Alignment"` → `"doraalignment"`. Used to match against the column-header set `_REQUIRED_HEADERS` in `services/questionnaire_generator.py`.
- `iter_body_blocks(doc)` is the **key trick** lifted from the original v11 script: `python-docx` exposes `doc.paragraphs` and `doc.tables` as flat lists, which destroys document order. This helper walks `doc.element.body.iterchildren()` and yields paragraphs and tables in true XML order.
- `iter_sectioned_tables(source)` yields `(section_heading, table_rows)` pairs. A "section heading" matches the regex `^\d+(\.\d+)*\.?\s+` (e.g. `1.`, `7.1.`, `10.2.3`). The latest heading is associated with every subsequent table until another heading appears.
- `extract_full_text(source, include_tables=True)` returns a single joined string with table rows flattened as `"cell1 | cell2 | cell3"`.

#### 5.2.4 `utils/pdf_parser.py`

**Business explanation.** Reads the text out of an uploaded regulation PDF. Designed to **fail soft**: if the file is encrypted or scanned (image-only), the parser returns an empty `text` plus a `warning_message` so the rest of the pipeline can continue.

**Technical explanation.**

- Uses **PyMuPDF** (`import fitz`). Hard-fails at import time if PyMuPDF is missing.
- `extract_pdf_text(source, max_chars=None)` returns a `PdfExtractionResult(text, page_count, is_encrypted, warning_message)`.
- Encrypted-but-no-password files: tries `doc.authenticate("")`; if that fails it returns an empty result with `warning_message="PDF is password-protected; no text extracted."`.
- Empty-text result triggers `warning_message="PDF appears to contain no extractable text. It may be a scanned image. Consider OCR (e.g. ocrmypdf) before re-uploading."`.

---

### 5.3 Services layer (`services/`)

#### 5.3.1 `services/persistence.py`

A thin re-export of `services/database.py` so the new pipeline can `from services import persistence as db` without renaming the underlying file.

#### 5.3.2 `services/database.py`

**Business explanation.** The single source of permanent state. Everything the user uploads, generates or answers is persisted in a local SQLite database (`data/app.db`). This means a user can close the browser, restart the app and resume an in-flight assessment.

**Technical explanation.**

- Connection: `sqlite3.connect(path, detect_types=PARSE_DECLTYPES)` with `row_factory = sqlite3.Row` and `PRAGMA foreign_keys = ON`.
- `session()` context manager auto-commits on clean exit.
- `init_db()` runs the entire schema via `executescript`, guarded by `CREATE TABLE IF NOT EXISTS`, so calling it on startup is idempotent.

**Schema (5 tables).** See section 8 for the SQL.

**Public API.**

| Function | Purpose |
|----------|---------|
| `save_document(name, kind, path, ...)` | Insert a row into `documents`. Validates `kind ∈ {regulation, brd, frd, other}`. |
| `list_documents(kind=None)` / `get_document(id)` | Reads. |
| `save_requirements(document_id, requirements)` | Replace-then-insert per document. Stores `impacted_areas` / `impacted_functions` as JSON arrays. |
| `list_requirements(document_id)` | Parses JSON-encoded list columns back into Python lists. |
| `save_questionnaire(name, package, document_id=None, regulation=None)` | Stores the full package as `package_json` (verbatim). Indexes `question_count`, `requirement_count`, `overall_confidence_pct`. |
| `list_questionnaires()` / `get_questionnaire(id)` | Reads. `get_questionnaire` re-parses `package_json` into a Python dict on `package` key. |
| `create_assessment(questionnaire_id, name)` | Creates an assessment row, returns its ID. |
| `update_assessment_snapshot(assessment_id, state_json, evaluation, recommendations, completed)` | Atomic update of state + evaluation + recs. Builds the `UPDATE` statement dynamically so unspecified columns are not touched. |
| `upsert_responses(assessment_id, responses)` | Replace-then-insert, skipping keys ending in `__display_sequence` or `__comments` (these are stored in `state_json` only). |
| `_jsonable(obj)` | Recursively converts tuple-keyed dicts (`pair_scores` from the scoring engine) and `set` instances into JSON-safe shapes. Tuple keys become `"a | b"`. |

#### 5.3.3 `services/document_parser.py`

**Business explanation.** A single entry point that turns any uploaded regulation/BRD/FRD file into a `ParsedDocument` so the rest of the application doesn't care whether the source was PDF or DOCX.

**Technical explanation.**

- `parse_document(path, kind="regulation")` dispatches on the file extension:
  - `.pdf` → `parse_pdf` → wraps `utils.pdf_parser.extract_pdf_text`.
  - `.docx` → `parse_docx` → wraps `utils.docx_parser.extract_full_text(include_tables=True)`.
  - anything else → returns an empty `ParsedDocument` with `warning_message=f"Unsupported file type for parser: {suffix}"`.
- Defensive: missing file returns an empty `ParsedDocument` with `warning_message=f"File not found: {p}"` instead of raising — so the orchestrator can choose whether to abort or continue with the offline fallback.

#### 5.3.4 `services/genai_service.py`

**Business explanation.** The "phone line" to the PwC GenAI Shared Service (Azure GPT-4o behind PwC's internal gateway). This module is **completely decoupled** from BRD generation so we can swap LLMs later without touching business logic.

**Technical explanation.**

- `GenAISettings.from_env()` snapshots every relevant env var into a `@dataclass(frozen=True)` so callers can introspect / mock the configuration.
- `get_llm_api_key()` reads `API_KEY` from `.env`. **Crucially does not call `input()`** (the original v5 script did), because Streamlit cannot block on stdin.
- `build_ssl_verify_setting(settings)` chooses the SSL verification strategy:
  - `OPENAI_VERIFY_SSL=false` → `verify=False`.
  - `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` set → use that PEM bundle.
  - `OPENAI_SSL_STRATEGY ∈ {auto, windows_store, system, os}` and `truststore` package installed → `truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)` (uses the OS root store; required on corporate Windows).
  - Otherwise → `certifi.where()`.
- `build_http_client(settings)` builds an `httpx.Client` with `trust_env=True`, `verify=<above>`, and explicit connect/read/write timeouts (45/180/60 s by default).
- `preflight_openai_connectivity(http_client, settings)` issues a 5-token `chat/completions` ping with `"Return the word OK."`. Returns `True` only if status code is exactly 200.
- `create_configured_llm(api_key, http_client, settings)` returns a LangChain `ChatOpenAI` bound to the PwC base URL.
- `generate_structured_component(llm, schema_model, component_name, instruction, context, system_instruction=...)` builds a `ChatPromptTemplate.from_messages(...) | structured_llm` chain and `.invoke()`s it. The `system_instruction` default (`_DEFAULT_SYSTEM_INSTRUCTION`) tells the model it is a "principal regulatory compliance architect, DORA SME, and senior business analyst" and that it must produce structured output bounded to Tier-2 proportionality.
- `GenAIClient` is the convenience facade. `GenAIClient.try_create()`:
  1. If `OPENAI_SKIP_API=true` → returns `None` (forces offline).
  2. If `API_KEY` is missing → returns `None`.
  3. Builds the HTTP client and runs the preflight.
  4. If the preflight fails → closes the client and returns `None`.
  5. Otherwise → constructs and returns a `GenAIClient` holding the configured `ChatOpenAI`.

**Why this matters.** Every other module that wants GenAI calls `GenAIClient.try_create()`, branches on `None`, and never has to reason about HTTP/SSL. This is the entire reason the offline fallback is reliable.

#### 5.3.5 `services/search_config.py`

**Business explanation.** The hard guardrail that prevents the application from ever consuming content from a non-authoritative web source. Wikipedia, blogs, news sites, generic search engines — all blocked. Only the 15 approved EU/UK/national regulators and 10 approved consulting firms can supply text into the BRD pipeline.

**Technical explanation.**

- `SOURCE_TYPE_OFFICIAL_REGULATOR`, `SOURCE_TYPE_OFFICIAL_LEGISLATION` (EUR-Lex), `SOURCE_TYPE_CONSULTING_GUIDANCE` — the only three allowed source types.
- `SOURCE_PRIORITY = [OFFICIAL_REGULATOR, OFFICIAL_LEGISLATION, CONSULTING_GUIDANCE]` — drives the ordering inside `RegulatoryIntelligencePackage.all_sources()`.
- `PUBLICATION_TYPES = ["Regulation", "Directive", "RTS", "ITS", "Guideline", "Technical Standard", "Q&A", "Consultation Paper", "Supervisory Statement", "Enforcement Publication", "Opinion", "Recommendation"]` — the controlled vocabulary used by the metadata extractor.
- `APPROVED_REGULATORS` (15 entries, in source order):
  - **EU**: EBA, ESMA, ECB, SSM, EIOPA, SRB, AMLA, DG_FISMA, EUR_LEX.
  - **UK**: FCA, PRA.
  - **National**: BAFIN (Germany), AMF_FR (France), CBI (Ireland), DNB (Netherlands).
- Each `RegulatorSource` is a frozen dataclass with `code`, `name`, `jurisdiction`, `website`, `domains`, `source_type`, `description`, `publication_hints`. The `.matches(url)` method returns `True` only when the URL's host equals one of the regulator's domains (or is a subdomain of one).
- `APPROVED_CONSULTING_FIRMS` (10 entries): PwC, Deloitte, EY, KPMG, Accenture, Capgemini, McKinsey, BCG, Oliver Wyman, Bain.
- `resolve_regulators(selection)` / `resolve_consulting_firms(selection)`:
  - `None` / empty / `["ALL"]` → every approved entry.
  - Else → match against the code lookup (case-insensitive), preserving the user's order, silently dropping unknown codes.
- `is_regulator_url(url, regulators=None)` / `is_consulting_url(url, firms=None)` — boolean guard called by every fetcher after the search engine returns. This is the actual guarantee.
- Runtime env getters (all defensively typed):
  - `is_regulatory_search_enabled()` (honours legacy `REGULATION_WEB_SEARCH`, `DORA_ENABLE_WEB_SEARCH`).
  - `is_consulting_search_enabled()`.
  - `search_backends()` — default `"duckduckgo"`; the project's `.env` sets `"brave,yandex"` because DuckDuckGo is firewalled on the corporate network.
  - `regulatory_max_results()` — default 12.
  - `consulting_max_results()` — default 4.
  - `search_timeout_seconds()` — default 8.

#### 5.3.6 `services/native_regulator_search.py`

**Business explanation.** When the corporate firewall blocks every general-purpose search engine, this module hits each regulator's own search page directly. EBA, ESMA, EIOPA and FCA all expose a public search URL we can call with `httpx` and parse for result links.

**Technical explanation.**

- Adapter registry: `@register("CODE")` decorator adds a function to the `_ADAPTERS` dict.
- `native_search(regulator_code, query, *, max_results=8, timeout=12)` looks up the adapter and calls it. Wraps every adapter in a try/except so one regulator's failure never breaks Stage 1.
- `_fetch(url, timeout)`: simple `httpx.get` with browser-ish headers, follows redirects, returns the body for any status < 500. (ESMA / EIOPA legitimately return 404 with valid results.)
- `_parse_drupal_style_results(html, domain, query, max_results)` is the generic regex-based parser. Filters out:
  - URLs containing any of `_DROP_HREF_TOKENS` (`/search?`, `/sitemap`, `/cookie`, `/error/`, `?f%5b`, …).
  - Titles in `_DROP_TITLE_LOWERCASE` (e.g. `"main menu"`, `"page 1"`, `"more"`).
  - Language-switcher anchors (regex `^[a-z]{2}\s+([a-zé…]+)$`).
- Requires either a Drupal `data-drupal-link-system-path` attribute **or** a URL path of ≥ 2 segments (so `/about-us` and `/extranet` are dropped, while `/activities/dora` survives).
- Relevance gate: at least one of `_relevance_terms(query)` must appear in the URL or title. The gate uses a synonym table `_REGULATION_SYNONYMS` so e.g. `"dora"` matches `"digital operational resilience"`, `"2022/2554"`, `"32022r2554"`.
- Registered adapters:
  - `EBA`: `https://www.eba.europa.eu/search?keywords={q}`
  - `ESMA`: `https://www.esma.europa.eu/search-results?keys={q}`
  - `EIOPA`: `https://www.eiopa.europa.eu/search?keys={q}`
  - `FCA`: `https://www.fca.org.uk/search-results?search_term={q}`

#### 5.3.7 `services/official_regulation_fetcher.py` — Stage 1

**Business explanation.** Stage 1 of the Regulatory Intelligence Pipeline. Given a regulation label (e.g. `"DORA"`) and an optional list of regulator codes, returns every authoritative publication retrieved from the regulator's own website.

**Technical explanation.**

- `OfficialRegulationResult` dataclass captures every per-publication field required by the BRD prompt (source_type, regulator name + code, title, URL, snippet, publication type, regulation_id, publication_date, version, executive_summary, key_obligations, impacted_business_functions, related_regulations, backend, query, confidence_score, retrieved_at).
- `.as_context_block()` renders the result as the prompt-ready text block fed to the LLM.

**`fetch_official_regulations(regulation, regulator_selection=None, *, max_results_per_query=None, status=_noop)`** is the entry point and returns:

```python
{
    "results": [OfficialRegulationResult],
    "regulators": [{"code", "name", "jurisdiction", "website"}],
    "diagnostics": [str],
    "queries": [{"query", "domain", "regulator_code"}],
    "errors": [str],
    "enabled": bool,
}
```

**Two-path strategy.**

1. **PRIMARY — native per-regulator site search.** Loops over the selected regulators; for any that have a `native_regulator_search` adapter, calls `native_search(...)`. Hits are URL-filtered with `is_regulator_url(...)`. If every selected regulator has a native adapter, OR if Stage 1 already has `early_stop_results` hits, the function returns and skips DDGS entirely.
2. **FALLBACK — DDGS** (DuckDuckGo Search wrapper, importable as `ddgs` or `duckduckgo_search`). Sequential dispatch with `inter_query_delay_ms` between queries to avoid rate limiting. Each query is run against the configured `backends` list (`brave,yandex,duckduckgo,bing,mojeek,…`) and the first backend that returns hits wins. Every returned URL is post-filtered with `is_regulator_url` so even a backend that ignores `site:` filters cannot leak Wikipedia into the BRD context.

**Query construction (`build_queries`):**

- Default templates (overridable via `REGULATORY_QUERY_TEMPLATES`):
  ```
  "{regulation} regulation official text"
  "{regulation} RTS technical standards guidelines"
  ```
- If the user selected **1-4 regulators** (not "ALL"), an extra **biased query** is prepended: `"<regulator names> {label} regulation"`.

**Metadata extraction (deterministic, snippet-only).**

- `_extract_regulation_id(text)` checks patterns in order: `EBA/RTS/2024/05`-style, `Regulation (EU) YYYY/N`, `Directive (EU) YYYY/N`, `CPxx/yy` / `PSxx/yy` / `SSxx/yy` / `DPxx/yy` / `FGxx/yy`, `ESMA70-xxxx-yyy`. First match wins.
- `_extract_publication_type(text, hints)` first scans the regulator's `publication_hints` tuple, then falls back to the global `PUBLICATION_TYPES` list.
- `_extract_publication_date(text)` parses three date formats (`D Month YYYY`, `YYYY-MM-DD`, `DD/MM/YYYY`).

**`_score_confidence(result, regulation)` — heuristic in [0.5, 0.99]:**
- Base 0.5 (any approved-domain match).
- +0.20 if regulation label appears in the title.
- +0.15 if a regulation_id is detected.
- +0.10 if a publication type is detected.
- +0.05 if a publication date is detected.
- Capped at 0.99.

**Runtime caps (`_runtime_caps`):**
- `REGULATORY_SEARCH_MAX_QUERIES` (default 4)
- `REGULATORY_SEARCH_EARLY_STOP` (default 10) — stops once we have this many unique URLs.
- `REGULATORY_SEARCH_MAX_SECONDS` (default 30) — wall-clock cap.
- `REGULATORY_SEARCH_DELAY_MS` (default 600) — sleep between sequential queries.

#### 5.3.8 `services/consulting_guidance_fetcher.py` — Stage 2

**Business explanation.** Stage 2 enriches the BRD with implementation guidance from PwC, Deloitte, EY, KPMG and the other approved consulting firms. **Currently disabled** in `.env` (`CONSULTING_SEARCH_ENABLED=false`); the fetcher and UI hooks remain in the codebase so re-enabling is a one-line change.

**Technical explanation.**

- `ConsultingGuidanceResult` mirrors `OfficialRegulationResult` and adds `anchor_regulation_title`, `anchor_regulation_id`, `anchor_regulator` so we can prove every consulting hit was anchored to a Stage 1 hit.
- `fetch_consulting_guidance(regulations, consulting_selection=None, *, regulation_label="", max_anchors=3, ...)`:
  - Returns an empty result if Stage 1 was empty — consulting firms are never searched independently.
  - Returns an empty result if `CONSULTING_SEARCH_ENABLED=false`.
  - Anchors on the top-N Stage 1 hits (default 3).
  - For each anchor × firm, builds two queries from templates `"{anchor} implementation guidance"` and `"{anchor} compliance roadmap"`, where `_anchor_phrase(...)` is `'"<title>" <regulation_id>'` so the search is tied to the exact publication.
  - Each query is dispatched through the configured DDGS backends, stops at the first backend that yields any hits, and post-filters every URL with `is_consulting_url(...)`.
- `_score_confidence(result, anchor_title, anchor_id)`: base 0.4; +0.30 if anchor title appears in title/snippet; +0.15 if anchor ID appears; +0.05 publication date; +0.05 firm matched. Cap 0.95 — **always below the Stage 1 ceiling of 0.99** so consulting can never out-rank a regulator.

#### 5.3.9 `services/regulatory_intelligence_service.py` — Coordinator

**Business explanation.** Combines Stage 1 + Stage 2 into a single bundle (`RegulatoryIntelligencePackage`) and produces the **prompt-ready context string** the BRD generator consumes.

**Technical explanation.**

- `RegulatoryIntelligencePackage` carries: regulation label, regulator/consulting selections, `official_results`, `consulting_results`, `context_text` (the combined prompt block), `source_summary`, `diagnostics`, `errors`, `stage1_enabled`, `stage2_enabled`.
  - `has_official_content` / `has_any_content` — convenience guards.
  - `all_sources()` — flat list ordered by `SOURCE_PRIORITY` then by descending confidence_score, with every field the UI dashboard wants.
- `RegulatoryIntelligenceService.gather(...)` runs Stage 1, then conditionally Stage 2, then composes:
  - **`_format_context(...)`** renders three sections with hard-coded banners:
    - `=== OFFICIAL REGULATORY CONTEXT (Primary Source of Truth) ===`
    - `=== SUPPLEMENTARY IMPLEMENTATION GUIDANCE (Not Authoritative) ===`
    - `=== OFFLINE REGULATORY BASELINE (Authoritative Sources Unavailable) ===`
  - Truncates the rendered block at `char_budget=12000` characters (followed by `… (context truncated)`).
- `offline_baseline_for(regulation)` returns a baked-in DORA baseline if `regulation.upper() == "DORA"`, else a generic placeholder warning that the content reflects "the LLM's pretrained knowledge rather than from an authoritative regulatory publication."
- Module-level helper `gather_regulatory_intelligence(...)` wraps a default `RegulatoryIntelligenceService` instance.

#### 5.3.10 `services/brd_frd_generator.py`

This is the largest single file (~1450 lines). It contains four logical layers.

##### 5.3.10.1 Pydantic schemas

| Class | Role |
|-------|------|
| `BulletItem` | `title` + `description`. |
| `RequirementItem` | `id`, `category`, `requirement`, `detailed_requirement`, `dora_alignment`, `priority`, `acceptance_criteria`, `confidence_level` (default `"95%"`). |
| `ControlCheckpointItem` | `stage` (Identify/Protect/Detect/Respond/Recover/Govern/Third Party), `control_checkpoint`, `requirement`, `tooling_expectation`, `evidence`. |
| `RiskItem` | `risk`, `impact`, `mitigation`, `owner`. |
| `DeliveryPhaseItem` | `phase`, `duration`, `objectives`, `activities[]`, `outputs[]`. |
| `StandardSection` | `description` + `items[BulletItem]`. |
| `RequirementSection` | `description` + `items[RequirementItem]`. |
| `ControlFrameworkSection` | `description`, `lifecycle_checkpoints`, `preventive_controls`, `detective_controls`, `corrective_controls`, `governance_controls`, `tooling_integration`. |
| `RiskSection` | `description` + `items[RiskItem]`. |
| `DeliveryPlanSection` | `description`, `phases[]`, `success_factors[]`. |
| `DoraDetailedBRD` | The top-level schema — 18 typed fields (executive_summary, objectives, scope, stakeholders, current_state_challenges, target_state_overview, process/data/reporting/functional/non_functional requirements, control_framework, assumptions, dependencies, risks_and_mitigations, success_criteria, appendix, workshop_delivery_plan). |
| `FrontMatterBundle`, `AnalysisBundle`, `SolutionRequirementsBundle`, `GovernanceBundle`, `ClosureBundle` | Sub-bundles used to chunk the LLM calls — see below. |

##### 5.3.10.2 Confidence helpers

- `normalize_confidence_level(value, dora_alignment="")`:
  1. If a numeric value is parseable from the string, clamp to **[90, 100]** and return as `"NN%"`.
  2. Otherwise, scan `dora_alignment.lower()` for "strong terms" (`article`, `ict risk`, `incident`, `third-party`, `register`, `resilience testing`, `auditability`, `governance`, `backup`, `recovery`, `critical`, `dora`, `rts`, `its`) → returns `"96%"`.
  3. Default → `"93%"`.
- `apply_confidence_floor(report)` runs `normalize_confidence_level` over every requirement row in every requirement section.
- `normalize_requirement_ids(report)` re-numbers requirements sequentially within each section: `BR-PRO-001`, `BR-PRO-002`, …, `BR-DAT-…`, `BR-REP-…`, `FR-…`, `NFR-…`.
- `calculate_overall_confidence(report)` returns `"NN%"`:
  1. Mean of every requirement's confidence value (each clamped to 90-100).
  2. If any section is below its **count gate** (process ≥ 14, data ≥ 14, reporting ≥ 10, functional ≥ 18, non-functional ≥ 10), the average is clamped at 90%.
  3. Result clamped to 90-100.
- `enforce_overall_confidence_floor(report)` repeats the floor with a guarantee that overall ≥ 90%.

##### 5.3.10.3 GenAI pipeline

`generate_detailed_dora_brd(context, tier="Tier-2", status=_noop, client=None)` runs **8 bundled LLM calls** (one per bundle so the model never hits a token-length limit):

1. `FrontMatterBundle` — exec summary + objectives + scope.
2. `AnalysisBundle` — stakeholders + current state + target state.
3. `RequirementSection` — 7.1 Process Requirements.
4. `RequirementSection` — 7.2 Data Requirements.
5. `RequirementSection` — 7.3 Reporting Requirements.
6. `SolutionRequirementsBundle` — 8 (FR) + 10 (NFR).
7. `GovernanceBundle` — controls + assumptions + dependencies + risks + success criteria.
8. `ClosureBundle` — appendix + workshop delivery plan.

Each call uses `client.generate(...)` (i.e. `langchain_openai.ChatOpenAI.with_structured_output(schema_model)`). The `_REQUIREMENT_COMMON` instruction explicitly tells the model how many items each section should produce so the count gates can pass.

Any of `APIConnectionError`, `APITimeoutError`, `APIStatusError`, generic `Exception` (including LangChain's `LengthFinishReasonError`) → returns `None` so the caller can fall back to offline.

##### 5.3.10.4 Offline fallback + enrichment

- `generate_offline_fallback_brd()` returns a hand-written `DoraDetailedBRD` containing 5-6 entries per section. It is deliberately less complete than the GenAI output because it then runs through `ensure_minimum_detail(...)`.
- `ensure_minimum_detail(report)` is the **deterministic enrichment** layer. It pads every section to a minimum count (executive_summary ≥ 6 bullets, objectives ≥ 8, scope ≥ 10, stakeholders ≥ 10, current_state_challenges ≥ 10, target_state_overview ≥ 10; process requirements ≥ 14, data ≥ 14, reporting ≥ 10, functional ≥ 18, non-functional ≥ 10; control_framework checkpoints ≥ 12 and preventive/detective/corrective/governance/tooling controls ≥ 6 each; risks ≥ 10; assumptions/dependencies/success_criteria/appendix ≥ 8). Padding rows are skipped if their ID/title already exists (case-insensitive set comparison).

##### 5.3.10.5 DOCX writer

`write_brd_docx(report, filename, tier="Tier-2")` writes a styled Word document via `python-docx`:
- 0.8" margins, Arial 11pt body, 14pt H1, 12pt H2.
- Title: "DORA Digital Operational Resilience Diagnostic".
- Subtitle: "Business Requirements Document (BRD) / Functional Requirements Document (FRD)".
- Compliance Framework / Tier banner.
- "Overall AI Confidence: NN% …" caveat at the top.
- Sections 1-16 with embedded `Table Grid` tables for the requirement sections.

##### 5.3.10.6 Top-level orchestrator

`build_brd_frd_report(regulation, tier, extra_context=None, status=_noop, client=None, *, regulator_selection, consulting_selection, include_consulting_guidance, intelligence_package=None)`:

1. If no `intelligence_package` supplied → call `gather_regulatory_intelligence(...)`.
2. Compose context: `intelligence_package.context_text or offline_baseline_for(regulation)`, with `extra_context` (uploaded PDF text) appended under a banner `--- Uploaded regulation document context ---`.
3. Try `generate_detailed_dora_brd(...)` → if `None`, fall back to `generate_offline_fallback_brd()`.
4. Run `ensure_minimum_detail` → `apply_confidence_floor` → `normalize_requirement_ids` → `enforce_overall_confidence_floor`.
5. Compute provenance flag (`regulation_source ∈ {official_regulator, uploaded_document, offline_baseline}`).
6. Return `(report, metadata_dict)` where `metadata` contains overall confidence, used_genai flag, source summary, official/consulting source lists, section counts, search diagnostics, and back-compat keys (`web_sources`, `web_source_count`) for UI code that has not migrated yet.

##### 5.3.10.7 Back-compat helpers

`monitor_regulation_updates(...)` and `monitor_dora_updates(...)` wrap the new pipeline and return the legacy `(context, sources[])` shape so older app code paths keep working without changes.

#### 5.3.11 `services/questionnaire_generator.py`

This is the second-largest file (~1620 lines). It builds the questionnaire from either a parsed BRD DOCX or an in-memory `DoraDetailedBRD`.

##### 5.3.11.1 Tunables

- `CONFIDENCE_FLOOR = QUESTION_CONFIDENCE_FLOOR or 90` — every question's confidence is clamped to ≥ 90.
- `OVERALL_CONFIDENCE_FLOOR = OVERALL_QUESTIONNAIRE_CONFIDENCE_FLOOR or 90` — package-level floor.
- `MAX_AREA_FUNCTION_PAIRS = 40` — caps the number of impact pairs the question bank can fan out over.
- `MIN_FREE_TEXT = 5`, `MAX_FREE_TEXT = 10` — bounds on the free-text section.

##### 5.3.11.2 Taxonomies (the "brain" of the question generator)

- `REGULATORY_TAXONOMY["DORA"]` — official source, 11 pillars (Governance, ICT risk management, ICT systems, IPDRR, Backup/recovery, Incident management, Resilience testing, Third-party risk, Contracts, Information security, Management reporting) and an article-hint dictionary.
- `AREA_KEYWORDS` (20 area labels → keyword lists) — used to classify a requirement into one or more **impacted areas** (Front Office, Middle Office, Back Office, Regulatory Reporting & Financial Reporting, Business Structure & Functions, Firm Type / Client Type, Operating Model, Risk & Controls framework, Governance Model, Internal Compliances, Third Party Risk Management / Dependency, Programme Maturity / Programme Ownership, Program Sponsorship / Budget Planning, People Policies & Processes, IT Systems & Technology, Data Reporting & Governance, IT Security / Cyber Security, Legal, HR, High Impact Pain Points).
- `FUNCTION_KEYWORDS` (13 functions → keyword lists).
- `DEFAULT_OPTIONS` — seven canonical option families (`maturity`, `yes_no_partial`, `coverage`, `risk_level`, `ownership`, `evidence`, `support`).
- `THEME_OPTIONS` — seven theme-specific option families used for the deep-dive question per pair (`incident_reporting`, `third_party`, `resilience_testing`, `security_access`, `governance`, `data_evidence`, `reporting`).
- `_THEME_TO_KEY` maps the canonical theme labels (e.g. `"Incident reporting"`) to their internal key (e.g. `"incident_reporting"`) — used to bridge the static funnel to the adaptive `branch_registry`.

##### 5.3.11.3 `ANSWER_SCORES` — the central scoring map

Used by `services/scoring_engine.score_value` whenever an option doesn't carry per-option `score_value` metadata:

| Answer | Score |
|--------|-------|
| `Yes`, `Complete`, `Measured / Optimised` | 100 |
| `Implemented` | 90 |
| `Mostly complete` | 85 |
| `Defined` | 75 |
| `Partially`, `Partially complete` | 55 |
| `Ad hoc` | 35 |
| `Not started`, `No`, `No owner assigned`, `No evidence available` | 0 |
| `Critical` | 15 |
| `High` | 35 |
| `Medium` | 65 |
| `Low` | 90 |
| `Named accountable owner` | 95 |
| `Shared ownership` | 70 |
| `Informal owner` | 40 |
| `Not applicable` | `None` (excluded from scoring) |
| `Unknown` | 25 |

##### 5.3.11.4 Dataclasses

- `Requirement`: `source_section`, `source_id`, `normalized_id` (`BR-PRO-001`-style), `category`, `requirement`, `detail`, `alignment`, `priority`, `acceptance`, `confidence`, `themes[]`.
- `ImpactPair`: `area`, `function`, `requirement_ids`, `regulatory_basis`, `confidence`.
- `Question`: `question_id`, `area`, `function`, `question_type` (`Single Select` / `Multi Select` / `Open Ended`), `question`, `options[]`, `mapped_requirement_ids[]`, `regulatory_basis`, `confidence`, `scoring_weight`, `funnel_parent_id`, `trigger_answers[]`, `rationale`, `is_free_text`, plus v12 adaptive-branching fields (`branch_theme`, `branch_rule_id`, `source_parent_id`, `dynamic_depth`).

##### 5.3.11.5 Pipeline (top-to-bottom)

1. **`read_docx_requirements(source)`** — uses `iter_sectioned_tables` to walk the DOCX. Validates every requirement table has the column headers `id, category, requirement, detailedrequirement, doraalignment, priority, acceptancecriteria` (AI confidence is optional). Calls `_section_prefix(...)` to pick `BR-PRO`/`BR-DAT`/`BR-REP`/`FR`/`NFR` based on the section heading or the source id prefix.
2. **`requirements_from_report(report)`** — closed-loop alternative that walks the Pydantic `DoraDetailedBRD` straight into `Requirement` records.
3. **`infer_themes(text)`** — returns up to 8 theme labels by scanning lowercase text for theme-specific keywords (Governance, ICT risk management, Incident reporting, Resilience testing, Third-party risk, Data and evidence, Security and access, Reporting; default `["General regulatory coverage"]`).
4. **`derive_impact_pairs(requirements, regulation)`** — for every requirement, runs `impacted_labels_for_requirement` against `AREA_KEYWORDS` and `FUNCTION_KEYWORDS`; the keyword scoring keeps each label whose count is within 1 of the maximum; caps each requirement to 3 areas × 3 functions; deduplicates the pairs to the **top 40** (`MAX_AREA_FUNCTION_PAIRS`) by requirement-count, alphabetical tie-break.
5. **`dedupe_impact_pairs(pairs)`** — drops a pair if a previously-kept pair on the same axis shares ≥ 80% of its requirement IDs.
6. **`build_closed_questions_for_pair(pair, requirements, start_idx)`** — synthesises a **funnel of 5-6 questions** per pair (Coverage → Ownership / Evidence / Risk → Remediation, plus an optional theme-specific deep-dive). All the question wording is generated from anchors extracted from the most informative mapped requirement:
   - `_select_anchor_requirement(reqs)` — picks the requirement with the most signal (article reference + metric + must-priority + acceptance + detail length).
   - `_extract_article(req)`, `_extract_metric(req)`, `_behavioural_anchor(req)`, `_evidence_anchor(req)`, `_format_req_label(req)` — feed real article references, metrics like `4 hours / RTO / RPO / TLPT`, verb phrases, and citation-style labels into the question text. Strips subject stems (`"The organization must"`) so the remaining phrase reads as a verb clause.
   - The seven theme deep-dives have their own bespoke wording functions (`_theme_question_text`).
7. **`build_free_text_questions(requirements, pairs, start_idx)`** — uses `_THEME_FREE_TEXT_PROMPTS` for the dominant themes, then pads with `_GENERIC_FREE_TEXT_PROMPTS`. Bounded to `[MIN_FREE_TEXT, MAX_FREE_TEXT]`.
8. **`dedupe_and_resequence_questions(questions)`** — drops near-duplicates (same `question_kind` + ≥ 70% requirement overlap + same area or function), then re-IDs every question to `Q-0001`, `Q-0002`, …, and rewires `funnel_parent_id` to the new IDs.
9. **`validate_and_score_package(requirements, pairs, questions)`** — computes the overall confidence:
   - Composite formula: `0.30 × req_cov + 0.18 × area_cov + 0.18 × fn_cov + 0.14 × pair_depth + 0.10 × free_text_ok + 0.10 × (avg_q_conf / 100)`, multiplied by 100, clamped to `[OVERALL_CONFIDENCE_FLOOR, 100]`.
   - `req_cov` = mapped requirements / total requirements.
   - `area_cov`, `fn_cov` = areas/functions touched by at least one closed question / total.
   - `pair_depth` = closed questions / (pair_count × 4), capped at 1.
   - `free_text_ok` = 1 if `MIN_FREE_TEXT ≤ free_text_count ≤ MAX_FREE_TEXT` else 0.
10. **`package_dict(...)`** — emits the dict consumed by the rest of the system. Top-level keys: `metadata`, `requirements`, `impact_pairs`, `questions`, `answer_scores`.
11. Convenience writers: `write_excel(...)` / `write_excel_from_package(...)` produce an Excel workbook with sheets `Summary`, `Impacted Functions Areas`, `Questionnaire`, `Free Text Questions`, `Funnel Logic`, `Scoring Rubric`, `Requirement Traceability`.

#### 5.3.12 `services/branch_registry.py`

**Business explanation.** Per-option, regulation-aware adaptive branching rules. When the user answers a baseline question, the engine looks up `(regulation, theme, question_kind, selected_answer_label)` in this registry and, if it finds a match, queues bespoke follow-up questions. Otherwise it falls back to the generic engine.

**Technical explanation.**

- `BranchKey = Tuple[str, str, str, str]` — `(regulation, theme, question_kind, selected_answer_label)`. Regulation is upper-cased; the other three are matched verbatim.
- `BRANCH_LIBRARY` is the public mapping. Today's vertical slice covers DORA × Incident reporting × coverage for answers `Not started`, `Partially complete`, `Mostly complete`, `Partially`, `Complete`, `Unknown`. (`Mostly complete` and `Partially` re-use the `Partially complete` spec list as defensive aliases.)
- Each spec is a dict with `question_id`, `question`, `options`, `question_type`, `rationale`, `branch_rule_id` (e.g. `dora_incident_coverage_not_started__classification`), and `scoring_weight`. The scoring engine materialises a spec into a queue-ready question via `materialize_branch_spec`.
- `lookup_branch(regulation, theme, question_kind, answer_label)` returns the spec list (deep-copied) or `[]`.

**Example branch (Not started → 3 follow-ups):**
1. Has an incident classification process been defined? (Single Select; weight 3) — branch_rule_id `dora_incident_coverage_not_started__classification`.
2. Named accountable owner for regulatory notifications? (Single Select; weight 3).
3. Reporting timelines agreed for initial / intermediate / final? (Single Select; weight 2).

#### 5.3.13 `services/scoring_engine.py`

This is the cockpit's brain — adaptive routing + scoring + heatmap helpers. Lives entirely in pure functions + an `AssessmentState` dataclass; never touches `streamlit.session_state` directly.

##### 5.3.13.1 Signal sets (lifted verbatim from v11)

| Set | Members |
|-----|---------|
| `GAP_SIGNALS` | `Partially`, `Partially complete`, `Mostly complete`, `Not started`, `No`, `Unknown`, `Ad hoc`, `No evidence available`, `No owner assigned`, `Informal owner`, `Medium`, `High`, `Critical` |
| `POSITIVE_SIGNALS` | `Yes`, `Complete`, `Implemented`, `Measured / Optimised`, `Named accountable owner`, `Low` |
| `WEAK_OWNERSHIP` | `No owner assigned`, `Informal owner`, `Shared ownership`, `Unknown` |
| `WEAK_EVIDENCE` | `No evidence available`, `Unknown` |
| `HIGH_RISK` | `Medium`, `High`, `Critical`, `Unknown` |
| `NEGATIVE_COVERAGE` | `Partially`, `Partially complete`, `Mostly complete`, `Not started`, `No`, `Ad hoc`, `Unknown` |

##### 5.3.13.2 Adaptive budget knobs (env-overridable)

- `MAX_DYNAMIC_FOLLOWUP_DEPTH = 3` — max chain length parent → child → grandchild.
- `MAX_DYNAMIC_FOLLOWUPS_PER_PARENT = 3` — max follow-ups any single parent can emit.
- `MAX_DYNAMIC_QUESTIONS_PER_ASSESSMENT = 50` — total dynamic Qs across the whole assessment.

##### 5.3.13.3 `AssessmentState`

Holds:
- `responses: Dict[str, Any]` — `question_id -> answer`.
- `dynamic_queue: List[Dict]` — pending follow-up questions.
- `skipped_ids: Set[str]` — questions skipped by the positive-answer optimisation.
- `display_numbers: Dict[str, int]` + `display_counter: int` — stable numbering for the "Question 003" label on Page 4.
- `history: List[str]` — order of answered question IDs.
- `branch_log: List[Dict]` — audit trail (v12).
- `dynamic_questions_emitted: int`, `emitted_dynamic_ids: Set[str]` — idempotency + cap enforcement.

Method `reset_responses()` wipes everything; `remaining_dynamic_budget()` returns `MAX_DYNAMIC_QUESTIONS_PER_ASSESSMENT - dynamic_questions_emitted`.

##### 5.3.13.4 Routing

`update_applicability_after_response(state, answered_q, value, base_questions, package_regulation=None)` is called after every answer submission. Routing order:

1. **Step 1.** Discard any unanswered pending dynamic children of this question's root (so we re-evaluate when the parent is re-answered).
2. **Early registry hit.** Call `registry_followups(...)`:
   - Resolve regulation (parent's explicit field, else `package_regulation`, else `"DORA"`).
   - Resolve theme from `parent["branch_theme"]` (falling back to `_infer_theme_from_parent` which scans the question text for 7 theme keyword bundles).
   - Look up each selected answer in `branch_registry.lookup_branch(...)`.
   - Materialise specs into queue-ready dicts via `materialize_branch_spec(...)` and append (subject to budget caps).
   - If anything was queued → log a `registry` branch decision and apply the positive-answer skip rule (so the legacy static funnel doesn't pile up extra generic questions).
3. **Step 3.** If the answered question is itself dynamic → stop.
4. **Step 4a.** If the answer contains any `GAP_SIGNALS` → call `dynamic_followups(...)` which can emit up to 4 generic follow-ups (`NEXT-GAP`, `NEXT-EVIDENCE`, `NEXT-OWNER`, `NEXT-RISK`) based on which signal sets the answer hit. Log as `generic` decision.
5. **Step 4b.** If `is_positive_answer(value)` → call `_apply_positive_skip(...)` (adds risk/remediation questions in the same context to `skipped_ids`) and queue a `NEXT-VALIDATE` follow-up (so positive claims aren't trusted blindly).

`_apply_positive_skip` only skips questions that share the parent's area + function and have ≥ 1 mapped requirement overlap. Allowed parent kinds and what they skip:

| Parent `question_kind` | Skippable kinds |
|------------------------|-----------------|
| `coverage` (default) | `risk`, `remediation` |
| `ownership` or `evidence` | `remediation` only |

##### 5.3.13.5 `choose_next_question(state, base_questions, focus_area="All")`

- Returns the first unanswered dynamic question (queue order) if any.
- Otherwise ranks the unanswered base questions by:
  1. `-answered_area_counts[q.area]` (prefer areas with the fewest answered questions so far).
  2. `-answered_function_counts[q.function]` (same idea for functions).
  3. `confidence + scoring_weight × 10` (descending).
  4. `question_id` (descending tie-break — chosen for determinism).
- Honours `focus_area` ("All" or one of the area labels).

##### 5.3.13.6 `score_value(value, question=None)`

- Empty answer → `None` (excluded).
- If the question's options include per-option `score_value` metadata, use the mean of the matched options' `score_value`s.
- Otherwise use the mean of `ANSWER_SCORES[v]` for every value found; unknown-but-non-empty answers default to **25**.

##### 5.3.13.7 `evaluate(questions, state)`

This is the core readiness calculation. For every closed question that isn't skipped:

- `weight = scoring_weight × (confidence / 100)`.
- `score = score_value(state.responses[qid], q)`.
- Accumulate `score × weight` and `100 × weight` into per-area, per-function, per-pair, per-requirement and grand-total totals.
- After the loop:
  - `compliance_score_pct = round(total_num / total_den * 100, 1)`.
  - `area_scores`, `function_scores`, `pair_scores`, `requirement_scores` — `round(num/den * 100, 1)` per key.
  - `evaluation_confidence_pct = round(max(90.0, min(99.0, avg_conf − coverage_penalty + min(3.0, answered_count × 0.05))), 1)` where `coverage_penalty = min(8.0, unanswered_count × 0.10)`.
- Wraps area/function scores in a summary dict that includes a `CXO status` and an action via `cxo_status(score)`.

##### 5.3.13.8 `cxo_status(score)`

- `≥ 85` → `("Ready", "Maintain evidence and periodic validation.")`
- `≥ 65` → `("Watch", "Resolve targeted gaps before executive sign-off.")`
- `≥ 40` → `("At risk", "Prioritise remediation plan, owners and evidence.")`
- `< 40` → `("Critical", "Escalate to governance and define funded remediation.")`

##### 5.3.13.9 Display helpers

- `rationale_text(q, responses)` returns the "Why this question was asked" copy shown under each question on Page 4.
- `summary_dataframe(summary, label)`, `heatmap_dataframe(area_summary)`, `pair_heatmap_rows(pair_scores)` — convert the scoring dicts into `pandas.DataFrame` for rendering on Page 5.

#### 5.3.14 `services/recommendation_service.py`

**Business explanation.** Turns a scored assessment into a structured list of recommendations with severity, owner, time horizon and back-traceability to the parent gaps. Optional GenAI "polish" rewrites the action wording into formal business English.

**Technical explanation.**

- `Recommendation` dataclass: `recommendation_id`, `title`, `severity`, `area`, `function`, `compliance_pct`, `rationale`, `suggested_action`, `suggested_owner`, `mapped_requirement_ids[]`, `horizon`, plus v12 fields `branch_evidence` (audit trail snippet) and `branch_rule_ids[]`.
- Severity playbook constants:
  - `_SEVERITY_ACTIONS` — one fixed sentence per severity (Critical / At risk / Watch / Ready).
  - `_SEVERITY_HORIZON` — `Immediate (0-30 days)` / `Short-term (30-90 days)` / `Medium-term (90-180 days)` / `Steady-state (periodic)`.
  - `_OWNER_BY_FUNCTION` — 13-entry mapping from impacted function to owner role (e.g. `"Cyber Security" → "Chief Information Security Officer"`). Falls back to `"Compliance / Programme Owner"`.

**`generate_recommendations(package, evaluation, *, min_severity="Watch", top_n_requirements=10, branch_log=None)`** produces three buckets:

1. **Pair recommendations** — every `(area, function)` pair in `evaluation["pair_scores"]` whose severity ≥ `min_severity` gets a `Recommendation`. Title pattern: `Improve <area> / <function> readiness`.
2. **Area-only recommendations** — fallback used when `pair_scores` is empty but `area_summary` is populated.
3. **Top-N weakest requirements** — sorted ascending by score; title `Close gap on <req_id>`.

If a `branch_log` is supplied, `_summarise_branch_log_for(...)` is called per recommendation to attach a `branch_evidence` snippet like `Branch trace: on Q-0042 the user selected 'Not started' [dora_incident_coverage_not_started__classification] (asked DQ-0042__DORA_INC_NS_001); …`.

After all three buckets are assembled, the list is sorted by `(severity_rank, compliance_pct ascending)` and re-numbered `REC-001`, `REC-002`, ….

`enrich_recommendations_with_genai(recommendations, package, *, client=None)` calls the LLM once per recommendation with a strict prompt: "Rewrite the supplied action as one concise paragraph (60-100 words) in formal business English. Do not invent new evidence, owners, or deadlines." Failures are silently swallowed — the deterministic baseline always wins.

---

### 5.4 Agents layer (`agents/`)

Each agent is a thin orchestration shim around the services. They exist so the Streamlit UI and any future CI/batch caller can compose the pipeline in a uniform way.

#### 5.4.1 `agents/regulatory_analysis_agent.py` — Agent 1

**Business role.** Reads the regulation (and optionally the uploaded PDF) and produces the obligation catalogue plus the BRD/FRD draft.

**Technical role.**

- `RegulatoryAnalysisAgent(client=client)` is stateless.
- `analyze(parsed_document=None, regulation="DORA", tier="Tier-2", status=..., regulator_selection=..., consulting_selection=..., include_consulting_guidance=True, intelligence_package=None)`:
  1. Builds `extra_context = parsed_document.text` (if non-empty).
  2. Delegates to `build_brd_frd_report(...)` (the heavy lifter).
  3. Calls `_extract_obligations(report)` to flatten the BRD into a list of `Obligation` records.
  4. Returns a `RegulatoryAnalysis` whose `summary` is `"Regulatory analysis for {regulation} ({tier}) produced N obligations across A areas and T themes. Overall BRD coverage confidence is X%."`.
- `_extract_obligations(report)`:
  - One obligation per `RequirementItem` across `process/data/reporting/functional/non_functional` sections.
  - One extra obligation per `ControlCheckpointItem` (theme = `f"Control: {cp.stage}"`, regulatory_basis = `f"Control framework / {cp.stage}"`).
- Helpers:
  - `_deadline_hint(text)` — searches for any of `within`, `no later than`, `by `, `deadline`, `calendar days`, `business days`, `annually`, `quarterly`, `monthly`, `weekly` and returns a small snippet around the match.
  - `_control_expectations(req, checkpoints)` — keyword-overlap matching between the requirement text and control checkpoints; falls back to `["Establish documented control aligned to the obligation."]`.
  - `_evidence_needs(req)` — theme-driven defaults (incident workflow record, contract clause assessment, etc.), padded with `_EVIDENCE_DEFAULTS`.
  - `_risk_implication(req, report)` — `priority`-driven phrase: "material regulatory exposure and possible supervisory finding" for `Must`; "operational maturity gap with limited regulatory exposure" for `Could`; "regulatory readiness gap" for anything else.

#### 5.4.2 `agents/brd_rtm_agent.py` — Agent 2

**Business role.** Wraps the BRD into a downloadable `BRDArtifact` and builds the Requirements Traceability Matrix.

**Technical role.**

- `build(analysis, *, docx_export_path=None, tier=None)` returns `{"brd": BRDArtifact, "rtm": RTMArtifact}`.
- `_wrap_brd(analysis, docx_export_path, tier)`:
  - If `docx_export_path` is provided → calls `write_brd_docx(...)` and stores the path on the artefact. Failures swallowed; `docx_path` set to `None`.
- `_build_rtm(obligations, brd_report)`:
  - Indexes every functional requirement by its concatenated `category + requirement + detailed_requirement` (lower-cased).
  - For each obligation, `_pick_functional_requirement(...)` does a simple token-overlap match (tokens with length > 4) and returns the best-scoring FR id and text. Score 0 → `None` + placeholder text "Functional requirement to be elaborated during BRD/FRD workshop.".
  - `_system_process_impact(obligation)` is a templated sentence: `"Implementation impacts {function} operating within the {area} domain. Expected impact spans process, data, and reporting controls aligned to {regulatory_basis}."`.
- Returned metadata: `entry_count`, `covered_functions` (sorted set), `covered_areas` (sorted set), `top_themes` (5 most common via `Counter`).

#### 5.4.3 `agents/questionnaire_agent.py` — Agent 3

**Business role.** Produces the questionnaire package the user will answer.

**Technical role.** Three entry points, each returning a `QuestionnairePackage`:

- `from_report(brd, regulation="DORA", name=None)` — closed-loop, calls `build_package_from_report(...)`. Source `"generated_brd"`.
- `from_docx(path, regulation="DORA", name=None)` — uploaded path, calls `build_questionnaire_package(...)`. Source `"uploaded_brd"`.
- `from_package(package, source="uploaded_json", name=None)` — validates via `validate_package_schema(...)`, refuses to load if there are any issues. Source defaults to `"uploaded_json"` but can be overridden (the orchestrator passes `"db"` when loading from SQLite).

#### 5.4.4 `agents/recommendation_agent.py` — Agent 4

**Business role.** Produces the CXO action list.

**Technical role.** `recommend(questionnaire, scoring, *, min_severity="Watch", top_n_requirements=10, enrich_with_genai=False, branch_log=None)`:

1. Always runs `generate_recommendations(...)` deterministically.
2. If `enrich_with_genai=True` AND `self.client is not None` → tries `enrich_recommendations_with_genai(...)`; failures are silently swallowed.
3. Returns a `RecommendationResult` with the (possibly enriched) list, the filter, the cap, and a `used_genai` boolean.

---

### 5.5 Orchestrator (`orchestrator.py`)

**Business explanation.** The single coordination object held by `app.py`. Every page calls orchestrator methods; the pages never reach into agents or services directly. This is what makes the workflow stages explicit and testable.

**Technical explanation.**

- `RegulatoryWorkflowOrchestrator(*, client=None)` builds the four agents up-front, sharing one configured `GenAIClient` across them (so the LLM-bearing agents — 1 and 4 — use the same authenticated client).
- Static `parse_document(path, kind="regulation")` exposes the Document Parser stage as `document_parser.parse_document(...)`.
- Instance methods (one per stage):
  - `run_regulatory_analysis(...)` → Agent 1.
  - `gather_regulatory_intelligence(...)` (static) → Stage 1 + Stage 2 preview, exposed so Page 1 can preview sources without going through the BRD pipeline.
  - `run_brd_rtm(analysis, docx_export_path=None, tier=None)` → Agent 2.
  - `run_questionnaire_from_report(brd, ...)`, `run_questionnaire_from_docx(path, ...)`, `load_questionnaire_package(package, source="uploaded_json", ...)` → Agent 3 (three entry points).
  - `run_rules_engine(questionnaire, state)` (static) → builds the full active question list = applicable base + dynamic queue, runs `evaluate(...)`, computes top-10 gaps as `(rid, score)` pairs sorted ascending.
  - `run_recommendations(questionnaire, scoring, *, min_severity, top_n_requirements, enrich_with_genai, branch_log)` → Agent 4.
- Convenience: `run_full_pipeline(...)` chains Agents 1 → 3 in one call, useful for non-UI smoke tests.

---

### 5.6 Streamlit UI (`app.py`)

A 2148-line single file. Organisation:

#### 5.6.1 Bootstrap

- `load_dotenv()` at import time.
- `ensure_dirs(UPLOAD_DIR, OUTPUT_DIR, SAMPLE_DIR, DATA_DIR)` and `db.init_db()` on every run.
- `st.set_page_config(page_title="Regulatory Impact & Readiness Cockpit", page_icon="OK", layout="wide")`.
- Massive `_HERO_CSS` string injected via `st.markdown(... unsafe_allow_html=True)` — high-contrast PwC-orange theme that overrides every BaseWeb / Streamlit selector individually so the page reads consistently regardless of the user's OS dark-mode preference.

#### 5.6.2 Session-state initialisation

- `_DEFAULT_STATE` (see Section 4.3) is applied by `_init_session_state()`.
- `_probe_genai()` runs once per session (gated by `_genai_probed`):
  1. If `OPENAI_SKIP_API=true` → message `"OPENAI_SKIP_API=true in .env — offline mode forced."`.
  2. Else try `get_llm_api_key()`. Missing → message `"API key missing: …"`.
  3. Build `http_client`, run `preflight_openai_connectivity(...)`. If it returns `False` → message `"Preflight HTTP call did not return 200. Check API_KEY, model name, network/VPN, and proxy."`.
  4. On success → construct `ChatOpenAI`, wrap in `GenAIClient`, store under `_genai_client`.
- `_get_orchestrator()` lazily constructs a singleton orchestrator and caches it in session state.

#### 5.6.3 Sidebar

`PAGES = ["1. Setup", "2. BRD / FRD", "3. Questionnaire", "4. Assessment", "5. Dashboard", "6. Export"]` and `_render_sidebar()` (see Section 4.2 above).

#### 5.6.4 Page 1 — Setup

Code: `render_setup_page()` (lines ≈ 938-1072).

Highlights:
- Two columns for `regulation` (text input) and `tier` (selectbox `Tier-1`/`Tier-2`/`Tier-3`).
- Optional regulation file uploader (PDF/DOCX) → `save_upload(...)` → `db.save_document(kind="regulation")`.
- Radio mode: `"Use existing BRD/FRD"` vs `"Generate BRD/FRD from regulation"`.
- If `Use existing` → uploader for BRD DOCX + button to load the bundled sample `sample_data/DORA_Tier2_Detailed_DetailedBRDFRD.docx`.
- If `Generate` → `_render_regulatory_intelligence_block()`:
  - Stage-1 toggle indicator (success or warning based on `is_regulatory_search_enabled()`).
  - `_render_regulator_selector()` — multiselect with `ALL` plus every code from `APPROVED_REGULATORS`.
  - "Preview regulator sources" button → calls `gather_regulatory_intelligence(...)`. Empty results are classified into four diagnostic categories (DNS error, no results found, timeout/connect error, other) so the user knows whether to retry, change selection or fall back to PDF upload.
  - Renders the retrieved sources as a `pd.DataFrame` with `Source Type`, `Regulator`, `Publication Type`, `Regulation ID`, `Title`, `Publication Date`, `Confidence`, `URL`.
- Existing artefacts table at the bottom (lists every questionnaire saved in SQLite); selecting one and clicking "Load selected questionnaire" hydrates session state without re-running Agents 1-3.
- "Next" button (`_render_next_button(...)`) is disabled unless `setup_ready` is true (a regulation/BRD has been uploaded, an existing questionnaire was loaded, or the mode is "Generate").

#### 5.6.5 Page 2 — BRD / FRD

Code: `render_brd_page()` + helpers (lines ≈ 1182-1551).

- If mode is "Use existing BRD/FRD": `_run_agent2_for_uploaded_brd()` → `read_docx_requirements(...)` + `derive_impact_pairs(...)` → persists rows via `db.save_requirements(...)`. The parsed requirements are rendered as a dataframe.
- If mode is "Generate BRD/FRD from regulation":
  - `_run_agent1_and_agent2_with_status()`:
    1. Parses the regulation PDF via `orch.parse_document(...)` if one is stored.
    2. Runs `orch.run_regulatory_analysis(...)` inside a `st.status` block with a live `_log(msg)` writer.
    3. Builds the timestamped DOCX path (`{regulation}_BRD_FRD_<ts>.docx`) and runs `orch.run_brd_rtm(...)`.
    4. Stores `analysis`, `brd_artifact`, `rtm_artifact`, `brd_source` in session state.
  - The page then renders 5 KPI metrics (completeness coverage, accuracy coverage, used GenAI, total requirements, obligations count), `_render_regulation_source_panel(...)` (provenance), obligations preview (first 50), RTM preview (first 50), parsed BRD requirements, and `_render_brd_download_panel(...)`.
- `_render_brd_download_panel(...)` lays out 2 columns of downloads:
  1. Combined BRD+FRD DOCX (written once during the Agents 1 + 2 run by `write_brd_docx(...)` and reused for every subsequent download), Structured report JSON (`report.model_dump_json(indent=2)`).
  2. Requirements CSV (`_requirements_csv(...)` flattens every requirement section), Obligations JSON, RTM JSON+CSV.

#### 5.6.6 Page 3 — Questionnaire

Code: `render_questionnaire_page()` + `_run_agent3()` (lines ≈ 1557-1684).

- Two columns: "Run Agent 3" button → `_run_agent3(...)` which branches on the current mode:
  - "Generate BRD/FRD from regulation" → `orch.run_questionnaire_from_report(brd_artifact, ...)`.
  - "Use existing BRD/FRD" → `orch.run_questionnaire_from_docx(path, ...)`.
  - On success → `db.save_questionnaire(...)`, stamps the returned `qid` onto the package, resets `assessment_state`.
- The other column lets the user upload a previously-saved package JSON, validates it via `validate_package_schema(...)`, and persists it.
- 5 KPI metrics (requirements, closed questions, free-text, coverage %, overall confidence).
- Preview table of the first 25 questions.

#### 5.6.7 Page 4 — Assessment

Code: `render_assessment_page()` + `_render_question_card()` + `_render_free_text_questions()` (lines ≈ 1690-1856).

- Top bar: focus area selectbox (`All` or one of the closed-question areas), `Start / continue` button (creates an assessment row in SQLite if none exists), `Restart` (clears in-memory answers + persists), `New session` (creates a fresh assessment row).
- 5 KPI metrics: answered/applicable, dynamic follow-ups pending, skipped by funnel, branch decisions, assessment ID.
- "Adaptive branch trace" expander shows the last 15 entries from `state.branch_log` as a dataframe (`Parent`, `Answer`, `Source` (registry/generic), `Rule`, `Theme`, `Children`).
- `choose_next_question(...)` picks the next Q. `None` → success message + free-text expander + persist as `completed=True`.
- `_render_question_card(...)`:
  - Header label: `"Question NNN | <area> / <function> | Confidence X%"`. Dynamic questions append `"| Adaptive branch follow-up | rule: …"`.
  - Form with `st.multiselect` (for Multi Select) or `st.radio` (for everything else). Comments text area.
  - On submit:
    1. Writes the answer + display sequence + optional comments to `state.responses`.
    2. Calls `update_applicability_after_response(state, question, value, base_questions, package_regulation=session["regulation"])` → may add dynamic questions + skip downstream Qs + append a branch_log entry.
    3. Appends qid to `state.history`.
    4. `st.rerun()`.
- After every interaction the page calls `_persist_assessment_snapshot()` so SQLite always mirrors session state.

#### 5.6.8 Page 5 — Dashboard

Code: `render_dashboard_page()` (lines ≈ 1863-1992).

- Calls `_refresh_scoring_snapshot()` which delegates to `orch.run_rules_engine(...)`.
- 4 top-row metrics: compliance %, evaluation confidence, answered closed questions, pairs scored.
- Severity filter radio (`All`, `Critical`, `At risk`, `Watch`, `Ready`) applied to the area and function tables.
- Two side-by-side dataframes: area summary, function summary — styled via `_df_with_styling(...)` which applies a `RdYlGn` background gradient on the `Compliance %` column from 0-100.
- Area × Function heatmap expander built from `pair_heatmap_rows(...)`.
- Top gaps table (built from `scoring.top_gaps`).
- Agent 4 panel: minimum severity, top requirements (1-30), `Use GenAI to refine action wording` checkbox (auto-disabled if GenAI is offline), `Run Agent 4` button → `orch.run_recommendations(...)`. Result rendered as a dataframe (`id`, `severity`, `title`, `compliance %`, `owner`, `horizon`, `action`).

#### 5.6.9 Page 6 — Export

Code: `render_export_page()` (lines ≈ 1999-2113).

- Two columns of downloads:
  1. Questionnaire package JSON, responses+state JSON (uses `_jsonable_eval(...)` to convert tuple-keyed `pair_scores` into `"area | function"` strings), obligations JSON, RTM JSON.
  2. Excel report (via `write_excel_from_package(...)`), generated BRD/FRD DOCX (if Page 2 produced one).

#### 5.6.10 Router

`main()` dispatches on `st.session_state["page"]`:

```
"1. Setup"      -> render_setup_page()
"2. BRD / FRD"  -> render_brd_page()
"3. Questionnaire" -> render_questionnaire_page()
"4. Assessment" -> render_assessment_page()
"5. Dashboard"  -> render_dashboard_page()
"6. Export"     -> render_export_page()
otherwise       -> st.warning(f"Unknown page: {page}")
```

`main()` is called at the bottom of the file (it's a Streamlit script, not a CLI program).

---

## 6. Logic, Algorithms and Calculations

### 6.1 Confidence scoring (BRD, questionnaire, question)

#### 6.1.1 Per-requirement confidence (`brd_frd_generator.normalize_confidence_level`)

- Extract the first integer in the value (regex `(\d{1,3})`); clamp to [90, 100]. **Why 90% floor?** The system instructions tell the LLM never to emit below 90%, and the count-gate logic relies on that floor to keep the overall number meaningful.
- Missing value → keyword scan on `dora_alignment.lower()` against `("article", "ict risk", "incident", "third-party", "third party", "register", "resilience testing", "auditability", "governance", "backup", "recovery", "critical", "dora", "rts", "its")` → `"96%"` if any match, else `"93%"`.

#### 6.1.2 Per-section count gates (`brd_frd_generator.calculate_overall_confidence`)

| Section | Minimum count |
|---------|---------------|
| Process Requirements | 14 |
| Data Requirements | 14 |
| Reporting Requirements | 10 |
| Functional Requirements | 18 |
| Non-Functional Requirements | 10 |

If any gate fails, the average is forced down to 90% (`average = min(average, 90)`) regardless of how high the row-level confidences were. The final number is clamped to `[90, 100]`.

**Business reasoning.** A BRD that has high-quality requirements but is missing whole sections is **not** a high-confidence BRD — count gates prevent the LLM from masking missing sections with high per-row confidence values.

#### 6.1.3 Questionnaire overall confidence (`questionnaire_generator.validate_and_score_package`)

```
overall = round(
    (0.30 * req_cov          # mapped-requirement coverage
     + 0.18 * area_cov         # impact-area coverage
     + 0.18 * fn_cov           # function coverage
     + 0.14 * pair_depth       # closed questions per pair
     + 0.10 * free_text_ok     # 1 if MIN_FREE_TEXT <= count <= MAX_FREE_TEXT
     + 0.10 * (avg_q_conf/100) # mean per-question confidence
    ) * 100
)
overall = max(OVERALL_CONFIDENCE_FLOOR, min(100, overall))
```

#### 6.1.4 Per-question confidence

- Clamped to `[CONFIDENCE_FLOOR, 100]` (default `[90, 100]`) at construction.
- Inherited from the dominant pair's `confidence` (which is the mean of mapped requirements' confidences).
- Decremented by 1-2 points for follow-ups (`clamp_confidence(base_conf - 1)` for theme deep-dives, `- 2` for ownership/evidence/risk/remediation).

### 6.2 Impact-pair derivation (`questionnaire_generator.derive_impact_pairs`)

Step-by-step:

1. For each `Requirement`:
   - `text = source_section + category + requirement + detail + alignment + acceptance`.
   - `score_keywords(text, AREA_KEYWORDS)` → `Counter({area_label: count})`.
   - Select areas whose count is within 1 of the maximum (`v >= max(1, max_score - 1)`), then cap to 3.
   - Repeat for `FUNCTION_KEYWORDS` with default `"Compliance & Legal"`.
   - For every (area, function) Cartesian product, append the requirement_id.
2. Rank pairs by `(-len(set(requirement_ids)), area, function)` — i.e. most-shared pairs first, alphabetical tie-break — keep top 40.
3. Build `ImpactPair` records: requirement_ids deduplicated by first occurrence, average confidence rounded, regulatory_basis = `regulatory_basis_for(reqs, regulation)` (up to 3 alignment strings joined with `" | "`).

### 6.3 Funnel question synthesis (`questionnaire_generator.build_closed_questions_for_pair`)

The deterministic funnel for each impact pair:

```
L1  Coverage           (always asked, root, options=DEFAULT_OPTIONS["coverage"], weight 3)
      |
      +-- L2  Ownership   (parent=Coverage, trigger=NEGATIVE_COVERAGE, weight 3)
      |       |
      |       +-- L3  Risk (shared parent, see below)
      |
      +-- L2  Evidence    (parent=Coverage, trigger=NEGATIVE_COVERAGE, weight 2)
      |
      +-- L2  Risk        (parent=Coverage, trigger=NEGATIVE_COVERAGE ∪ WEAK_OWNERSHIP, weight 3)
      |       |
      |       +-- L3  Remediation  (parent=Risk, trigger=HIGH_RISK, weight 2)
      |
      +-- L2  Theme deep-dive (parent=Coverage, trigger=NEGATIVE_COVERAGE, weight 2)
                             (only when a dominant theme is detected for the pair)
```

`trigger_answers` are stored on each question so the static funnel can short-circuit branches when the parent answer is positive.

### 6.4 Adaptive branching (registry + generic)

See Section 5.3.13.4 for the full routing order. Key constraints (all env-overridable):

| Cap | Default | Where |
|-----|---------|-------|
| Max chain depth (parent → grandchild) | 3 | `MAX_DYNAMIC_FOLLOWUP_DEPTH` |
| Max follow-ups per parent question | 3 | `MAX_DYNAMIC_FOLLOWUPS_PER_PARENT` |
| Max dynamic Qs per whole assessment | 50 | `MAX_DYNAMIC_QUESTIONS_PER_ASSESSMENT` |
| Idempotency | Yes | `state.emitted_dynamic_ids` |

**Branch decision logging** (`_log_branch_decision`) appends to `state.branch_log` with `parent_question_id`, `selected_answer`, `branch_rule_id`, `branch_source` (`registry`/`generic`), `child_question_ids`, regulation, theme, question_kind, area, function, mapped_requirement_ids, depth — so every routing decision can be audited later (and rendered on Page 4 + Page 6 export).

### 6.5 Live scoring engine (readiness / compliance %)

`scoring_engine.evaluate(questions, state)`:

```
weight       = scoring_weight × (confidence / 100)
score        = score_value(answer, question)
total_num   += score × weight
total_den   += 100  × weight
area_num[a] += score × weight
area_den[a] += 100  × weight
# repeated per function, per (area, function), per requirement_id
```

```
compliance_score_pct = round(total_num / total_den × 100, 1)
```

**Evaluation confidence:**

```
avg_conf         = mean(confidence over all scored questions)
coverage_penalty = min(8.0, unanswered_count × 0.10)
bonus            = min(3.0, answered_count × 0.05)
evaluation_confidence_pct = round(
    max(90.0, min(99.0, avg_conf - coverage_penalty + bonus)), 1
)
```

**Worked example (illustrative).** Three closed questions, all with `confidence=92` and `scoring_weight=2`:
- Q1 answered "Complete" (100), Q2 "Partially" (55), Q3 unanswered.
- `weight = 2 × 0.92 = 1.84`.
- `total_num = (100 + 55) × 1.84 = 285.2`; `total_den = 2 × 100 × 1.84 = 368`.
- `compliance = round(285.2 / 368 × 100, 1) = 77.5%`.
- `avg_conf = 92.0`, `coverage_penalty = min(8.0, 1 × 0.10) = 0.10`, `bonus = min(3.0, 2 × 0.05) = 0.10`.
- `evaluation_confidence = round(max(90, min(99, 92 − 0.10 + 0.10)), 1) = 92.0%`.

### 6.6 CXO status thresholds & heatmap

| Score range | CXO status | Recommended executive action (verbatim) |
|-------------|------------|------------------------------------------|
| ≥ 85 | Ready | Maintain evidence and periodic validation. |
| 65-84 | Watch | Resolve targeted gaps before executive sign-off. |
| 40-64 | At risk | Prioritise remediation plan, owners and evidence. |
| < 40 | Critical | Escalate to governance and define funded remediation. |

Heatmap: `pair_heatmap_rows(pair_scores)` returns a DataFrame indexed by area with one column per function, populated with the rounded pair score or `None` if no questions were answered for that pair. Page 5 styles this DataFrame with the `RdYlGn` colormap (0→100).

### 6.7 Recommendation severity & ranking

`generate_recommendations(...)` builds three buckets (pair, area, top-N requirements) and sorts them by:

```python
(severity_rank, compliance_pct ascending)
where severity_order = {"Critical": 0, "At risk": 1, "Watch": 2, "Ready": 3}
```

Each recommendation's `suggested_owner` is picked via `_OWNER_BY_FUNCTION` (Cyber Security → CISO, Risk Management → CRO, etc.) with fallback `"Compliance / Programme Owner"`. Requirement-level recs are always owned by `"DORA Programme Manager"`.

`_SEVERITY_HORIZON`:
- Critical → `Immediate (0-30 days)`
- At risk → `Short-term (30-90 days)`
- Watch → `Medium-term (90-180 days)`
- Ready → `Steady-state (periodic)`

### 6.8 Stage 1 / Stage 2 source ranking

`RegulatoryIntelligencePackage.all_sources()` sorts by `(priority_index, -confidence_score)` where the priority index comes from `SOURCE_PRIORITY = [Official Regulator, Official Legislation, Consulting Guidance]`. This guarantees a Stage 2 consulting article (max confidence 0.95) can never out-rank a Stage 1 regulator article (max confidence 0.99) even when the consulting article's confidence happens to be higher.

---

## 7. Decision Points (Every `if/else`, threshold, constant)

This section enumerates the deterministic decisions sprinkled across the codebase that are not already obvious from the algorithms above.

### 7.1 Configuration switches (`.env`)

| Variable | Default | Decision driven |
|----------|---------|-----------------|
| `OPENAI_SKIP_API` | `false` | When `true`, `GenAIClient.try_create()` returns `None` immediately → forces offline BRD. |
| `OPENAI_SSL_STRATEGY` | `auto` | Selects truststore (`auto`/`os`/`windows_store`/`system`) vs certifi. |
| `REGULATORY_SEARCH_ENABLED` | `true` | Stage 1 master switch. Legacy `REGULATION_WEB_SEARCH` and `DORA_ENABLE_WEB_SEARCH` are also honoured. |
| `CONSULTING_SEARCH_ENABLED` | `false` (project default) | Stage 2 master switch. |
| `REGULATORY_SEARCH_BACKENDS` | `brave,yandex` (project) | Ordered DDGS backend list. |
| `REGULATORY_SEARCH_MAX_RESULTS` | `12` | Hits requested per query. |
| `REGULATORY_SEARCH_MAX_QUERIES` | `4` | Hard cap on total queries dispatched. |
| `REGULATORY_SEARCH_EARLY_STOP` | `10` | Stop dispatching once N unique URLs have been kept. |
| `REGULATORY_SEARCH_MAX_SECONDS` | `30` | Wall-clock cap for the whole Stage 1 dispatch loop. |
| `REGULATORY_SEARCH_DELAY_MS` | `600` | Sleep between sequential queries (anti-rate-limit). |
| `QUESTION_CONFIDENCE_FLOOR` | `90` | Per-question confidence clamp. |
| `OVERALL_QUESTIONNAIRE_CONFIDENCE_FLOOR` | `90` | Package-level clamp. |
| `MAX_AREA_FUNCTION_PAIRS` | `40` | Caps the funnel breadth. |
| `MIN_FREE_TEXT_QUESTIONS` | `5` | Minimum count used by the package scorer. |
| `MAX_FREE_TEXT_QUESTIONS` | `10` | Maximum count enforced by `build_free_text_questions`. |
| `MAX_DYNAMIC_FOLLOWUP_DEPTH` | `3` | Branching depth cap. |
| `MAX_DYNAMIC_FOLLOWUPS_PER_PARENT` | `3` | Branching width cap per parent. |
| `MAX_DYNAMIC_QUESTIONS_PER_ASSESSMENT` | `50` | Whole-assessment cap. |

### 7.2 Defensive defaults inside the code

| Location | Decision | Default |
|----------|----------|---------|
| `Obligation.priority` | When LLM doesn't specify | `"Should"` |
| `Obligation.confidence` | When LLM doesn't specify | `92` |
| `BRDArtifact.source` | When source is unknown | `"generated"` |
| `score_value` (no match) | Known-but-unscorable answer | `25.0` |
| `score_value` (empty) | Skip | `None` |
| `_apply_positive_skip` | `coverage`/default parent | Skips `{risk, remediation}` |
| `_apply_positive_skip` | `ownership`/`evidence` parent | Skips `{remediation}` only |
| `_resolve_regulation` | When parent + package both lack a label | Default `"DORA"` |
| `_pick_functional_requirement` (Agent 2) | When no FR overlap | `(None, "Functional requirement to be elaborated during BRD/FRD workshop.")` |
| `parse_document` | Unsupported extension | Empty `ParsedDocument` + warning |
| `parse_pdf` | Encrypted PDF | Empty result + warning `"PDF is password-protected; no text extracted."` |
| `parse_pdf` | No extractable text | Warning `"PDF appears to contain no extractable text. It may be a scanned image. Consider OCR (e.g. ocrmypdf) before re-uploading."` |
| `apply_confidence_floor` | Anything below 90% | Forced to 90% |
| `enforce_overall_confidence_floor` | If overall < 90% after enrichment | Every row forced to `"90%"` |

### 7.3 Heuristic ranking & deduplication

| Function | Behaviour |
|----------|-----------|
| `dedupe_impact_pairs` | Drops a pair whose requirement-IDs overlap ≥ 80% with an already-kept pair on the same axis. |
| `dedupe_and_resequence_questions` | Drops near-duplicates: same `question_kind`, ≥ 70% requirement overlap, AND same area or function. Also re-IDs every survivor sequentially. |
| `choose_next_question` ranking key | `(area_need DESC, function_need DESC, confidence + 10×weight DESC, question_id DESC)` |
| `is_positive_answer` | Returns True only if every selected value is in `POSITIVE_SIGNALS` ∪ `{"Not applicable"}`. |

---

## 8. Database schema (SQLite)

From `services/database.py` (`_SCHEMA`). All `CREATE TABLE IF NOT EXISTS` so init is idempotent.

```sql
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    kind            TEXT    NOT NULL CHECK (kind IN ('regulation','brd','frd','other')),
    path            TEXT    NOT NULL,
    mime            TEXT,
    size_bytes      INTEGER,
    regulation      TEXT,
    uploaded_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents(kind);

CREATE TABLE IF NOT EXISTS requirements (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id         INTEGER NOT NULL,
    requirement_id      TEXT    NOT NULL,
    section             TEXT,
    description         TEXT,
    impacted_areas      TEXT,      -- JSON array
    impacted_functions  TEXT,      -- JSON array
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_requirements_doc ON requirements(document_id);

CREATE TABLE IF NOT EXISTS questionnaires (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id             INTEGER,
    regulation              TEXT,
    name                    TEXT    NOT NULL,
    package_json            TEXT    NOT NULL,   -- full package, verbatim
    question_count          INTEGER,
    requirement_count       INTEGER,
    overall_confidence_pct  REAL,
    created_at              TEXT    NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_questionnaires_doc ON questionnaires(document_id);

CREATE TABLE IF NOT EXISTS assessments (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    questionnaire_id            INTEGER NOT NULL,
    name                        TEXT    NOT NULL,
    created_at                  TEXT    NOT NULL,
    updated_at                  TEXT    NOT NULL,
    completed_at                TEXT,
    compliance_score_pct        REAL,
    evaluation_confidence_pct   REAL,
    answered_count              INTEGER,
    state_json                  TEXT,   -- full AssessmentState
    evaluation_json             TEXT,
    recommendations_json        TEXT,
    FOREIGN KEY (questionnaire_id) REFERENCES questionnaires(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_assessments_q ON assessments(questionnaire_id);

CREATE TABLE IF NOT EXISTS responses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id   INTEGER NOT NULL,
    question_id     TEXT    NOT NULL,
    answer_json     TEXT,
    answered_at     TEXT    NOT NULL,
    FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE,
    UNIQUE (assessment_id, question_id)
);
CREATE INDEX IF NOT EXISTS idx_responses_assess ON responses(assessment_id);
```

Notes:
- Foreign-key cascades: deleting a document removes its requirements; deleting a questionnaire removes its assessments and responses.
- `package_json`, `state_json`, `evaluation_json`, `recommendations_json` carry full JSON so no schema migration is needed when those structures evolve.
- `responses` is a denormalised mirror of `state.responses` for fine-grained reporting; `state.responses` is also serialised into `assessments.state_json` losslessly (including dynamic follow-ups and skipped IDs).

The `docker-compose.yml` exposes a read-write `sqlite-web` browser at `http://localhost:8080` for ad-hoc inspection.

---

## 9. Configuration (`.env`)

The `.env` shipped with the repository contains every knob the runtime reads. Highlights:

```
API_KEY=<bearer token for PwC GenAI Shared Service>
GENAI_SHARED_SERVICE_BASE=https://genai-sharedservice-americas.pwcinternal.com
GENAI_SHARED_SERVICE_MODEL=azure.gpt-4o
OPENAI_TIMEOUT_SECONDS=180
OPENAI_MAX_TOKENS=2200
GENAI_CONTEXT_CHARS=6000
OPENAI_VERIFY_SSL=true
OPENAI_SKIP_API=false
OPENAI_SSL_STRATEGY=auto

REGULATORY_SEARCH_ENABLED=true
CONSULTING_SEARCH_ENABLED=false   # Stage 2 disabled pending team review
REGULATORY_SEARCH_BACKENDS=brave,yandex   # corporate network: DDG blocked
REGULATORY_SEARCH_MAX_RESULTS=12
CONSULTING_SEARCH_MAX_RESULTS=8
REGULATORY_SEARCH_TIMEOUT=10
REGULATORY_SEARCH_MAX_QUERIES=4
REGULATORY_SEARCH_EARLY_STOP=10
REGULATORY_SEARCH_MAX_SECONDS=30
REGULATORY_SEARCH_DELAY_MS=600

QUESTION_CONFIDENCE_FLOOR=90
OVERALL_QUESTIONNAIRE_CONFIDENCE_FLOOR=90
MAX_AREA_FUNCTION_PAIRS=40
MIN_FREE_TEXT_QUESTIONS=5
MAX_FREE_TEXT_QUESTIONS=10
```

`.gitignore` excludes `.env` from version control. Secrets must never be committed — `.env.example` is the only env file allowed in git.

---

## 10. Export Surface & File Outputs

| Artefact | Source | Format | Where produced |
|----------|--------|--------|----------------|
| BRD/FRD combined | Agents 1 + 2 | DOCX | Page 2 download + Page 6 download |
| Structured BRD report | `report.model_dump_json` | JSON | Page 2 download |
| Requirements flat list | `requirements_from_report(report)` | CSV | Page 2 download |
| Obligations | Agent 1 | JSON | Page 2, Page 6 downloads |
| RTM | Agent 2 | JSON + CSV | Page 2, Page 6 downloads |
| Questionnaire package | Agent 3 | JSON | Page 6 download |
| Responses + live results | Page 4 + Page 5 | JSON | Page 6 download |
| Questionnaire workbook | `write_excel_from_package` | XLSX with sheets `Summary`, `Impacted Functions Areas`, `Questionnaire`, `Free Text Questions`, `Funnel Logic`, `Scoring Rubric`, `Requirement Traceability` | Page 6 download |
| Stage 1 sources | Regulatory Intelligence Pipeline | JSON | Page 2 (per BRD) |

All file names are routed through `utils.file_utils.timestamped_name(...)` so re-running doesn't overwrite earlier artefacts.

---

## 11. Known Limitations / Not Implemented

These callouts are explicit in the code (or are observably absent) and should be communicated to senior management to set expectations:

- **Stage 2 (consulting guidance) is disabled** by default (`.env` `CONSULTING_SEARCH_ENABLED=false`). The fetcher code in `services/consulting_guidance_fetcher.py` is ready, but the Streamlit UI block that exposes the consulting selector is intentionally **commented out**; re-enabling requires both a config change and a UI snippet restoration (the inline comment in `app.py` calls this out).
- **Adaptive branch library is a vertical slice.** `services/branch_registry.py` only registers branches for `("DORA", "Incident reporting", "coverage", <answer>)` today. All other `(regulation, theme, kind, answer)` combinations fall through to the generic engine. Adding new branches is a pure-data change.
- **Native regulator adapters cover 4 regulators.** `services/native_regulator_search.py` registers EBA, ESMA, EIOPA and FCA only. The other 11 approved regulators fall through to the DDGS path; on a locked-down corporate network where DDGS is unreachable, those regulators will appear empty in Page 1 previews.
- **Schema validation depth.** `utils/json_utils.validate_package_schema` only deep-checks the first 5 items of each list. A questionnaire JSON could pass validation and still contain malformed rows beyond index 5.
- **Questionnaire generator targets DORA semantics.** Many helpers (`REGULATORY_TAXONOMY`, theme keyword bundles, theme deep-dive wording, theme free-text prompts) are DORA-specific even though the field `regulation` is free-form. Generating a questionnaire for another regulation will work technically but the theme deep-dives and free-text prompts will not be tailored to that regulation.
- **BRD generator schemas are DORA-shaped.** `DoraDetailedBRD` carries field names like `dora_alignment`. Other regulations are accepted as a free-form label but the schema does not rename per regulation.
- **No multi-user concurrency.** SQLite is a single-writer database. `docker-compose.yml` even warns: "If you are actively answering questions in the Streamlit cockpit AND editing the same rows through this UI, you may see 'database is locked' errors on the writer that loses the race."
- **No authentication / authorisation.** The app trusts whoever can reach its URL; there is no user model, no role separation, and no audit of who answered what.
- **Streamlit session state is in-memory.** While SQLite persists assessments, transient session keys (`_genai_client`, `_orchestrator`, intelligence package cache) are lost on `Reset everything` or on a process restart. Re-probing GenAI on every cold start is intentional.
- **Free-text answers are not scored.** They are stored in `state.responses` and exported, but the scoring engine excludes them from compliance / heatmap maths (`is_free_text` short-circuit). This is by design but should be highlighted when defending the readiness score.
- **Recommendations don't always carry mapped requirement IDs.** Area-only fallback recommendations have `mapped_requirement_ids=[]`. Use the pair-level recommendations for the most actionable evidence trail.
- **No automated tests.** There is no `tests/` directory. All validation is via the Streamlit UI and the bundled sample data.

---

*End of documentation. Every assertion above is grounded in the source code currently in the repository. When the code is updated, this document must be updated in lock-step.*
