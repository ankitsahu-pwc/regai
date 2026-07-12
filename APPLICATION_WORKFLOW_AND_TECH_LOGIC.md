# Reg AI RAP — Application Workflow & Technical Logic

> Panel-ready technical documentation.
> Every claim is anchored to source code (`file:line`). Nothing here is invented — where a feature is a placeholder, that is stated explicitly.

**Application name (as rendered):** *Reg AI RAP — A Complete Regulatory Impact Assessment & Readiness Platform* (`app.py:157`, `app.py:2009`)

---

## Table of Contents

1. [Feature / UI Action Catalogue](#1-feature--ui-action-catalogue)
2. [Where Each Feature Is Triggered](#2-where-each-feature-is-triggered)
3. [End-to-End Flow (per action)](#3-end-to-end-flow-per-action)
4. [Technical Logic Used](#4-technical-logic-used)
5. [Agent Details](#5-agent-details)
6. [BRD / FRD Generation Details](#6-brd--frd-generation-details)
7. [RTM Generation Details](#7-rtm-generation-details)
8. [Questionnaire Generation Details](#8-questionnaire-generation-details)
9. [Scoring Logic Details](#9-scoring-logic-details)
10. [Dashboard Logic](#10-dashboard-logic)
11. [Gap Identification Logic](#11-gap-identification-logic)
12. [Regulatory Intelligence Hub Logic](#12-regulatory-intelligence-hub-logic)
13. [Evidence & Reference Logic](#13-evidence--reference-logic)
14. [Guardrails / Hallucination Prevention](#14-guardrails--hallucination-prevention)
15. [Database / Storage Logic](#15-database--storage-logic)
16. [Export / Download Logic](#16-export--download-logic)
17. [Tech Stack Summary](#17-tech-stack-summary)
18. [End-to-End Workflow Diagram](#18-end-to-end-workflow-diagram)
19. [Panel Q&A Preparation](#19-panel-qa-preparation)
20. [Implementation Gaps (fully/partial/placeholder/hardcoded)](#20-implementation-gaps)

---

# 1. Feature / UI Action Catalogue

The Streamlit UI (`app.py`, ~8,509 lines) is a **six-page sidebar-navigated wizard**. All workflow logic is fronted by a single orchestrator singleton.


| #   | Page               | Sidebar label           | Renderer                                         | Purpose                                                                                  |
| --- | ------------------ | ----------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| 1   | Setup              | `1. Setup`              | `render_setup_page` (`app.py:3824-3956`)         | Institution type, client profile, regulator selection, regulation upload / sample loader |
| 2   | BRD / FRD          | `2. Generate BRD / FRD` | `render_brd_page` (`app.py:4139-4508`)           | Run Agents 1 + 2 (or parse an uploaded BRD) and render analysis / RTM / obligations      |
| 3   | Questionnaire      | `3. Questionnaire`      | `render_questionnaire_page` (`app.py:4820-4991`) | Auto-runs Agent 3, collects answers, submits to scoring                                  |
| 4   | Dashboard          | `4. Dashboard`          | `render_dashboard_page` (`app.py:5889-6076`)     | Impact + readiness scores, heatmap, recommendations                                      |
| 5   | Gap Identification | `5. Gap Identification` | `render_gap_page` (`app.py:8286-8395`)           | Five gap tabs + HITL review queue                                                        |
| 6   | Export             | `6. Export`             | `render_export_page` (`app.py:8402-8474`)        | Questionnaire JSON, Responses JSON, XLSX report                                          |


The router lives in `main()` at `app.py:8489-8508`; the sidebar is `_render_sidebar()` at `app.py:3694-3770`.

**Feature list (as exposed by real buttons / widgets):**

- Upload Regulation (Page 1, generate mode) — `st.file_uploader` inside `_render_optional_regulation_card` (`app.py:3777-3821`).
- Load Bundled Sample BRD (Page 1, upload mode) — `st.button "Use Bundled Sample BRD"` (`app.py:3870`).
- Regulator Scope selector (Page 1, generate mode) — `st.multiselect` in `_render_regulator_selector` (`app.py:2656-2679`).
- Preview Sources (auto, no button) — `_auto_fetch_regulatory_intelligence` (`app.py:2682-2708`) whenever regulator selection changes.
- Client Role-Aware selector (Page 1) — `st.multiselect "Institution Type(s)"` (`app.py:2542-2624`).
- Client Profile keywords (Page 1) — 6 multiselects via `_render_client_profile_selector` (`app.py:2481-2539`).
- Generate BRD / FRD (Page 2) — `st.button "Generate BRD / FRD"` (`app.py:4127-4134`). This single CTA drives Agents 1 → 2 → 3.
- Load Existing BRD from JSON (Page 3) — `st.button "Load Uploaded JSON"` (`app.py:4867`).
- Re-run Agent 3 (Page 3) — `st.button "Re-run Agent 3"` (`app.py:4849-4850`).
- Clear My Answers (Page 3) — `st.button "Clear My Answers"` (`app.py:4852-4856`).
- Answer Questions (Page 3) — bulk-render selectbox / multiselect / textarea per question (`_render_question_input_widget`, `app.py:5387-5503`).
- Calculate Impact & Readiness (Page 3, primary CTA) — `st.button` with `on_click=_submit_and_go_to_dashboard` (`app.py:4977-4986`).
- Regenerate Recommendations (Page 4, advanced) — `st.button` (`app.py:6051-6068`).
- Save (Page 5, HITL review) — `st.button "Save"` (`app.py:8361-8369`).
- Build Excel And Prepare Download → Download Excel Report (Page 6) — `st.button` + `st.download_button` (`app.py:8454-8474`).
- Download BRD + FRD (DOCX / JSON) (Page 2, Downloads expander) — `st.download_button` (`app.py:4745-4762`).
- Download Requirements CSV / Obligations JSON / RTM CSV / RTM JSON (Page 2) — `st.download_button` (`app.py:4772-4809`).
- Download Questionnaire JSON / Responses JSON (Page 6) — `st.download_button` (`app.py:8420-8449`).
- Download Gap Report JSON (Page 5) — `st.download_button` (`app.py:8388-8395`).
- Reset Everything (Sidebar) — `st.button` (`app.py:3765-3770`).
- Next → {page} — centered navigation `st.button` (`app.py:3628-3637`).

**Retired / hidden features** (present in code but suppressed):

- Consulting-firm Stage 2 selector — env `CONSULTING_SEARCH_ENABLED=false` (`.env:46`); UI wiring removed (`app.py:2043-2044`, `app.py:2349-2352`).
- Guardrail audit panel is intentionally hidden on Page 2 (`app.py:4334-4339`).

---

# 2. Where Each Feature Is Triggered

Comprehensive map (button label → file:line → callback → downstream module).


| Feature / Button                             | Widget file:line                                                                         | Callback function                                                                                    | Downstream code path                                                                                                     |
| -------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Institution Type multiselect                 | `app.py:2542-2624` (`_render_client_roles_selector`)                                     | inline write to `st.session_state["client_roles"]`                                                   | `services.client_roles.normalize_client_roles` (`client_roles.py:246-271`)                                               |
| Client Profile field multiselect (×6)        | `app.py:2481-2539` (`_render_client_profile_selector`)                                   | inline via `_keyword_multiselect`                                                                    | `services.client_profile.normalize_client_profile` (`client_profile.py:531-565`)                                         |
| Regulator Scope multiselect                  | `app.py:2656-2679` (`_render_regulator_selector`)                                        | inline write to `regulator_selection`                                                                | `services.search_config.APPROVED_REGULATORS` (`search_config.py:119-261`)                                                |
| **Auto fetch regulator sources** (no button) | `app.py:2682-2708` (`_auto_fetch_regulatory_intelligence`)                               | called from `render_setup_page` when fingerprint changes                                             | `services.regulatory_intelligence_service.gather_regulatory_intelligence` (`regulatory_intelligence_service.py:421-436`) |
| Upload Regulation                            | `app.py:3777-3821` (`_render_optional_regulation_card`)                                  | `utils.file_utils.save_upload` → `services.database.save_document`                                   | Files land in `uploads/`; DB row keyed by `id`                                                                           |
| Use Bundled Sample BRD                       | `app.py:3870`                                                                            | inline copy of `sample_data/DORA_Tier2_Detailed_DetailedBRDFRD.docx` → `save_document`               | Sets `brd_source="sample"`                                                                                               |
| **Generate BRD / FRD** (Mode B)              | `app.py:4127-4134` (`_render_step2_cta`)                                                 | `_run_agent1_and_agent2_with_status` (`app.py:3963-4050`)                                            | `orchestrator.run_regulatory_analysis` → `run_brd_rtm` → `_run_agent3`                                                   |
| Generate BRD / FRD (Mode A, upload)          | `app.py:4127-4134`                                                                       | `_run_agent2_for_uploaded_brd` (`app.py:4053-4110`)                                                  | `services.questionnaire_generator.read_docx_requirements` → `derive_impact_pairs` → `_run_agent3`                        |
| Re-run Agent 3                               | `app.py:4849-4850`                                                                       | `_run_agent3` (`app.py:5820-5882`)                                                                   | `orchestrator.run_questionnaire_from_report/docx`                                                                        |
| Load Uploaded JSON (questionnaire)           | `app.py:4867`                                                                            | inline validation                                                                                    | `utils.json_utils.validate_package_schema` → `orchestrator.load_questionnaire_package`                                   |
| Clear My Answers                             | `app.py:4852-4856`                                                                       | `_clear_questionnaire_answers` (`app.py:4994-5007`)                                                  | wipes `assessment_state.responses`, refreshes scoring, persists                                                          |
| Answer widgets (per question)                | `app.py:5387-5503` (`_render_question_input_widget`)                                     | inline `on_change`: `assessment_state.responses[qid] = value`                                        | `_refresh_scoring_snapshot` (`app.py:2196-2301`)                                                                         |
| **Calculate Impact & Readiness**             | `app.py:4977-4986`                                                                       | `_submit_and_go_to_dashboard` (`app.py:5172-5190`)                                                   | Creates SQLite assessment row → `_refresh_scoring_snapshot` → routes to page 4                                           |
| Regenerate Recommendations (advanced)        | `app.py:6051-6068`                                                                       | inline                                                                                               | `orchestrator.run_recommendations` (`orchestrator.py:286-307`)                                                           |
| Gap page tab widgets                         | `app.py:8322` (`st.tabs`)                                                                | inline HTML render                                                                                   | `services.gap_analysis.build_gap_report` (`gap_analysis.py:344-397`)                                                     |
| Save (HITL review row)                       | `app.py:8361-8369`                                                                       | inline write to `st.session_state["gap_review_state"]`                                               | in-memory only (not persisted to DB)                                                                                     |
| Build Excel                                  | `app.py:8454-8463`                                                                       | `services.questionnaire_generator.write_excel_from_package` (`questionnaire_generator.py:1510-1518`) | Writes to `outputs/…xlsx`                                                                                                |
| Reset Everything                             | `app.py:3765-3770`                                                                       | deletes all non-`_` state keys, `_init_session_state`, `st.rerun`                                    | SQLite is preserved                                                                                                      |
| Next →                                       | `app.py:3628-3637` (`_render_next_button`)                                               | `on_click=_set_page` (`app.py:3573-3578`)                                                            | Advances `st.session_state["page"]`                                                                                      |
| Reset regulation intel cache                 | Implicit — happens whenever the regulator+regulation fingerprint changes (`app.py:2708`) | `_fresh_intelligence_package` (`app.py:2636-2653`)                                                   | Avoids re-using stale package for Agent 1                                                                                |


---

# 3. End-to-End Flow (per action)

## 3.1 Click **Generate BRD / FRD** (Mode B — Generate from regulation)

Trigger: `app.py:4223-4235` (button rendered in the section for generate-from-regulation mode) → `_run_agent1_and_agent2_with_status` (`app.py:3963-4050`).

1. **Get orchestrator singleton** — `_get_orchestrator()` (`app.py:2166-2172`) constructs `RegulatoryWorkflowOrchestrator(client=_genai_client())` (`orchestrator.py:94-99`).
2. **Parse regulation document** (if any was uploaded on Page 1) — `orchestrator.parse_document(path)` (`orchestrator.py:105-108`) → `services.document_parser.parse_document` (`services/document_parser.py:67-104`) → PyMuPDF (`utils/pdf_parser.py`) or python-docx (`utils/docx_parser.py`).
3. **Take a fresh regulatory intelligence package** — `_fresh_intelligence_package()` (`app.py:2636-2653`) returns the cached `RegulatoryIntelligencePackage` iff its `regulation` and `regulator_selection` match current UI state; otherwise `None`.
4. **Run Agent 1 — Regulatory Analysis** — `orchestrator.run_regulatory_analysis(...)` (`orchestrator.py:114-139`) → `RegulatoryAnalysisAgent.analyze(...)` (`agents/regulatory_analysis_agent.py:101-355`), which:
  - Calls `services.brd_frd_generator.build_brd_frd_report(...)` (`brd_frd_generator.py:1920-2169`) → 8-call GPT-4o pipeline (`brd_frd_generator.py:722-795`).
  - Extracts `Obligation` objects deterministically from BRD requirement rows (`_extract_obligations`, `agents/regulatory_analysis_agent.py:442-510`).
  - Builds a Client-Role-Aware Interpretation via `services.client_roles.build_role_aware_interpretation` (`client_roles.py:772-942`) — deterministic keyword scoring.
  - Runs a **guardrail sweep** on every obligation string field via `services.guardrails.apply_text_guardrails` (`agents/regulatory_analysis_agent.py:265-340`, `guardrails.py:976-1018`).
5. **Run Agent 2 — BRD + RTM** — `orchestrator.run_brd_rtm(analysis, docx_export_path=...)` (`orchestrator.py:171-183`) → `BRDRTMAgent.build(...)` (`agents/brd_rtm_agent.py:54-86`):
  - `_wrap_brd(...)` calls `services.brd_frd_generator.write_brd_docx(...)` (`brd_frd_generator.py:1774-1913`) to persist a `.docx` under `outputs/{REGULATION}_BRD_FRD_{ts}.docx`.
  - `_build_rtm(...)` (`brd_rtm_agent.py:144-219`) creates one `RTMEntry` per obligation (traceability_id `TR-####`) by matching keywords against the BRD's functional-requirements section.
6. **Chain Agent 3 — Questionnaire** — `_run_agent3()` (`app.py:5820-5882`) invokes `orchestrator.run_questionnaire_from_report(...)` (`orchestrator.py:189-223`).
7. **Persist artefacts to SQLite** — questionnaire package saved via `services.database.save_questionnaire` (`database.py:404-452`); Agent 1 metadata and Agent 3 payload flow through the pre-persistence guardrail (`database.py:73-127`).
8. **Post-generation UI rendering** (`app.py:4241-4508`): confidence metrics row, Regulation Source panel, Source References panel, Role-Aware Interpretation panel, Regulatory Obligations expander (`app.py:4348-4413`), RTM expander (`app.py:4417-4461`), Parsed BRD Requirements expander (`app.py:4500-4504`), Downloads expander (`app.py:4735-4813`).
9. **Fallback path** — if the shared GenAI service is unreachable, `GenAIClient.try_create()` returns `None` (`genai_service.py:397-417`), the orchestrator carries `client=None`, and `build_brd_frd_report` falls back to `generate_offline_fallback_brd(regulation)` (`brd_frd_generator.py:850-1165`). Page 2 shows an *offline fallback* caption (`app.py:4312-4324`).

## 3.2 Click **Calculate Impact & Readiness** (Page 3)

Trigger: `_submit_and_go_to_dashboard` (`app.py:5172-5190`).

1. Ensure a SQLite `assessment` row exists — `_ensure_assessment_row_for_bulk_answers` (`app.py:5156-5169`) calls `database.create_assessment(...)` (`database.py:485-501`).
2. Persist responses — `database.upsert_responses(...)` (`database.py:594-616`).
3. Recompute all scores — `_refresh_scoring_snapshot()` (`app.py:2196-2301`), which internally:
  - `orchestrator.run_rules_engine(...)` (`orchestrator.py:264-280`) → `services.scoring_engine.evaluate(...)` (`scoring_engine.py:934-1085`).
  - `orchestrator.assess_impact_intelligence(analysis)` (`orchestrator.py:329-335`) → `services.ai_assessment_intelligence.assess_impact` (`ai_assessment_intelligence.py:708-712`).
  - `orchestrator.assess_readiness_intelligence(...)` (`orchestrator.py:337-353`) → `assess_readiness`.
  - `orchestrator.assess_confidence_intelligence(...)` (`orchestrator.py:313-327`) → `assess_confidence`.
  - `services.readiness_score.compute_weighted_readiness(...)` (`readiness_score.py:812-1008`) — DORA-weighted readiness.
  - `database.update_assessment_snapshot(...)` (`database.py:504-591`) — evaluation JSON + recommendations JSON, guardrail-swept.
4. Set `st.session_state["page"] = "4. Dashboard"` and `st.rerun()`.
5. On first entry to Page 4, `_autorun_recommendations_if_needed()` (`app.py:6079-6119`) invokes Agent 4 for the current scoring fingerprint.

## 3.3 Answer a single question

Trigger: change event in `_render_question_input_widget` (`app.py:5387-5503`).

1. New answer written into `assessment_state.responses[qid]`.
2. `_refresh_scoring_snapshot()` re-runs the rules engine end-to-end.
3. `_persist_assessment_snapshot()` (`app.py:2304-2333`) writes state, evaluation, recs to SQLite.
4. Live severity pill under the question is repainted by `_render_question_score_badge` (`app.py:5538-5638`).

## 3.4 Preview Sources (Regulatory Intelligence Hub)

There is **no explicit "Preview Sources" button** — sources auto-fetch when the fingerprint (regulation label + selected regulator codes) changes (`app.py:2682-2708`). The user sees a live `st.status("Fetching …")` block, and the resulting `RegulatoryIntelligencePackage` is rendered by `_render_intelligence_sources_table` (`app.py:2772-2820`).

## 3.5 Download BRD DOCX

Trigger: `st.download_button("Download BRD + FRD (DOCX)", ...)` (`app.py:4745-4751`). Reads (or lazily rebuilds) the DOCX via `_build_or_get_brd_docx(brd_artifact)` (`app.py:4643-4671`) which delegates to `services.brd_frd_generator.write_brd_docx` (`brd_frd_generator.py:1774-1913`) using `python-docx`.

## 3.6 Generate / Export Questionnaire (Page 6)

- **Build Excel** — `write_excel_from_package(path, package)` (`questionnaire_generator.py:1510-1518` → `write_excel`, `questionnaire_generator.py:1386-1501`) using `openpyxl` (`requirements.txt:6`). Contains Metadata, Requirements, Impact Pairs, Questions (with per-option scoring), Answer Scores legend.
- **Download Questionnaire JSON** — `json.dumps(pkg)` (`app.py:8420-8426`).
- **Download Responses JSON** — envelope with responses, evaluation snapshot, and recommendations (`app.py:8443-8449`).

---

# 4. Technical Logic Used

Concentrated per concern; every claim is line-referenced.

### 4.1 Python / dataclass / Pydantic model architecture

- **Workflow contracts** live in `models/workflow_models.py` (`ParsedDocument`, `Obligation`, `RegulatoryAnalysis`, `BRDArtifact`, `RTMEntry`, `RTMArtifact`, `QuestionnairePackage`, `AssessmentResponse`, `ScoringResult`, `RichRecommendation`, `RecommendationResult`, `ConfidenceAssessment`, `ImpactDimension`, `ImpactAssessment`, `ReadinessDimension`, `ReadinessAssessment`).
- **BRD content schema** is Pydantic v2 (via `langchain-openai.with_structured_output`) — `BulletItem`, `RequirementItem`, `ControlCheckpointItem`, `RiskItem`, `DeliveryPhaseItem`, and the top `DoraDetailedBRD` (`brd_frd_generator.py:167-321`).
- **Questionnaire AI schemas** — `AIOption`, `AIQuestion`, `AIFreeTextQuestion`, `AIQuestionBank`, `AIFreeTextBank` (`ai_questionnaire_generator.py:139-473`).

### 4.2 LangChain / LLM call plumbing

- Provider: **PwC GenAI Shared Service** (OpenAI-compatible endpoint) reached via LangChain `ChatOpenAI` (`genai_service.py:256-271`). Default base URL `https://genai-sharedservice-americas.pwcinternal.com` (`genai_service.py:128-131`), default model `azure.gpt-4o` (`.env:7`, `genai_service.py:132`).
- Structured output is enforced with `llm.with_structured_output(schema_model)` (`genai_service.py:346`) so each LLM call returns a validated Pydantic instance.
- `GenAIClient.try_create()` (`genai_service.py:397-417`) returns `None` when `OPENAI_SKIP_API=true`, `API_KEY` missing, or the pre-flight HTTP check fails — the whole pipeline degrades to deterministic offline fallbacks.
- Retry / timeout: `ChatOpenAI(max_retries=3, timeout=180s)` (`genai_service.py:262-263`), `httpx` connect=45 s, read=180 s, write=60 s (`genai_service.py:201-206`), plus a one-shot length-limit retry `generate_with_length_retry(max_retry_tokens=12000)` (`genai_service.py:458-510`).
- Every LLM call in the BRD pipeline is wrapped in `services.guardrails.safe_generate` (`guardrails.py:1310-1502`) which hardens the prompt, validates the response, and rejects payloads whose verified-citation ratio falls below `min_citation_ratio` (default 0.5).

### 4.3 Web / RAG / Search logic

- **Stage 1** (approved regulators) — real HTTP calls. Native adapters (`services.native_regulator_search.py:306-331`) use `httpx.get` (`native_regulator_search.py:27`) against EBA/ESMA/EIOPA/FCA site-search URLs. Fallback: `DDGS` (from `ddgs` or `duckduckgo-search`) — `services.official_regulation_fetcher.py:47-53`. Every hit is post-filtered by `services.search_config.is_regulator_url` (`search_config.py:413-417`) against a hostname allow-list.
- **Stage 2** (consulting guidance) — DDGS only, always anchored on Stage 1 hits (`services.consulting_guidance_fetcher.py:131-144`). Currently **disabled** at runtime (`.env:46`).
- No vector database, no embeddings, no semantic search — this is **not a classical RAG**. Ground truth is passed to the LLM as **plain-text context** (`context_text` field of `RegulatoryIntelligencePackage`, capped at ~12,000 chars in `_format_context`, `regulatory_intelligence_service.py:216-267`) and Agent 1's `build_brd_frd_report` truncates further to `settings.context_chars` (default 6000) before sending (`genai_service.py:454`).
- **Source–requirement matching** is deterministic token/regex overlap (`services.source_traceability.SourceCatalogue.match`, `source_traceability.py:227-278`).

### 4.4 SQLite logic

- File: `data/app.db` (`services.database.DEFAULT_DB_PATH`, `database.py:39`). Overridable via `APP_DB_PATH` env var (`database.py:134-140`).
- Tables (idempotent schema, `database.py:168-238`): `documents`, `requirements`, `questionnaires`, `assessments`, `responses` — with FK cascade.
- Every write goes through `session()` context manager (`database.py:153-161`) that commits on exit.
- Pre-persistence guardrail sweep on every generative payload (`database.py:64-127`), controlled by `APP_PERSIST_GUARDRAIL` env (`off` | `warn` (default) | `strict`).

### 4.5 Scoring formulas

Compact form; see §9 for the full derivation.

- Per-question **weight** = `scoring_weight × max(1, impact_weight) × (confidence/100)` (`scoring_engine.py:983-986`).
- **Compliance score** = `Σ(score × weight) / Σ(100 × weight) × 100` (`scoring_engine.py:1046-1050`).
- Free-text answers get their weight multiplied by 0.3 (`scoring_engine.py:998`).
- Evaluation confidence = `avg_conf + coverage_bonus − coverage_penalty + quant_bonus`, clamped to `[40, 99]` (`scoring_engine.py:1055-1063`).
- **Weighted readiness** = area-mean × area-weight, summed; weights sum to 100 (`readiness_score.py:83-91`, `812-1008`).
- **Readiness band** (from `services.severity.py:65-66`): ≥75 Ready, ≥50 Watch, ≥25 At risk, <25 Critical.

### 4.6 Dataframe / table generation

- **Streamlit** renders parsed requirements, obligations, RTM as either `st.dataframe` (`app.py:4348-4413`, `4417-4461`, `4519-4551`) or a custom HTML table with hyperlinked source cells (`_render_parsed_requirements_html`, `app.py:4595-4636`).
- **DOCX** requirement/control tables built with `python-docx` `Table Grid` style; header 9pt bold, body 8.5pt Arial (`brd_frd_generator.py:1433-1474`).
- **Excel** built via `openpyxl` in `questionnaire_generator.write_excel` (`questionnaire_generator.py:1386-1501`).
- **CSV** built with Python `csv` module (`app.py:4674-4712`).

### 4.7 Validation rules

- `utils.json_utils.validate_package_schema` (`utils/json_utils.py:80-130`) — enforces the questionnaire package contract (top-level keys, required fields per requirement / pair / question).
- `services.guardrails.check_before_persist` (`guardrails.py:1230-1263`) — walks every string leaf of a payload before DB write.

### 4.8 Confidence calculation

Two coexisting notions:

- `evaluation_confidence_pct` — coverage-driven, computed inside `scoring_engine.evaluate` (`scoring_engine.py:1055-1063`).
- `ConfidenceAssessment.overall_score` — evidence-driven, computed by `services.ai_assessment_intelligence.assess_confidence` (`ai_assessment_intelligence.py:297-303`, deterministic baseline at 188-294).

Overall confidence surfaced on the BRD headers is `services.brd_frd_generator.calculate_overall_confidence(report)` (`brd_frd_generator.py:528-559`).

### 4.9 Fallback logic

- LLM offline → orchestrator instantiated with `client=None`; every generation call has a deterministic path (BRD offline scaffold at `brd_frd_generator.py:850-1165`; questionnaire manual-review placeholders at `ai_questionnaire_generator.py:1296-1391`; recommendations at `rich_recommendation_service.py` `_deterministic_rich_recommendations`).
- Network broken → `preflight_openai_connectivity()` (`genai_service.py:239-247`) short-circuits `GenAIClient.try_create()`.
- Regulation intelligence empty → `services.regulatory_intelligence_service.offline_baseline_for(regulation)` inline hardcoded DORA baseline (`regulatory_intelligence_service.py:176-188`) or neutral disclaimer for other regulations.
- Guardrail veto → payload rejected, deterministic baseline retained.

### 4.10 Error handling

- Try/except at every LLM entry point returns `(None, reason)` and never raises to Streamlit (`brd_frd_generator.py:821-843`).
- File parse failures return an empty `ParsedDocument` with `warning_message` set (`services/document_parser.py:89-104`).
- Streamlit-visible errors: `st.error(...)` blocks in `render_brd_page` (`app.py:4271-4273`), the questionnaire loader (`app.py:4881-4893`), the gap page (`app.py:8308-8319`), and export (`app.py:8410-8412`).

---

# 5. Agent Details

The four agents are declared in `agents/__init__.py:14-24` and wired through `orchestrator.py:94-99`.

## Agent 1 — Regulatory Analysis

- **Class:** `RegulatoryAnalysisAgent` (`agents/regulatory_analysis_agent.py:91-355`)
- **Purpose:** parse regulation + retrieved intelligence into structured `RegulatoryAnalysis` (obligations, themes, impacted areas, role-aware interpretation, back-referenced BRD).
- **Input:** `parsed_document?`, `regulation`, `tier`, `regulator_selection`, `include_consulting_guidance`, `intelligence_package?`, `client_roles`, `client_profile`.
- **Output:** `RegulatoryAnalysis` (`models/workflow_models.py:174-227`).
- **Prompt:** eight bundle prompts inside `brd_frd_generator.generate_detailed_dora_brd` (`brd_frd_generator.py:722-795`) — see §6 for verbatim text.
- **Model / settings:** default `azure.gpt-4o`, `max_tokens=6000`, `context_chars=6000`. Temperature not sent by default; if `OPENAI_SEND_TEMPERATURE=true`, temperature = 0.10 (`genai_service.py:259-270`).
- **Tools used:** regulatory intelligence package (Stage 1 official regulators via httpx + DDGS, `services.regulatory_intelligence_service.gather_regulatory_intelligence`), source-traceability matcher, guardrails, client-roles engine, client-profile prompt block.
- **Deterministic vs LLM:** LLM-driven BRD generation, deterministic obligation extraction (`_extract_obligations`, `agents/regulatory_analysis_agent.py:442-510`), deterministic guardrail sweep on every string field.
- **Uses regulatory evidence:** yes — `context_text` from `RegulatoryIntelligencePackage`.
- **Hallucination guardrails:** yes — `harden_instruction`, `safe_generate`, per-field `apply_text_guardrails` post-pass (`agents/regulatory_analysis_agent.py:265-340`).
- **Stores results:** yes — analysis dict is persisted on the DB via the questionnaire/assessment writes; the BRD DOCX lands in `outputs/`.

## Agent 2 — BRD + RTM

- **Class:** `BRDRTMAgent` (`agents/brd_rtm_agent.py:51-219`)
- **Purpose:** wrap Agent 1's BRD into a `BRDArtifact` + build the `RTMArtifact` (traceability rows) + write DOCX.
- **Input:** `RegulatoryAnalysis`, optional DOCX export path, optional tier.
- **Output:** `{"brd": BRDArtifact, "rtm": RTMArtifact}`.
- **Model / settings:** none — Agent 2 does **not** call an LLM. Pure deterministic joins.
- **Tools used:** `services.brd_frd_generator.write_brd_docx` (`brd_frd_generator.py:1774-1913`).
- **Uses regulatory evidence:** yes — carries `analysis.metadata["source_references_by_item"]` into the DOCX and each RTM row.
- **Hallucination guardrails:** by construction — no LLM.
- **Stores results:** DOCX file to `outputs/`; artefact objects live in `st.session_state` and are written into the questionnaire package metadata.

## Agent 3 — Questionnaire

- **Class:** `QuestionnaireAgent` (`agents/questionnaire_agent.py:74-226`)
- **Purpose:** produce a structured questionnaire package tailored to the BRD, obligations, RTM, impact severity, and selected institution types.
- **Input:** `BRDArtifact` (or DOCX / JSON), `analysis`, `rtm`, `impact`, `readiness`, `client_roles`, `client_profile`.
- **Output:** `QuestionnairePackage` — a dict validated by `utils/json_utils.py`.
- **Prompt:** verbatim system prompt at `ai_questionnaire_generator.py:679-742` and user prompt template at `ai_questionnaire_generator.py:745-857` (see §8).
- **Model / settings:** `azure.gpt-4o`, temperature 0.10 or omitted (as above), `max_tokens=6000`.
- **Tools used:** `services.ai_questionnaire_generator.generate_ai_questionnaire`, `services.questionnaire_generator.build_package_from_report/build_questionnaire_package`, `services.questionnaire_enhancer.enhance_questionnaire_package`, `services.question_style_enhancer.diversify_question_styles`.
- **Deterministic vs LLM:** LLM generation; deterministic post-processing for dedup/resequencing, style diversification (Multi Select conversion, quantitative brackets), impact-severity re-weighting, role filter.
- **Uses regulatory evidence:** yes — BRD requirement snippets, obligation snippets, RTM entries, and source references appear in every prompt (`_funnel_prompt`).
- **Hallucination guardrails:** system prompt rule set at `ai_questionnaire_generator.py:679-742`, plus `harden_instruction` + `safe_generate`.
- **Stores results:** `services.database.save_questionnaire` (`database.py:404-452`), guardrail-swept.

## Agent 4 — Recommendations

- **Class:** `RecommendationAgent` (`agents/recommendation_agent.py:52-178`)
- **Purpose:** produce compact + rich recommendations from the scoring output.
- **Input:** `QuestionnairePackage`, `ScoringResult`, optional `analysis`, `client_roles`, `enrich_with_genai`.
- **Output:** `RecommendationResult` with compact + rich lists (`models/workflow_models.py:602-618`).
- **Prompt:**
  - Compact rewrite: verbatim at `recommendation_service.py:386-393`.
  - Rich recommendation: verbatim at `rich_recommendation_service.py:740-750`.
- **Model / settings:** `azure.gpt-4o`, same defaults.
- **Tools used:** `services.recommendation_service.generate_recommendations` (deterministic backbone), `services.rich_recommendation_service.build_rich_recommendations`, `services.recommendation_evaluator.attach_evaluations` (deterministic scoring of each rec across coverage / specificity / actionability / grounding).
- **Deterministic vs LLM:** always produces a deterministic draft first; LLM is optional and only rewrites text.
- **Uses regulatory evidence:** rich path passes `regulation_snippets` (top obligation snippets) as context (`rich_recommendation_service.py:711-739`).
- **Hallucination guardrails:** wrapped in `safe_generate` with a source-corpus built from obligation text; on veto, deterministic draft is retained.
- **Stores results:** JSON in `assessments.recommendations_json` via `services.database.update_assessment_snapshot` (`database.py:504-591`).

Note: **Adaptive follow-up branch generator** — `services.ai_branch_generator.generate_option_followups` (`ai_branch_generator.py:434-527`) — is a Fifth agent-like helper called from `scoring_engine.dynamic_followups`. It generates 1–3 option-aware child questions (LLM prompt at `ai_branch_generator.py:255-270`, offline templates at `49-202`), then registered through `scoring_engine.materialize_branch_spec`.

---

# 6. BRD / FRD Generation Details

The generator lives entirely in `services/brd_frd_generator.py` (~2,211 lines). Output object is `DoraDetailedBRD` (`brd_frd_generator.py:270-291`).

## 6.1 Sections produced

Rendered by `write_brd_docx` (`brd_frd_generator.py:1774-1913`):


| #       | DOCX heading                                                        | Source field                              | Content type                                                                                                                                                                                                                  |
| ------- | ------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| —       | Title / subtitle / tier banner                                      | inline                                    | Centered header                                                                                                                                                                                                               |
| —       | Overall AI Confidence paragraph                                     | `calculate_overall_confidence(report)`    | Auto-computed %                                                                                                                                                                                                               |
| 1       | Executive Summary                                                   | `executive_summary`                       | Bullets                                                                                                                                                                                                                       |
| 2       | Objectives                                                          | `objectives`                              | Bullets                                                                                                                                                                                                                       |
| 3       | Scope                                                               | `scope`                                   | Bullets                                                                                                                                                                                                                       |
| 4       | Stakeholders                                                        | `stakeholders`                            | Bullets                                                                                                                                                                                                                       |
| 5       | Current State Challenges                                            | `current_state_challenges`                | Bullets                                                                                                                                                                                                                       |
| 6       | Target State Overview                                               | `target_state_overview`                   | Bullets                                                                                                                                                                                                                       |
| 7.1     | Process Requirements                                                | `process_business_requirements`           | Requirement table                                                                                                                                                                                                             |
| 7.2     | Data Requirements                                                   | `data_business_requirements`              | Requirement table                                                                                                                                                                                                             |
| 7.3     | Reporting Requirements                                              | `reporting_business_requirements`         | Requirement table                                                                                                                                                                                                             |
| 8       | Functional Requirements                                             | `functional_requirements`                 | Requirement table                                                                                                                                                                                                             |
| 9.1     | Control Checkpoints Across {regulation} Lifecycle                   | `control_framework.lifecycle_checkpoints` | Table (Stage / Checkpoint / Requirement / Tooling / Evidence / Source Refs)                                                                                                                                                   |
| 9.2–9.6 | Preventive / Detective / Corrective / Governance / Tooling controls | bullet lists                              | Bullets only                                                                                                                                                                                                                  |
| 10      | Non-Functional Requirements                                         | `non_functional_requirements`             | Requirement table                                                                                                                                                                                                             |
| 11      | Assumptions                                                         | `assumptions`                             | Bullets                                                                                                                                                                                                                       |
| 12      | Dependencies                                                        | `dependencies`                            | Bullets                                                                                                                                                                                                                       |
| 13      | Risks & Mitigations                                                 | `risks_and_mitigations`                   | Risk table                                                                                                                                                                                                                    |
| 14      | Success Criteria                                                    | `success_criteria`                        | Bullets                                                                                                                                                                                                                       |
| 15      | Appendix                                                            | `appendix`                                | Bullets (includes Requirement Catalogue, Data Dictionary, Rule Library, Dashboard Catalogue, Glossary, Evidence Taxonomy, Workshop Templates, Control Mapping, Traceability Matrix, KPI/KRI Dictionary, Control Test Scripts) |
| 16      | Workshop Delivery Plan & Timelines                                  | `workshop_delivery_plan`                  | Phases + success factors                                                                                                                                                                                                      |
| 17      | Source References (optional)                                        | metadata                                  | Master catalogue table + per-requirement traceability                                                                                                                                                                         |


**Note:** There is *no populated RTM grid in the DOCX*. RTM appears as an appendix bullet reference; the actual RTM grid is emitted separately as JSON/CSV via Page 2 Downloads.

## 6.2 Business Requirements table — every column

Renderer: `_add_requirements_table` (`brd_frd_generator.py:1516-1563`).


| Column                                       | Code field                             | Generation source                                                                                                                                                                                                                             | Downstream usage                                                                                                                                              |
| -------------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ID                                           | `RequirementItem.id`                   | **LLM** initial → **deterministic** rewrite `normalize_requirement_ids` to `BR-PRO-###`, `BR-DAT-###`, `BR-REP-###`, `FR-###`, `NFR-###` (`brd_frd_generator.py:373-385`). Offline path uses hardcoded IDs (`brd_frd_generator.py:947-1035`). | Traceability key in RTM rows (`obligation.source_requirement_id`), in questionnaire `mapped_requirement_ids`, and in per-requirement source-reference blocks. |
| Category                                     | `RequirementItem.category`             | LLM (or offline hardcoded).                                                                                                                                                                                                                   | Grouping in area/function classification (`services.questionnaire_generator.impacted_labels_for_requirement`).                                                |
| Requirement                                  | `RequirementItem.requirement`          | LLM (or offline).                                                                                                                                                                                                                             | BRD row title, funnels into questionnaire snippets.                                                                                                           |
| Detailed Requirement                         | `RequirementItem.detailed_requirement` | LLM (or offline).                                                                                                                                                                                                                             | Feeds obligation `compliance_requirement`.                                                                                                                    |
| {Regulation} Alignment (e.g. DORA Alignment) | `RequirementItem.regulation_alignment` | LLM (or offline). Header label is dynamic per regulation (`brd_frd_generator.py:1527-1529`). For non-DORA runs, `_relabel_for_regulation` scrubs DORA-specific citations (`brd_frd_generator.py:103-125`).                                    | Cited by RTM's `regulatory_basis` and by questionnaire `regulatory_basis`.                                                                                    |
| Priority                                     | `RequirementItem.priority`             | LLM MoSCoW (Must / Should / Could / Won't).                                                                                                                                                                                                   | Feeds `_risk_implication` in Agent 1 (`agents/regulatory_analysis_agent.py:587-599`) and `obligation.priority`.                                               |
| Acceptance Criteria                          | `RequirementItem.acceptance_criteria`  | LLM (or offline).                                                                                                                                                                                                                             | Feeds `_control_expectations` / `_evidence_needs` heuristics in Agent 1.                                                                                      |
| AI Confidence                                | `normalize_confidence_level(...)`      | **Deterministic** normalization to `[90%, 100%]` (`brd_frd_generator.py:328-352`). If LLM omits or provides a weak value, defaults to `93%` or `96%` depending on whether the alignment mentions strong terms.                                | Header confidence badge; drives `calculate_overall_confidence`.                                                                                               |
| Source References                            | `_format_sources_cell(refs)`           | **Deterministic** — matched by `services.source_traceability.SourceCatalogue.match` (`source_traceability.py:227-278`) after Agent 1 has populated `metadata["source_references_by_item"]`.                                                   | Displayed in DOCX + JSON. Also underpins the Section 17 catalogue.                                                                                            |


**Why these columns exist:** they are the union of (a) BABOK v3 business-analysis conventions and (b) IREB CPRE requirement-quality attributes — dictated by the system prompt at `genai_service.py:274-310`.

## 6.3 Functional / Non-Functional table columns

Same schema as above (single Pydantic `RequirementItem`). IDs are namespaced `FR-###` / `NFR-###`.

## 6.4 Control Framework table (Section 9.1)

Renderer: `_add_control_framework_section` (`brd_frd_generator.py:1565-1614`).


| Column              | Code field                                                  | Source                |
| ------------------- | ----------------------------------------------------------- | --------------------- |
| Stage               | `ControlCheckpointItem.stage`                               | LLM / offline         |
| Control Checkpoint  | `.control_checkpoint`                                       | LLM / offline         |
| Requirement         | `.requirement`                                              | LLM / offline         |
| Tooling Expectation | `.tooling_expectation`                                      | LLM / offline         |
| Evidence            | `.evidence`                                                 | LLM / offline         |
| Source References   | `source_references_by_item[control_key(stage, checkpoint)]` | Deterministic matcher |


## 6.5 Risks table (Section 13)

Renderer: `_add_risk_section` (`brd_frd_generator.py:1616-1634`).


| Column     | Source                                |
| ---------- | ------------------------------------- |
| Risk       | `RiskItem.risk` (LLM / offline)       |
| Impact     | `RiskItem.impact` (LLM / offline)     |
| Mitigation | `RiskItem.mitigation` (LLM / offline) |
| Owner      | `RiskItem.owner` (LLM / offline)      |


No Source-References column in the DOCX; metadata still keys sources by `RISK:<first 120 chars>` (`source_traceability.py:437-438`).

## 6.6 The eight LLM calls (verbatim component instructions)

Located at `brd_frd_generator.py:748-795`.

1. **Executive Summary, Objectives, Scope** — `FrontMatterBundle`.
2. **Stakeholders, Current State Challenges, Target State Overview** — `AnalysisBundle`.
3. **7.1 Process Requirements** — `RequirementSection`.
4. **7.2 Data Requirements** — `RequirementSection`.
5. **7.3 Reporting Requirements** — `RequirementSection`.
6. **8 & 10. Functional + Non-Functional Requirements** — `SolutionRequirementsBundle`.
7. **9 & 11-14. Controls, Assumptions, Dependencies, Risks, Success Criteria** — `GovernanceBundle`.
8. **15-16. Appendix & Workshop Delivery Plan** — `ClosureBundle`.

Each call passes:

- `system_instruction` — the anti-hallucination-hardened default at `genai_service.py:274-310`.
- `component_name` / `component_instruction` — bundle-specific text.
- `context` — assembled by `build_brd_frd_report` from the intelligence package (`context_text`), the uploaded regulation text (if any), the **client-role directive** (`brd_frd_generator.py:1987-2003`), and the **client-profile directive** (`services.client_profile.client_profile_prompt_block`).

## 6.7 Deterministic post-processing pipeline

Always runs after LLM (or offline fallback) — `brd_frd_generator.py:2039-2050`:

1. `ensure_minimum_detail(report, regulation)` — pad rows/bullets up to minimums (`process 14, data 14, reporting 10, functional 18, non-functional 10, lifecycle checkpoints 12, risks 10`) (`brd_frd_generator.py:388-394`).
2. `apply_confidence_floor(report)` — normalize per-row confidence to `[90, 100]`.
3. `normalize_requirement_ids(report)` — reassign IDs sequentially by section.
4. `enforce_overall_confidence_floor(report)` — raise low rows so overall ≥ 90 %.
5. `_relabel_pydantic_strings(report, regulation)` — replace DORA-specific text for non-DORA runs.

## 6.8 `calculate_overall_confidence` — exact formula

`brd_frd_generator.py:528-559`:

1. Collect every `confidence_level` from the five requirement sections.
2. Normalize each via `normalize_confidence_level` (regex extract, clamp to `[90, 100]`, default 95).
3. **Count gate:** every section must reach its minimum count (see §6.7).
4. `average = round(sum / len)` else `90`.
5. If count gate fails, cap `average` at 90.
6. Return `f"{clamp(average, 90, 100)}%"`.

## 6.9 DOCX rendering

- Library: `python-docx` (>=1.1, `requirements.txt:3`).
- Style: Arial 11pt default, `Table Grid` style, 0.8″ margins (`brd_frd_generator.py:1798-1839`).
- **No images**, **no live hyperlinks** — URLs appear as plain text (`brd_frd_generator.py:1494-1513`).
- Section 17 (Source References) is optional; only rendered if `source_catalogue` non-empty (`brd_frd_generator.py:1656-1771`).

---

# 7. RTM Generation Details

Implemented in `agents/brd_rtm_agent.py:144-219`. Fully deterministic — no LLM.

## 7.1 How RTM rows are built

For every obligation in `RegulatoryAnalysis.obligations` (in order):

1. Traceability ID `TR-####` assigned sequentially.
2. Best-matching functional-requirement row picked via keyword overlap (`_pick_functional_requirement`, `agents/brd_rtm_agent.py:239-261`).
3. If **client roles** were selected, per-role applicability is computed from the obligation's `role_applicability` list; roles are split into `applicable_roles`, `partial_roles`, `uncertain_roles`, `not_applicable_roles`, `out_of_scope` flag.
4. `business_interpretation` and `business_justification` are composed from `obligation.compliance_requirement`, `obligation.risk_implication`, and the in-scope roles (`agents/brd_rtm_agent.py:274-303`).

## 7.2 Every column in the RTM

Defined by `RTMEntry` (`models/workflow_models.py:249-305`):


| Column                                                       | Source                                                                              | Deterministic / LLM                                                      |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `traceability_id`                                            | `TR-####` counter                                                                   | Deterministic                                                            |
| `obligation_id`                                              | `Obligation.obligation_id` (`OBL-###`)                                              | Deterministic (numbered by Agent 1)                                      |
| `business_requirement_id`                                    | `Obligation.source_requirement_id` (BRD-normalized ID)                              | Deterministic                                                            |
| `functional_requirement_id`                                  | Best-match FR row from BRD                                                          | Deterministic keyword-overlap                                            |
| `business_requirement`                                       | `Obligation.compliance_requirement`                                                 | LLM (via BRD)                                                            |
| `functional_requirement`                                     | Matched FR `detailed_requirement`                                                   | LLM (via BRD)                                                            |
| `impacted_area`                                              | `Obligation.impacted_area` (from `impacted_labels_for_requirement`)                 | Deterministic keyword classification                                     |
| `impacted_function`                                          | `Obligation.impacted_function`                                                      | Deterministic                                                            |
| `system_process_impact`                                      | `_system_process_impact(obligation)`                                                | Deterministic template (`agents/brd_rtm_agent.py:264-271`)               |
| `evidence_required`                                          | Joined `Obligation.evidence_needs`                                                  | Deterministic heuristics (`agents/regulatory_analysis_agent.py:564-584`) |
| `regulatory_basis`                                           | `Obligation.regulatory_basis` (from BRD `regulation_alignment`)                     | LLM-provided but validated by guardrails                                 |
| `priority`                                                   | `Obligation.priority` (MoSCoW)                                                      | LLM                                                                      |
| `obligation_verb`                                            | `services.obligation_verb.classify_verb_from_sources` (`obligation_verb.py:75-102`) | Deterministic regex on Must / Shall / Should / May / Can                 |
| `source_references`                                          | Copied from obligation (which copied from BRD source-references-by-item map)        | Deterministic                                                            |
| `applicable_roles` / `not_applicable_roles` / `out_of_scope` | From Client-Role-Aware Interpretation                                               | Deterministic keyword scoring                                            |
| `business_interpretation` / `business_justification`         | `_business_interpretation` / `_business_justification`                              | Deterministic templates                                                  |
| `role_rationale`                                             | `{role → "[applicability] rationale"}`                                              | Deterministic                                                            |


## 7.3 Traceability chain

`Article/RTS → BRD requirement (BR-XXX / FR-XXX) → Obligation (OBL-###) → RTM row (TR-####) → Question (mapped_requirement_ids / mapped_obligation_ids / explainability.obligation_id) → Recommendation (mapped_requirement_ids / mapped_obligation_ids)`.

Everything is rule-based / deterministic once the BRD IDs exist.

---

# 8. Questionnaire Generation Details

Three modules cooperate:

- `**services.questionnaire_generator`** (~1,562 lines) — dataclasses, BRD parsing, impact-pair derivation, package assembly, Excel export.
- `**services.ai_questionnaire_generator**` (~1,827 lines) — the AI generator itself.
- `**services.questionnaire_enhancer**` + `**services.question_style_enhancer**` — deterministic post-processors.

## 8.1 Question sourcing

- Not from BRD alone. Every question is scoped to an **impact pair** = `(area, function)` derived by `derive_impact_pairs` (`questionnaire_generator.py:547-564`) from the BRD requirements' impacted-area/function classification.
- For each pair, `_prioritise_pairs` (`ai_questionnaire_generator.py:658-672`) orders by impact severity (Critical → Low) then confidence, and takes up to 12 pairs (`DEFAULT_MAX_PAIRS`).
- Per pair, the LLM is asked to produce a fixed number of impact + readiness parents (`FUNNELS_PER_PAIR`, `ai_questionnaire_generator.py:128-133`):
  - Critical: 2 impact + 3 readiness → 5 parents
  - High: 1 + 2 → 3
  - Medium: 1 + 1 → 2
  - Low: 0 + 1 → 1
- A separate "free text" call adds 5–10 cross-cutting SME narratives (`.env:91-92`).

## 8.2 Full package JSON shape (emitted by `package_dict`, `questionnaire_generator.py:1202-1248`)

Top-level keys:

- `metadata` — `generated_at`, `regulation`, `overall_confidence_pct`, `coverage_pct`, `confidence_note`, `metrics`, `regulatory_taxonomy`.
- `requirements` — list of `Requirement` (`questionnaire_generator.py:241-259`).
- `impact_pairs` — list of `ImpactPair` (`questionnaire_generator.py:262-268`).
- `questions` — list of `Question` (`questionnaire_generator.py:272-360`).
- `answer_scores` — legacy label→score map.

**After Agent 3's post-processing:** `metadata.client_roles`, `metadata.role_filter_applied`, `metadata.role_interpretation`, `metadata.impact_enhanced`, `metadata.area_severity_map`, `out_of_scope_questions` are also present (`agents/questionnaire_agent.py:355-428`).

## 8.3 Every question field / column


| Field                                                                          | Meaning                                                                                                                                                                                 | Origin                                                                                                                                                              |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `question_id`                                                                  | `Q-0001` after final resequencing                                                                                                                                                       | Deterministic (`questionnaire_generator.py:711-760`)                                                                                                                |
| `area` / `function`                                                            | Business area × business function of the impact pair                                                                                                                                    | Deterministic pair selection                                                                                                                                        |
| `question_type`                                                                | `Single Select` / `Multi Select` / `Open Ended`                                                                                                                                         | LLM + style enhancer (auto-detects multi-select wording, forces Single Select for quantitative brackets)                                                            |
| `question`                                                                     | Business-friendly wording                                                                                                                                                               | LLM                                                                                                                                                                 |
| `options`                                                                      | List of strings **or** dicts `{label, score_value, readiness_interpretation, triggers_followup, followup_question_id, option_rationale, ...}`                                           | LLM                                                                                                                                                                 |
| `mapped_requirement_ids`                                                       | BRD IDs the question addresses (e.g. `BR-PRO-003`)                                                                                                                                      | LLM (constrained by prompt to use IDs supplied in the pair context)                                                                                                 |
| `mapped_obligation_ids`                                                        | Agent 1 `OBL-###` IDs                                                                                                                                                                   | LLM (from pair context)                                                                                                                                             |
| `confidence`                                                                   | 0–100, floor 90 (env `QUESTION_CONFIDENCE_FLOOR`)                                                                                                                                       | LLM + clamp                                                                                                                                                         |
| `scoring_weight`                                                               | Composite scoring multiplier                                                                                                                                                            | LLM initial; `questionnaire_enhancer` bumps to `weight_from_band(severity)` (Critical=5, High=4, Medium=3, Low=2) when higher (`questionnaire_enhancer.py:197-219`) |
| `impact_weight`                                                                | Sync of severity band                                                                                                                                                                   | Deterministic enhancer                                                                                                                                              |
| `impact_severity` / `impact_level` / `impact_reason`                           | Severity metadata                                                                                                                                                                       | LLM primary, enhancer fills missing                                                                                                                                 |
| `priority_rank`                                                                | Sort key                                                                                                                                                                                | Deterministic (`_severity_rank`)                                                                                                                                    |
| `funnel_parent_id` / `source_parent_id`                                        | Follow-up wiring                                                                                                                                                                        | Deterministic materialiser (`ai_questionnaire_generator._materialise_child_question`)                                                                               |
| `trigger_answers`                                                              | Option label(s) that reveal the child                                                                                                                                                   | LLM (from parent option's `triggers_followup=true`)                                                                                                                 |
| `child_question_ids`                                                           | Parent → children list                                                                                                                                                                  | Deterministic                                                                                                                                                       |
| `is_parent` / `is_child` / `dynamic_depth` / `branch_theme` / `branch_rule_id` | Branch metadata                                                                                                                                                                         | Deterministic                                                                                                                                                       |
| `regulatory_basis`                                                             | Article / clause                                                                                                                                                                        | LLM (from pair context)                                                                                                                                             |
| `rationale`                                                                    | "Why this question?" text                                                                                                                                                               | LLM                                                                                                                                                                 |
| `explainability`                                                               | Bundle: `regulation, regulator, article, obligation_id, brd_requirement_ids, business_function, control_objective, reason, expected_evidence, risk_if_negative, source_references, ...` | LLM + deterministic hydration                                                                                                                                       |
| `owning_team`                                                                  | Front / Middle / Back Office / Risk / Compliance / etc.                                                                                                                                 | LLM (constrained list in prompt)                                                                                                                                    |
| `team_rationale`                                                               | Why that team owns the question                                                                                                                                                         | LLM                                                                                                                                                                 |
| `plain_language_explainer`                                                     | Non-jargon restatement                                                                                                                                                                  | LLM                                                                                                                                                                 |
| `evidence_expectations`                                                        | List of expected evidence artefacts                                                                                                                                                     | LLM                                                                                                                                                                 |
| `question_purpose`                                                             | `impact` / `readiness` / `impact+readiness`                                                                                                                                             | LLM (per system prompt rules 9–13)                                                                                                                                  |
| `targets_impact_dimension` / `targets_readiness_dimension`                     | Which AI-assessment dimension the question probes                                                                                                                                       | LLM                                                                                                                                                                 |
| `requires_manual_review`                                                       | True only for the offline manual-review placeholder                                                                                                                                     | Deterministic                                                                                                                                                       |
| `generated_by_ai`                                                              | Provenance flag                                                                                                                                                                         | Deterministic                                                                                                                                                       |
| `quantitative_type`                                                            | `budget` / `timeline` / `coverage` / `frequency` / `team_size` / `sla`                                                                                                                  | Deterministic style enhancer (`question_style_enhancer.py:164-229`)                                                                                                 |
| `is_free_text`                                                                 | True for open-ended narrative questions                                                                                                                                                 | Deterministic                                                                                                                                                       |
| `role_applicability`                                                           | `{verdict, applicable_roles, partial_roles, uncertain_roles, not_applicable_roles, rationales, client_roles}`                                                                           | Deterministic role filter (`agents/questionnaire_agent.py:263-352`)                                                                                                 |


## 8.4 Per-option scoring

- LLM assigns `score_value` per parent option on the 0-100 scale (0 = not implemented, 50 = partial, 100 = fully implemented; `null` = N/A) — rule 6 of the system prompt.
- Style enhancer replaces options with curated **scored brackets** when the question is quantitative (`question_style_enhancer.py:96-153`): BUDGET_OPTIONS, TIMELINE_OPTIONS, COVERAGE_OPTIONS, FREQUENCY_OPTIONS, TEAM_SIZE_OPTIONS, SLA_OPTIONS.
- Legacy fallback: `ANSWER_SCORES` label map (`questionnaire_generator.py:220-234`) — `Yes=100`, `Partially=55`, `Not started=0`, etc.

## 8.5 Parent–child follow-up logic

- LLM marks 1–3 parent options with `triggers_followup=true`, provides a `followup_question` (+ `followup_options` or `followup_is_free_text=True`).
- `_generate_for_pair` reserves child IDs and hooks them onto the parent's `child_question_ids` and each option's `followup_question_id`.
- `_materialise_child_question` sets `funnel_parent_id`, `trigger_answers=[option label]`, `dynamic_depth=1`, and `branch_rule_id=ai_option_followup::{parent_qid}`.
- Note: dynamic follow-ups **are not surfaced live during answering** in the current UI — child questions are informational only in `_render_question_explainer` (`app.py:5773-5809`). The dynamic queue is still persisted in `AssessmentState.dynamic_queue` for scoring.

## 8.6 Impact-severity weighting

- `_prioritise_pairs` reorders pairs by severity first.
- `_weight_for_severity` (`ai_questionnaire_generator.py:1463-1469`) sets initial `scoring_weight` per parent.
- `enhance_questionnaire_package` (`questionnaire_enhancer.py:161-242`) bumps `scoring_weight` to at least `weight_from_band(area_severity)` and sorts questions by `(-priority_rank, is_free_text, -scoring_weight, area, question_id)`.

## 8.7 Manual-review fallback

When `client is None`, when `impact_count + readiness_count <= 0`, when the LLM raises, or when the returned bank is empty, one placeholder question per impact pair is created (`ai_questionnaire_generator._manual_review_placeholder`, `ai_questionnaire_generator.py:1296-1391`) with `requires_manual_review=True`, `generated_by_ai=False`, `confidence=50`, `scoring_weight=1`.

---

# 9. Scoring Logic Details

Two coexisting scoring engines; both are surfaced on the Dashboard.

## 9.1 Rules-engine scoring (`services.scoring_engine.evaluate`)

`scoring_engine.py:934-1085`. Called by `orchestrator.run_rules_engine` (`orchestrator.py:264-280`).

### Inputs

- Merged base questions + `state.dynamic_queue` (deduped, skipped IDs removed).
- `state.responses` — `{question_id: answer}`.

### Formulas

- **Per-question weight**
  ```
  weight = scoring_weight × max(1, impact_weight) × (confidence / 100)
  free_text_weight = weight × 0.3
  ```
  (`scoring_engine.py:983-998`)
- **Per-answer score** — `score_value(raw, q)` (`scoring_engine.py:720-803`):
  1. N/A tokens → `None` (excluded).
  2. Per-option `score_value` metadata → mean of picked options (0–100).
  3. Legacy `ANSWER_SCORES` table → mean.
  4. Enumeration fallback → pick-ratio with directionality.
- **Compliance score**
  ```
  compliance_score_pct = round(total_num / total_den × 100, 1)
  where  total_num += score × weight,  total_den += 100 × weight
  ```
- **Area / function / pair / requirement scores** — same weighted-mean pattern per bucket (`scoring_engine.py:1046-1078`).
- **Evaluation confidence**
  ```
  avg_conf = mean(question.confidence)   # default 90
  coverage_ratio = answered / max(1, answered + unanswered)
  coverage_bonus = coverage_ratio × 10
  coverage_penalty = (1 − coverage_ratio) × 12
  quant_bonus = min(4, quant_scored × 0.4)
  evaluation_confidence_pct = clamp(avg_conf + coverage_bonus − coverage_penalty + quant_bonus, 40, 99)
  ```
  (`scoring_engine.py:1055-1063`).

### Bands (from `services.severity.py:65-66`)


| Score | Readiness label |
| ----- | --------------- |
| ≥ 75  | Ready           |
| ≥ 50  | Watch           |
| ≥ 25  | At risk         |
| < 25  | Critical        |


Executive-action strings are hardcoded — see `recommendation_service.py:53-70`.

### Example (100 % worked)

Two closed questions:

- Q1: `scoring_weight=3`, `impact_weight=4`, `confidence=90`, answer scores 100 (Yes)
- Q2: `scoring_weight=2`, `impact_weight=2`, `confidence=90`, answer scores 0 (No)
- Q1 weight = `3 × max(1, 4) × 0.9 = 10.8`, contribution 100 × 10.8 = 1080, max = 1080.
- Q2 weight = `2 × max(1, 2) × 0.9 = 3.6`, contribution 0 × 3.6 = 0, max = 360.
- `compliance_score_pct = round(1080 / (1080 + 360) × 100, 1) = round(75.0, 1) = 75.0` → *Ready*.
- `avg_conf = 90`, coverage_ratio = 1 → confidence = `clamp(90 + 10 − 0 + 0, 40, 99) = 99`.

## 9.2 DORA weighted readiness (`services.readiness_score.compute_weighted_readiness`)

`readiness_score.py:812-1008`. Called from `_refresh_scoring_snapshot`.

Areas (weights sum to 100, `readiness_score.py:83-91`): ICT Governance & Risk Management 20 %, ICT Policies & Standards 15 %, ICT Processes & Operating Model 15 %, ICT Controls & Compliance Controls 20 %, ICT Technology & Architecture 15 %, Documentation & Evidence 10 %, Training & Awareness 5 %.

- `area_score = mean(scores)` per area (0-100).
- `weighted_score = area_score × (weight / 100)`.
- `overall_readiness_score = Σ weighted_score`.
- **Rating** (`readiness_score.py:310-332`): ≥90 Highly Ready, ≥75 Largely Ready, ≥60 Moderately Ready, ≥40 Needs Significant Improvement, <40 Not Ready.
- **Coverage gap severity** (`readiness_score.py:318-341`): ≤10 % Low, ≤25 % Medium, ≤40 % High, >40 % Critical.
- **Completeness** = `applicable_answered / applicable × 100` (`readiness_score.py:600-625`).
- **Accuracy** = `0.4 × evidence_coverage + 0.3 × answer_consistency + 0.3 × requirement_mapping_coverage` (`readiness_score.py:628-642`).

Verified by the smoke suite `tests/test_readiness_score.py` (reference example: overall = 76.0, rating "Largely Ready").

## 9.3 AI Confidence / Impact / Readiness intelligence

`services.ai_assessment_intelligence.py`.

- `**assess_confidence*`* (`297-303`) — weighted composite `0.30×completeness + 0.25×quality + 0.25×evidence + 0.20×clarity`, clamped `[45, 97]`, LLM-augmented, verdicted by `services.llm_judge.voted_generate` when `APP_LLM_VOTING` is enabled.
Verbatim instruction: *"You are a Big Four regulatory technology partner reviewing a regulatory impact analysis produced by an AI agent. Given the signals below (regulation coverage, evidence density, response coverage) produce a confidence assessment on a 0-100 scale…"* (`ai_assessment_intelligence.py:334-343`).
- `**assess_impact*`* (`708-712`) — six dimensions (business_functions, processes, systems, data, controls, stakeholders). Deterministic severity ladder based on item count: ≥8 items → Critical 92, ≥5 → High 78, ≥3 → Medium 58, ≥1 → Low 32, else Low 18 (`ai_assessment_intelligence.py:557-567`). Verbatim LLM instruction at `751-761`.
- `**assess_readiness**` (`1058-1065`) — seven consulting-standard dimensions (existing_controls, process_maturity, policy_coverage, technology_readiness, documentation_completeness, implementation_gaps, organizational_preparedness). Maturity level (`_maturity_from_score`, `863-872`): ≥85 Optimised, ≥70 Managed, ≥55 Defined, ≥35 Developing, <35 Initial. Verbatim instruction at `1106-1117`.

## 9.4 Completeness score

Two sources both live in the codebase:

- BRD completeness — `services.brd_frd_generator.calculate_completeness_coverage` (`brd_frd_generator.py:462-497`).
- Questionnaire completeness — `services.readiness_score.compute_completeness` (embedded in `compute_weighted_readiness`, `readiness_score.py:600-625`) = `applicable_answered / applicable × 100`.

## 9.5 Hardcoded thresholds

- Readiness bands 75/50/25 — `services.severity.py:65-66` (verified).
- Severity → weight — `weight_from_band`: Critical=5, At risk=4, Watch=3, Ready=2 (`severity.py`).
- Confidence floor 90 — `services.brd_frd_generator.apply_confidence_floor` and env `QUESTION_CONFIDENCE_FLOOR=90` (`.env:88`).
- Overall confidence clamp `[90, 100]` (`brd_frd_generator.py:559`).
- Evaluation confidence clamp `[40, 99]` (`scoring_engine.py:1063`).
- `min_citation_ratio=0.5` for guardrail veto (`guardrails.py`).
- `MAX_DYNAMIC_QUESTIONS_PER_ASSESSMENT=50`, `MAX_DYNAMIC_FOLLOWUP_DEPTH=3`, `MAX_DYNAMIC_FOLLOWUPS_PER_PARENT=3` (`scoring_engine.py`, all overridable via env).

---

# 10. Dashboard Logic

Renderer `render_dashboard_page` (`app.py:5889-6076`). **No Plotly / no matplotlib** — all charts are custom HTML/CSS injected via `st.markdown(..., unsafe_allow_html=True)`.

## 10.1 Sections and their data


| Section                           | Renderer                                                                                               | Data source                                                                                                          | Viz type                                   |
| --------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| Hero (Overall Impact + Readiness) | `_render_dashboard_hero` (`6746-6788`)                                                                 | `compliance_score_pct` (from rules engine, overwritten by weighted readiness), `confidence_assessment.overall_score` | HTML tiles + progress bars                 |
| Confidence rationale caption      | inline (`5945-5951`)                                                                                   | `confidence_assessment.reasoning`                                                                                    | Text                                       |
| Weighted Readiness (DORA)         | `_render_weighted_readiness_panel` (`6542-6732`)                                                       | `WeightedReadinessResult` from `compute_weighted_readiness`                                                          | KPI cards, `st.dataframe`, chips, expander |
| Regulatory Impact Assessment      | `_render_impact_intelligence_panel` (`6371-6431`)                                                      | `ImpactAssessment`                                                                                                   | HTML cards per dimension                   |
| Regulatory Readiness Assessment   | `_render_readiness_intelligence_panel` (`6434-6500`)                                                   | `ReadinessAssessment`                                                                                                | HTML cards per dimension                   |
| KPI row                           | `_render_dashboard_kpis` (`6871-6917`)                                                                 | confidence, answered/total, high-impact area count                                                                   | HTML KPI tiles                             |
| Severity distribution strip       | `_render_dashboard_legend` (`6941-7020`)                                                               | area/function/pair score buckets                                                                                     | 4-band live strip                          |
| Area-Wise Readiness Overview      | `_render_dashboard_readiness_cards` (`6791-6829`)                                                      | `evaluation.area_summary`                                                                                            | HTML cards                                 |
| Impact Assessment By Area         | `_render_dashboard_impact_cards` (`6832-6868`)                                                         | `100 − readiness` per area                                                                                           | HTML cards                                 |
| Area × Function Heatmap           | `_render_dashboard_pair_heatmap` (`7059-7129`)                                                         | `evaluation.pair_scores`                                                                                             | CSS tile heatmap `.dash-heatmap`           |
| Area-Detailed Recommendations     | `_render_rich_recommendations` (`6226-6368`) or `_render_dashboard_area_recommendations` (`7170-7250`) | Agent 4 output + hardcoded DORA playbook (`app.py:7289-7844`)                                                        | HTML rich cards                            |
| Top gaps expander                 | `_render_dashboard_top_gap_cards` (`7132-7167`)                                                        | `scoring.top_gaps` from orchestrator (`orchestrator.py:275-279`)                                                     | HTML cards                                 |
| Question-level detail expander    | `_render_dashboard_question_scoring_table` (`8110-8195`)                                               | questions + `requirement_scores`                                                                                     | Custom HTML table                          |


## 10.2 Labels

Colour classes come from `_severity_class`, `_impact_class`, `_severity_label_from_status`, `_readiness_severity_from_score`, `_impact_severity_from_score` (`app.py:6126-6223`). Palette lives in `_HERO_CSS` (`app.py:506-517`).

## 10.3 Autorun Agent 4

`_autorun_recommendations_if_needed` (`app.py:6079-6119`) is called on every dashboard render. It fingerprints `(compliance_pct, question_count, response_count, evaluation)` and only re-runs `orchestrator.run_recommendations(min_severity="Watch", top_n_requirements=10, enrich_with_genai=False)` when the fingerprint changes.

---

# 11. Regulatory Intelligence Hub Logic

Located inline on Page 1 (generate mode) — no separate top-level tab. Rendered by `_render_regulatory_intelligence_block` (`app.py:2711-2769`).

## 11.1 Is monitoring real?

**Real HTTP search, not simulated.** Two stages:

- **Stage 1 — Approved regulators.** `services.regulatory_intelligence_service._stage1` (`regulatory_intelligence_service.py:291-395`) calls `services.official_regulation_fetcher.fetch_official_regulations` (`official_regulation_fetcher.py`) which uses:
  - **Native httpx** adapters for `EBA, ESMA, EIOPA, FCA` (`services/native_regulator_search.py:306-331`).
  - **DDGS fallback** (from `ddgs` or `duckduckgo-search`) for the remaining 11 regulators (`official_regulation_fetcher.py:47-53`).
  - Every hit is post-filtered against a hostname allow-list (`search_config.is_regulator_url`, `search_config.py:413-417`).
- **Stage 2 — Consulting guidance.** `services.consulting_guidance_fetcher.py` — DDGS only, always anchored on Stage 1 hits (`consulting_guidance_fetcher.py:131-144`). **Currently disabled** in `.env:46` (`CONSULTING_SEARCH_ENABLED=false`); UI selector removed.

## 11.2 Regulator catalog

15 regulators (from `services/search_config.py:119-261`): EBA, ESMA, ECB, SSM, EIOPA, SRB, AMLA, DG_FISMA, EUR_LEX, FCA, PRA, BAFIN, AMF_FR, CBI, DNB. Each has a `domains` allow-list.

Consulting catalog (Stage 2): 10 firms — PWC, DELOITTE, EY, KPMG, ACCENTURE, CAPGEMINI, MCKINSEY, BCG, OLIVER_WYMAN, BAIN (`search_config.py:339-350`).

## 11.3 Version comparison / alerts / monitoring

**All three are placeholders.**

- `OfficialRegulationResult.version` is defined but **always `None`** (`official_regulation_fetcher.py:437, 606`).
- `monitor_regulation_updates` in `brd_frd_generator.py:622-644` is a *back-compat shim* that just re-calls `gather_regulatory_intelligence`.
- No delta detection, no subscription cadence, no notification pipeline anywhere in the codebase.
- No alerting is implemented — the "Hub" is a one-shot search pipeline invoked when the user changes regulator/regulation on Page 1.

## 11.4 How summaries are created

`_format_context` (`regulatory_intelligence_service.py:216-267`) concatenates fetched titles + snippets under three headers, capped at ~12,000 characters:

- `=== OFFICIAL REGULATORY CONTEXT (Primary Source of Truth) ===`
- `=== SUPPLEMENTARY IMPLEMENTATION GUIDANCE (Not Authoritative) ===` (Stage 2 only, currently unused)
- `=== OFFLINE REGULATORY BASELINE (Authoritative Sources Unavailable) ===` (fallback)

## 11.5 What is implemented vs placeholder


| Feature                                        | Status                    |
| ---------------------------------------------- | ------------------------- |
| Live regulator search (Stage 1)                | Implemented               |
| Domain allow-list post-filter                  | Implemented               |
| Consulting-firm search (Stage 2)               | Implemented but disabled  |
| Version comparison / diff                      | **Placeholder** (no code) |
| Alerts / notifications                         | **Not implemented**       |
| Periodic monitoring                            | **Not implemented**       |
| Offline baseline for DORA / neutral disclaimer | Implemented               |


---

# 12. Evidence & Reference Logic

Owned by `services.source_traceability` (`source_traceability.py`, 583 lines).

## 12.1 URL capture

Stage 1 fetchers return `OfficialRegulationResult.url` (`official_regulation_fetcher.py:67-94`); Stage 2 returns `ConsultingGuidanceResult.url`. `build_source_catalogue` (`source_traceability.py:311-348`) rolls these into `SourceReference` dicts.

## 12.2 SourceReference shape (`source_traceability.py:67-116`)

`source_url`, `title`, `regulator`, `publication_date`, `regulation_reference` (article / RTS / ITS / clause), `source_type` (Official Regulator / Official Legislation / Consulting Guidance / Uploaded Document / `No live source available`), `publication_type`, `confidence`.

## 12.3 Attaching to BRD items

`attach_source_references(report, catalogue, max_per_item=3)` (`source_traceability.py:441-530`) walks the BRD in place and calls `SourceCatalogue.match(*texts)` (`source_traceability.py:227-278`) per row:

- Requirement rows → keyed `REQ:{BR-XXX}` or `REQ:{FR-XXX}`.
- Control checkpoints → keyed `CTRL:{stage}:{checkpoint}`.
- Bullets (standard sections + control subsections) → keyed `BUL:{section}:{title}`.
- Risks → keyed `RISK:{first 120 chars}`.

Matching is **deterministic token/regex overlap** — never invents a citation. Fallback priority (`source_traceability.py:146-201`): (1) uploaded document if used, (2) `SOURCE_TYPE_NONE` when offline baseline was used, (3) highest-confidence official publication.

## 12.4 Hyperlinks

**Not preserved in DOCX** — URLs render as plain text in tables and Section 17 (`brd_frd_generator.py:1494-1513`, `1722-1726`, `1766-1771`).

Streamlit renders full clickable HTML `<a href>` links inside `_render_parsed_requirements_html` (`app.py:4595-4636`) and inside RTM/obligation tables (`_sources_cell_html`, `app.py:4554-4592`).

## 12.5 Validation of output against evidence

- Every LLM response is pushed through `services.guardrails.CitationValidator` (`guardrails.py:503-592`) which extracts citations (`Article`, `RTS`, `ITS`, `Regulation (EU)`, `Directive`, `Chapter`, `Section`, `Paragraph`, `Recital`, `Annex`, per `_CITATION_PATTERN_SPECS`, `guardrails.py:405-416`) and replaces any citation not found in the `source_corpus` with `"[citation not verified against source]"` (`guardrails.py:521`).
- `safe_generate` rejects the entire response when the ratio of verified citations falls below `min_citation_ratio` (default 0.5, `guardrails.py:1470-1487`).

---

# 13. Guardrails / Hallucination Prevention

Central module: `services/guardrails.py` (1,529 lines).

## 13.1 What is implemented

Six documented categories (`guardrails.py:11-45`):

1. **Instruction hardening** — `harden_instruction` (`guardrails.py:334-393`) prepends the `ANTI_HALLUCINATION_DIRECTIVE` (verbatim at `guardrails.py:95-122`) plus regulation- and role-scope lines. Applied to both system prompt and every component prompt.
2. **Citation validation** — `CitationValidator` (`guardrails.py:503-592`) matches citations against the `source_corpus` (Stage 1 text). Unverified → `"[citation not verified against source]"`. Strict mode raises the finding to `critical`.
3. **Regulation-scope validation** — `RegulationScopeValidator` (`guardrails.py:600-658`) flags any mention of a regulation name not equal to the one in scope (e.g., a DORA run mentioning MiFID II). Token list `_REGULATION_NAME_TOKENS` (`guardrails.py:211-228`).
4. **Client-role scope validation** — `RoleScopeValidator` (`guardrails.py:666-726`) flags any mention of an institution type not on the user's selection (leveraging `INSTITUTION_TYPE_NAMES` from `client_roles.py:215-216`).
5. **Meta-leakage detection** — `scrub_meta_leakage` (`guardrails.py:932-968`) strips AI meta-language (`as an AI language model`, `openai`, `chatgpt`, `gpt-N`, `[insert…]`, `{{placeholder}}`, apology sentences) using regex patterns at `guardrails.py:131-155`.
6. **Safe generation wrapper** — `safe_generate` (`guardrails.py:1310-1502`) is the sole gateway for all LLM output. On critical findings, it rejects the payload and forces the offline fallback.

Additional validators: **SpeculationValidator** (density of hedging language, `guardrails.py:734-784`), **UrlValidator** (fabricated / placeholder domains: example.com, tbd.com, etc., `guardrails.py:792-859`), **NumericValidator** (percentages / durations / years / dates / thresholds not in source, `guardrails.py:867-924`).

## 13.2 Pre-persistence sweep

`check_before_persist` (`guardrails.py:1230-1263`) is invoked by every DB writer (`database.py:73-127`). Modes:

- `off` — no sweep.
- `warn` (default) — sweep and log, write continues.
- `strict` — sweep and raise `PersistenceGuardrailError`, write refused.

Env: `APP_PERSIST_GUARDRAIL`.

## 13.3 Human-review flags

- Any `critical` finding from a validator → `report.ok = False` → payload rejected in `safe_generate`.
- LLM judge (`services.llm_judge.voted_generate`) verdict `REVIEW` → recorded on the confidence assessment and surfaced as a *Human review required* gap on Page 5.

## 13.4 Schema / JSON validation

- Pydantic + LangChain `with_structured_output` guarantees each LLM response conforms to the target schema before it is returned to the caller.
- Package writes: `utils.json_utils.validate_package_schema` (`utils/json_utils.py:80-130`).

## 13.5 Fallback handling

- No client (`GenAIClient.try_create() is None`) → offline BRD scaffold, manual-review questions, deterministic recommendations, deterministic AI-assessment baselines.
- API error / length-limit → `safe_generate` catches, returns `(None, report)`; caller either retries (with `generate_with_length_retry`) or falls back.
- Guardrail veto → deterministic path retained; user sees a caption "GenAI service unreachable/ blocked …" on Page 2 (`app.py:4312-4324`).

## 13.6 How we prevent AI hallucination — panel-ready answer

*(Grounded strictly in code — safe to say verbatim to judges.)*

> We ship a defense-in-depth hallucination stack:
>
> 1. **Deterministic grounding.** Regulatory context is fetched from a live search pipeline that is restricted to a hard-coded allow-list of 15 approved regulator domains (EBA, ESMA, ECB, FCA, …). Every URL is post-filtered against that allow-list — we never accept a hit from an off-list domain. No RAG-database; the context passed to the LLM is exactly what those regulators published.
> 2. **Anti-hallucination directive.** Every LLM call is wrapped by `services.guardrails.harden_instruction`, which prepends an explicit rule set: cite only from supplied context, say "Not stated in the source" when uncertain, do not reference other frameworks, do not claim applicability to institution types the user did not select, do not use AI meta-language.
> 3. **Structured output.** Every call is Pydantic-schema-locked via LangChain `with_structured_output`. Malformed responses fail loudly.
> 4. **Citation validator.** After the LLM responds, `CitationValidator` extracts every article / RTS / ITS / Regulation (EU) / Directive / chapter / section reference and cross-checks it against the source corpus. Any unverified citation is replaced with `"[citation not verified against source]"`. If the ratio of verified citations falls below 0.5, `safe_generate` rejects the whole payload and forces the deterministic offline fallback.
> 5. **Regulation- and client-role scope validators.** Any mention of an off-scope regulation or institution type is flagged; a critical finding kills the response.
> 6. **Meta-leakage scrubber.** Strips 15+ regex patterns for "as an AI language model", "OpenAI", "ChatGPT", "gpt-4o", `[insert …]` placeholders, apology sentences.
> 7. **Speculation / URL / numeric validators.** Density of "generally / typically / most firms" wording, placeholder URL domains, and unsourced dates / percentages all raise findings.
> 8. **Pre-persistence sweep.** `check_before_persist` walks every string leaf of the payload before SQLite write. In `warn` mode we log; in `strict` mode we refuse the write.
> 9. **LLM-as-judge voting.** For high-stakes payloads (`ConfidenceAssessment`), `services.llm_judge.voted_generate` runs a 2-of-3 vote between a deterministic baseline, the LLM candidate, and (optionally) an independent judge that returns `A / B / TIE / REVIEW`. `REVIEW` verdicts surface as human-review gaps.
> 10. **Deterministic fallbacks.** If any step fails, the system degrades to hand-written offline scaffolds (BRD, questionnaire manual-review placeholders, per-area recommendation playbooks) so the user is never shown a hallucinated artefact.

---

# 14. Database / Storage Logic

Implemented in `services/database.py` (~~717 lines). Sole storage engine is **SQLite** (`sqlite3` stdlib module), single file `data/app.db` (~~9.8 MB in the working copy).

## 14.1 Tables


| Table            | Purpose                                                                                            | Insert path                                                                                                                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `documents`      | Uploaded regulation PDF/DOCX + BRD/FRD DOCX                                                        | `save_document(...)` (`database.py:282-310`), called on Page 1 upload/sample.                                                                                     |
| `requirements`   | Denormalised per-BRD requirements                                                                  | `save_requirements(...)` (`database.py:339-376`), called by `_run_agent2_for_uploaded_brd`. Guardrail sweep before write.                                         |
| `questionnaires` | One row per generated questionnaire; **full package JSON stored verbatim** in `package_json`       | `save_questionnaire(...)` (`database.py:404-452`), called at the end of `_run_agent3`. Guardrail sweep before write; report embedded as `_persistence_guardrail`. |
| `assessments`    | One row per session; keeps serialised `AssessmentState`, latest evaluation, latest recommendations | `create_assessment(...)` + `update_assessment_snapshot(...)` (`database.py:485-591`). Guardrail sweep for evaluation and recommendations.                         |
| `responses`      | One row per answered question                                                                      | `upsert_responses(...)` (`database.py:594-616`), replaces the response set on each save.                                                                          |


Schema at `database.py:168-238` includes indices on `documents(kind)`, `requirements(document_id)`, `questionnaires(document_id)`, `assessments(questionnaire_id)`, `responses(assessment_id)` and a unique `(assessment_id, question_id)` constraint.

## 14.2 Read paths

`list_documents`, `get_document`, `list_requirements`, `list_questionnaires`, `get_questionnaire`, `list_assessments`, `get_assessment`, `get_responses` (`database.py:313-676`). Each rehydrates the stored JSON and returns plain dicts (not `sqlite3.Row`) for Streamlit JSON safety.

## 14.3 Audit / guardrail log

- Every guardrail sweep result is embedded inside `package_json._persistence_guardrail` / `evaluation_json._persistence_guardrail` (`database.py:428-432, 522-547`).
- No separate `audit_log` table exists — the guardrail records live alongside the payload.

## 14.4 File storage

- `uploads/` — every user upload (unique-suffixed via `utils/file_utils.py:save_upload`).
- `outputs/` — every generated BRD DOCX (`{REGULATION}_BRD_FRD_{ts}.docx`) and every built XLSX (`{regulation}_readiness_{ts}.xlsx`).
- `sample_data/` — bundled DORA sample BRD DOCX + sample questionnaire JSON.

---

# 15. Export / Download Logic

Six export formats are supported:


| Format                      | Where                                            | Library / function                                                                                                                                                                                          |
| --------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DOCX (BRD + FRD)**        | Page 2 → Downloads expander (`app.py:4745-4751`) | `python-docx` via `services.brd_frd_generator.write_brd_docx`                                                                                                                                               |
| **JSON (BRD + FRD)**        | Page 2 → Downloads (`app.py:4756-4762`)          | `json.dumps(BRDArtifact.report.model_dump())`                                                                                                                                                               |
| **CSV (Requirements)**      | Page 2 → Downloads (`app.py:4772-4778`)          | Python `csv` via `_requirements_csv` (`app.py:4674-4707`)                                                                                                                                                   |
| **JSON (Obligations)**      | Page 2 → Downloads (`app.py:4785-4791`)          | `json.dumps` of `analysis.obligations`                                                                                                                                                                      |
| **JSON + CSV (RTM)**        | Page 2 → Downloads (`app.py:4796-4809`)          | `json.dumps` + `_rtm_csv` (`app.py:4710-4712`)                                                                                                                                                              |
| **JSON (Gap Report)**       | Page 5 (`app.py:8388-8395`)                      | `json.dumps(GapReport)`                                                                                                                                                                                     |
| **JSON (Questionnaire)**    | Page 6 (`app.py:8420-8426`)                      | `json.dumps(pkg)`                                                                                                                                                                                           |
| **JSON (Responses)**        | Page 6 (`app.py:8443-8449`)                      | `json.dumps(envelope)`                                                                                                                                                                                      |
| **XLSX (Readiness Report)** | Page 6 (`app.py:8454-8474`)                      | `openpyxl` via `write_excel_from_package` → `write_excel` (`questionnaire_generator.py:1386-1501`). Contains Metadata, Requirements, Impact Pairs, Questions with per-option scoring, Answer Scores legend. |


**No PDF export**. **No CSV of responses on Page 6** (CSV export is Page-2-only). Downloads are triggered by `st.download_button` (native Streamlit).

---

# 16. Tech Stack Summary


| Layer            | Technology                                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UI framework     | Streamlit ≥ 1.39 (`requirements.txt:1`)                                                                                                                       |
| Backend language | Python 3.13 (as run)                                                                                                                                          |
| Database         | SQLite (stdlib `sqlite3`), file `data/app.db`                                                                                                                 |
| AI model         | `azure.gpt-4o` via **PwC GenAI Shared Service** (OpenAI-compatible endpoint)                                                                                  |
| LLM SDKs         | `langchain-openai ≥ 0.2`, `openai ≥ 1.40`                                                                                                                     |
| HTTP client      | `httpx ≥ 0.27` (native regulator search + LLM), `certifi`, `truststore ≥ 0.9`                                                                                 |
| Search           | `ddgs ≥ 9.0`, `duckduckgo-search ≥ 6.3` (fallback for DDGS)                                                                                                   |
| Document parsing | `PyMuPDF ≥ 1.24` (PDF), `python-docx ≥ 1.1` (DOCX)                                                                                                            |
| Excel export     | `openpyxl ≥ 3.1`                                                                                                                                              |
| DOCX export      | `python-docx`                                                                                                                                                 |
| PDF export       | *(none)*                                                                                                                                                      |
| Visualization    | Custom HTML/CSS (no Plotly, no matplotlib); `plotly ≥ 5.22` is in `requirements.txt:7` but unused in `app.py`                                                 |
| Data             | `pandas ≥ 2.2`                                                                                                                                                |
| Data validation  | `pydantic ≥ 2.7`, LangChain `with_structured_output`                                                                                                          |
| Environment      | `python-dotenv ≥ 1.0`                                                                                                                                         |
| Orchestration    | Custom `RegulatoryWorkflowOrchestrator` (`orchestrator.py`), no CrewAI / AutoGen / LangGraph                                                                  |
| Deployment       | Azure App Service Linux (`startup.sh`, `AZURE_DEPLOYMENT.md`); local dev via `streamlit run app.py`; SQLite browser via Docker Compose (`docker-compose.yml`) |
| Config           | `.streamlit/config.toml` (theme), `.env` (secrets + feature flags)                                                                                            |


---

# 17. End-to-End Workflow Diagram

Actual code path, not idealised.

```
+---------------------------------------------------------------+
| Page 1: Setup                                                 |
|   * Client Roles multiselect          -> normalize_client_roles|
|   * Client Profile 6 fields           -> normalize_client_profile
|   * Regulator scope multiselect       -> APPROVED_REGULATORS   |
|   * (optional) Regulation PDF/DOCX    -> uploads/, DB          |
|   * Auto-fetch sources on fingerprint change                   |
|         -> gather_regulatory_intelligence(regulation)          |
|              -> Stage 1 (httpx + DDGS, allow-listed regulators)|
|              -> (Stage 2 disabled)                             |
|              -> RegulatoryIntelligencePackage (cached)         |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
| Page 2: Generate BRD / FRD                                    |
|   Click "Generate BRD / FRD"                                  |
|     -> _run_agent1_and_agent2_with_status()                   |
|          -> orch.parse_document(regulation_doc?)              |
|          -> orch.run_regulatory_analysis(...)                 |
|               [Agent 1]                                       |
|               -> build_brd_frd_report(...)                    |
|                    -> gather_regulatory_intelligence (cached) |
|                    -> 8 LLM calls (Pydantic-locked, guarded)  |
|                    -> ensure_minimum_detail (deterministic)   |
|                    -> attach_source_references                |
|               -> _extract_obligations() (deterministic)       |
|               -> build_role_aware_interpretation (kw scoring) |
|               -> per-obligation guardrail sweep               |
|          -> orch.run_brd_rtm(analysis, docx_export_path)      |
|               [Agent 2]                                       |
|               -> write_brd_docx (python-docx) -> outputs/     |
|               -> _build_rtm() (deterministic joins)           |
|          -> chained _run_agent3()                             |
|               [Agent 3]                                       |
|               -> orch.run_questionnaire_from_report(...)      |
|                    -> build_package_from_report               |
|                         -> generate_ai_questionnaire (LLM)    |
|                         -> _funnel_prompt per pair            |
|                         -> _free_text_prompt cross-cutting    |
|                    -> enhance_questionnaire_package (kw)      |
|                    -> _apply_role_filter (kw)                 |
|               -> db.save_questionnaire (guarded)              |
|   Render: obligations, RTM, source refs, downloads            |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
| Page 3: Questionnaire                                         |
|   Auto-seed demo answers on first entry                       |
|   User edits answers -> _refresh_scoring_snapshot() per event |
|   Click "Calculate Impact & Readiness"                        |
|     -> _submit_and_go_to_dashboard()                          |
|          -> db.create_assessment / upsert_responses           |
|          -> _refresh_scoring_snapshot()                       |
|               -> orch.run_rules_engine (evaluate)             |
|               -> orch.assess_impact_intelligence              |
|               -> orch.assess_readiness_intelligence           |
|               -> orch.assess_confidence_intelligence          |
|               -> compute_weighted_readiness (DORA weights)    |
|          -> db.update_assessment_snapshot (guarded)           |
|          -> page = "4. Dashboard"                             |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
| Page 4: Dashboard                                             |
|   _autorun_recommendations_if_needed()                        |
|     -> orch.run_recommendations()                             |
|          [Agent 4]                                            |
|          -> generate_recommendations (compact, deterministic) |
|          -> build_rich_recommendations                        |
|               (deterministic baseline + optional GenAI)       |
|          -> attach_evaluations (coverage/spec/action/ground)  |
|   Render: hero, weighted readiness, AI impact + readiness,    |
|           KPIs, severity strip, area/impact cards,            |
|           area x function heatmap, rich recs, top gaps        |
+---------------------------------------------------------------+
                            |
                            v

+---------------------------------------------------------------+
| Page 5: Export                                                |
|   Download Questionnaire JSON                                 |
|   Download Responses JSON (+ evaluation + recommendations)    |
|   Build Excel Report -> openpyxl -> outputs/                  |
+---------------------------------------------------------------+
```

Guardrails / persistence / offline fallback bracket every stage transparently.

---

# 18. Panel Q&A Preparation

### Q1. What happens when I click **Generate BRD / FRD**?

See §3.1. In short: the button triggers `_run_agent1_and_agent2_with_status`, which sequentially calls the orchestrator's Agent 1 (regulatory analysis + BRD via 8 LLM calls with Pydantic structured output + guardrails), Agent 2 (RTM build + DOCX export), then chains Agent 3 (questionnaire generation). Every step degrades to a deterministic fallback if the GenAI service is unavailable.

### Q2. How do we ensure the output is evidence-backed?

- Regulatory context comes from a live search restricted to a hard-coded allow-list of 15 regulator domains — never from open web scraping.
- Every LLM response is passed through `CitationValidator`, which cross-checks each citation against the fetched source corpus. Unverified citations are marked `[citation not verified against source]` and the payload is rejected if less than 50 % of citations verify.
- Every BRD requirement, obligation, control checkpoint, and RTM row carries a `source_references` list produced by deterministic token overlap between the row text and the fetched publications.
- Section 17 of the BRD DOCX reproduces the full source catalogue plus per-requirement traceability.

### Q3. How is readiness calculated?

Two coexisting engines. See §9.1 and §9.2.

- **Rules engine** (`scoring_engine.evaluate`): weighted mean per question, with `weight = scoring_weight × max(1, impact_weight) × (confidence/100)`. Compliance % = `Σ(score × weight) / Σ(100 × weight) × 100`. Bands are Ready ≥ 75, Watch ≥ 50, At risk ≥ 25, Critical < 25.
- **DORA weighted readiness** (`readiness_score.compute_weighted_readiness`): mean of scored questions per area × official DORA weight (Governance 20 %, Policies 15 %, Processes 15 %, Controls 20 %, Technology 15 %, Documentation 10 %, Training 5 %); rating: ≥ 90 Highly Ready, ≥ 75 Largely Ready, ≥ 60 Moderately Ready, ≥ 40 Needs Significant Improvement, < 40 Not Ready.

### Q4. How are questions generated?

Not from a template. `services.ai_questionnaire_generator.generate_ai_questionnaire`:

1. Derives area × function impact pairs from the BRD.
2. Prioritises the top 12 pairs by impact severity.
3. Runs one LLM funnel call per pair with a system prompt that forces plain-language, per-option scoring, adaptive branching, and team routing, and an impact-vs-readiness split.
4. Runs a separate LLM call for 5-10 cross-cutting free-text SME questions.
5. Applies deterministic post-processing: style diversification (Multi-Select detection + quantitative brackets), impact-severity re-weighting, dedup, role filter, sequencing.

Offline fallback: one `Manual review required` placeholder per pair — never invents content.

### Q5. How do we prevent hallucination?

See §14.6 — the full 10-layer stack.

### Q6. Is this true RAG or simple indexed retrieval?

Neither. It is a **search-augmented generation** pipeline: real HTTP calls to allow-listed regulator sites (native httpx + DDGS fallback), no vector DB, no embeddings. The retrieved plain text is concatenated into a `context_text` (12 k-char cap) and passed as the LLM's context on every call. There is no similarity search.

### Q7. Is the Regulatory Intelligence Hub live or simulated?

**Live search, real HTTP calls.** But *versioning*, *alerts*, and *monitoring* are placeholders — the version field is always `None`, no cron / subscription code exists, and `monitor_regulation_updates` is a legacy shim that just re-searches. See §12.

### Q8. How does client role affect requirement generation?

Two entry points:

- **Prompt-side**: the client-role directive is appended to every LLM prompt (`brd_frd_generator.py:1987-2003`) — the model is told to interpret each obligation for the selected institution type(s), mark out-of-scope items explicitly, and adjust proportionality.
- **Deterministic side**: `services.client_roles.build_role_aware_interpretation` (`client_roles.py:772-942`) scores each obligation against a JSON catalogue of 53 institution types (in `services/data/institution_types.json`) using per-role keyword and typical-obligation lists. Verdicts: Applicable / Partially Applicable / Uncertain (Not Applicable requires explicit source evidence). Verdicts flow into: RTM `applicable_roles` / `out_of_scope` columns, questionnaire `role_applicability` blocks (out-of-scope questions dropped), and recommendation scope banners.

### Q9. What is rule-based and what is LLM-based?


| Rule-based / deterministic                                                              | LLM-based                                                                               |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Client-Role-Aware Interpretation scoring (keyword substring + thresholds)               | BRD/FRD content generation (8 calls)                                                    |
| Obligation extraction from BRD rows                                                     | Questionnaire questions, options, per-option scoring, follow-ups                        |
| RTM row construction (traceability_id, business_interpretation, business_justification) | Optional recommendation rewrite (`enrich_with_genai=True`)                              |
| Source-reference matching (token overlap)                                               | AI Confidence assessment (LLM verdict optional; deterministic baseline always computed) |
| Rules-engine scoring (weighted means, thresholds, bands)                                | AI Impact assessment (LLM optional; deterministic ladder always computed)               |
| DORA weighted readiness                                                                 | AI Readiness assessment (same pattern)                                                  |
| Gap detection (all 5 families)                                                          | Adaptive follow-up branch generator (LLM primary, deterministic templates as fallback)  |
| Guardrails (regex + Jaccard)                                                            | Judge for high-stakes payloads (`llm_judge.py`)                                         |
| Recommendation deterministic drafts                                                     | —                                                                                       |
| Package validation, style enhancer, dedup, resequencing                                 | —                                                                                       |
| Excel / DOCX / CSV / JSON export                                                        | —                                                                                       |
| Session state, DB writes, page navigation                                               | —                                                                                       |


### Q10. Current limitations

See §20.

### Q11. What model, temperature, and token budgets do you use?

- Model: `azure.gpt-4o` (env-configurable).
- Temperature: **not sent by default** (deterministic). If `OPENAI_SEND_TEMPERATURE=true`, temperature = 0.10 (`genai_service.py:259-270`).
- Max completion tokens: 6000 (env `OPENAI_MAX_TOKENS`), with one length-limit retry to 12000.
- Context cap fed into the LLM: 6000 chars (env `GENAI_CONTEXT_CHARS`), truncated from the ≤ 12 000-char `RegulatoryIntelligencePackage.context_text`.

### Q12. What data is stored, and where?

See §15. All artefacts live in a single SQLite file (`data/app.db`) with 5 tables, plus uploaded files in `uploads/` and generated artefacts in `outputs/`. Every generative payload is scrubbed by the pre-persistence guardrail before it hits the DB.

### Q13. What is `impact_weight` vs `scoring_weight`?

`scoring_weight` is the composite scoring multiplier assigned by the LLM per question. `impact_weight` mirrors the severity-band weight (Critical=5, High=4, Medium=3, Low=2) computed by `services.questionnaire_enhancer` from the area's impact severity. The rules engine uses `scoring_weight × max(1, impact_weight) × (confidence/100)` so the two multiply — heavy questions in critical areas dominate the compliance score.

### Q14. Why does `Ready` start at 75 % and not 80 %?

Canonical thresholds live in `services/severity.py:65-66`. The value 75 was chosen so that "Ready" tracks the Largely-Ready band (75-89) in the DORA weighted readiness rating (`readiness_score.py:310-332`), keeping the two views aligned.

### Q15. Why does the DOCX not have live hyperlinks?

Deliberate simplicity: `python-docx` supports hyperlinks only via raw XML manipulation. To keep the export deterministic and re-openable across Word versions, URLs are rendered as plain text (which Word auto-parses as clickable when opened). The Streamlit UI does render clickable HTML `<a>` tags in RTM/obligation tables.

---

# 19. Implementation Gaps

Grouped by status. **Only concrete, code-verified statements.**

## 19.1 Fully implemented

- Six-page Streamlit UI with sidebar router and Reset (`app.py`).
- Document Parser stage (PDF via PyMuPDF, DOCX via python-docx).
- Agent 1 — Regulatory Analysis (LLM + deterministic obligation extraction + role-aware interpretation + guardrail sweep).
- Agent 2 — BRD + RTM (LLM BRD wrapping + deterministic RTM + DOCX writer).
- Agent 3 — Questionnaire (AI generation + deterministic post-processing + role filter).
- Agent 4 — Recommendations (compact + rich, deterministic baseline + optional GenAI rewrite + evaluator).
- Rules-engine scoring, DORA weighted readiness, AI confidence/impact/readiness intelligence.
- Guardrails stack (11 validators + `safe_generate` + pre-persistence sweep).
- LLM-judge voting for confidence assessment.
- Source-traceability matcher + DOCX Section 17.
- Live Stage-1 regulator search (httpx + DDGS) with allow-list post-filter.
- Adaptive branch generation (LLM primary + deterministic templates + generic signal-banded fallback).
- SQLite persistence (5 tables + FK cascade).
- Client-role-aware and client-profile-aware BRD and questionnaire.
- Gap identification (all 5 families).
- DOCX / JSON / CSV / XLSX exports (§16).
- Streamlit theming + `_HERO_CSS` block.
- Offline fallbacks for every LLM stage.

## 19.2 Partially implemented

- **Recommendations LLM enrichment** — only rewrites text (`suggested_action` / `what / why / how / …`); does not add new evidence or steps beyond the deterministic draft (by design).
- **AI Assessment (Confidence / Impact / Readiness)** — always computes deterministic baseline; LLM path is optional and can be vetoed by the judge; user does not see when the judge fell back.
- **HITL review queue** on the Gap page — UI stores state but does **not persist to SQLite** (`app.py:8271-8272`).
- **Dynamic follow-up questions** — generated, materialised, and stored in `AssessmentState.dynamic_queue`, but the UI does not surface child questions during answering; they are only visible in the *Why this question?* explainer.
- **Client-Role-Aware Interpretation table** — engine still runs and self-heals metadata, but the per-role table has been removed from the visible UI (`app.py:3312-3316`).
- **Guardrail audit panel** — data collected on the analysis, but panel is intentionally hidden (`app.py:4334-4339`).

## 19.3 Placeholder only

- **Regulatory Intelligence Hub — monitoring / alerts / diff / version comparison.** No code. `version` field is always `None`. `monitor_regulation_updates` is a re-search shim (`brd_frd_generator.py:622-644`).
- **Stage 2 Consulting Guidance search** — fully coded, but disabled (`.env:46`); UI selector removed.
- **Native regulator search** — implemented for only 4 of 15 regulators (EBA, ESMA, EIOPA, FCA). The other 11 rely on DDGS (`native_regulator_search.py:306-331`).

## 19.4 Hardcoded

- **Default regulation** `"DORA"` (`app.py:2028`); **default tier** `"Tier-2"` (`app.py:2029`); **default client role** `"Commercial Bank"` (`app.py:2037`).
- **DORA area weights** and **readiness rating bands** (`readiness_score.py:83-91, 310-332`).
- **Severity thresholds** 75/50/25 in `services/severity.py:65-66`.
- **BRD section minimum row counts** (process 14, data 14, reporting 10, functional 18, non-functional 10, checkpoints 12, risks 10) — `brd_frd_generator.py:388-394`.
- **Per-row AI confidence floor** 90 % (`brd_frd_generator.py:328-352`) — LLM values are clamped upward if the model returns nothing meaningful.
- **AI confidence normalisation defaults** — 93 % / 96 % if the LLM omits a value (`brd_frd_generator.py:328-352`).
- **Owner-by-function table** (13 rows) in `services/owner_registry.py:28-42`.
- **Severity → action / horizon** strings in `services/recommendation_service.py:53-76`.
- **BRD offline fallback** — a ~300-line hand-written DORA scaffold in `brd_frd_generator.py:850-1165`.
- **DORA area playbook** for Dashboard area recommendations — `app.py:7289-7844`.
- **Client-profile keyword catalogues** (Business Lines 58, Products 75, Countries 96, Legal Entities 50, Vendors 58) — `services/client_profile.py`.
- **Institution-types catalogue** (53 entries, `services/data/institution_types.json`).
- **DORA regulator/consulting catalogues** (15 + 10) — `services/search_config.py:119-350`.
- **Impact-severity → weight mapping** (Critical=5, High=4, Medium=3, Low=2) — `services/severity.py` `weight_from_band`.
- **Ready threshold demo seed answers** for the questionnaire (`app.py:5014`).
- **Path to bundled sample BRD** — `sample_data/DORA_Tier2_Detailed_DetailedBRDFRD.docx` (`app.py:3872`).

## 19.5 Not implemented

- PDF export.
- CSV export of questionnaire responses on Page 6.
- Real-time / cron-based regulation monitoring, delta alerts, version diffs.
- Persistent HITL review queue.
- Live rendering of dynamic follow-up questions during answering.
- Per-user authentication / RBAC.
- Multi-tenant separation.
- Vector-store RAG.
- Streaming LLM output.
- Native search for 11 of 15 regulators.
- Consulting-firm Stage-2 UI selector (turned off pending product decision).

---

*End of document.*