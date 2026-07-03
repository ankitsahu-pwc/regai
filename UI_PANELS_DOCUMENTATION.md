# UI Blocks & Panels — Executive Reference

Concise companion to `DOCUMENTATION.md`. One row per UI element. **What it is**, **how it's calculated**, and **which code produces it**. All facts grounded in the current repository source.

Legend: `app.py` is the UI layer. `se` = `services/scoring_engine.py`. `qg` = `services/questionnaire_generator.py`. `bfg` = `services/brd_frd_generator.py`. `db` = `services/database.py`.

---

## Sidebar (every page)

| Element | What it shows | How it's calculated |
|---------|----------------|--------------------|
| Navigation radio | 6 pages (`1. Setup` … `6. Export`) | Keyed to `st.session_state["page"]` so the `Next →` callback survives reruns. |
| GenAI status pill | `Connected` / `Offline / not configured` | Set by `_probe_genai()` — green only if `API_KEY` present AND `preflight_openai_connectivity` returns 200. |
| Probe diagnostics | One sentence | One of: `OPENAI_SKIP_API=true …`, `API key missing …`, `Preflight HTTP call did not return 200 …`, `GenAI connectivity OK`. |
| Re-check GenAI | Button | Clears `_genai_probed`, `_genai_client`, `_orchestrator`; re-runs probe. |
| Agent 1/2/3 captions | `Agent N ready - X obligations / RTM rows / questions` or `Agent N: not run` | Reads `len(analysis.obligations)`, `len(rtm.entries)`, `questionnaire.question_count` from session state. |
| Questionnaire questions | tile | `len(package["questions"])`. |
| Requirements | tile | `len(package["requirements"])`. |
| Package confidence | tile `NN%` | `package["metadata"]["overall_confidence_pct"]` — **content-correctness grade** in `qg._validate_content_correctness` (weights: 0.25 article-citation match, 0.20 explainability completeness, 0.15 traceability, 0.15 behaviour anchoring, 0.10 specificity, 0.10 evidence anchoring, 0.05 L1 option grounding). Floor controlled by `OVERALL_QUESTIONNAIRE_CONFIDENCE_FLOOR` (default 0). Set `PACKAGE_CONFIDENCE_MODE=structural` to revert to the legacy v11 formula. Both numbers are always emitted under `metadata.metrics.content_correctness_pct` / `metadata.metrics.structural_completeness_pct`. |
| Reset everything | Button | Deletes every `st.session_state` key not starting with `_`, then `_init_session_state()`. SQLite untouched. |

---

## Page 1 — Setup

| Element | What it shows | How it's calculated |
|---------|----------------|--------------------|
| Regulation code | Text input | Stored in `st.session_state["regulation"]`. Free-form. |
| Tier | Selectbox `Tier-1/2/3` | Stored in `st.session_state["tier"]`. |
| Regulation document uploader | PDF/DOCX | `save_upload(file, UPLOAD_DIR)` → `db.save_document(kind="regulation", …)`; returned id stored as `regulation_doc_id`. |
| Mode radio | `Use existing BRD/FRD` vs `Generate BRD/FRD from regulation` | Stored in `st.session_state["mode"]`; drives Page 2 + 3 code paths. |
| BRD/FRD DOCX uploader (existing-mode) | DOCX | `db.save_document(kind="brd", …)`; sets `brd_source = "uploaded"`. |
| Use bundled sample BRD | Button | Loads `sample_data/DORA_Tier2_Detailed_DetailedBRDFRD.docx`; sets `brd_source = "sample"`. |
| Stage-1 toggle (generate-mode) | Green/yellow banner | Green when `is_regulatory_search_enabled()` reads `REGULATORY_SEARCH_ENABLED=true`. |
| Regulator scope multiselect | `ALL` + 15 regulator codes | Options = `APPROVED_REGULATORS` codes. Defaults to `["ALL"]` if empty. |
| Preview regulator sources | Button | Calls `gather_regulatory_intelligence(...)` inside a live `st.status(...)` panel that streams stage-by-stage progress. Result cached in `regulatory_intelligence_package`. |
| Empty-result warnings | 4 conditional messages | Branched on whether the diagnostic strings contain `dns error`, `no results found`, `timeout`/`connecterror`, else generic. |
| Retrieved regulator sources table | Ranked publications | `package.all_sources()` filtered to non-Consulting. Sorted by `(SOURCE_PRIORITY, -confidence_score)`. Confidence from `official_regulation_fetcher._score_confidence` ∈ [0.5, 0.99]. |
| Existing artefacts table | Saved questionnaires | `db.list_questionnaires()` → columns `id, name, regulation, question_count, requirement_count, overall_confidence_pct, created_at`. |
| Load selected questionnaire | Button | Hydrates `questionnaire`, `package`, `questionnaire_id`; resets `assessment_state = AssessmentState()`. |
| Next → | Disabled until `regulation_doc_id` OR `brd_doc_id` OR `questionnaire` is set, OR mode is `Generate`. | `setup_ready` bool. |

---

## Page 2 — BRD / FRD

### Buttons

| Button | Action | Calls |
|--------|--------|-------|
| Parse uploaded BRD (existing-mode) | Reads requirement tables from the saved DOCX | `qg.read_docx_requirements` → `qg.derive_impact_pairs` → `db.save_requirements`. |
| Run Agents 1 + 2 (generate-mode) | Generates BRD/FRD + RTM | `orch.parse_document` → `orch.run_regulatory_analysis` → `orch.run_brd_rtm`. Writes DOCX to `outputs/{regulation}_BRD_FRD_<ts>.docx`. |

### Five KPI tiles

| Tile | Formula |
|------|---------|
| Completeness coverage | `metadata["completeness_coverage_pct"]`. For each requirement section computes `min(1.0, actual_items / DORA-tier minimum)` (minimums: process 14, data 14, reporting 10, functional 18, non-functional 10), averages across sections, rounds to a percent in `[0, 100]`. Captures "did we cover enough of the regulation's surface area?". |
| Accuracy coverage | `metadata["accuracy_coverage_pct"]`. Mean of per-row confidences (each clamped to `[90, 100]`) across all five requirement sections, rounded, floored at 90%. Captures "how accurately does each captured requirement map back to DORA / RTS / ITS?". |
| Used GenAI | `Yes` if `metadata["used_genai_shared_service"]`, else `No`. |
| Requirements | Sum of `section_counts[process_requirements + data_requirements + reporting_requirements + functional_requirements + non_functional_requirements]`. |
| Obligations (Agent 1) | `len(analysis.obligations)` — one obligation per requirement + one per control checkpoint. |

> `metadata["overall_confidence_pct"]` (mean of per-row confidences capped at 90% if any section misses its count gate) is still emitted for back-compat — it is what the BRD DOCX header and saved-questionnaire index column use — but the UI now surfaces the two-dimension split above.

### Regulation source (provenance) panel

| Sub-element | Source / formula |
|-------------|-------------------|
| Banner | Switches on `metadata["regulation_source"]`: `official_regulator` → green; `uploaded_document` → blue; `offline_baseline` → yellow; else grey caption. |
| Official sources tile | `summary["official_count"]`. |
| Regulators hit tile | `len(summary["regulators_hit"])`. |
| Approved-source publications expander | Columns: Source Type, Regulator, Publication Type, Regulation ID, Title (≤160 ch), Publication Date, Confidence, URL. Plus a "Download sources JSON" button. |

> `metadata["search_diagnostics"]` is still emitted by the Regulatory Intelligence Pipeline and persisted in the BRD artefact, but is no longer rendered as a UI expander — stage-by-stage progress is shown only inside the live `st.status(...)` panel during the run.

### Preview tables (first 50/50, parsed list of requirements)

| Table | Columns / source |
|-------|-------------------|
| Obligations | `id, theme, title (≤100ch), area, function, priority, regulatory_basis, deadline ("—" if empty)`. Source order (process → data → reporting → functional → non-functional → controls). |
| RTM | `trace_id, obligation, BR id, FR id ("—" if None), area, function, evidence_required`. From `rtm_artifact.entries`. FR id is `—` when token-overlap (`_pick_functional_requirement`) returns 0. |
| Parsed BRD requirements | `requirement_id, section, description (≤240ch), impacted_areas, impacted_functions`. Built from `requirements_from_report` + `derive_impact_pairs`. |

### Downloads (2 columns)

| Item | Source |
|------|--------|
| BRD + FRD DOCX | `_build_or_get_brd_docx` — written once during the Agents 1 + 2 run and reused for every subsequent download. |
| BRD + FRD JSON | `brd_artifact.report.model_dump_json(indent=2)`. |
| Requirements CSV | `_requirements_csv` — flattens to `requirement_id, source_id, section, category, requirement, detail, alignment, priority, acceptance, confidence`. |
| Obligations JSON | `json.dumps([asdict(o) for o in analysis.obligations], indent=2)`. |
| RTM JSON / CSV | `[asdict(e) for e in rtm.entries]` → JSON or CSV via pandas. |

---

## Page 3 — Questionnaire

### Five KPI tiles

| Tile | Formula |
|------|---------|
| Requirements | `len(package["requirements"])`. |
| Closed questions | `len([q for q in questions if not q["is_free_text"]])`. |
| Free-text questions | `len([q for q in questions if q["is_free_text"]])`. |
| Coverage (closed) | `metadata["coverage_pct"]` = `round(mapped_requirements / total_requirements × 100, 1)`. |
| Overall confidence | `metadata["overall_confidence_pct"]` = content-correctness grade (`0.25·article_match + 0.20·explainability_complete + 0.15·traceability + 0.15·behaviour_anchor + 0.10·specificity + 0.10·evidence_anchor + 0.05·l1_option_grounded`, ×100). Floor 0 by default. Legacy structural number available under `metadata.metrics.structural_completeness_pct`. |

### Other elements

| Element | Notes |
|---------|-------|
| Run Agent 3 (primary) | Branches on `mode` → `orch.run_questionnaire_from_report` or `orch.run_questionnaire_from_docx`. Saves via `db.save_questionnaire`; resets `assessment_state`. |
| Upload questionnaire JSON | Validates with `utils.json_utils.validate_package_schema`; errors are printed as bullets. |
| Question preview | First 25 rows: `id, area, function, type, question, confidence`. |

---

## Page 4 — Assessment

### Action bar (4 columns)

| Control | Effect |
|---------|--------|
| Focus area selectbox | `All` + sorted areas of closed base Qs. Drives `choose_next_question(..., focus_area=focus)`. |
| Start / continue | Creates an `assessments` row in SQLite if none exists. |
| Restart | `state.reset_responses()` + persist. |
| New session | Resets state AND creates a fresh `assessments` row. |

### Five KPI tiles

```python
active = applicable_base_questions(state, base_questions) + list(state.dynamic_queue)
applicable_count = len([q for q in active if not q["is_free_text"]])
answered_count   = sum(1 for q in active if answered(q, state.responses))
```

| Tile | Formula |
|------|---------|
| Answered / Applicable | `f"{answered_count} / {applicable_count}"`. |
| Dynamic follow-ups pending | Unanswered count in `state.dynamic_queue`. |
| Skipped by funnel | `len(state.skipped_ids)`. Grows when `_apply_positive_skip` marks downstream Qs unnecessary after a positive answer. |
| Branch decisions | `len(state.branch_log)`. |
| Assessment ID | `st.session_state["assessment_id"]`. |

### Adaptive branch trace expander

Shows last **15** entries: `Parent, Answer, Source (registry/generic), Rule (branch_rule_id), Theme, Children`.

### Question card

| Sub-element | Source |
|-------------|--------|
| Label | `f"Question {N:03d} | {area} / {function} | Confidence {conf}%"`. Dynamic Qs append `"| Adaptive branch follow-up [| rule: <id>]"`. |
| Response widget | `Multi Select` → `st.multiselect`; else `st.radio` defaulting to `Unknown` when present (prevents accidental Yes). |
| Comments | Stored at `state.responses[f"{qid}__comments"]`. |
| Why-asked expander | Text from `se.rationale_text(q, responses)`. |
| On submit | `state.responses[qid]=value` → `se.update_applicability_after_response(...)` (enqueues follow-ups, skips downstream Qs, appends `branch_log`) → `state.history.append(qid)` → `st.rerun()`. |

Free-text answers live in `state.responses` but are **never scored** by `se.evaluate`. Persistence (`_persist_assessment_snapshot`) writes full state JSON + per-question rows after every action.

---

## Page 5 — Dashboard

Every render: `_refresh_scoring_snapshot()` → `orch.run_rules_engine` → `se.evaluate(active_questions, state)`.

`evaluate` loop per non-skipped, non-free-text question:
```
weight = scoring_weight × (confidence / 100)
score  = se.score_value(answer, q)
# accumulate score×weight and 100×weight into total, area, function, pair, requirement
```

### Four KPI tiles

| Tile | Formula | Range |
|------|---------|-------|
| Readiness / Compliance | `round(total_num / total_den × 100, 1)` | 0.0–100.0 |
| Evaluation confidence | `round(max(90, min(99, avg_conf − min(8, unans·0.1) + min(3, ans·0.05))), 1)` | 90.0–99.0 |
| Answered closed questions | `f"{answered_count} / {answered_count + unanswered_count}"` | int |
| Pairs scored | `len(pair_scores)` | int |

### Filter radio

`All / Critical / At risk / Watch / Ready`. Filters area & function tables on `CXO status` column.

`se.cxo_status(score)`:
- ≥ 85 → `Ready`, `Maintain evidence and periodic validation.`
- ≥ 65 → `Watch`, `Resolve targeted gaps before executive sign-off.`
- ≥ 40 → `At risk`, `Prioritise remediation plan, owners and evidence.`
- < 40 → `Critical`, `Escalate to governance and define funded remediation.`

### Tables / heatmap

| Table | Columns | Source |
|-------|---------|--------|
| Aggregate by impacted area | `Impacted Area, Compliance %, CXO status, Questions scored, Recommended executive action` | `se.summary_dataframe(area_summary, "Impacted Area")`; sorted ascending. |
| Aggregate by function | Same columns | `se.summary_dataframe(function_summary, "Function")`. |
| Area × Function heatmap | One row per area, one column per function | `se.pair_heatmap_rows(pair_scores)`. Empty cells = no answers for that pair. |
| Top gaps | `Requirement, Compliance %` | `scoring.top_gaps` — lowest 10 `requirement_scores`. |

All score columns styled by `_df_with_styling(df, cols)`:
```python
df.style.background_gradient(subset=cols, cmap="RdYlGn", vmin=0, vmax=100)
        .format({c: "{:.1f}%" for c in cols})
```

### Agent 4 panel

| Control | Default | Effect |
|---------|---------|--------|
| Minimum severity | `Watch` | Filters which `(area, function)` pairs become recommendations. |
| Top requirements | `10` (1–30) | Cap for the lowest-scoring-requirement bucket. |
| Use GenAI to refine action wording | unchecked; disabled when GenAI offline | If checked, calls `enrich_recommendations_with_genai`. |
| Run Agent 4 (primary) | → `orch.run_recommendations(..., branch_log=state.branch_log)`. |

Recommendations table columns: `id (REC-NNN), severity, title, compliance %, owner (_OWNER_BY_FUNCTION map), horizon (_SEVERITY_HORIZON map), action`. Sorted by `(severity_rank, compliance_pct asc)`.

---

## Page 6 — Export

| Button | Output | Source |
|--------|--------|--------|
| Download questionnaire JSON | `<regulation>_questionnaire_package.json` | `json.dumps(pkg, indent=2)`. |
| Download responses JSON | `<regulation>_responses.json` | Payload includes `responses, skipped_by_funnel, history, branch_log, dynamic_queue, evaluation (tuple keys flattened to "area | function" via `_jsonable_eval`), recommendations`. |
| Download obligations JSON | `<regulation>_obligations.json` | Rendered only when Agent 1 has run. |
| Download RTM JSON | `<regulation>_RTM.json` | Rendered only when `rtm_artifact.entries` non-empty. |
| Build Excel and prepare download | `outputs/{regulation}_Readiness_Report_<ts>.xlsx` | `write_excel_from_package(target, pkg)` — sheets: Summary, Impacted Functions Areas, Questionnaire, Free Text Questions, Funnel Logic, Scoring Rubric, Requirement Traceability. |
| Download generated BRD/FRD DOCX | Path from `brd_artifact.docx_path` | Only when Agents 1 + 2 ran on Page 2. |

---

## Styling (Page 5 tables)

All score cells: `RdYlGn` gradient (red = 0, green = 100), `{:.1f}%` formatting. Empty score cells stay blank. If styling fails the raw dataframe is rendered (no crash).

---

## Quick reference — every numeric metric on screen

| Where | Metric | Formula | Range |
|-------|--------|---------|-------|
| Sidebar | Package confidence | Content-correctness weighted sum, ×100 | 0–100 |
| Page 2 | Completeness coverage | Mean of `min(1, actual/minimum)` per section, ×100 | 0–100 |
| Page 2 | Accuracy coverage | Row-mean of per-requirement confidences, floored | 90–100 |
| Page 3 | Coverage (closed) | mapped / total × 100 | 0–100 |
| Page 3 | Overall confidence | Content-correctness weighted formula | 0–100 |
| Page 4 | Answered / Applicable | counts on `active` set | int |
| Page 4 | Dynamic follow-ups pending | unanswered in queue | int |
| Page 4 | Skipped by funnel | `len(state.skipped_ids)` | int |
| Page 4 | Branch decisions | `len(state.branch_log)` | int |
| Page 5 | Readiness / Compliance | `Σ(score·w) / Σ(100·w) × 100` | 0–100 |
| Page 5 | Evaluation confidence | `max(90, min(99, avg_conf − 0.1·unans + 0.05·ans))` | 90–99 |
| Page 5 | Area / Function / Pair % | bucket `Σ(score·w)/Σ(100·w)·100` | 0–100 |
| Page 5 | Top gaps % | per-requirement `req_num/req_den·100` | 0–100 |
