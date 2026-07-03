"""Regulatory Impact & Readiness Assessment — Streamlit cockpit.

This module is the Streamlit UI layer only. All workflow logic flows through
the :class:`~orchestrator.RegulatoryWorkflowOrchestrator`, which coordinates
the agentic pipeline:

    Upload Regulation
        -> Document Parser
            -> Agent 1: Regulatory Analysis -> Obligations
                -> Agent 2: BRD + RTM
                    -> Agent 3: Questionnaire Generation
                        -> User Responses
                            -> Python Rules Engine
                                -> Agent 4: Recommendations -> Dashboard

Six pages remain available from the sidebar (Setup / BRD-FRD / Questionnaire /
Assessment / Dashboard / Export) so existing users do not have to relearn the
cockpit. Each page now calls orchestrator methods instead of reaching into
individual services.

The app is robust to:

* GenAI Shared Service being unreachable (offline fallback BRD).
* Missing uploads (clear inline messages, no crashes).
* Re-runs in the middle of an assessment (state is restored from SQLite).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from models.workflow_models import (
    BRDArtifact,
    Obligation,
    QuestionnairePackage,
    RegulatoryAnalysis,
    RTMArtifact,
    ScoringResult,
)
from orchestrator import RegulatoryWorkflowOrchestrator
from services import persistence as db
from services.genai_service import GenAIClient
from services.brd_frd_generator import (
    DoraDetailedBRD,
    write_brd_docx,
)
from services.regulatory_intelligence_service import (
    RegulatoryIntelligencePackage,
    gather_regulatory_intelligence,
)
from services.search_config import (
    APPROVED_REGULATORS,
    is_regulatory_search_enabled,
)
from services.questionnaire_generator import (
    option_label,
    option_labels,
    write_excel_from_package,
)
from services.recommendation_service import Recommendation
from services.scoring_engine import (
    AssessmentState,
    answered,
    applicable_base_questions,
    choose_next_question,
    pair_heatmap_rows,
    rationale_text,
    summary_dataframe,
    update_applicability_after_response,
)
from utils.file_utils import ensure_dirs, save_upload, timestamped_name
from utils.json_utils import validate_package_schema


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DIR = PROJECT_ROOT / "sample_data"
DATA_DIR = PROJECT_ROOT / "data"

ensure_dirs(UPLOAD_DIR, OUTPUT_DIR, SAMPLE_DIR, DATA_DIR)
db.init_db()

st.set_page_config(
    page_title="Regulatory Impact & Readiness Cockpit",
    page_icon="OK",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Styling (kept compact; mirrors the original PwC cockpit visual identity)
# ---------------------------------------------------------------------------

_HERO_CSS = """
<style>
/* ------------------------------------------------------------------ */
/* Light-background regions: dark text                                 */
/* ------------------------------------------------------------------ */
.stApp {
    background: linear-gradient(180deg, #fff8f2 0%, #ffffff 34%, #ffffff 100%);
    color: #1a1a1a;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #2d2d2d !important;
    font-weight: 700;
}
.stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span, [data-testid="stMarkdownContainer"] strong {
    color: #1a1a1a !important;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
    color: #4a4a4a !important;
}

/* Inputs — light background, dark text */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stDateInput input, .stTimeInput input {
    color: #1a1a1a !important;
    background-color: #ffffff !important;
    border: 1px solid #d9c3b3 !important;
}
/* Selectbox / multiselect: BaseWeb wraps content in a [role="combobox"] */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border: 1px solid #d9c3b3 !important;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] input,
div[data-baseweb="select"] svg {
    color: #1a1a1a !important;
    fill: #1a1a1a !important;
}
/* Selectbox dropdown menu (BaseWeb popover) */
div[data-baseweb="popover"], div[data-baseweb="menu"],
div[data-baseweb="popover"] li, div[data-baseweb="menu"] li {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}
div[data-baseweb="menu"] li:hover {
    background-color: #fff0e6 !important;
}

/* Input labels */
.stRadio label, .stCheckbox label,
.stSelectbox label, .stMultiSelect label,
.stTextInput label, .stNumberInput label,
.stTextArea label, .stFileUploader label,
.stDateInput label, .stTimeInput label,
label[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] * {
    color: #1a1a1a !important;
    font-weight: 500;
}

/* File uploader dropzone */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] * {
    color: #1a1a1a !important;
    background-color: #ffffff !important;
}
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed #d04a02 !important;
}
[data-testid="stFileUploader"] small {color: #4a4a4a !important;}

/* Buttons: white + orange border, primary = orange fill + white text */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {
    color: #1a1a1a !important;
    background-color: #ffffff !important;
    border: 1px solid #d04a02 !important;
    font-weight: 600;
}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"],
.stFormSubmitButton button[kind="primary"] {
    color: #ffffff !important;
    background-color: #d04a02 !important;
    border: 1px solid #b03d00 !important;
}
.stButton button[kind="primary"] *, .stDownloadButton button[kind="primary"] *,
.stFormSubmitButton button[kind="primary"] * {
    color: #ffffff !important;
}
.stButton button:hover, .stDownloadButton button:hover {
    border-color: #b03d00 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {color: #1a1a1a !important;}

/* DataFrames + metrics */
[data-testid="stDataFrame"] * {color: #1a1a1a !important;}
[data-testid="stMetric"] {background: #ffffff; border: 1px solid #f0d7c8;
           border-left: 6px solid #d04a02; padding: 0.7rem; border-radius: 12px;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    color: #4a4a4a !important; font-weight: 600;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    color: #2d2d2d !important; font-weight: 700;
}

/* Alerts (info/warning/success/error) keep their tinted backgrounds */
[data-testid="stAlert"], [data-testid="stAlert"] * {color: #1a1a1a !important;}

/* Expanders */
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
    color: #1a1a1a !important;
    font-weight: 600;
}
[data-testid="stExpander"] details > div {color: #1a1a1a !important;}

/* Executive card (in-page panel) */
.exec-card {background: #ffffff; border: 1px solid #ead8cc; padding: 1rem;
           border-radius: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);}
.exec-card, .exec-card * {color: #1a1a1a !important;}

/* ------------------------------------------------------------------ */
/* Dark-background regions: light text                                 */
/* ------------------------------------------------------------------ */
.pwc-hero {background: linear-gradient(90deg, #2d2d2d 0%, #4a4a4a 45%, #d04a02 100%);
           padding: 1.0rem 1.3rem; border-radius: 14px; margin-bottom: 1rem;
           box-shadow: 0 6px 18px rgba(0,0,0,0.12);}
.pwc-hero, .pwc-hero p, .pwc-hero span, .pwc-hero h1, .pwc-hero h2,
.pwc-hero h3, .pwc-hero a {color: #ffffff !important;}
.pwc-title {font-size: 1.7rem; font-weight: 800; margin: 0; color: #ffffff !important;}
.pwc-subtitle {font-size: 0.95rem; margin-top: .25rem; opacity: .95; color: #f7e6dc !important;}

/* Code blocks (dark background, light monospace text) */
[data-testid="stCodeBlock"], [data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code, .stCodeBlock pre, .stCodeBlock code {
    background-color: #1f2937 !important;
    color: #f3f4f6 !important;
}

/* Status pills */
.status-pill {border-radius: 999px; padding: .15rem .55rem; font-weight: 700;}
.status-Critical {color:#ffffff !important;background:#b00020;}
.status-At-risk  {color:#2d2d2d !important;background:#ffb600;}
.status-Watch    {color:#2d2d2d !important;background:#ffd966;}
.status-Ready    {color:#ffffff !important;background:#2e7d32;}

/* ------------------------------------------------------------------ */
/* Sidebar — light background, dark text                               */
/* ------------------------------------------------------------------ */
section[data-testid="stSidebar"] {
    background: #fff8f2;
    border-right: 1px solid #ead8cc;
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] li {
    color: #1a1a1a !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {color: #2d2d2d !important;}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {color: #4a4a4a !important;}
section[data-testid="stSidebar"] .stRadio label {color: #1a1a1a !important; font-weight: 500;}

/* "Next" button is centred + slightly larger for hero placement */
.next-button {margin-top: 1.5rem;}
</style>
<div class="pwc-hero">
  <p class="pwc-title">Regulatory Impact & Readiness Cockpit</p>
  <p class="pwc-subtitle">Upload a regulation, run the agentic workflow (Regulatory Analysis → BRD+RTM → Questionnaire → Scoring → Recommendations), and produce CXO-grade output.</p>
</div>
"""

st.markdown(_HERO_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

_DEFAULT_STATE: Dict[str, Any] = {
    # Setup
    "regulation": "DORA",
    "tier": "Tier-2",
    "mode": "Use existing BRD/FRD",
    "regulation_doc_id": None,
    "brd_doc_id": None,
    # Regulatory Intelligence Pipeline (Stage 1 only; Stage 2 is currently
    # disabled pending team review -- see CONSULTING_SEARCH_ENABLED in .env).
    "regulator_selection": ["ALL"],          # codes from search_config.APPROVED_REGULATORS
    "regulatory_intelligence_package": None, # cached RegulatoryIntelligencePackage
    # Agent 1 / Agent 2 outputs
    "analysis": None,             # RegulatoryAnalysis | None
    "brd_artifact": None,         # BRDArtifact | None
    "rtm_artifact": None,         # RTMArtifact | None
    "brd_source": None,           # 'uploaded' | 'generated' | 'sample'
    # Questionnaire
    "questionnaire": None,        # QuestionnairePackage | None
    "package": None,              # Dict (kept for backward-compat with helpers)
    "package_source": None,
    "questionnaire_id": None,
    # Assessment
    "assessment_state": AssessmentState(),
    "assessment_id": None,
    "focus_area": "All",
    "dashboard_filter": "All",
    # Live evaluation + recommendations
    "scoring_result": None,       # ScoringResult | None
    "evaluation": None,           # dict (legacy mirror of scoring_result.evaluation)
    "recommendations": [],
    # Page
    "page": "1. Setup",
    # GenAI
    "_genai_probed": False,
    "genai_available": False,
    # Orchestrator
    "_orchestrator": None,
}


def _init_session_state() -> None:
    for key, default in _DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default


_init_session_state()


def _probe_genai(*, force_reload_env: bool = False) -> None:
    """Probe the GenAI Shared Service once per session (cached).

    Captures the failure reason in ``genai_probe_message`` so the sidebar can
    show why we are offline (missing key, network error, etc.) instead of a
    generic "Offline" pill.
    """
    if st.session_state["_genai_probed"]:
        return

    import os
    if force_reload_env:
        load_dotenv(override=True)

    from services.genai_service import (
        build_http_client,
        get_llm_api_key,
        get_settings,
        preflight_openai_connectivity,
        create_configured_llm,
        GenAIConfigError,
    )

    message = ""
    settings = get_settings()
    if settings.skip_api:
        message = "OPENAI_SKIP_API=true in .env — offline mode forced."
    else:
        try:
            api_key = get_llm_api_key()
        except GenAIConfigError as exc:
            api_key = ""
            message = f"API key missing: {exc}"

        if api_key:
            http_client = build_http_client(settings)
            try:
                ok = preflight_openai_connectivity(http_client, settings)
            except Exception as exc:
                ok = False
                message = f"Preflight raised {type(exc).__name__}: {exc}"
            if ok:
                try:
                    llm = create_configured_llm(api_key, http_client, settings)
                    st.session_state["_genai_client"] = GenAIClient(
                        api_key=api_key, http_client=http_client,
                        settings=settings, llm=llm,
                    )
                    message = (f"Connected to {settings.base_url} "
                               f"using model {settings.model}.")
                except Exception as exc:
                    http_client.close()
                    message = f"LLM init failed: {type(exc).__name__}: {exc}"
            else:
                http_client.close()
                if not message:
                    message = ("Preflight HTTP call did not return 200. "
                               "Check API_KEY, model name, network/VPN, and proxy.")

    st.session_state["genai_available"] = st.session_state.get("_genai_client") is not None
    st.session_state["genai_probe_message"] = message
    st.session_state["_genai_probed"] = True


_probe_genai()


def _genai_client() -> Optional[GenAIClient]:
    return st.session_state.get("_genai_client")


def _get_orchestrator() -> RegulatoryWorkflowOrchestrator:
    """Return a singleton orchestrator bound to the current GenAI client."""
    orch = st.session_state.get("_orchestrator")
    if orch is None:
        orch = RegulatoryWorkflowOrchestrator(client=_genai_client())
        st.session_state["_orchestrator"] = orch
    return orch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_pill(status: str) -> str:
    safe = status.replace(" ", "-")
    return f'<span class="status-pill status-{safe}">{status}</span>'


def _df_with_styling(df: pd.DataFrame, score_cols: List[str]) -> Any:
    if df.empty:
        return df
    try:
        styled = df.style.background_gradient(
            subset=score_cols, cmap="RdYlGn", vmin=0, vmax=100
        ).format({c: "{:.1f}%" for c in score_cols})
        return styled
    except Exception:
        return df


def _refresh_scoring_snapshot() -> Optional[ScoringResult]:
    """Re-run the Python Rules Engine against the current responses."""
    questionnaire: Optional[QuestionnairePackage] = st.session_state.get("questionnaire")
    if questionnaire is None:
        return None
    state: AssessmentState = st.session_state["assessment_state"]
    orch = _get_orchestrator()
    scoring = orch.run_rules_engine(questionnaire, state)
    st.session_state["scoring_result"] = scoring
    st.session_state["evaluation"] = scoring.evaluation
    return scoring


def _persist_assessment_snapshot(completed: bool = False) -> None:
    """Push the current assessment state + evaluation + recs to SQLite."""
    if st.session_state.get("assessment_id") is None:
        return
    state: AssessmentState = st.session_state["assessment_state"]
    scoring: Optional[ScoringResult] = st.session_state.get("scoring_result")
    eval_result = scoring.evaluation if scoring else st.session_state.get("evaluation")
    recs = st.session_state.get("recommendations") or []
    state_payload = {
        "responses": state.responses,
        "dynamic_queue": list(state.dynamic_queue),
        "skipped_ids": sorted(state.skipped_ids),
        "display_numbers": state.display_numbers,
        "display_counter": state.display_counter,
        "history": list(state.history),
        "branch_log": list(state.branch_log),
        "dynamic_questions_emitted": state.dynamic_questions_emitted,
        "emitted_dynamic_ids": sorted(state.emitted_dynamic_ids),
    }
    db.update_assessment_snapshot(
        assessment_id=st.session_state["assessment_id"],
        state_json=json.dumps(state_payload, ensure_ascii=False, default=str),
        evaluation=eval_result,
        recommendations=[_rec_to_dict(r) for r in recs],
        completed=completed,
    )
    db.upsert_responses(
        assessment_id=st.session_state["assessment_id"],
        responses=state.responses,
    )


def _rec_to_dict(r: Any) -> Dict[str, Any]:
    if isinstance(r, dict):
        return r
    if isinstance(r, Recommendation) or is_dataclass(r):
        return asdict(r)
    return dict(r)


# ---------------------------------------------------------------------------
# Regulatory Intelligence Pipeline — UI helpers (Stage 1 only)
# ---------------------------------------------------------------------------
#
# Stage 1 = approved regulator domains (EBA, ESMA, ECB, FCA, BaFin, ...).
# Stage 2 = approved consulting firms — currently DISABLED pending team
#   confirmation. The fetcher modules still ship in services/ so re-enabling
#   is a config-only change (set CONSULTING_SEARCH_ENABLED=true and restore
#   the Stage 2 widgets here).
# Generic internet search is intentionally NOT supported.


_ALL_REGULATOR_CODE = "ALL"


def _selected_regulator_codes() -> List[str]:
    sel = st.session_state.get("regulator_selection") or [_ALL_REGULATOR_CODE]
    return list(sel) if isinstance(sel, list) else [str(sel)]


def _regulator_label(code: str) -> str:
    if code == _ALL_REGULATOR_CODE:
        return f"All regulators ({len(APPROVED_REGULATORS)})"
    for r in APPROVED_REGULATORS:
        if r.code == code:
            return f"{r.name} ({r.code}) - {r.jurisdiction}"
    return code


def _fresh_intelligence_package() -> Optional[RegulatoryIntelligencePackage]:
    """Return the cached :class:`RegulatoryIntelligencePackage` only if its
    inputs still match the current session state.

    Prevents Agent 1 from being fed a stale Stage 1 package after the user
    has changed the regulation or regulator selection since the last
    "Preview" click.
    """
    pkg: Optional[RegulatoryIntelligencePackage] = st.session_state.get(
        "regulatory_intelligence_package"
    )
    if pkg is None:
        return None
    if pkg.regulation != (st.session_state.get("regulation") or ""):
        return None
    if list(pkg.regulator_selection) != _selected_regulator_codes():
        return None
    return pkg


def _render_regulator_selector() -> None:
    """Stage 1 selector. Lets the user scope search to specific regulator(s)."""
    options = [_ALL_REGULATOR_CODE] + [r.code for r in APPROVED_REGULATORS]
    current = _selected_regulator_codes()
    cleaned = [c for c in current if c in options] or [_ALL_REGULATOR_CODE]

    st.session_state["regulator_selection"] = st.multiselect(
        "Regulator scope",
        options=options,
        default=cleaned,
        format_func=_regulator_label,
        help=(
            "Search is restricted to the official websites of the selected regulators. "
            "Choose 'ALL' to query every approved regulator. "
            "Wikipedia, blogs and generic search are never used."
        ),
        key="regulator_selection_widget",
    )
    if not st.session_state["regulator_selection"]:
        st.session_state["regulator_selection"] = [_ALL_REGULATOR_CODE]


def _render_regulatory_intelligence_block() -> None:
    """Stage 1-only intelligence panel rendered on Page 1.

    Lets the user pick which official regulators to query and previews the
    publications retrieved before the full BRD generation runs.
    """
    regulation = st.session_state.get("regulation") or "DORA"
    stage1_enabled = is_regulatory_search_enabled()

    with st.expander("Regulatory Intelligence — Official Regulator Search", expanded=True):
        if stage1_enabled:
            st.success(
                f"Regulator search is **ON** for `{regulation}`. "
                "Only approved regulator domains will be searched."
            )
        else:
            st.warning(
                "Regulator search is **OFF**. Set `REGULATORY_SEARCH_ENABLED=true` in `.env` "
                "to enable live regulator search."
            )

        _render_regulator_selector()

        c1, c2 = st.columns([1, 3])
        with c1:
            preview = st.button(
                "Preview regulator sources",
                help="Run the regulator search now and show the URLs.",
            )
        with c2:
            st.caption(
                "Hits each regulator's own site-search directly (EBA, ESMA, EIOPA, FCA) "
                "and only falls back to a general web search if needed. Completes in 2-15 "
                "seconds depending on the selection. Manual regulation document upload "
                "(above) remains supported."
            )

        if preview:
            with st.status(
                f"Searching approved regulators for `{regulation}`...", expanded=True
            ) as preview_status:
                def _preview_log(msg: str) -> None:
                    preview_status.write(msg)

                package = gather_regulatory_intelligence(
                    regulation,
                    regulator_selection=_selected_regulator_codes(),
                    consulting_selection=None,
                    include_consulting=False,
                    status=_preview_log,
                )
                preview_status.update(
                    label="Regulator search complete",
                    state="complete",
                    expanded=False,
                )
            st.session_state["regulatory_intelligence_package"] = package
            if not package.has_any_content:
                # Classify why Stage 1 came back empty. The pipeline now tries
                # each regulator's *native* site-search first (no third-party
                # search engine), and only falls back to DDGS if needed. So
                # the diagnostic should report what *actually* failed, not
                # blame DuckDuckGo unconditionally.
                errs = [e or "" for e in package.errors]
                dns_hit = any("dns error" in e.lower() or "decoding error" in e.lower() for e in errs)
                no_results_hit = any("no results found" in e.lower() for e in errs)
                timeout_hit = any("timeout" in e.lower() or "connecttimeout" in e.lower() for e in errs)
                connect_hit = any("connecterror" in e.lower() or "connect error" in e.lower() for e in errs)

                if dns_hit:
                    st.warning(
                        "**No live regulator publications retrieved.** "
                        "Both the regulator-native search and the DDGS fallback returned "
                        "DNS protocol errors (`record type OPT only allowed in additional section`). "
                        "This is a corporate DNS-resolver bug, not an app issue. "
                        "Try again in a minute, or upload the regulation PDF on Page 1 -- "
                        "Agent 1 will use that as primary context."
                    )
                elif no_results_hit:
                    st.warning(
                        "**No live regulator publications retrieved.** "
                        "Native search found no DORA-relevant anchors on the selected "
                        "regulator sites, and the DDGS fallback was rate-limited. "
                        "Try a different regulator selection, or upload the regulation PDF."
                    )
                elif timeout_hit or connect_hit:
                    st.warning(
                        "**No live regulator publications retrieved.** "
                        "Every backend timed out / could not connect. Check VPN/proxy connectivity, "
                        "or upload the regulation PDF and Agent 1 will use it as primary context."
                    )
                else:
                    st.error(
                        "No publications retrieved from approved regulator domains. "
                        "Check the status log above, or upload the regulation PDF and Agent 1 will "
                        "use it as primary context."
                    )
                try:
                    import importlib
                    spec_new = importlib.util.find_spec("ddgs")
                    spec_old = importlib.util.find_spec("duckduckgo_search")
                    st.caption(
                        f"DDGS package presence — new `ddgs`: "
                        f"{'OK' if spec_new else 'MISSING'} | "
                        f"legacy `duckduckgo_search`: {'present (deprecated)' if spec_old else 'absent'}."
                    )
                except Exception:
                    pass
            else:
                st.success(
                    f"Retrieved {len(package.official_results)} official publication(s) "
                    f"from approved regulator domains."
                )

        package: Optional[RegulatoryIntelligencePackage] = st.session_state.get("regulatory_intelligence_package")
        if package and package.has_official_content:
            _render_intelligence_sources_table(package)


def _render_intelligence_sources_table(package: RegulatoryIntelligencePackage) -> None:
    """Render every retrieved official regulator source as a ranked table."""
    rows = [r for r in package.all_sources() if r.get("source_type") != "Consulting Guidance"]
    if not rows:
        return
    df = pd.DataFrame([{
        "Source Type": r["source_type"],
        "Regulator": r["regulator"],
        "Publication Type": r["publication_type"],
        "Regulation ID": r["regulation_id"],
        "Title": (r["title"] or "")[:120],
        "Publication Date": r["publication_date"],
        "Confidence": r["confidence_score"],
        "URL": r["source_url"],
    } for r in rows])
    st.markdown("**Retrieved regulator sources (ranked by confidence)**")
    st.dataframe(df, width="stretch", hide_index=True)


def _format_source_label(ref: Dict[str, Any]) -> str:
    """Compact ``Regulator - Reference - Date`` label for the UI."""
    parts: List[str] = []
    if ref.get("regulator"):
        parts.append(str(ref["regulator"]))
    ref_label = ref.get("regulation_reference") or ref.get("publication_type") or ""
    if ref_label and ref_label not in parts:
        parts.append(str(ref_label))
    if ref.get("publication_date"):
        parts.append(str(ref["publication_date"]))
    if not parts:
        title = ref.get("title") or ref.get("source_type") or "Source"
        parts.append(str(title)[:80])
    return " - ".join(parts)


def _format_sources_inline(refs: List[Dict[str, Any]]) -> str:
    """One-line summary of citations used in DataFrame cells / preview tables.

    Renderers truncate the list to the top three references; the full list is
    surfaced by the "Source References" panel below the table.
    """
    if not refs:
        return "(no live source matched - see panel below)"
    pieces: List[str] = []
    for ref in refs[:3]:
        pieces.append(_format_source_label(ref))
    suffix = "" if len(refs) <= 3 else f" (+{len(refs) - 3} more)"
    return " | ".join(pieces) + suffix


def _render_source_references_panel(brd_artifact: BRDArtifact) -> None:
    """Render the dedicated Source References panel on Page 2.

    Shows two views:
    1. Master catalogue - every unique publication that contributed to the
       BRD (Source Type, Regulator, Title, Reference, Date, URL).
    2. Per-requirement traceability - each BRD requirement ID with its
       cited source(s) rendered as a compact bullet list. Requirements
       that could not be anchored to a retrieved publication are flagged
       explicitly so reviewers see the gap.
    """
    metadata = brd_artifact.metadata or {}
    catalogue: List[Dict[str, Any]] = metadata.get("source_references_catalogue") or []
    refs_by_item: Dict[str, List[Dict[str, Any]]] = (
        metadata.get("source_references_by_item") or {}
    )

    st.markdown("#### Source References")
    if not catalogue and not refs_by_item:
        st.warning(
            "No source-reference metadata is attached to this BRD. The "
            "regulator search returned no usable publications and no "
            "regulation document was uploaded, so the BRD is running on the "
            "offline baseline. Validate every requirement against the "
            "official regulation text before sign-off."
        )
        return

    used_uploaded = bool(metadata.get("source_references_used_uploaded_document"))
    used_offline = bool(metadata.get("source_references_used_offline_baseline"))
    total_unique = int(metadata.get("source_references_total_unique") or len(catalogue))

    summary_cols = st.columns(3)
    summary_cols[0].metric("Unique sources cited", total_unique)
    summary_cols[1].metric(
        "Uploaded regulation",
        "Yes" if used_uploaded else "No",
        help="Did the BRD generator consume text from a user-uploaded regulation document?",
    )
    summary_cols[2].metric(
        "Offline baseline",
        "Yes" if used_offline else "No",
        help="True when no live regulator publication was retrieved. "
             "Citations fall back to a sentinel 'No live source available' marker.",
    )

    with st.expander("Master source catalogue", expanded=False):
        if not catalogue:
            st.info(
                "No live regulatory publications were retrieved for this run. "
                "The BRD content reflects the offline baseline and/or the "
                "uploaded regulation document."
            )
        else:
            df = pd.DataFrame([{
                "#": idx + 1,
                "Source Type": r.get("source_type", ""),
                "Regulator / Issuer": r.get("regulator", ""),
                "Title": (r.get("title", "") or "")[:160],
                "Reference": r.get("regulation_reference", "") or r.get("publication_type", ""),
                "Publication Date": r.get("publication_date", ""),
                "URL": r.get("source_url", ""),
                "Confidence": r.get("confidence", ""),
            } for idx, r in enumerate(catalogue)])
            st.dataframe(df, width="stretch", hide_index=True)
            st.download_button(
                "Download source catalogue (JSON)",
                data=json.dumps(catalogue, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{metadata.get('regulation', 'regulation')}_source_references.json",
                mime="application/json",
            )

    requirement_refs = {
        key.split(":", 1)[1]: refs
        for key, refs in refs_by_item.items() if key.startswith("REQ:")
    }
    if requirement_refs:
        with st.expander(
            f"Per-requirement traceability ({len(requirement_refs)} requirements)",
            expanded=False,
        ):
            rows: List[Dict[str, Any]] = []
            for req_id in sorted(requirement_refs.keys()):
                refs = requirement_refs[req_id]
                if not refs:
                    rows.append({
                        "Requirement ID": req_id,
                        "Sources": "[!] No live source available",
                        "URL(s)": "",
                    })
                    continue
                labels = " | ".join(_format_source_label(r) for r in refs)
                urls = "\n".join(r.get("source_url", "") for r in refs if r.get("source_url"))
                rows.append({
                    "Requirement ID": req_id,
                    "Sources": labels,
                    "URL(s)": urls,
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_regulation_source_panel(brd_artifact: BRDArtifact) -> None:
    """Show the provenance of the BRD's regulatory context on Page 2."""
    metadata = brd_artifact.metadata or {}
    source = metadata.get("regulation_source", "unknown")
    official_sources: List[Dict[str, Any]] = metadata.get("official_sources") or []
    summary: Dict[str, Any] = metadata.get("source_summary") or {}
    used_uploaded = bool(metadata.get("used_uploaded_document"))

    st.markdown("#### Regulation source (provenance)")
    if source == "official_regulator":
        st.success(
            f"This BRD was generated from **{len(official_sources)} official regulatory publication(s)** "
            f"retrieved from approved regulator domains for `{metadata.get('regulation')}`."
        )
    elif source == "uploaded_document":
        st.info(
            "This BRD was generated using the **regulation document you uploaded** on Page 1 as "
            "primary context. The regulator search returned no usable results."
        )
    elif source == "offline_baseline":
        st.warning(
            "The regulator search returned no usable results and no regulation document was "
            "uploaded, so Agent 1 fell back to the **offline baseline**. The BRD content reflects "
            "the LLM's pretrained knowledge rather than live regulatory sources."
        )
    else:
        st.caption("Regulation source metadata unavailable.")

    if used_uploaded and source == "official_regulator":
        st.caption("Note: your uploaded document was also appended to the prompt context.")

    cols = st.columns(2)
    cols[0].metric("Official sources", summary.get("official_count", len(official_sources)))
    cols[1].metric("Regulators hit", len(summary.get("regulators_hit") or []))

    ranked_rows: List[Dict[str, Any]] = [
        r for r in (metadata.get("all_sources_ranked") or [])
        if r.get("source_type") != "Consulting Guidance"
    ]
    if ranked_rows:
        with st.expander(
            f"Show the {len(ranked_rows)} approved-source publication(s) used",
            expanded=False,
        ):
            df = pd.DataFrame([{
                "Source Type": r.get("source_type", ""),
                "Regulator": r.get("regulator", ""),
                "Publication Type": r.get("publication_type", ""),
                "Regulation ID": r.get("regulation_id", ""),
                "Title": (r.get("title", "") or "")[:160],
                "Publication Date": r.get("publication_date", ""),
                "Confidence": r.get("confidence_score", ""),
                "URL": r.get("source_url", ""),
            } for r in ranked_rows])
            st.dataframe(df, width="stretch", hide_index=True)
            st.download_button(
                "Download sources JSON",
                data=json.dumps(ranked_rows, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{metadata.get('regulation', 'regulation')}_regulatory_sources.json",
                mime="application/json",
            )


def _set_page(target_page: str) -> None:
    """Callback for the Next-button. Runs BEFORE the next script rerun, which
    is the only safe time to mutate ``st.session_state["page"]`` now that the
    sidebar radio is keyed to the same slot.
    """
    st.session_state["page"] = target_page


def _render_next_button(current_page: str, *, disabled: bool = False,
                        help_text: Optional[str] = None) -> None:
    """Render a 'Next: <page>' button at the bottom of a page.

    Uses an ``on_click`` callback to advance ``st.session_state["page"]``;
    direct assignment inside the button's if-block raises ``StreamlitAPIException``
    because the sidebar radio (key=``page``) is instantiated earlier in the run.
    """
    if current_page not in PAGES:
        return
    idx = PAGES.index(current_page)
    if idx >= len(PAGES) - 1:
        return
    next_page = PAGES[idx + 1]
    st.markdown('<div class="next-button"></div>', unsafe_allow_html=True)
    st.divider()
    cols = st.columns([3, 1])
    with cols[1]:
        st.button(
            f"Next → {next_page}",
            type="primary",
            disabled=disabled,
            help=help_text or f"Advance to {next_page}",
            width="stretch",
            key=f"next_btn_{current_page}",
            on_click=_set_page,
            args=(next_page,),
        )


def _restore_assessment_from_db(assessment_id: int) -> bool:
    rec = db.get_assessment(assessment_id)
    if not rec:
        return False
    qrec = db.get_questionnaire(rec["questionnaire_id"])
    if not qrec or not qrec.get("package"):
        return False
    questionnaire = _get_orchestrator().load_questionnaire_package(
        qrec["package"], source="db", name=qrec.get("name"),
    )
    st.session_state["questionnaire"] = questionnaire
    st.session_state["package"] = questionnaire.package
    st.session_state["questionnaire_id"] = qrec["id"]
    st.session_state["assessment_id"] = assessment_id

    state = AssessmentState()
    raw_state = rec.get("state_json")
    if raw_state:
        try:
            data = json.loads(raw_state)
            state.responses = dict(data.get("responses") or {})
            state.dynamic_queue = list(data.get("dynamic_queue") or [])
            state.skipped_ids = set(data.get("skipped_ids") or [])
            state.display_numbers = dict(data.get("display_numbers") or {})
            state.display_counter = int(data.get("display_counter") or 0)
            state.history = list(data.get("history") or [])
        except json.JSONDecodeError:
            pass
    st.session_state["assessment_state"] = state
    if rec.get("evaluation"):
        st.session_state["evaluation"] = rec["evaluation"]
        st.session_state["scoring_result"] = ScoringResult(evaluation=rec["evaluation"])
    if rec.get("recommendations"):
        st.session_state["recommendations"] = rec["recommendations"]
    return True


# ---------------------------------------------------------------------------
# Sidebar — global nav + status
# ---------------------------------------------------------------------------

PAGES = [
    "1. Setup",
    "2. BRD / FRD",
    "3. Questionnaire",
    "4. Assessment",
    "5. Dashboard",
    "6. Export",
]


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Navigation")
        # Bind the radio's state DIRECTLY to session_state["page"] by reusing the
        # same key. Without this, the Next-button (which writes to
        # session_state["page"]) is silently overwritten on the next rerun by the
        # radio widget's own cached value.
        st.radio(
            "Workflow stage",
            PAGES,
            label_visibility="collapsed",
            key="page",
        )
        st.divider()
        st.markdown("### GenAI Shared Service")
        if st.session_state["genai_available"]:
            st.success("Connected", icon=":material/check:")
        else:
            st.warning("Offline / not configured", icon=":material/cloud_off:")
            st.caption("Set `API_KEY` in `.env` to enable GenAI BRD generation. "
                       "All features still work via deterministic fallbacks.")
        probe_msg = st.session_state.get("genai_probe_message")
        if probe_msg:
            with st.expander("Probe diagnostics", expanded=not st.session_state["genai_available"]):
                st.code(probe_msg, language=None)
        if st.button("Re-check GenAI", help="Re-probe the GenAI Shared Service "
                     "after changing .env or rotating the API key."):
            st.session_state["_genai_probed"] = False
            st.session_state.pop("_genai_client", None)
            st.session_state.pop("_orchestrator", None)
            _probe_genai(force_reload_env=True)
            st.rerun()
        st.divider()
        st.markdown("### Agentic workflow")
        analysis: Optional[RegulatoryAnalysis] = st.session_state.get("analysis")
        if analysis:
            st.caption(f"Agent 1 ready - {len(analysis.obligations)} obligations")
        else:
            st.caption("Agent 1: not run")
        rtm: Optional[RTMArtifact] = st.session_state.get("rtm_artifact")
        if rtm:
            st.caption(f"Agent 2 ready - {len(rtm.entries)} RTM rows")
        else:
            st.caption("Agent 2: not run")
        questionnaire: Optional[QuestionnairePackage] = st.session_state.get("questionnaire")
        if questionnaire:
            st.caption(f"Agent 3 ready - {questionnaire.question_count} questions")
        else:
            st.caption("Agent 3: not run")
        st.divider()
        if questionnaire is not None:
            pkg = questionnaire.package
            meta = pkg.get("metadata") or {}
            st.metric("Questionnaire questions", questionnaire.question_count)
            st.metric("Requirements", questionnaire.requirement_count)
            st.metric("Package confidence", f"{meta.get('overall_confidence_pct', 0)}%")
        else:
            st.info("No questionnaire loaded yet.")
        st.divider()
        if st.button("Reset everything", help="Clear all in-memory state. SQLite data is preserved."):
            for k in list(st.session_state.keys()):
                if not k.startswith("_"):
                    del st.session_state[k]
            _init_session_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Page 1 — Setup (Upload Regulation)
# ---------------------------------------------------------------------------

def render_setup_page() -> None:
    st.subheader("1. Setup — Upload Regulation")
    st.write("Upload a regulation document and/or an existing BRD/FRD, then choose how to source requirements. "
             "The Document Parser stage runs automatically when downstream agents need text from these files.")

    col_reg, col_tier = st.columns([2, 1])
    with col_reg:
        st.session_state["regulation"] = st.text_input(
            "Regulation code", st.session_state["regulation"],
            help="Free-form label used in reports and exports (e.g. DORA, MiFID II).",
        )
    with col_tier:
        st.session_state["tier"] = st.selectbox(
            "Tier", ["Tier-1", "Tier-2", "Tier-3"],
            index=["Tier-1", "Tier-2", "Tier-3"].index(st.session_state["tier"]),
        )

    st.markdown("#### Regulation document (optional)")
    reg_file = st.file_uploader(
        "Upload regulation document (PDF or DOCX). Used as additional context for the BRD generator (Agent 1).",
        type=["pdf", "docx"], key="reg_uploader",
    )
    if reg_file is not None:
        saved = save_upload(reg_file, UPLOAD_DIR)
        doc_id = db.save_document(
            name=reg_file.name, kind="regulation", path=str(saved),
            mime=getattr(reg_file, "type", None),
            size_bytes=saved.stat().st_size,
            regulation=st.session_state["regulation"],
        )
        st.session_state["regulation_doc_id"] = doc_id
        st.success(f"Saved regulation document `{reg_file.name}` (id={doc_id}).")

    st.markdown("#### Source of requirements")
    st.session_state["mode"] = st.radio(
        "Mode",
        ["Use existing BRD/FRD", "Generate BRD/FRD from regulation"],
        index=["Use existing BRD/FRD", "Generate BRD/FRD from regulation"].index(st.session_state["mode"]),
        horizontal=True,
    )

    if st.session_state["mode"] == "Use existing BRD/FRD":
        st.markdown("##### Upload BRD/FRD DOCX")
        brd_file = st.file_uploader(
            "Upload BRD/FRD .docx", type=["docx"], key="brd_uploader",
            help="Should follow the standard requirement table layout.",
        )
        col_a, col_b = st.columns([1, 2])
        with col_a:
            use_sample = st.button("Use bundled sample BRD")
        with col_b:
            st.caption(f"Sample: `{(SAMPLE_DIR / 'DORA_Tier2_Detailed_DetailedBRDFRD.docx').name}`")

        target_path: Optional[Path] = None
        if brd_file is not None:
            target_path = save_upload(brd_file, UPLOAD_DIR)
            doc_id = db.save_document(
                name=brd_file.name, kind="brd", path=str(target_path),
                mime=getattr(brd_file, "type", None),
                size_bytes=target_path.stat().st_size,
                regulation=st.session_state["regulation"],
            )
            st.session_state["brd_doc_id"] = doc_id
            st.session_state["brd_source"] = "uploaded"
            st.success(f"Saved BRD `{brd_file.name}` (id={doc_id}).")
        elif use_sample:
            sample = SAMPLE_DIR / "DORA_Tier2_Detailed_DetailedBRDFRD.docx"
            if not sample.exists():
                st.error("Bundled sample BRD is missing. Drop a DOCX into `sample_data/`.")
            else:
                target_path = sample
                doc_id = db.save_document(
                    name=sample.name, kind="brd", path=str(sample),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    size_bytes=sample.stat().st_size,
                    regulation=st.session_state["regulation"],
                )
                st.session_state["brd_doc_id"] = doc_id
                st.session_state["brd_source"] = "sample"
                st.success(f"Loaded sample BRD `{sample.name}`.")

        if target_path is not None:
            st.info("Move to **Page 2 — BRD/FRD** to parse this document and build the RTM.")

    else:
        st.info(
            "BRD will be generated when you open **Page 2 — BRD/FRD** (Agent 1 runs the GenAI pipeline). "
            "If the GenAI Shared Service is unavailable, a deterministic fallback BRD is used."
        )

        _render_regulatory_intelligence_block()

    st.markdown("#### Existing artefacts in this workspace")
    qs = db.list_questionnaires()
    if qs:
        df = pd.DataFrame(qs)[
            ["id", "name", "regulation", "question_count", "requirement_count",
             "overall_confidence_pct", "created_at"]
        ]
        st.dataframe(df, width="stretch", hide_index=True)
        ids = [q["id"] for q in qs]
        labels = [f"#{q['id']} — {q['name']} ({q['question_count']} questions)" for q in qs]
        selection = st.selectbox(
            "Load an existing questionnaire (skips Page 3 generation):",
            options=[None] + ids,
            format_func=lambda v: "(none)" if v is None else labels[ids.index(v)],
        )
        if selection is not None and st.button("Load selected questionnaire"):
            qrec = db.get_questionnaire(int(selection))
            if qrec and qrec.get("package"):
                questionnaire = _get_orchestrator().load_questionnaire_package(
                    qrec["package"], source="db", name=qrec.get("name"),
                )
                st.session_state["questionnaire"] = questionnaire
                st.session_state["package"] = questionnaire.package
                st.session_state["questionnaire_id"] = qrec["id"]
                st.session_state["package_source"] = "db"
                st.session_state["assessment_state"] = AssessmentState()
                st.session_state["assessment_id"] = None
                st.success(f"Loaded questionnaire #{qrec['id']} into session.")
    else:
        st.caption("No questionnaires have been generated yet.")

    setup_ready = bool(
        st.session_state.get("regulation_doc_id")
        or st.session_state.get("brd_doc_id")
        or st.session_state.get("questionnaire")
        or st.session_state["mode"] == "Generate BRD/FRD from regulation"
    )
    _render_next_button(
        "1. Setup",
        disabled=not setup_ready,
        help_text=("Upload a regulation/BRD or choose 'Generate BRD/FRD from regulation' "
                   "to enable the next stage." if not setup_ready else None),
    )


# ---------------------------------------------------------------------------
# Page 2 — BRD / FRD (runs Agents 1 + 2)
# ---------------------------------------------------------------------------

def _run_agent1_and_agent2_with_status() -> None:
    """Run Agent 1 (Regulatory Analysis) + Agent 2 (BRD + RTM) with a live status panel."""
    orch = _get_orchestrator()
    parsed_doc = None
    reg_id = st.session_state.get("regulation_doc_id")
    if reg_id:
        reg = db.get_document(int(reg_id))
        if reg:
            try:
                parsed_doc = orch.parse_document(Path(reg["path"]), kind="regulation")
            except Exception as exc:
                st.warning(f"Could not parse regulation document `{reg['name']}`: {exc}")

    with st.status(
        "Running Agent 1 (Regulatory Analysis) and Agent 2 (BRD + RTM)...",
        expanded=True,
    ) as status:
        def _log(msg: str) -> None:
            status.write(msg)
        try:
            analysis = orch.run_regulatory_analysis(
                parsed_document=parsed_doc,
                regulation=st.session_state["regulation"],
                tier=st.session_state["tier"],
                status=_log,
                regulator_selection=_selected_regulator_codes(),
                consulting_selection=None,
                include_consulting_guidance=False,
                intelligence_package=_fresh_intelligence_package(),
            )
        except Exception as exc:
            status.update(label="Agent 1 failed", state="error")
            st.error(f"Regulatory analysis failed: {exc}")
            return

        st.session_state["analysis"] = analysis
        status.write(f"Agent 1 produced {len(analysis.obligations)} obligations.")

        docx_path = OUTPUT_DIR / timestamped_name(
            f"{st.session_state['regulation']}_BRD_FRD", ".docx"
        )
        try:
            bundle = orch.run_brd_rtm(
                analysis, docx_export_path=docx_path, tier=st.session_state["tier"],
            )
        except Exception as exc:
            status.update(label="Agent 2 failed", state="error")
            st.error(f"BRD/RTM generation failed: {exc}")
            return

        brd_artifact: BRDArtifact = bundle["brd"]
        rtm_artifact: RTMArtifact = bundle["rtm"]
        st.session_state["brd_artifact"] = brd_artifact
        st.session_state["rtm_artifact"] = rtm_artifact
        st.session_state["brd_source"] = brd_artifact.source
        status.write(f"Agent 2 produced {len(rtm_artifact.entries)} RTM entries.")
        status.update(label="Agents 1 + 2 complete", state="complete", expanded=False)


def _run_agent2_for_uploaded_brd() -> None:
    """Parse an uploaded BRD into requirements via the existing parser path."""
    doc_id = st.session_state.get("brd_doc_id")
    if not doc_id:
        st.warning("Upload a BRD/FRD DOCX on Page 1 first.")
        return
    rec = db.get_document(int(doc_id))
    if not rec:
        st.error("BRD document record is missing from the database.")
        return
    path = Path(rec["path"])
    if not path.exists():
        st.error(f"Saved BRD file is missing on disk: {path}")
        return
    from services.questionnaire_generator import (
        derive_impact_pairs,
        read_docx_requirements,
    )
    from utils.docx_parser import DocxSource
    try:
        reqs = read_docx_requirements(DocxSource(path=str(path)))
    except Exception as exc:
        st.error(f"Failed to parse BRD: {exc}")
        return
    pairs = derive_impact_pairs(reqs, st.session_state["regulation"])
    area_lookup: Dict[str, List[str]] = {}
    function_lookup: Dict[str, List[str]] = {}
    for pair in pairs:
        for rid in pair.requirement_ids:
            area_lookup.setdefault(rid, []).append(pair.area)
            function_lookup.setdefault(rid, []).append(pair.function)
    db.save_requirements(
        document_id=int(doc_id),
        requirements=[
            {
                "requirement_id": r.normalized_id,
                "section": r.source_section,
                "description": r.requirement or r.detail,
                "impacted_areas": sorted(set(area_lookup.get(r.normalized_id, []))),
                "impacted_functions": sorted(set(function_lookup.get(r.normalized_id, []))),
            }
            for r in reqs
        ],
    )
    st.success(f"Parsed {len(reqs)} requirements from `{path.name}`.")


def render_brd_page() -> None:
    st.subheader("2. BRD / FRD — Agent 1 (Regulatory Analysis) + Agent 2 (BRD + RTM)")
    mode = st.session_state["mode"]
    col_a, col_b = st.columns(2)

    if mode == "Use existing BRD/FRD":
        with col_a:
            if st.button("Parse uploaded BRD", type="primary",
                         disabled=not st.session_state.get("brd_doc_id"),
                         help="Read requirement tables from the DOCX uploaded on Page 1."):
                _run_agent2_for_uploaded_brd()
        with col_b:
            st.caption("Need to generate from a regulation instead? Switch the mode on Page 1.")
        doc_id = st.session_state.get("brd_doc_id")
        reqs_ready = False
        if doc_id:
            reqs = db.list_requirements(int(doc_id))
            if reqs:
                _render_parsed_requirements(reqs)
                reqs_ready = True
            else:
                st.info("Click **Parse uploaded BRD** to extract requirements.")
        else:
            st.warning("No BRD uploaded yet. Use Page 1 to upload one or load the sample.")
        _render_next_button(
            "2. BRD / FRD",
            disabled=not reqs_ready,
            help_text="Parse the uploaded BRD first." if not reqs_ready else None,
        )
        return

    # Generate-from-regulation mode (runs Agents 1 + 2)
    with col_a:
        if st.button("Run Agents 1 + 2 (Regulatory Analysis -> BRD + RTM)", type="primary"):
            _run_agent1_and_agent2_with_status()
    with col_b:
        st.caption("Agent 1 uses the GenAI Shared Service when available, "
                   "deterministic offline content otherwise. Agent 2 builds the RTM deterministically.")

    analysis: Optional[RegulatoryAnalysis] = st.session_state.get("analysis")
    brd_artifact: Optional[BRDArtifact] = st.session_state.get("brd_artifact")
    rtm_artifact: Optional[RTMArtifact] = st.session_state.get("rtm_artifact")

    if analysis is None or brd_artifact is None:
        st.info("Click **Run Agents 1 + 2** to produce the regulatory analysis, BRD/FRD, and RTM.")
        _render_next_button(
            "2. BRD / FRD",
            disabled=True,
            help_text="Run Agents 1 + 2 first.",
        )
        return

    metadata = brd_artifact.metadata or {}
    cols = st.columns(5)
    cols[0].metric(
        "Completeness coverage",
        f"{metadata.get('completeness_coverage_pct') or '0%'}",
        help="Mean of per-section coverage = min(1.0, actual_items / DORA-tier minimum) "
             "across Process, Data, Reporting, Functional, and Non-Functional sections. "
             "Measures how much of the regulation's expected requirement surface area "
             "the BRD has captured.",
    )
    cols[1].metric(
        "Accuracy coverage",
        f"{metadata.get('accuracy_coverage_pct') or '0%'}",
        help="Mean of per-requirement AI confidence (each clamped to 90%-100%). "
             "Measures how accurately each captured requirement maps to DORA "
             "Regulation (EU) 2022/2554 and relevant RTS/ITS guidance.",
    )
    cols[2].metric("Used GenAI", "Yes" if metadata.get("used_genai_shared_service") else "No")
    section_counts: Dict[str, int] = metadata.get("section_counts") or {}
    total_reqs = (
        section_counts.get("process_requirements", 0)
        + section_counts.get("data_requirements", 0)
        + section_counts.get("reporting_requirements", 0)
        + section_counts.get("functional_requirements", 0)
        + section_counts.get("non_functional_requirements", 0)
    )
    cols[3].metric("Requirements", total_reqs)
    cols[4].metric("Obligations (Agent 1)", len(analysis.obligations))

    # Surface a clear reason whenever GenAI was configured but the run still
    # fell back to the deterministic offline content. Without this, the UI
    # silently shows "Used GenAI: No" even though the sidebar reports
    # "Connected", which is confusing.
    if (
        metadata.get("genai_was_attempted")
        and not metadata.get("used_genai_shared_service")
    ):
        reason = metadata.get("genai_failure_reason") or (
            "The GenAI Shared Service was configured but one of the 8 bundled "
            "BRD generation calls did not succeed. The BRD shown below was built "
            "from the deterministic offline fallback."
        )
        st.warning(
            "**Used GenAI: No** — the GenAI Shared Service was reachable at probe "
            f"time but the BRD generation fell back to offline content.\n\n"
            f"Reason reported by the generator: `{reason}`"
        )

    _render_regulation_source_panel(brd_artifact)
    _render_source_references_panel(brd_artifact)

    # Obligations preview - includes the cited source(s) for each row so
    # reviewers can validate traceability without leaving Page 2.
    st.markdown("#### Obligations (Agent 1 output)")
    obl_rows = [
        {
            "id": o.obligation_id,
            "theme": o.theme,
            "title": (o.title[:100] + "...") if len(o.title) > 100 else o.title,
            "area": o.impacted_area,
            "function": o.impacted_function,
            "priority": o.priority,
            "regulatory_basis": o.regulatory_basis,
            "deadline": o.deadline or "—",
            "sources": _format_sources_inline(list(getattr(o, "source_references", []) or [])),
        }
        for o in analysis.obligations[:50]
    ]
    st.dataframe(pd.DataFrame(obl_rows), width="stretch", hide_index=True)
    if len(analysis.obligations) > 50:
        st.caption(f"Showing first 50 of {len(analysis.obligations)} obligations.")

    # RTM preview
    if rtm_artifact is not None and rtm_artifact.entries:
        st.markdown("#### RTM (Agent 2 output)")
        rtm_rows = [
            {
                "trace_id": e.traceability_id,
                "obligation": e.obligation_id,
                "BR id": e.business_requirement_id,
                "FR id": e.functional_requirement_id or "—",
                "area": e.impacted_area,
                "function": e.impacted_function,
                "evidence_required": e.evidence_required,
                "sources": _format_sources_inline(list(getattr(e, "source_references", []) or [])),
            }
            for e in rtm_artifact.entries[:50]
        ]
        st.dataframe(pd.DataFrame(rtm_rows), width="stretch", hide_index=True)
        if len(rtm_artifact.entries) > 50:
            st.caption(f"Showing first 50 of {len(rtm_artifact.entries)} RTM rows.")

    # BRD requirements table (for parity with the previous UI)
    from services.questionnaire_generator import (
        derive_impact_pairs,
        requirements_from_report,
    )
    report = brd_artifact.report
    if report is not None:
        # Pass the BRD's source-reference map so every flattened
        # requirement carries its citations into the on-screen table.
        flat = requirements_from_report(
            report, metadata.get("source_references_by_item") or {},
        )
        pairs = derive_impact_pairs(flat, st.session_state["regulation"])
        area_lookup: Dict[str, List[str]] = {}
        function_lookup: Dict[str, List[str]] = {}
        for pair in pairs:
            for rid in pair.requirement_ids:
                area_lookup.setdefault(rid, []).append(pair.area)
                function_lookup.setdefault(rid, []).append(pair.function)
        rows = [
            {
                "requirement_id": r.normalized_id,
                "section": r.source_section,
                "description": ((r.requirement or r.detail)[:240]
                                + ("..." if len(r.requirement or r.detail) > 240 else "")),
                "impacted_areas": ", ".join(sorted(set(area_lookup.get(r.normalized_id, [])))),
                "impacted_functions": ", ".join(sorted(set(function_lookup.get(r.normalized_id, [])))),
                "sources": _format_sources_inline(r.source_references or []),
            }
            for r in flat
        ]
        st.markdown("#### Parsed BRD requirements")
        _render_parsed_requirements(rows)

    _render_brd_download_panel(analysis, brd_artifact, rtm_artifact)

    _render_next_button("2. BRD / FRD")


def _render_parsed_requirements(reqs: List[Dict[str, Any]]) -> None:
    if not reqs:
        st.info("No requirements parsed.")
        return
    df = pd.DataFrame(reqs)
    keep_cols = [c for c in ["requirement_id", "section", "description",
                             "impacted_areas", "impacted_functions", "sources"]
                 if c in df.columns]
    st.dataframe(df[keep_cols], width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Page 2 — BRD/FRD export helpers
# ---------------------------------------------------------------------------

def _build_or_get_brd_docx(brd_artifact: BRDArtifact) -> Optional[Path]:
    """Return the on-disk path of the BRD/FRD DOCX, regenerating if needed.

    Forwards the BRD's source-reference map / catalogue so any rebuild keeps
    the per-row "Source References" column and the dedicated traceability
    section.
    """
    if brd_artifact.report is None:
        return None
    existing = brd_artifact.docx_path
    if existing and Path(existing).exists():
        return Path(existing)
    target = OUTPUT_DIR / timestamped_name(
        f"{st.session_state['regulation']}_BRD_FRD", ".docx"
    )
    metadata = brd_artifact.metadata or {}
    try:
        path = write_brd_docx(
            brd_artifact.report,
            str(target),
            tier=st.session_state["tier"],
            source_references_by_item=metadata.get("source_references_by_item"),
            source_catalogue=metadata.get("source_references_catalogue"),
        )
        brd_artifact.docx_path = path
        return Path(path)
    except Exception as exc:
        st.error(f"Could not (re)build BRD/FRD DOCX: {exc}")
        return None


def _requirements_csv(
    report: DoraDetailedBRD,
    source_refs_by_item: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> bytes:
    """Flatten all requirement tables into a single CSV blob.

    When ``source_refs_by_item`` is supplied (the BRD metadata produced by
    :func:`services.brd_frd_generator.build_brd_frd_report`) two extra columns
    are emitted -- ``source_references`` (human-readable summary) and
    ``source_urls`` (newline-separated URLs) -- so the CSV export is
    self-contained.
    """
    from services.questionnaire_generator import requirements_from_report
    reqs = requirements_from_report(report, source_refs_by_item or {})
    df = pd.DataFrame([{
        "requirement_id": r.normalized_id,
        "source_id": r.source_id,
        "section": r.source_section,
        "category": r.category,
        "requirement": r.requirement,
        "detail": r.detail,
        "alignment": r.alignment,
        "priority": r.priority,
        "acceptance": r.acceptance,
        "confidence": r.confidence,
        "source_references": " | ".join(
            _format_source_label(ref) for ref in (r.source_references or [])
        ) or "No live source available",
        "source_urls": "\n".join(
            ref.get("source_url", "") for ref in (r.source_references or [])
            if ref.get("source_url")
        ),
    } for r in reqs])
    return df.to_csv(index=False).encode("utf-8")


def _rtm_csv(rtm: RTMArtifact) -> bytes:
    df = pd.DataFrame([asdict(e) for e in rtm.entries])
    return df.to_csv(index=False).encode("utf-8")


def _render_brd_download_panel(
    analysis: RegulatoryAnalysis,
    brd_artifact: BRDArtifact,
    rtm_artifact: Optional[RTMArtifact],
) -> None:
    """Multi-format download surface for the BRD/FRD + agentic artefacts."""
    if brd_artifact.report is None:
        return

    st.markdown("#### Downloads")
    regulation = st.session_state["regulation"]
    tier = st.session_state["tier"]
    stem_base = f"{regulation}_{tier}".replace(" ", "_")

    col1, col2 = st.columns(2)

    # --- Column 1: combined BRD/FRD DOCX + full structured JSON ---
    with col1:
        st.markdown("**Combined BRD + FRD (DOCX)**")
        docx_path = _build_or_get_brd_docx(brd_artifact)
        if docx_path and docx_path.exists():
            with open(docx_path, "rb") as fh:
                st.download_button(
                    "Download BRD + FRD (DOCX)",
                    data=fh.read(),
                    file_name=docx_path.name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    width="stretch",
                )

        st.markdown("**Structured report (JSON)**")
        try:
            report_json = brd_artifact.report.model_dump_json(indent=2).encode("utf-8")
            st.download_button(
                "Download BRD + FRD (JSON)",
                data=report_json,
                file_name=f"{stem_base}_BRD_FRD_report.json",
                mime="application/json",
                width="stretch",
            )
        except Exception as exc:
            st.warning(f"JSON dump failed: {exc}")

    # --- Column 2: requirements CSV, obligations + RTM ---
    with col2:
        st.markdown("**Requirements (CSV)**")
        try:
            csv_bytes = _requirements_csv(
                brd_artifact.report,
                (brd_artifact.metadata or {}).get("source_references_by_item"),
            )
            st.download_button(
                "Download requirements CSV",
                data=csv_bytes,
                file_name=f"{stem_base}_requirements.csv",
                mime="text/csv",
                width="stretch",
            )
        except Exception as exc:
            st.warning(f"CSV export failed: {exc}")

        st.markdown("**Obligations (JSON)**")
        obligations_payload = [asdict(o) if is_dataclass(o) else dict(o)
                               for o in analysis.obligations]
        st.download_button(
            "Download obligations JSON",
            data=json.dumps(obligations_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{stem_base}_obligations.json",
            mime="application/json",
            width="stretch",
        )

        st.markdown("**RTM (JSON / CSV)**")
        if rtm_artifact is not None and rtm_artifact.entries:
            rtm_payload = [asdict(e) for e in rtm_artifact.entries]
            st.download_button(
                "Download RTM JSON",
                data=json.dumps(rtm_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{stem_base}_RTM.json",
                mime="application/json",
                width="stretch",
            )
            st.download_button(
                "Download RTM CSV",
                data=_rtm_csv(rtm_artifact),
                file_name=f"{stem_base}_RTM.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.caption("RTM not available — re-run Agents 1 + 2.")


# ---------------------------------------------------------------------------
# Page 3 — Questionnaire (Agent 3)
# ---------------------------------------------------------------------------

def render_questionnaire_page() -> None:
    st.subheader("3. Questionnaire — Agent 3 (Questionnaire Generation)")

    col_gen, col_load = st.columns(2)
    with col_gen:
        st.markdown("#### Generate a new questionnaire")
        if st.button("Run Agent 3", type="primary"):
            _run_agent3()
    with col_load:
        st.markdown("#### Load from saved package JSON")
        uploaded = st.file_uploader("Upload questionnaire JSON", type=["json"], key="pkg_uploader")
        if uploaded is not None and st.button("Load uploaded JSON"):
            try:
                content = json.loads(uploaded.read().decode("utf-8"))
                errors = validate_package_schema(content)
                if errors:
                    st.error("Package JSON failed validation:")
                    for e in errors:
                        st.write(f"- {e}")
                else:
                    questionnaire = _get_orchestrator().load_questionnaire_package(
                        content, source="uploaded_json", name=uploaded.name,
                    )
                    st.session_state["questionnaire"] = questionnaire
                    st.session_state["package"] = questionnaire.package
                    st.session_state["package_source"] = "uploaded_json"
                    st.session_state["assessment_state"] = AssessmentState()
                    st.session_state["assessment_id"] = None
                    qid = db.save_questionnaire(
                        name=uploaded.name, package=content,
                        regulation=st.session_state["regulation"],
                    )
                    st.session_state["questionnaire_id"] = qid
                    st.success(f"Loaded `{uploaded.name}` and saved as questionnaire #{qid}.")
            except Exception as exc:
                st.error(f"Could not parse JSON: {exc}")

    questionnaire: Optional[QuestionnairePackage] = st.session_state.get("questionnaire")
    if questionnaire is None:
        st.info("Run Agent 3 or load a saved package to continue.")
        _render_next_button(
            "3. Questionnaire",
            disabled=True,
            help_text="Build or load a questionnaire first.",
        )
        return

    pkg = questionnaire.package
    meta = pkg.get("metadata") or {}
    questions = list(pkg.get("questions") or [])
    closed = [q for q in questions if not q.get("is_free_text")]
    free_text = [q for q in questions if q.get("is_free_text")]
    requirements = list(pkg.get("requirements") or [])

    # Headline "Coverage (closed)" = fraction of BRD requirements that have at
    # least one closed/mapped question. Newer packages emit ``coverage_pct``
    # explicitly; older packages saved before that alias was added still carry
    # the same value under ``requirement_coverage_pct``, so read both.
    coverage_pct = (
        meta.get("coverage_pct")
        if meta.get("coverage_pct") is not None
        else meta.get("requirement_coverage_pct", 0)
    )

    cols = st.columns(5)
    cols[0].metric("Requirements", len(requirements))
    cols[1].metric("Closed questions", len(closed))
    cols[2].metric("Free-text questions", len(free_text))
    cols[3].metric("Coverage (closed)", f"{coverage_pct}%")
    cols[4].metric("Overall confidence", f"{meta.get('overall_confidence_pct', 0)}%")

    st.markdown("#### Question preview")
    preview_rows = [
        {
            "id": q["question_id"],
            "area": q.get("area"),
            "function": q.get("function"),
            "type": q.get("question_type"),
            "question": q.get("question"),
            "confidence": q.get("confidence"),
        }
        for q in questions[:25]
    ]
    st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)
    if len(questions) > 25:
        st.caption(f"Showing first 25 of {len(questions)} questions.")

    _render_next_button("3. Questionnaire")


def _run_agent3() -> None:
    mode = st.session_state["mode"]
    regulation = st.session_state["regulation"]
    orch = _get_orchestrator()
    with st.spinner("Building questionnaire package..."):
        try:
            if mode == "Generate BRD/FRD from regulation":
                brd_artifact: Optional[BRDArtifact] = st.session_state.get("brd_artifact")
                if brd_artifact is None or brd_artifact.report is None:
                    st.error("Run Agents 1 + 2 on Page 2 before building the questionnaire.")
                    return
                questionnaire = orch.run_questionnaire_from_report(
                    brd_artifact, regulation=regulation,
                )
                source = "generated_brd"
                name = questionnaire.name
            else:
                doc_id = st.session_state.get("brd_doc_id")
                if not doc_id:
                    st.error("Upload a BRD on Page 1 before building the questionnaire.")
                    return
                rec = db.get_document(int(doc_id))
                if not rec:
                    st.error("Saved BRD record is missing from the database.")
                    return
                questionnaire = orch.run_questionnaire_from_docx(
                    Path(rec["path"]), regulation=regulation,
                    name=f"{regulation} — from {Path(rec['name']).stem}",
                )
                source = "uploaded_brd"
                name = questionnaire.name
        except Exception as exc:
            st.error(f"Questionnaire build failed: {exc}")
            return
    st.session_state["questionnaire"] = questionnaire
    st.session_state["package"] = questionnaire.package
    st.session_state["package_source"] = source
    st.session_state["assessment_state"] = AssessmentState()
    st.session_state["assessment_id"] = None
    qid = db.save_questionnaire(
        name=name, package=questionnaire.package,
        document_id=st.session_state.get("brd_doc_id"),
        regulation=regulation,
    )
    questionnaire.questionnaire_id = qid
    st.session_state["questionnaire_id"] = qid
    st.success(f"Agent 3 built {questionnaire.question_count} questions and saved as questionnaire #{qid}.")


# ---------------------------------------------------------------------------
# Page 4 — Assessment (User Responses)
# ---------------------------------------------------------------------------

def render_assessment_page() -> None:
    st.subheader("4. Assessment — User Responses")
    questionnaire: Optional[QuestionnairePackage] = st.session_state.get("questionnaire")
    if questionnaire is None:
        st.warning("No questionnaire loaded. Go to Page 3 to run Agent 3 or load one.")
        _render_next_button(
            "4. Assessment",
            disabled=True,
            help_text="Load a questionnaire first.",
        )
        return
    pkg = questionnaire.package

    state: AssessmentState = st.session_state["assessment_state"]
    base_questions = list(pkg.get("questions") or [])
    base_closed = [q for q in base_questions if not q.get("is_free_text")]
    areas = sorted({q.get("area") for q in base_closed if q.get("area")})

    bar_a, bar_b, bar_c, bar_d = st.columns([2, 1, 1, 1])
    with bar_a:
        focus = st.selectbox(
            "Focus area (hard filter)", ["All"] + areas,
            index=(["All"] + areas).index(st.session_state.get("focus_area", "All"))
            if st.session_state.get("focus_area", "All") in (["All"] + areas) else 0,
            key="focus_select",
        )
        st.session_state["focus_area"] = focus
    with bar_b:
        if st.button("Start / continue", type="primary", help="Ensure an assessment session exists."):
            if st.session_state.get("assessment_id") is None:
                aid = db.create_assessment(
                    questionnaire_id=int(st.session_state["questionnaire_id"]),
                    name=f"Assessment {time.strftime('%Y-%m-%d %H:%M:%S')}",
                )
                st.session_state["assessment_id"] = aid
            st.rerun()
    with bar_c:
        if st.button("Restart", help="Clear answers but keep the questionnaire."):
            state.reset_responses()
            _persist_assessment_snapshot()
            st.rerun()
    with bar_d:
        if st.button("New session", help="Create a fresh assessment row in SQLite."):
            state.reset_responses()
            aid = db.create_assessment(
                questionnaire_id=int(st.session_state["questionnaire_id"]),
                name=f"Assessment {time.strftime('%Y-%m-%d %H:%M:%S')}",
            )
            st.session_state["assessment_id"] = aid
            st.rerun()

    if st.session_state.get("assessment_id") is None:
        st.info("Click **Start / continue** to begin (this creates a row in SQLite).")
        return

    active = applicable_base_questions(state, base_questions) + list(state.dynamic_queue)
    applicable_count = len([q for q in active if not q.get("is_free_text")])
    answered_count = sum(1 for q in active if answered(q, state.responses))
    cols = st.columns(5)
    cols[0].metric("Answered / Applicable", f"{answered_count} / {applicable_count}")
    cols[1].metric("Dynamic follow-ups pending",
                   len([q for q in state.dynamic_queue if not answered(q, state.responses)]))
    cols[2].metric("Skipped by funnel", len(state.skipped_ids))
    cols[3].metric("Branch decisions", len(state.branch_log))
    cols[4].metric("Assessment ID", st.session_state["assessment_id"])

    if state.branch_log:
        with st.expander(f"Adaptive branch trace ({len(state.branch_log)} decisions)"):
            trace_rows = []
            for entry in state.branch_log[-15:]:
                trace_rows.append({
                    "Parent": entry.get("parent_question_id", ""),
                    "Answer": ", ".join(entry.get("selected_answer", [])) if isinstance(entry.get("selected_answer"), list) else entry.get("selected_answer", ""),
                    "Source": entry.get("branch_source", ""),
                    "Rule": entry.get("branch_rule_id", ""),
                    "Theme": entry.get("theme", ""),
                    "Children": ", ".join(entry.get("child_question_ids", [])),
                })
            st.dataframe(pd.DataFrame(trace_rows), width="stretch", hide_index=True)

    next_q = choose_next_question(state, base_questions, focus_area=focus)
    if next_q is None:
        st.success(
            f"All currently applicable closed-ended questions for "
            f"{'all areas' if focus == 'All' else focus} are complete. "
            f"Move to **Page 5 — Dashboard** to view scores."
        )
        _persist_assessment_snapshot(completed=True)
        _render_free_text_questions(base_questions, state)
        _render_next_button("4. Assessment")
        return

    _render_question_card(next_q, base_questions, state)
    _render_free_text_questions(base_questions, state)
    _persist_assessment_snapshot()

    has_any_answer = bool(state.responses)
    _render_next_button(
        "4. Assessment",
        disabled=not has_any_answer,
        help_text=("Answer at least one question to view scores."
                   if not has_any_answer else "Jump to the live dashboard."),
    )


def _render_question_card(question: Dict[str, Any], base_questions: List[Dict[str, Any]],
                          state: AssessmentState) -> None:
    """Render the v13 minimal question card.

    Default view shows only:
      - lightweight progress chip (Question N / Total)
      - the question text
      - the answer widget (radio / multiselect)
      - a primary submit button
      - a single, collapsible 'Why am I being asked this?' affordance

    All rich metadata (regulation, regulator, article, obligation, BRD/RTM
    IDs, business function, control objective, reason, expected evidence,
    risk if unanswered negatively) is rendered inside the expander so the
    default screen stays clean.
    """
    qid = question["question_id"]
    display_no = state.assign_display_number(qid)
    active = applicable_base_questions(state, base_questions) + list(state.dynamic_queue)
    total = max(display_no, len([q for q in active if not q.get("is_free_text")]))

    chip = f"Question {display_no} of ~{total}"
    if question.get("dynamic"):
        chip += " · Adaptive follow-up"

    st.markdown('<div class="exec-card">', unsafe_allow_html=True)
    st.caption(chip)
    st.markdown(f"### {question['question']}")
    raw_options = question.get("options") or ["Unknown"]
    labels = option_labels(raw_options) or ["Unknown"]
    with st.form(f"form_{qid}_{len(state.history)}"):
        if question.get("question_type") == "Multi Select":
            value: Any = st.multiselect("Select all that apply", labels, default=[])
        else:
            default_idx = len(labels) - 1 if "Unknown" in labels else 0
            value = st.radio(
                "Choose one",
                labels,
                index=default_idx,
                horizontal=False,
                label_visibility="collapsed",
            )
        comments = st.text_area(
            "Comments or evidence reference (optional)",
            "",
            placeholder="Add SME notes, evidence links or context if relevant…",
        )
        submitted = st.form_submit_button("Submit", type="primary", width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    _render_explainability_panel(question, state)

    if submitted:
        state.responses[qid] = value
        state.responses[f"{qid}__display_sequence"] = state.display_numbers.get(qid)
        if comments:
            state.responses[f"{qid}__comments"] = comments
        package_regulation = st.session_state.get("regulation")
        update_applicability_after_response(
            state, question, value, base_questions,
            package_regulation=package_regulation,
        )
        state.history.append(qid)
        st.rerun()


def _render_explainability_panel(question: Dict[str, Any], state: AssessmentState) -> None:
    """Render the collapsible 'Why am I being asked this?' panel.

    Reads the structured ``explainability`` bundle attached to each Question
    by the v13 generator. Falls back to the free-form ``rationale_text``
    when the bundle is missing (legacy v10/v11 packages).
    """
    explain = question.get("explainability") or {}
    with st.expander("Why am I being asked this?", expanded=False):
        if not explain:
            st.write(rationale_text(question, state.responses))
            return

        # One-line summary banner
        summary_parts = []
        if explain.get("regulation"):
            summary_parts.append(f"**{explain['regulation']}**")
        if explain.get("article"):
            summary_parts.append(explain["article"])
        if explain.get("control_objective"):
            summary_parts.append(explain["control_objective"])
        if summary_parts:
            st.markdown(" · ".join(summary_parts))
            st.markdown("---")

        rows = [
            ("Regulator", explain.get("regulator")),
            ("Article / clause", explain.get("article")),
            ("Obligation ID", explain.get("obligation_id")),
            ("Business function", explain.get("business_function")),
            ("Business area", explain.get("business_area")),
            ("Control objective", explain.get("control_objective")),
        ]
        list_rows = [
            ("BRD requirement IDs", explain.get("brd_requirement_ids") or []),
            ("RTM trace IDs", explain.get("rtm_trace_ids") or []),
        ]
        long_rows = [
            ("Why this question exists", explain.get("reason")),
            ("Expected evidence", explain.get("expected_evidence")),
            ("Risk if answered negatively", explain.get("risk_if_negative")),
        ]

        col_a, col_b = st.columns(2)
        for idx, (key, value) in enumerate(rows):
            target = col_a if idx % 2 == 0 else col_b
            with target:
                if value:
                    st.markdown(f"**{key}**  \n{value}")

        for key, items in list_rows:
            if items:
                st.markdown(f"**{key}**: {', '.join(items)}")

        for key, value in long_rows:
            if value:
                st.markdown(f"**{key}**")
                st.write(value)

        # Source-reference traceability surfaced for every question. When the
        # anchor BRD requirement carries no live source we flag the gap so the
        # user knows the citation chain is incomplete.
        source_refs = explain.get("source_references") or []
        st.markdown("**Source references**")
        if not source_refs:
            st.caption(
                "[!] No live regulatory publication was matched to the BRD "
                "requirement that produced this question. Validate the wording "
                "against the official regulation text before relying on it."
            )
        else:
            for ref in source_refs:
                label = _format_source_label(ref)
                url = ref.get("source_url") or ""
                if url:
                    st.markdown(f"- {label}  \n  [{url}]({url})")
                else:
                    st.markdown(f"- {label}")

        if question.get("dynamic"):
            rule = question.get("branch_rule_id", "")
            triggers = ", ".join(question.get("trigger_answers", [])) or "the prior response"
            st.caption(
                f"This is an adaptive follow-up — triggered by **{triggers}** on the previous "
                f"question. Branch rule: `{rule or 'generic'}`."
            )


def _render_free_text_questions(base_questions: List[Dict[str, Any]], state: AssessmentState) -> None:
    free_text = [q for q in base_questions if q.get("is_free_text")]
    if not free_text:
        return
    with st.expander(f"Free-text questions ({len(free_text)})"):
        for q in free_text:
            key = q["question_id"]
            text_val = st.text_area(
                f"{key} | {q['question']}",
                value=state.responses.get(key, ""), key=f"text_{key}",
            )
            if text_val:
                state.responses[key] = text_val
            elif key in state.responses:
                state.responses.pop(key, None)


# ---------------------------------------------------------------------------
# Page 5 — Dashboard (Python Rules Engine + Agent 4)
# ---------------------------------------------------------------------------

def render_dashboard_page() -> None:
    st.subheader("5. Dashboard — Python Rules Engine + Agent 4 (Recommendations)")
    questionnaire: Optional[QuestionnairePackage] = st.session_state.get("questionnaire")
    if questionnaire is None:
        st.warning("No questionnaire loaded.")
        _render_next_button("5. Dashboard", disabled=True,
                            help_text="Load a questionnaire first.")
        return
    scoring = _refresh_scoring_snapshot()
    if scoring is None:
        st.warning("No evaluation available yet.")
        _render_next_button("5. Dashboard", disabled=True,
                            help_text="Answer some questions first.")
        return

    result = scoring.evaluation
    score = result["compliance_score_pct"]
    eval_conf = result["evaluation_confidence_pct"]
    cols = st.columns(4)
    cols[0].metric("Readiness / Compliance score", f"{score}%")
    cols[1].metric("Evaluation confidence", f"{eval_conf}%")
    cols[2].metric("Answered closed questions",
                   f"{result['answered_count']} / "
                   f"{result['answered_count'] + result['unanswered_count']}")
    cols[3].metric("Pairs scored", len(result["pair_scores"]))

    filt = st.radio(
        "Filter", ["All", "Critical", "At risk", "Watch", "Ready"],
        index=["All", "Critical", "At risk", "Watch", "Ready"].index(
            st.session_state.get("dashboard_filter", "All")),
        horizontal=True, key="dash_filter_radio",
    )
    st.session_state["dashboard_filter"] = filt

    def _apply_filter(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or filt == "All" or "CXO status" not in df.columns:
            return df
        return df[df["CXO status"] == filt]

    area_df = _apply_filter(summary_dataframe(result["area_summary"], "Impacted Area"))
    func_df = _apply_filter(summary_dataframe(result["function_summary"], "Function"))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Aggregate by impacted area")
        if area_df.empty:
            st.info("No area scores match the current filter.")
        else:
            st.dataframe(_df_with_styling(area_df, ["Compliance %"]),
                         width="stretch", hide_index=True)
    with right:
        st.markdown("#### Aggregate by function")
        if func_df.empty:
            st.info("No function scores match the current filter.")
        else:
            st.dataframe(_df_with_styling(func_df, ["Compliance %"]),
                         width="stretch", hide_index=True)

    with st.expander("Detailed impacted area × function heatmap", expanded=True):
        pair_df = pair_heatmap_rows(result["pair_scores"])
        if pair_df.empty:
            st.info("No area×function scores yet.")
        else:
            score_cols = [c for c in pair_df.columns if c != "Impacted Area"]
            st.dataframe(_df_with_styling(pair_df, score_cols),
                         width="stretch", hide_index=True)

    st.markdown("#### Top gaps (lowest-scoring requirements)")
    if scoring.top_gaps:
        gap_df = pd.DataFrame([
            {"Requirement": g["requirement_id"], "Compliance %": g["compliance_pct"]}
            for g in scoring.top_gaps
        ])
        st.dataframe(_df_with_styling(gap_df, ["Compliance %"]),
                     width="stretch", hide_index=True)
    else:
        st.caption("No requirement scores yet — answer more closed questions.")

    st.markdown("#### Agent 4 — Recommendations")
    rcol_a, rcol_b, rcol_c = st.columns([1, 1, 2])
    with rcol_a:
        min_sev = st.selectbox(
            "Minimum severity",
            ["Critical", "At risk", "Watch", "Ready"],
            index=2,
        )
    with rcol_b:
        top_n = st.number_input("Top requirements", 1, 30, 10, 1)
    with rcol_c:
        run_genai = st.checkbox(
            "Use GenAI to refine action wording",
            value=False,
            disabled=not st.session_state["genai_available"],
            help="Disabled when the GenAI Shared Service is unavailable.",
        )

    if st.button("Run Agent 4 (Generate recommendations)", type="primary"):
        rec_state: AssessmentState = st.session_state["assessment_state"]
        recommendation_result = _get_orchestrator().run_recommendations(
            questionnaire,
            scoring,
            min_severity=min_sev,
            top_n_requirements=int(top_n),
            enrich_with_genai=bool(run_genai),
            branch_log=list(rec_state.branch_log),
        )
        if recommendation_result.used_genai:
            st.toast("Recommendations enriched via GenAI.")
        st.session_state["recommendations"] = recommendation_result.recommendations
        _persist_assessment_snapshot()

    recs = st.session_state.get("recommendations") or []
    if recs:
        rec_df = pd.DataFrame([
            {
                "id": getattr(r, "recommendation_id", None) or r.get("recommendation_id"),
                "severity": getattr(r, "severity", None) or r.get("severity"),
                "title": getattr(r, "title", None) or r.get("title"),
                "compliance %": getattr(r, "compliance_pct", None) or r.get("compliance_pct"),
                "owner": getattr(r, "suggested_owner", None) or r.get("suggested_owner"),
                "horizon": getattr(r, "horizon", None) or r.get("horizon"),
                "action": getattr(r, "suggested_action", None) or r.get("suggested_action"),
            }
            for r in recs
        ])
        st.dataframe(rec_df, width="stretch", hide_index=True)
    else:
        st.caption("Click **Run Agent 4** to produce the action list.")

    _render_next_button("5. Dashboard")


# ---------------------------------------------------------------------------
# Page 6 — Export
# ---------------------------------------------------------------------------

def render_export_page() -> None:
    st.subheader("6. Export")
    questionnaire: Optional[QuestionnairePackage] = st.session_state.get("questionnaire")
    if questionnaire is None:
        st.warning("Generate or load a questionnaire first.")
        return
    pkg = questionnaire.package

    state: AssessmentState = st.session_state["assessment_state"]
    scoring = _refresh_scoring_snapshot()
    eval_result = scoring.evaluation if scoring else None
    recs = st.session_state.get("recommendations") or []

    st.markdown("#### Downloads")
    cols = st.columns(2)

    with cols[0]:
        st.markdown("**Questionnaire package (JSON)**")
        st.download_button(
            "Download questionnaire JSON",
            data=json.dumps(pkg, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{st.session_state['regulation']}_questionnaire_package.json",
            mime="application/json",
            width="stretch",
        )

        st.markdown("**Responses & live results (JSON)**")
        responses_payload = {
            "regulation": st.session_state["regulation"],
            "tier": st.session_state["tier"],
            "questionnaire_id": st.session_state.get("questionnaire_id"),
            "assessment_id": st.session_state.get("assessment_id"),
            "responses": state.responses,
            "skipped_by_funnel": sorted(state.skipped_ids),
            "history": list(state.history),
            "branch_log": list(state.branch_log),
            "dynamic_queue": list(state.dynamic_queue),
            "dynamic_questions_emitted": state.dynamic_questions_emitted,
            "evaluation": _jsonable_eval(eval_result) if eval_result else None,
            "recommendations": [_rec_to_dict(r) for r in recs],
        }
        st.download_button(
            "Download responses JSON",
            data=json.dumps(responses_payload, ensure_ascii=False, indent=2,
                             default=str).encode("utf-8"),
            file_name=f"{st.session_state['regulation']}_responses.json",
            mime="application/json",
            width="stretch",
        )

        # Optional: Obligations + RTM exports
        analysis: Optional[RegulatoryAnalysis] = st.session_state.get("analysis")
        if analysis is not None:
            obligations_payload = [asdict(o) if is_dataclass(o) else dict(o)
                                   for o in analysis.obligations]
            st.markdown("**Obligations (Agent 1)**")
            st.download_button(
                "Download obligations JSON",
                data=json.dumps(obligations_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{st.session_state['regulation']}_obligations.json",
                mime="application/json",
                width="stretch",
            )

        rtm_artifact: Optional[RTMArtifact] = st.session_state.get("rtm_artifact")
        if rtm_artifact is not None and rtm_artifact.entries:
            rtm_payload = [asdict(e) if is_dataclass(e) else dict(e) for e in rtm_artifact.entries]
            st.markdown("**Requirements Traceability Matrix (Agent 2)**")
            st.download_button(
                "Download RTM JSON",
                data=json.dumps(rtm_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{st.session_state['regulation']}_RTM.json",
                mime="application/json",
                width="stretch",
            )

    with cols[1]:
        st.markdown("**Excel report (questionnaire + responses + scores)**")
        if st.button("Build Excel and prepare download", type="primary"):
            target = OUTPUT_DIR / timestamped_name(
                f"{st.session_state['regulation']}_Readiness_Report", ".xlsx"
            )
            try:
                write_excel_from_package(str(target), pkg)
                st.session_state["_excel_export_path"] = str(target)
                st.success(f"Wrote `{target.name}`.")
            except Exception as exc:
                st.error(f"Excel export failed: {exc}")

        excel_path = st.session_state.get("_excel_export_path")
        if excel_path and Path(excel_path).exists():
            with open(excel_path, "rb") as fh:
                st.download_button(
                    "Download Excel report",
                    data=fh.read(),
                    file_name=Path(excel_path).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )

        st.markdown("**Generated BRD/FRD (DOCX)**")
        brd_artifact: Optional[BRDArtifact] = st.session_state.get("brd_artifact")
        docx_path = brd_artifact.docx_path if brd_artifact else None
        if docx_path and Path(docx_path).exists():
            with open(docx_path, "rb") as fh:
                st.download_button(
                    "Download generated BRD/FRD DOCX",
                    data=fh.read(),
                    file_name=Path(docx_path).name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    width="stretch",
                )
        else:
            st.caption("Only available when Agents 1 + 2 were run on Page 2.")


def _jsonable_eval(result: Dict[str, Any]) -> Dict[str, Any]:
    """Pair-score dict has tuple keys — JSON can't represent those."""
    out = dict(result)
    pair_scores = out.get("pair_scores") or {}
    out["pair_scores"] = {f"{a} | {f}": s for (a, f), s in pair_scores.items()}
    out["skipped_by_funnel"] = sorted(out.get("skipped_by_funnel", []) or [])
    return out


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def main() -> None:
    _render_sidebar()
    page = st.session_state["page"]
    if page == "1. Setup":
        render_setup_page()
    elif page == "2. BRD / FRD":
        render_brd_page()
    elif page == "3. Questionnaire":
        render_questionnaire_page()
    elif page == "4. Assessment":
        render_assessment_page()
    elif page == "5. Dashboard":
        render_dashboard_page()
    elif page == "6. Export":
        render_export_page()
    else:
        st.warning(f"Unknown page: {page}")


main()
