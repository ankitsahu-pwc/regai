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

Five pages are available from the sidebar (Setup / Generate BRD-FRD /
Questionnaire / Dashboard / Export). Page 3's *Calculate Impact & Readiness*
button now routes users straight to the Dashboard - the previous standalone
Assessment page has been retired. Each page calls orchestrator methods
instead of reaching into individual services.

The app is robust to:

* GenAI Shared Service being unreachable (offline fallback BRD).
* Missing uploads (clear inline messages, no crashes).
* Re-runs in the middle of an assessment (state is restored from SQLite).
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import html
import json
import logging
import math
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components
from dotenv import load_dotenv

from services.logging_config import setup_logging

_LOG_FILE = setup_logging()
logger = logging.getLogger(__name__)
logger.info("Reg AI RAP starting up. Log file at %s", _LOG_FILE)

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
from services.ai_questionnaire_generator import (
    AIAnswerRefinement,
    evaluate_answer_and_generate_followup,
)
from services.genai_service import GenAIClient
from services.brd_frd_generator import (
    DoraDetailedBRD,
    write_brd_docx,
)
from services.client_profile import (
    CLIENT_PROFILE_FIELDS,
    CLIENT_PROFILE_KEYS,
    ClientProfileField,
    default_client_profile,
    empty_client_profile,
    is_client_profile_populated,
    normalize_client_profile,
)
from services.client_roles import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    APPLICABILITY_PARTIAL,
    APPLICABILITY_UNCERTAIN,
    INSTITUTION_TYPES,
    INSTITUTION_TYPE_NAMES,
    get_institution_type,
    normalize_client_roles,
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
    option_metadata,
    write_excel_from_package,
)
from services.recommendation_service import Recommendation
from services.scoring_engine import (
    AssessmentState,
    answered,
    evaluate as _scoring_evaluate,
    pair_heatmap_rows,
    score_free_text_answer,
    score_value,
    summary_dataframe,
)
from services.severity import (
    css_class as _severity_css_class,
    from_label as _severity_from_label,
    impact_band as _severity_impact_band,
    impact_label as _severity_impact_label,
    readiness_band as _severity_readiness_band,
)
from services.gap_analysis import GapItem, GapReport, build_gap_report
from services.readiness_score import (
    DORA_AREA_WEIGHTS,
    WeightedReadinessResult,
    compute_weighted_readiness,
    demo_result as _readiness_demo_result,
)
from services.impact_score import (
    DORA_IMPACT_FACTOR_WEIGHTS,
    WeightedImpactResult,
    compute_weighted_impact,
    priority_score as _priority_score,
    demo_result as _impact_demo_result,
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
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "regai_logo.png"

ensure_dirs(UPLOAD_DIR, OUTPUT_DIR, SAMPLE_DIR, DATA_DIR)
db.init_db()


def _load_logo_data_uri(path: Path = LOGO_PATH) -> str:
    """Return the RegAI RAP logo as a base64 data URI.

    Embedding the logo inline keeps the hero HTML self-contained -- no static
    file server or ``st.image`` call is needed and the same markup renders on
    every page. Returns an empty string if the asset is missing so the hero
    still renders (title-only) instead of raising.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "image/svg+xml" if suffix == "svg" else f"image/{suffix}"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


_LOGO_DATA_URI = _load_logo_data_uri()

st.set_page_config(
    page_title="Reg AI RAP – A Complete Regulatory Impact Assessment & Readiness Platform",
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
/* Trim the default Streamlit top padding so the hero banner sits close to
   the toolbar instead of leaving ~6rem of empty space above it. Covers the
   modern ``data-testid`` selector plus the legacy ``.block-container``
   fallback so the rule holds across Streamlit versions. */
.stApp [data-testid="stMainBlockContainer"],
.stApp [data-testid="stAppViewContainer"] > .main > .block-container,
.stApp .main .block-container,
.stApp .block-container {
    padding-top: 1.25rem !important;
}
/* The header strip Streamlit reserves for the deploy / kebab menu is
   transparent; keep it that way so the gradient shows through, but drop
   its default height so it doesn't force extra space above content. */
.stApp [data-testid="stHeader"] {
    background: transparent;
    height: 2.25rem;
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
.stFormSubmitButton button[kind="primary"],
.stButton [data-testid="stBaseButton-primary"],
.stDownloadButton [data-testid="stBaseButton-primary"],
.stFormSubmitButton [data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"] {
    color: #ffffff !important;
    background: #d04a02 !important;
    background-color: #d04a02 !important;
    background-image: none !important;
    border: 1px solid #b03d00 !important;
    box-shadow: 0 2px 8px rgba(208, 74, 2, 0.28) !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
.stButton button[kind="primary"] *, .stDownloadButton button[kind="primary"] *,
.stFormSubmitButton button[kind="primary"] *,
button[data-testid="stBaseButton-primary"] * {
    color: #ffffff !important;
}
.stButton button[kind="primary"]:hover,
.stDownloadButton button[kind="primary"]:hover,
.stFormSubmitButton button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background: #b03d00 !important;
    background-color: #b03d00 !important;
    border-color: #8f3100 !important;
}
.stButton button:hover, .stDownloadButton button:hover {
    border-color: #b03d00 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {color: #1a1a1a !important;}

/* DataFrames + metrics */
[data-testid="stDataFrame"] * {color: #1a1a1a !important;}

/* Bold, Title-Case dataframe column headers with a solid dark outer
   border around every table. Wrapper (.rap-table-wrap) keeps horizontal
   scroll OUTSIDE the table so the bar never overlaps text. */
.rap-table-wrap {
    border: 2px solid #1a1a1a;
    border-radius: 8px;
    /* No bottom padding — the last body row sits flush against the
       outer border. Any horizontal scrollbar that appears will render
       inside the scroll gutter (see ``scrollbar-gutter``) rather than
       forcing an empty band beneath the last row. */
    padding: 0;
    background: #ffffff;
    margin: 0.35rem 0 0.9rem;
    overflow-x: auto;
    overflow-y: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    scrollbar-gutter: stable;
}
/* Streamlit renders each ``st.markdown('<div class="rap-table-wrap">')``
   call in its own DOM block, so the wrapper div often doesn't actually
   contain the dataframe. To guarantee every ``st.dataframe`` still gets
   a strong dark outer border + rounded corners we apply the same
   treatment directly to the ``stDataFrame`` container. */
.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stDataFrameResizable"] {
    border: 2px solid #1a1a1a !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    overflow: hidden !important;
}
.rap-table-wrap [data-testid="stDataFrame"],
.rap-table-wrap [data-testid="stDataFrameResizable"] {
    /* When the wrapper does contain the dataframe (custom HTML fallback
       paths), suppress the double border. */
    border: none !important;
    box-shadow: none !important;
}
/* Header row — Streamlit exposes the header via multiple selectors
   depending on the Glide Data Grid version. We cover them all so the
   header always reads bold with a darker off-white band and a strong
   bottom divider. */
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrame"] [data-testid="stDataFrameHeaderCell"],
[data-testid="stDataFrame"] thead th,
[data-testid="stDataFrame"] .row-header,
[data-testid="stDataFrame"] .header-cell {
    font-weight: 800 !important;
    text-transform: capitalize;
    background: #e8d9c6 !important;
    border-bottom: 2px solid #1a1a1a !important;
    border-right: 1px solid #1a1a1a !important;
    color: #1a1a1a !important;
    letter-spacing: 0.3px;
    font-size: 14px !important;
}
[data-testid="stDataFrame"] [role="columnheader"] *,
[data-testid="stDataFrame"] [data-testid="stDataFrameHeaderCell"] *,
[data-testid="stDataFrame"] thead th *,
[data-testid="stDataFrame"] .row-header *,
[data-testid="stDataFrame"] .header-cell * {
    font-weight: 800 !important;
    color: #1a1a1a !important;
}
/* Vertical column separators + row lines between data cells rendered
   as darker, higher-contrast strokes so columns read as distinct
   swim-lanes at a glance. */
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] tbody td {
    border-right: 1px solid #8a7a6c !important;
    border-bottom: 1px solid #b7a597 !important;
}
[data-testid="stDataFrame"] [role="row"] [role="gridcell"]:last-child,
[data-testid="stDataFrame"] [role="row"] [role="columnheader"]:last-child,
[data-testid="stDataFrame"] tbody tr td:last-child,
[data-testid="stDataFrame"] thead tr th:last-child {
    border-right: none !important;
}

/* Custom HTML table used by the Parsed BRD Requirements renderer so the
   Sources column can carry per-cell hyperlinks. Visual language matches
   the sibling st.dataframe tables (bold Title-Case headers, black outer
   border via .rap-table-wrap, soft off-white header band). */
/* Wrapper variant specifically used by the Parsed BRD custom HTML table,
   so it gets its own vertical scroll bar (kept INSIDE the border, right at
   the table's right edge, matching Streamlit's native dataframe scroll). */
.rap-table-wrap.rap-table-scroll {
    max-height: 380px;
    overflow-y: auto;
    overflow-x: auto;
    /* No bottom padding here either — the wrapper is already handling
       its own scroll gutter, so any horizontal scrollbar renders at
       the very bottom of the scroll area without leaving an empty
       strip beneath the last row. */
    padding-bottom: 0;
    /* ``stable both-edges`` reserved scrollbar-width space on BOTH
       sides of the wrapper, which pushed the first column ~10px to
       the right of the outer border. Reserve only on the trailing
       edge (right in LTR layouts) so the leading column aligns flush
       with the outer border. */
    scrollbar-gutter: stable;
}
/* ------------------------------------------------------------------ */
/* Unified table scrollbar system — every scrollable table wrapper in  */
/* the app shares the Regulatory Obligations pattern:                  */
/*    - slim 10px scrollbar with #bfae9a thumb on a #f4ece2 track      */
/*    - scrollbar-gutter reserves space so the bar never overlaps text */
/*    - scrollbar-width: thin (Firefox) + scrollbar-color fallback     */
/* Applies to:                                                         */
/*    .rap-table-wrap                (Regulatory Obligations et al.)   */
/*    .rap-table-wrap.rap-table-scroll (Parsed BRD requirements)       */
/*    .reg-src-table-wrap             (Regulator Sources)              */
/*    .dash-qtable-wrap               (Question-Level Scoring Detail)  */
/*    [data-testid="stDataFrame"]     (any bare st.dataframe)          */
/* ------------------------------------------------------------------ */
.rap-table-wrap::-webkit-scrollbar,
.rap-table-wrap.rap-table-scroll::-webkit-scrollbar,
.reg-src-table-wrap::-webkit-scrollbar,
.dash-qtable-wrap::-webkit-scrollbar,
[data-testid="stDataFrame"] ::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
.rap-table-wrap::-webkit-scrollbar-track,
.rap-table-wrap.rap-table-scroll::-webkit-scrollbar-track,
.reg-src-table-wrap::-webkit-scrollbar-track,
.dash-qtable-wrap::-webkit-scrollbar-track,
[data-testid="stDataFrame"] ::-webkit-scrollbar-track {
    background: #f4ece2;
    border-radius: 8px;
}
.rap-table-wrap::-webkit-scrollbar-thumb,
.rap-table-wrap.rap-table-scroll::-webkit-scrollbar-thumb,
.reg-src-table-wrap::-webkit-scrollbar-thumb,
.dash-qtable-wrap::-webkit-scrollbar-thumb,
[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {
    background: #bfae9a;
    border-radius: 8px;
    border: 2px solid #f4ece2;
}
.rap-table-wrap::-webkit-scrollbar-thumb:hover,
.rap-table-wrap.rap-table-scroll::-webkit-scrollbar-thumb:hover,
.reg-src-table-wrap::-webkit-scrollbar-thumb:hover,
.dash-qtable-wrap::-webkit-scrollbar-thumb:hover,
[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb:hover {
    background: #a0895f;
}
.rap-table-wrap,
.rap-table-wrap.rap-table-scroll,
.reg-src-table-wrap,
.dash-qtable-wrap {
    scrollbar-width: thin;
    scrollbar-color: #bfae9a #f4ece2;
}
/* Non-scrolling wrapper variants (used by tables that fit vertically
   without ``rap-table-scroll``) should still keep their first column
   flush against the outer border — the base rule above sets
   ``overflow-x: auto`` on ``.rap-table-wrap`` which is what enables
   the same ``scrollbar-gutter`` reservation. */
.rap-table-wrap {
    scrollbar-gutter: stable;
}
.rap-table-wrap table.rap-html-table {
    width: 100%;
    /* Use ``separate`` so every cell owns its own border independently.
       With ``border-collapse: collapse`` the sticky header ``<th>``'s
       ``border-bottom`` collapses into the first ``<tr>``'s
       ``border-top`` — as soon as the user scrolls the first row up,
       the shared border goes with it and the sticky header loses its
       divider. ``separate`` + ``border-spacing: 0`` gives the same
       visual look while keeping the header's bottom border anchored
       to the header. */
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.88rem;
    color: #1a1a1a;
    background: #ffffff;
    /* ``<table>`` is baseline-aligned by default, so the parent block's
       line-height reserves a couple of pixels of descender space
       *below* the table — visible as a thin strip between the last
       row and the outer wrapper border. ``vertical-align: top`` +
       ``margin: 0`` collapse that ghost strip without breaking the
       sticky header (which requires ``display: table`` to remain
       intact). */
    vertical-align: top;
    margin: 0;
}
/* Kill the descender-line-height reservation on the wrapper so no
   phantom baseline space can accumulate beneath the table. The
   ``<td>``/``<th>`` cells restore their own text line-height
   explicitly, so cell text still reads normally. */
.rap-table-wrap {
    line-height: 0;
}
.rap-table-wrap table.rap-html-table thead,
.rap-table-wrap table.rap-html-table tbody,
.rap-table-wrap table.rap-html-table tr,
.rap-table-wrap table.rap-html-table th,
.rap-table-wrap table.rap-html-table td {
    line-height: 1.35;
}
.rap-table-wrap table.rap-html-table thead th.rap-th {
    background: #f0e6da;
    color: #1a1a1a;
    font-weight: 800;
    text-transform: capitalize;
    /* Header divider matches the outer wrapper border weight (2px
       solid #1a1a1a) — anything thicker reads as an outlier next to
       the surrounding table borders. With ``border-collapse: separate``
       above, this border stays anchored to the sticky header during
       scroll on its own, so we no longer paint an additional
       ``box-shadow`` underneath (which was stacking a second 2px black
       stripe onto the divider and making it read ~4px thick). */
    border-bottom: 2px solid #1a1a1a;
    border-right: 1px solid #1a1a1a;
    padding: 0.6rem 0.75rem;
    text-align: left;
    letter-spacing: 0.25px;
    position: sticky;
    top: 0;
    /* Sit clearly above scrolling body rows so nothing bleeds through
       the header band during rapid scroll. */
    z-index: 5;
}
.rap-table-wrap table.rap-html-table thead th.rap-th:last-child {
    border-right: none;
}
.rap-table-wrap table.rap-html-table tbody td.rap-td {
    padding: 0.5rem 0.75rem;
    /* Darker vertical + horizontal grid lines so every column and
       row reads as a distinct swim-lane at a glance. */
    border-bottom: 1px solid #8a7a6c;
    border-right: 1px solid #8a7a6c;
    vertical-align: top;
    line-height: 1.35;
}
.rap-table-wrap table.rap-html-table tbody td.rap-td:last-child {
    border-right: none;
}
.rap-table-wrap table.rap-html-table tbody tr:last-child td.rap-td {
    border-bottom: none;
}
.rap-table-wrap table.rap-html-table tbody tr:hover td.rap-td {
    background: #fdf6f0;
}
.rap-table-wrap table.rap-html-table td.rap-td-id {
    /* ID cells keep their compact single-line layout but render in the
       same regular weight as every other body cell — the ID pattern
       itself (e.g. ``REQ-042``, ``TR-0007``) already reads as a stable
       identifier without needing bold. */
    color: #2d2d2d;
    white-space: nowrap;
}
.rap-table-wrap table.rap-html-table td.rap-td-desc {
    max-width: 480px;
}
.rap-table-wrap table.rap-html-table td.rap-td-src {
    min-width: 220px;
}
.rap-table-wrap table.rap-html-table a.rap-src-link {
    color: #d04a02;
    text-decoration: none;
    font-weight: 600;
}
.rap-table-wrap table.rap-html-table a.rap-src-link:hover {
    text-decoration: underline;
}
.rap-table-wrap table.rap-html-table .rap-src-plain {
    color: #4a4a4a;
}
.rap-table-wrap table.rap-html-table .rap-src-plain.rap-src-none {
    color: #8a8a8a;
    font-style: italic;
}
.rap-table-wrap table.rap-html-table .rap-src-sep {
    color: #b6b6b6;
    margin: 0 2px;
}
.rap-table-wrap table.rap-html-table .rap-src-more {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 8px;
    border-radius: 999px;
    background: #ead8cc;
    color: #6a3300;
    font-size: 0.75rem;
    font-weight: 700;
}

/* Compact section headings used to visually highlight the
   Per-requirement Traceability table (matches Source References style). */
.rap-section-hd {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin: 0.55rem 0 0.35rem;
}
.rap-section-hd-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #2d2d2d;
    letter-spacing: 0.1px;
}
.rap-section-hd-badge {
    font-size: 0.72rem;
    font-weight: 700;
    color: #ffffff;
    background: #d04a02;
    padding: 2px 9px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* Lighter, tighter page 2: reduce vertical rhythm between blocks. */
.rap-tight-hdr h4, .rap-tight-hdr h3 {
    margin-top: 0.4rem !important;
    margin-bottom: 0.35rem !important;
}

/* ------------------------------------------------------------------ */
/* Rules-Engine dashboard (Page 5) — palette + card / heatmap system.  */
/* Symmetric four-band ladder (aligned across the app):                */
/*   Red   #ffb3b3 tile / #e52528 accent = Critical (readiness < 25%)   */
/*   Amber #f2b91b tile                    = At risk  (readiness 25-50%)*/
/*   Green #a8e6a8 tile                    = Watch    (readiness 50-75%)*/
/*   Green #b7e4c0 tile / #14572d accent = Ready    (readiness >= 75%)  */
/* Tile *body* text is always #111 for readability; only the dark-grey  */
/* group-heading strip above uses white text.                          */
/* ------------------------------------------------------------------ */
:root {
    --dash-red: #e52528;
    --dash-amber: #f2b91b;
    --dash-peach: #fde7d6;
    --dash-green: #2e7d32;
    --dash-blue: #0e4b73;
    --dash-orange: #d04a02;
    --dash-border: #d9dee4;
    --dash-panel: #ffffff;
    --dash-bg: #f7f8fa;
    --dash-ink: #1c1c1c;
    --dash-muted: #6c757d;
}

/* Section legend strip. */
.dash-legend {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    padding: 8px 12px;
    background: #ffffff;
    border: 1px solid var(--dash-border);
    border-radius: 10px;
    margin: 0.35rem 0 0.6rem;
    font-size: 0.85rem;
}
.dash-legend b { color: var(--dash-ink); margin-right: 6px; }
.dash-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.2px;
}
.dash-pill.crit  {
    background: linear-gradient(135deg, #ff5b5f 0%, #b00020 100%);
    color: #ffffff;
    box-shadow: 0 2px 6px rgba(176,0,32,0.25);
}
.dash-pill.risk  {
    background: linear-gradient(135deg, #ffd166 0%, #f2a900 100%);
    color: #3a2500;
    box-shadow: 0 2px 6px rgba(242,169,0,0.25);
}
.dash-pill.watch {
    background: linear-gradient(135deg, #a8e6a8 0%, #6ec06e 100%);
    color: #0f3d0f;
    box-shadow: 0 2px 6px rgba(110,192,110,0.25);
}
.dash-pill.ready {
    background: linear-gradient(135deg, #2e7d32 0%, #14572d 100%);
    color: #ffffff;
    box-shadow: 0 2px 6px rgba(20,87,45,0.32);
}

/* ------------------------------------------------------------------ */
/* Live "severity distribution" strip — replaces the flat static legend.
   Each card shows: gradient icon dot, severity name + score band,
   live count of areas/questions in that band, and a segmented progress
   bar giving the % share out of the total items scored.               */
/* ------------------------------------------------------------------ */
.sev-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(160px, 1fr));
    gap: 10px;
    margin: 0.4rem 0 0.75rem;
}
.sev-card {
    position: relative;
    background: #ffffff;
    border: 1px solid var(--dash-border);
    border-left: 5px solid var(--dash-border);
    border-radius: 12px;
    padding: 0.55rem 0.7rem 0.6rem 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.sev-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}
.sev-card .sev-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}
.sev-card .sev-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 800;
    font-size: 0.86rem;
    color: var(--dash-ink);
    letter-spacing: 0.2px;
}
.sev-card .sev-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.85), 0 0 8px rgba(0,0,0,0.18);
    animation: sev-dot-pulse 2.2s ease-in-out infinite;
}
@keyframes sev-dot-pulse {
    0%   { transform: scale(1); opacity: 1; }
    50%  { transform: scale(1.18); opacity: 0.85; }
    100% { transform: scale(1); opacity: 1; }
}
.sev-card .sev-count {
    font-weight: 800;
    font-size: 1.05rem;
    color: var(--dash-ink);
    line-height: 1;
}
.sev-card .sev-range {
    font-size: 0.72rem;
    color: var(--dash-muted);
    margin-top: 2px;
    font-weight: 600;
    letter-spacing: 0.2px;
}
.sev-card .sev-bar {
    position: relative;
    height: 8px;
    background: #eef1f5;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 8px;
}
.sev-card .sev-bar > span {
    display: block;
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease-out;
}
.sev-card .sev-share {
    font-size: 0.72rem;
    color: var(--dash-muted);
    margin-top: 3px;
    font-weight: 600;
}
/* Severity-specific colour accents */
.sev-card.crit  { border-left-color: #b00020; background: linear-gradient(180deg, #fff5f6 0%, #ffffff 60%); }
.sev-card.crit  .sev-dot { background: linear-gradient(135deg, #ff5b5f 0%, #b00020 100%); }
.sev-card.crit  .sev-bar > span { background: linear-gradient(90deg, #ff5b5f 0%, #b00020 100%); }
.sev-card.risk  { border-left-color: #f2a900; background: linear-gradient(180deg, #fff9ec 0%, #ffffff 60%); }
.sev-card.risk  .sev-dot { background: linear-gradient(135deg, #ffd166 0%, #f2a900 100%); }
.sev-card.risk  .sev-bar > span { background: linear-gradient(90deg, #ffd166 0%, #f2a900 100%); }
.sev-card.watch { border-left-color: #6ec06e; background: linear-gradient(180deg, #f2fbf3 0%, #ffffff 60%); }
.sev-card.watch .sev-dot { background: linear-gradient(135deg, #a8e6a8 0%, #6ec06e 100%); }
.sev-card.watch .sev-bar > span { background: linear-gradient(90deg, #a8e6a8 0%, #6ec06e 100%); }
.sev-card.ready { border-left-color: #14572d; background: linear-gradient(180deg, #e6f5ea 0%, #ffffff 60%); }
.sev-card.ready .sev-dot { background: linear-gradient(135deg, #2e7d32 0%, #14572d 100%); }
.sev-card.ready .sev-bar > span { background: linear-gradient(90deg, #2e7d32 0%, #14572d 100%); }

/* Optional caption shown above the severity strip */
.sev-caption {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
    margin-top: 0.35rem;
}
.sev-caption .sev-caption-title {
    font-weight: 800;
    color: var(--dash-ink);
    letter-spacing: 0.2px;
    font-size: 0.9rem;
}
.sev-caption .sev-caption-hint {
    font-size: 0.75rem;
    color: var(--dash-muted);
}
.sev-caption .sev-caption-live {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #2e7d32;
    background: rgba(46,125,50,0.08);
    border: 1px solid rgba(46,125,50,0.28);
    padding: 2px 8px;
    border-radius: 999px;
}
.sev-caption .sev-caption-live::before {
    content: "";
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #2e7d32;
    box-shadow: 0 0 0 3px rgba(46,125,50,0.22);
    animation: sev-dot-pulse 1.6s ease-in-out infinite;
}
@media (max-width: 900px) {
    .sev-strip { grid-template-columns: repeat(2, minmax(150px, 1fr)); }
}

/* KPI tiles with an inline mini progress bar. */
.dash-kpis {
    display: grid;
    grid-template-columns: repeat(5, minmax(150px, 1fr));
    gap: 10px;
    margin: 0.25rem 0 0.6rem;
}
.dash-kpi {
    background: #ffffff;
    border: 1px solid var(--dash-border);
    border-left: 6px solid var(--dash-orange);
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.dash-kpi-label {
    font-size: 0.72rem;
    color: var(--dash-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
}
.dash-kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #2d2d2d;
    margin-top: 4px;
    line-height: 1.1;
}
.dash-kpi-bar {
    height: 8px;
    background: #eef0f3;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 8px;
}
.dash-kpi-bar > span {
    display: block;
    height: 100%;
    background: var(--dash-blue);
    transition: width 0.25s ease;
}
.dash-kpi-bar.crit  > span { background: var(--dash-red); }
.dash-kpi-bar.risk  > span { background: var(--dash-amber); }
.dash-kpi-bar.watch > span { background: #c47e00; }
.dash-kpi-bar.ready > span { background: var(--dash-green); }

/* Hero row - two-tile overall Impact + Readiness banner at the top of the
   dashboard. Bigger typography than the standard KPI tiles so the numbers
   read as executive summary. */
.dash-hero {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 14px;
    margin: 0.25rem 0 0.75rem;
}
.dash-hero-tile {
    background: #ffffff;
    border: 1px solid var(--dash-border);
    border-left: 8px solid var(--dash-blue);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.dash-hero-tile.crit  { border-left-color: var(--dash-red);   background: #fff7f7; }
.dash-hero-tile.risk  { border-left-color: var(--dash-amber); background: #fffaf0; }
.dash-hero-tile.watch { border-left-color: #6ec06e;           background: #f2fbf3; }
.dash-hero-tile.ready { border-left-color: #14572d;           background: #e6f5ea; }
.dash-hero-cap {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem;
    color: var(--dash-muted);
    font-weight: 700;
}
.dash-hero-value {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    color: #1a1a1a;
    letter-spacing: -0.01em;
}
.dash-hero-sub {
    font-size: 0.82rem;
    color: #333333;
}
.dash-help-hint {
    display: inline-block;
    margin-left: 4px;
    font-size: 0.72rem;
    color: #7a3d00;
    cursor: help;
    font-weight: 700;
    line-height: 1;
    opacity: 0.85;
}
.dash-help-hint:hover { opacity: 1; }
.dash-hero-sub[title], .dash-kpi[title] { cursor: help; }
.dash-hero-bar {
    height: 10px;
    background: #eef0f3;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 4px;
}
.dash-hero-bar > span {
    display: block;
    height: 100%;
    background: var(--dash-blue);
}
.dash-hero-bar.crit  > span { background: var(--dash-red); }
.dash-hero-bar.risk  > span { background: var(--dash-amber); }
.dash-hero-bar.watch > span { background: #6ec06e; }
.dash-hero-bar.ready > span { background: #14572d; }

/* AI-driven impact & readiness intelligence panels */
.dash-impact-header,
.dash-readiness-header {
    background: linear-gradient(135deg, #f2f5fb, #eaeff8);
    border: 1px solid var(--dash-border);
    border-radius: 10px;
    padding: 12px 18px;
    display: grid;
    grid-template-columns: auto auto auto 1fr;
    gap: 16px;
    align-items: center;
    margin: 6px 0 4px;
}
.dash-impact-cap {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem;
    color: var(--dash-muted);
    font-weight: 700;
}
.dash-impact-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #1a1a1a;
}
.dash-impact-sub .dash-pill { font-size: 0.82rem; }
.dash-impact-src {
    justify-self: end;
    font-size: 0.72rem;
    color: var(--dash-muted);
    font-style: italic;
}
.impact-int-grid,
.readiness-int-grid {
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}
.impact-int-grid .dash-card,
.readiness-int-grid .dash-card {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.impact-int-grid .dash-card-body ul,
.readiness-int-grid .dash-card-body ul {
    margin: 4px 0 6px 18px;
    padding: 0;
}
.impact-int-grid .dash-card-body li,
.readiness-int-grid .dash-card-body li {
    font-size: 0.82rem;
    line-height: 1.35;
    color: #2d2d2d;
}

/* Rich recommendation cards (What/Why/How/Priority/Outcome/Deps) */
.dash-rich-rec-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 14px;
    margin: 0.25rem 0 0.75rem;
}
.dash-rich-rec-card {
    background: #ffffff;
    border: 1px solid var(--dash-border);
    border-left: 6px solid var(--dash-orange);
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.dash-rich-rec-card.crit  { border-left-color: var(--dash-red);   background: #fff7f7; }
.dash-rich-rec-card.risk  { border-left-color: var(--dash-amber); background: #fffaf0; }
.dash-rich-rec-card.watch { border-left-color: #6ec06e;           background: #f2fbf3; }
.dash-rich-rec-card.ready { border-left-color: #14572d;           background: #e6f5ea; }
.dash-rich-rec-header {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.dash-rich-rec-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #1f1f1f;
    margin-right: auto;
}
.dash-rich-rec-body { margin-top: 6px; }
.dash-rich-rec-section {
    background: rgba(255,255,255,0.55);
    border-left: 3px solid rgba(0,0,0,0.08);
    padding: 6px 12px;
    margin: 4px 0;
    border-radius: 6px;
    font-size: 0.88rem;
    line-height: 1.4;
    color: #2d2d2d;
}
.dash-rich-rec-section b { color: #1a1a1a; }
.dash-rich-rec-meta {
    font-size: 0.78rem;
    color: var(--dash-muted);
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 6px;
}
.dash-rich-rec-list {
    margin: 4px 0 4px 18px;
    padding: 0;
    font-size: 0.85rem;
}
.dash-rich-rec-list li { margin: 2px 0; }
.dash-rich-rec-badges {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    font-size: 0.78rem;
}

/* Area recommendation cards - one card per impacted area, each holding
   a header (name + severity pills + score ribbon), an executive action
   line, and a bulleted playbook (3-4 bullets). */
.dash-rec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 14px;
    margin: 0.25rem 0 0.75rem;
}
.dash-rec-card {
    background: #ffffff;
    border: 1px solid var(--dash-border);
    border-left: 6px solid var(--dash-orange);
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.dash-rec-card.crit  { border-left-color: var(--dash-red);   background: #fff7f7; }
.dash-rec-card.risk  { border-left-color: var(--dash-amber); background: #fffaf0; }
.dash-rec-card.watch { border-left-color: #6ec06e;           background: #f2fbf3; }
.dash-rec-card.ready { border-left-color: #14572d;           background: #e6f5ea; }
.dash-rec-hdr {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.dash-rec-title {
    font-size: 1.02rem;
    font-weight: 800;
    color: #2d2d2d;
    line-height: 1.2;
}
.dash-rec-tags {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    color: var(--dash-muted);
}
.dash-rec-scores { margin-left: auto; }
.dash-rec-scores b { color: #2d2d2d; }
.dash-rec-exec {
    font-size: 0.85rem;
    color: #1f1f1f;
    background: rgba(255,255,255,0.6);
    border-left: 3px solid var(--dash-orange);
    padding: 6px 10px;
    border-radius: 4px;
}
.dash-rec-bullets {
    margin: 0;
    padding-left: 1.15rem;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.dash-rec-bullets li {
    font-size: 0.85rem;
    color: #333333;
    line-height: 1.4;
}
.dash-rec-bullets li b { color: #1a1a1a; }

/* Card grid (used by aggregate-by-area / by-function / top gaps). */
.dash-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
    margin: 0.25rem 0 0.75rem;
}
.dash-card {
    background: #ffffff;
    border: 1px solid var(--dash-border);
    border-left: 6px solid var(--dash-orange);
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.dash-card.crit  { border-left-color: var(--dash-red);   background: #fff7f7; }
.dash-card.risk  { border-left-color: var(--dash-amber); background: #fffaf0; }
.dash-card.watch { border-left-color: #6ec06e;           background: #f2fbf3; }
.dash-card.ready { border-left-color: #14572d;           background: #e6f5ea; }
.dash-card-title {
    font-size: 0.95rem;
    font-weight: 800;
    color: #2d2d2d;
    line-height: 1.2;
}
.dash-card-meta {
    font-size: 0.78rem;
    color: var(--dash-muted);
    line-height: 1.35;
}
.dash-card-meta b { color: #2d2d2d; }
.dash-card-body {
    font-size: 0.82rem;
    color: #333333;
    line-height: 1.35;
}
.dash-card-bar {
    height: 8px;
    background: #eef0f3;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 2px;
}
.dash-card-bar > span {
    display: block;
    height: 100%;
    background: var(--dash-blue);
}
.dash-card-bar.crit  > span { background: var(--dash-red); }
.dash-card-bar.risk  > span { background: var(--dash-amber); }
.dash-card-bar.watch > span { background: #6ec06e; }
.dash-card-bar.ready > span { background: #14572d; }

/* Grouped tile heatmap (Area × Function). Each group is one area,
   containing a tile grid of functions coloured by pair score. */
.dash-heatmap {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 12px;
    margin: 0.25rem 0 0.75rem;
}
.dash-heatgroup {
    background: #ffffff;
    border: 1px solid var(--dash-border);
    border-radius: 10px;
    overflow: hidden;
}
.dash-heatgroup-title {
    background: #2d2d2d;
    color: #ffffff !important;
    text-align: left;
    font-weight: 800;
    padding: 10px 14px;
    font-size: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    letter-spacing: 0.2px;
}
/* Streamlit's global stMarkdownContainer span rule (color: #1a1a1a) wins by
   specificity, so we scope the white override to the same container. */
.dash-heatgroup-title,
.dash-heatgroup-title *,
[data-testid="stMarkdownContainer"] .dash-heatgroup-title,
[data-testid="stMarkdownContainer"] .dash-heatgroup-title *,
[data-testid="stMarkdownContainer"] .dash-heatgroup-title span,
[data-testid="stMarkdownContainer"] .dash-heatgroup-title strong {
    color: #ffffff !important;
}
.dash-heatgroup-title .dash-heatgroup-avg {
    font-size: 0.82rem;
    font-weight: 700;
    color: #ffffff !important;
    opacity: 1;
    background: rgba(255,255,255,0.12);
    padding: 3px 10px;
    border-radius: 999px;
}
.dash-heat-tiles {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 2px;
    padding: 2px;
}
.dash-heat-tile {
    min-height: 68px;
    padding: 8px 6px;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: #111111;
    line-height: 1.15;
    border: 1px solid #ffffff;
}
/* Tile backgrounds keep their four-band colour, but the tile *body* text
   (function name + Impact / Readiness lines) is always dark so it reads
   as data. Only the group heading strip above uses white on dark grey. */
.dash-heat-tile.crit  { background: #ffb3b3;           color: #111111 !important; }
.dash-heat-tile.risk  { background: var(--dash-amber); color: #111111 !important; }
.dash-heat-tile.watch { background: #a8e6a8;           color: #111111 !important; }
.dash-heat-tile.ready { background: #b7e4c0;           color: #111111 !important; }
.dash-heat-tile.none  { background: #eef0f3;           color: #6a6a6a !important; }
/* High-specificity overrides so Streamlit's stMarkdownContainer span rule
   (color: #1a1a1a !important) doesn't leak white into these tiles. */
.dash-heat-tile.crit,
.dash-heat-tile.crit  *,
.dash-heat-tile.risk,
.dash-heat-tile.risk  *,
.dash-heat-tile.watch,
.dash-heat-tile.watch *,
.dash-heat-tile.ready,
.dash-heat-tile.ready *,
[data-testid="stMarkdownContainer"] .dash-heat-tile.crit,
[data-testid="stMarkdownContainer"] .dash-heat-tile.crit *,
[data-testid="stMarkdownContainer"] .dash-heat-tile.risk,
[data-testid="stMarkdownContainer"] .dash-heat-tile.risk *,
[data-testid="stMarkdownContainer"] .dash-heat-tile.watch,
[data-testid="stMarkdownContainer"] .dash-heat-tile.watch *,
[data-testid="stMarkdownContainer"] .dash-heat-tile.ready,
[data-testid="stMarkdownContainer"] .dash-heat-tile.ready * {
    color: #111111 !important;
    text-shadow: none !important;
}
[data-testid="stMarkdownContainer"] .dash-heat-tile.none,
[data-testid="stMarkdownContainer"] .dash-heat-tile.none * {
    color: #6a6a6a !important;
}
.dash-heat-cap   { font-size: 0.80rem; font-weight: 800; }
.dash-heat-score { font-size: 0.74rem; font-weight: 700; margin-top: 4px; }

/* Question-Level Scoring Detail — dense reference-style table.
   Wrapper follows the Regulatory Obligations pattern (.rap-table-wrap)
   so its scrollbars, border and shadow match every other table in the
   app; visual scrollbar styling is defined once in the unified block
   above so this rule only sets layout + sizing. */
.dash-qtable-wrap {
    max-height: 480px;
    overflow: auto;
    border: 2px solid #1a1a1a;
    border-radius: 8px;
    background: #ffffff;
    margin: 0.35rem 0 0.9rem;
    padding-bottom: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    scrollbar-gutter: stable both-edges;
}
.dash-qtable {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    background: #ffffff;
}
.dash-qtable thead th {
    position: sticky;
    top: 0;
    background: var(--dash-blue);
    color: #ffffff;
    text-align: left;
    padding: 8px 12px;
    font-weight: 800;
    text-transform: capitalize;
    letter-spacing: 0.2px;
    z-index: 1;
    border-bottom: 1.5px solid #0b3a5a;
}
.dash-qtable thead th:first-child { padding-left: 14px; }
.dash-qtable thead th:last-child  { padding-right: 14px; }
.dash-qtable tbody td {
    border-top: 1px solid var(--dash-border);
    padding: 7px 12px;
    vertical-align: top;
    color: #1a1a1a;
}
.dash-qtable tbody td:first-child { padding-left: 14px; }
.dash-qtable tbody td:last-child  { padding-right: 14px; }
.dash-qtable tbody tr:nth-child(even) td { background: #fbfbfb; }
.dash-qtable tbody tr:hover td { background: #fdf6f0; }

/* ------------------------------------------------------------------ */
/* Questionnaire preview cards (Page 3) — matches the reference        */
/* assessment tool: one card per question with tag pill, bold heading, */
/* and a compact answer-option list.                                   */
/* ------------------------------------------------------------------ */
.qprev-group-hdr {
    margin: 0.85rem 0 0.35rem;
    padding: 8px 12px;
    background: #eef2f7;
    border-radius: 8px;
    border-left: 4px solid var(--dash-blue);
    color: #13293d;
    font-size: 0.98rem;
    font-weight: 800;
    letter-spacing: 0.1px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.qprev-group-hdr .qprev-group-count {
    background: var(--dash-blue);
    color: #ffffff;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
/* Questionnaire summary tiles — replaces ``st.metric`` for the row of
   four count tiles at the top of Page 3 so we can attach a native
   ``title`` attribute for hover-tooltips. Streamlit's ``st.metric``
   only exposes tooltips via a small ``?`` icon which is easy to miss
   when the label ("Closed Questions (Quantitative)") is truncated. */
.qgen-metric-tile {
    background: #ffffff;
    border: 1px solid #ead8cc;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    cursor: default;
    min-height: 78px;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}
.qgen-metric-tile:hover {
    border-color: #d04a02;
    box-shadow: 0 2px 8px rgba(208,74,2,0.15);
}
.qgen-metric-label {
    font-size: 0.85rem;
    color: #4a4a4a;
    font-weight: 600;
    letter-spacing: 0.1px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.qgen-metric-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #d04a02;
    line-height: 1.1;
    letter-spacing: -0.5px;
}
/* Top-level section headers used by the flattened Quantitative /
   Qualitative buckets on the Questionnaire page. */
.q-section-hdr {
    margin: 1.4rem 0 0.5rem;
    padding: 12px 16px;
    background: linear-gradient(90deg, #d04a02 0%, #a53400 100%);
    color: #ffffff;
    border-radius: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12);
}
.q-section-hdr.q-section-hdr-alt {
    background: linear-gradient(90deg, #2d5a87 0%, #1c3d5c 100%);
}
.q-section-hdr .q-section-title {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 0.3px;
    color: #ffffff !important;
}
.q-section-hdr .q-section-count {
    background: rgba(255,255,255,0.18);
    color: #ffffff !important;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.qprev-card {
    background: #ffffff;
    border: 1px solid var(--dash-border);
    border-left: 4px solid var(--dash-orange);
    border-radius: 10px;
    padding: 12px 14px;
    margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
/* Answer widget highlight. Applies to every ``st.selectbox`` /
   ``st.multiselect`` / ``st.text_area`` in the app (BaseWeb wraps
   selects in ``div[data-baseweb="select"]``, text areas render a
   plain ``<textarea>``). Earlier iterations tried to scope this to
   the questionnaire only via a marker div + ``:has()`` + adjacent
   sibling, but Streamlit's actual DOM nests elements more deeply
   than the sibling combinator could bridge, so the rule never
   matched. Applying it globally is deliberate — the same visual
   language flows to the setup page multiselects too, which reads as
   consistent design rather than one-off flash. */
div[data-baseweb="select"] > div,
.stTextArea textarea,
textarea[data-testid="stTextArea"] {
    border: 2px solid var(--dash-orange, #d04a02) !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(208,74,2,0.12) !important;
    /* Explicit longhand properties (not the ``background`` shorthand)
       so we deterministically override the earlier
       ``background-color: #ffffff !important`` rule higher up in the
       stylesheet — CSS shorthands can leave older longhand
       ``!important`` declarations in place depending on cascade order. */
    background-color: #fff6ee !important;
    background-image: linear-gradient(180deg, #ffffff 0%, #fff6ee 100%) !important;
    font-weight: 600 !important;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}
div[data-baseweb="select"] > div:hover,
.stTextArea textarea:hover {
    border-color: #a63a02 !important;
    box-shadow: 0 3px 10px rgba(208,74,2,0.22) !important;
}
div[data-baseweb="select"]:focus-within > div,
.stTextArea textarea:focus {
    border-color: #a63a02 !important;
    box-shadow: 0 0 0 3px rgba(208,74,2,0.18) !important;
    outline: none !important;
}
.qprev-card.free-text {
    border-left-color: var(--dash-blue);
    background: #f8fbff;
}
.qprev-tag-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 6px;
}
.qprev-tag {
    display: inline-block;
    background: #e7eef7;
    color: #13293d;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.2px;
}
.qprev-tag.type-single { background: #fff1e6; color: #6a3300; }
.qprev-tag.type-multi  { background: #ffe6ee; color: #7a1636; }
.qprev-tag.type-free   { background: #e7f5ee; color: #16523c; }
/* Nested follow-up (child) cards — surface only after the parent's
   answer triggers them. Named ``qprev-child`` (not ``qprev-followup``)
   because ``qprev-followup`` is already taken by the purple "brief
   answer nudge" box rendered under short free-text responses, and we
   don't want the two rulesets to bleed into one another. Visual
   language:
   - left-inset via ``margin-left`` (indented children)
   - lighter background + tinted left border (accent line)
   - smaller pad so the card feels like a sub-item, not a peer.
   Depth 2+ nests further so a follow-up-of-a-follow-up is still
   distinguishable. */
.qprev-card.qprev-child {
    margin-left: 22px;
    margin-top: 10px;
    padding: 10px 12px;
    background: #fff8f2;
    border-color: #f2d4b8;
    border-left: 3px solid var(--dash-orange);
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    position: relative;
}
.qprev-card.qprev-child::before {
    /* Elbow connector from the parent card to the child, drawn just
       inside the parent's left border so the branch is obvious even
       without whitespace. */
    content: "";
    position: absolute;
    left: -14px;
    top: 20px;
    width: 12px;
    height: 2px;
    background: var(--dash-orange, #d04a02);
    opacity: 0.55;
    border-radius: 2px;
}
.qprev-card.qprev-child.qprev-child-depth-2 {
    margin-left: 40px;
    background: #fff3e6;
    border-left-color: #a63a02;
}
.qprev-card.qprev-child.qprev-child-depth-3 {
    margin-left: 58px;
    background: #ffedd8;
    border-left-color: #7a2a01;
}
.qprev-card.qprev-child.free-text {
    /* Free-text follow-ups keep the orange accent (still a follow-up)
       while borrowing the qualitative-blue background so the qual /
       quant distinction stays legible. */
    background: #f2f8ff;
    border-color: #cddff0;
}
.qprev-tag.qprev-tag-followup {
    background: var(--dash-orange, #d04a02);
    color: #ffffff;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.qprev-qhead {
    font-weight: 700;
    color: #1a1a1a;
    font-size: 0.95rem;
    line-height: 1.35;
    margin-bottom: 6px;
}
.qprev-options {
    margin: 6px 0 4px 0;
    padding-left: 18px;
    color: #333333;
    font-size: 0.85rem;
    line-height: 1.4;
}
.qprev-options li { margin: 1px 0; }
.qprev-options .qprev-opt-score {
    display: inline-block;
    background: #eef0f3;
    color: #4a4a4a;
    border-radius: 4px;
    padding: 0px 6px;
    font-size: 0.7rem;
    margin-left: 6px;
    font-weight: 700;
}
.qprev-footer {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 0.75rem;
    color: var(--dash-muted);
    margin-top: 6px;
}
.qprev-footer b { color: #2d2d2d; }
.qprev-more {
    color: var(--dash-muted);
    font-size: 0.8rem;
    padding: 6px 0 0;
}

/* Live per-question scoring pill rendered underneath each dropdown once
   the user picks an answer. Reuses the .dash-pill palette so the colour
   coding stays consistent with the dashboard. */
.qprev-score {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    color: #4a4a4a;
    margin: 2px 0 4px;
}
.qprev-score b { color: #2d2d2d; }
.qprev-score.unanswered {
    color: var(--dash-muted);
    font-style: italic;
}

/* Adaptive follow-up prompt shown under a free-text answer when the user's
   response is too brief, ambiguous, or contains only filler tokens. */
.qprev-followup {
    background: #f6f4ff;
    border-left: 4px solid #6b5cff;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    color: #35306b;
    margin: 6px 0 10px;
    line-height: 1.35;
}
.qprev-followup-badge {
    display: inline-block;
    background: #6b5cff;
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    margin-right: 8px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* Live scoring summary strip pinned above the question grid on Page 3.
   Matches the Page 5 KPI-tile visual language so users see the same
   metrics update as they answer. */
.qprev-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin: 0.35rem 0 0.6rem;
}
.qprev-summary-tile {
    background: #ffffff;
    border: 1px solid var(--dash-border);
    border-left: 6px solid var(--dash-blue);
    border-radius: 10px;
    padding: 10px 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.qprev-summary-tile.crit  { border-left-color: var(--dash-red); }
.qprev-summary-tile.risk  { border-left-color: var(--dash-amber); }
.qprev-summary-tile.watch { border-left-color: var(--dash-blue); }
.qprev-summary-tile.ready { border-left-color: var(--dash-green); }
.qprev-summary-label {
    font-size: 0.7rem;
    color: var(--dash-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
}
.qprev-summary-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: #2d2d2d;
    margin-top: 2px;
    line-height: 1.1;
}
.qprev-summary-bar {
    height: 6px;
    background: #eef0f3;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 6px;
}
.qprev-summary-bar > span {
    display: block;
    height: 100%;
    background: var(--dash-blue);
}
.qprev-summary-bar.crit  > span { background: var(--dash-red); }
.qprev-summary-bar.risk  > span { background: var(--dash-amber); }
.qprev-summary-bar.watch > span { background: #c47e00; }
.qprev-summary-bar.ready > span { background: var(--dash-green); }

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
           padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;
           box-shadow: 0 6px 18px rgba(0,0,0,0.12);
           display: flex; flex-direction: column; align-items: flex-start;}
.pwc-hero, .pwc-hero p, .pwc-hero span, .pwc-hero h1, .pwc-hero h2,
.pwc-hero h3, .pwc-hero a {color: #ffffff !important;}
.pwc-hero-logo {display: inline-block;
                margin-bottom: 0.85rem;
                border-radius: 12px;
                overflow: hidden;
                line-height: 0;
                box-shadow: 0 3px 14px rgba(0,0,0,0.35),
                            0 0 0 1px rgba(255, 154, 74, 0.35);}
.pwc-hero-logo img {display: block; height: 78px; width: auto;}
@media (max-width: 640px) {
    .pwc-hero-logo img {height: 58px;}
}
.pwc-title {font-size: 1.55rem; font-weight: 800; margin: 0; letter-spacing: 0.2px;
            color: #ffffff !important;}
.pwc-title .pwc-title-accent {color: #ffd7b8 !important; font-weight: 800;}
.pwc-subtitle {font-size: 0.98rem; margin-top: .35rem; opacity: .95;
               color: #f7e6dc !important; font-weight: 400;
               text-transform: none !important;}
.pwc-subtitle::first-letter {text-transform: uppercase;}

/* ------------------------------------------------------------------ */
/* Optional regulation document — colourful side card                  */
/* ------------------------------------------------------------------ */
.opt-reg-card {
    background: linear-gradient(135deg, #fff2e6 0%, #ffe0c2 45%, #ffd0a3 100%);
    border: 1px solid #f0b27a;
    border-left: 5px solid #d04a02;
    border-radius: 14px;
    padding: 1rem 1.1rem 0.9rem 1.1rem;
    box-shadow: 0 4px 14px rgba(208, 74, 2, 0.12);
    margin-top: 0.15rem;
}
.opt-reg-card .opt-reg-badge {
    display: inline-block;
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #ffffff !important;
    background: #d04a02;
    padding: 2px 10px;
    border-radius: 999px;
    margin-bottom: 0.5rem;
}
.opt-reg-card .opt-reg-title {
    font-size: 1.02rem;
    font-weight: 800;
    color: #4a1f00 !important;
    margin: 0 0 0.25rem 0;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.opt-reg-card .opt-reg-title .opt-reg-icon {
    font-size: 1.1rem;
}
.opt-reg-card .opt-reg-desc {
    font-size: 0.83rem;
    color: #6b3410 !important;
    line-height: 1.35;
    margin: 0 0 0.55rem 0;
}
.opt-reg-card .opt-reg-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-bottom: 0.35rem;
}
.opt-reg-card .opt-reg-chip {
    font-size: 0.7rem;
    font-weight: 700;
    color: #7a3a0d !important;
    background: rgba(255,255,255,0.75);
    border: 1px solid #f0b27a;
    padding: 2px 8px;
    border-radius: 999px;
}
/* Restyle the file_uploader dropzone inside the card */
.opt-reg-card [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.85) !important;
    border: 1.5px dashed #d04a02 !important;
    border-radius: 10px !important;
}
.opt-reg-card [data-testid="stFileUploaderDropzone"] * {
    color: #4a1f00 !important;
}
.opt-reg-card [data-testid="stFileUploader"] label,
.opt-reg-card [data-testid="stFileUploader"] small {
    color: #4a1f00 !important;
}
.opt-reg-card .opt-reg-saved {
    display: block;
    margin-top: 0.45rem;
    padding: 0.4rem 0.6rem;
    background: rgba(46, 125, 50, 0.12);
    border: 1px solid #a5d6a7;
    border-radius: 8px;
    color: #1b5e20 !important;
    font-size: 0.8rem;
    font-weight: 600;
}

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

/* Centred large "Next" button wrapper. Applied to every page footer via
   ``_render_next_button``. The wrapper enlarges the button font,
   thickens the border and centres the label inside the middle column of
   the surrounding ``st.columns`` so it does not sit tucked into the
   bottom-right corner anymore. */
.rap-next-btn-wrap { margin: 0.85rem 0 0.25rem; }
.rap-next-btn-wrap + div .stButton button,
.rap-next-btn-wrap ~ div .stButton button {
    font-size: 1.15rem !important;
    padding: 0.65rem 1.2rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.2px;
}
/* Streamlit renders the columns as a sibling of the wrapper div, so
   target every button that follows it. */
.rap-next-btn-wrap ~ [data-testid="stHorizontalBlock"] .stButton button {
    font-size: 1.2rem !important;
    padding: 0.7rem 1.3rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px;
    min-height: 3rem;
}

/* Global body / heading rhythm. Every heading (h1..h6) is rendered at
   ``body + 3px`` and bold so the hierarchy stays uniform across the
   app while still reading as a heading (regulator ask). */
html, body, .stApp, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    font-size: 15px;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4,
.stApp [data-testid="stMarkdownContainer"] h5,
.stApp [data-testid="stMarkdownContainer"] h6 {
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* Page-level subheader (``st.subheader(...)`` renders as ``<h3>``) —
   the top heading on every page ("1. Setup", "2. Generate BRD / FRD",
   "4. Dashboard — Impact & Readiness", …). Rendered noticeably larger
   than the ``body + 3px`` heading baseline (18px) so it reads as the
   dominant page title, and uppercased per product spec. */
.stApp h3,
.stApp [data-testid="stMarkdownContainer"] h3 {
    font-size: 34px !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px;
    margin-top: 0.25rem !important;
    margin-bottom: 0.9rem !important;
}
/* Streamlit auto-appends a chain-link anchor icon next to every
   heading which becomes visible on hover. Hide it on the page-level
   subheader so the "1. SETUP" line reads as a clean title. */
.stApp h3 a[href^="#"],
.stApp [data-testid="stMarkdownContainer"] h3 a[href^="#"] {
    display: none !important;
}

/* Streamlit renders widget labels ("Regulation Code", "Source Of
   Requirements", …) inside ``<label data-testid="stWidgetLabel">``
   wrappers instead of real ``<h*>`` tags, so the h1..h6 rule above
   does not touch them. We force the same body+3px / bold treatment
   here so every visual heading — including form-field labels and
   expander summaries — reads uniformly. */
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label,
.stApp [data-testid="stWidgetLabel"] label > div p {
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* Expander headers ("Regulatory Intelligence — Official Regulator
   Search", "Downloads", "Answer Questions", …) live inside a
   ``<summary>`` element that wraps a markdown container. Target the
   inner paragraph so only the summary label is styled, leaving the
   expander body content at the default body-text size. Rendered 2px
   smaller than the standard heading baseline so the expander summary
   reads as a secondary heading rather than a top-level page heading. */
.stApp details > summary [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stExpander"] details > summary [data-testid="stMarkdownContainer"] p {
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* Dashboard section headings keep the coloured strip / border-left
   treatment but honour the global heading size + weight so nothing
   pops larger than the ``body + 3px`` baseline. */
.stApp h4.rap-dash-hdr,
.stApp [data-testid="stMarkdownContainer"] h4.rap-dash-hdr {
    font-size: 18px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15px;
    color: #1a1a1a !important;
    margin-top: 2rem !important;
    margin-bottom: 0.9rem !important;
    padding: 0.55rem 0 0.35rem 0.75rem;
    border-top: none;
    border-left: 5px solid #d04a02;
    background: linear-gradient(90deg, #fff5ec 0%, rgba(255,245,236,0) 65%);
    border-radius: 4px;
}
.stApp h4.rap-dash-hdr:first-of-type {
    margin-top: 0.75rem !important;
}
.stApp h4.rap-dash-hdr a[href^="#"] {
    display: none !important;
}

/* Ensure every native Streamlit dataframe defers its horizontal scrollbar
   to the wrapping .rap-table-wrap container so the bar renders *outside*
   the cell text. Scrollbar styling itself lives in the unified block near
   the top of this stylesheet (Regulatory Obligations pattern). */
[data-testid="stDataFrame"] { overflow: visible !important; }
[data-testid="stDataFrameResizable"] { overflow: visible !important; }

/* Sidebar Agentic Workflow tiles — progressive reveal, one tile per agent  */
.agent-tile {
    background: #ffffff;
    border: 1px solid #ead8cc;
    border-left: 4px solid #2e7d32;
    border-radius: 10px;
    padding: 0.55rem 0.7rem;
    margin: 0.35rem 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.agent-tile:hover {
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    transform: translateY(-1px);
}
.agent-tile .agent-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.4px;
    color: #ffffff !important;
    background: #2e7d32;
    padding: 1px 8px;
    border-radius: 999px;
    align-self: flex-start;
    text-transform: uppercase;
}
.agent-tile .agent-name {
    font-size: 0.88rem;
    font-weight: 700;
    color: #2d2d2d !important;
    margin-top: 3px;
}
.agent-tile .agent-metric {
    font-size: 0.78rem;
    color: #4a4a4a !important;
    font-weight: 500;
}

/* Prominent, centered primary CTA for Step 2 */
.step-cta-wrap {display: flex; justify-content: center; margin: 0.6rem 0 0.4rem;}
.step-cta-wrap .stButton {width: 100%;}
.step-cta-wrap .stButton > button,
.step-cta-wrap ~ [data-testid="stHorizontalBlock"] .stButton > button {
    width: 100%;
    padding: 0.85rem 1.6rem !important;
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px;
    border-radius: 10px;
    box-shadow: 0 3px 10px rgba(208, 74, 2, 0.18);
    min-height: 3rem;
}
.step-cta-wrap .stButton > button:hover {
    box-shadow: 0 5px 14px rgba(208, 74, 2, 0.28);
    transform: translateY(-1px);
}

/* Regulator sources table — Title is the hyperlink, no separate URL column.
   Wrapper follows the Regulatory Obligations pattern (.rap-table-wrap)
   for a unified look-and-feel; visual scrollbar styling lives in the
   unified block near the top of the stylesheet. */
.reg-src-table-wrap {
    background: #ffffff;
    border: 2px solid #1a1a1a;
    border-radius: 8px;
    padding: 0.75rem 0.9rem 1.1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    margin: 0.35rem 0 0.9rem;
    max-height: 380px;
    overflow-x: auto;
    overflow-y: auto;
    scrollbar-gutter: stable both-edges;
}
.reg-src-caption {
    font-size: 0.92rem;
    font-weight: 700;
    color: #2d2d2d;
    margin-bottom: 0.5rem;
}
.reg-src-caption-hint {
    font-weight: 400;
    color: #6a6a6a;
    font-size: 0.82rem;
}
.reg-src-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.88rem;
    color: #1a1a1a;
}
.reg-src-table thead th {
    text-align: left;
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: #4a4a4a;
    background: #fff8f2;
    border-bottom: 2px solid #ead8cc;
    padding: 0.55rem 0.7rem;
    position: sticky;
    top: 0;
}
.reg-src-table tbody td {
    padding: 0.5rem 0.7rem;
    border-bottom: 1px solid #f2e4d8;
    vertical-align: top;
}
.reg-src-table tbody tr:last-child td {border-bottom: none;}
.reg-src-table tbody tr:hover td {background: #fff8f2;}
.reg-src-title {min-width: 260px;}
.reg-src-conf, .reg-src-conf-h {white-space: nowrap; text-align: right; width: 90px;}
.reg-src-type, .reg-src-reg {white-space: nowrap; color: #4a4a4a;}
.reg-src-link {
    color: #d04a02 !important;
    font-weight: 600;
    text-decoration: none;
    border-bottom: 1px dotted rgba(208, 74, 2, 0.45);
    transition: color 0.12s ease, border-bottom-color 0.12s ease;
}
.reg-src-link:hover {
    color: #b03d00 !important;
    border-bottom: 1px solid #b03d00;
}
.reg-src-link:visited {color: #7a3a00 !important;}
.reg-src-notitle {color: #8a8a8a; font-style: italic;}

/* -------- Client Role-Aware selector (Page 1 · Step 1) -------- */
.client-roles-card {
    background: linear-gradient(135deg, #fff3e6 0%, #ffffff 65%);
    border: 2px solid #d04a02;
    border-radius: 12px;
    padding: 1rem 1.15rem 0.85rem;
    margin: 0.4rem 0 0.9rem;
    box-shadow: 0 3px 10px rgba(208, 74, 2, 0.12);
}
.client-roles-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #ffffff !important;
    background: #d04a02;
    padding: 2px 10px;
    border-radius: 999px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.client-roles-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #2d2d2d !important;
    margin: 0.1rem 0 0.25rem !important;
}
.client-roles-desc {
    color: #4a4a4a !important;
    font-size: 0.9rem;
    margin: 0 0 0.6rem !important;
    line-height: 1.4;
}
.client-role-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 0.5rem;
    /* Breathing room so the tile does not visually collide with the
       "Regulation Code" heading (and the rest of the Setup form) that
       renders directly below it. */
    margin-bottom: 1.25rem;
}
.client-role-chip {
    flex: 1 1 240px;
    max-width: 320px;
    background: #ffffff;
    border: 1px solid #ead8cc;
    border-left: 4px solid #d04a02;
    border-radius: 8px;
    padding: 0.55rem 0.7rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.client-role-chip-title {
    font-weight: 800;
    font-size: 0.95rem;
    color: #2d2d2d;
}
.client-role-chip-cat {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: #d04a02;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.client-role-chip-desc {
    font-size: 0.82rem;
    color: #4a4a4a;
    margin-bottom: 0.3rem;
    line-height: 1.35;
}
.client-role-chip-meta {
    font-size: 0.78rem;
    color: #4a4a4a;
    margin-top: 2px;
}
.client-role-chip-meta b {color: #2d2d2d;}

/* -------- Client Profile keyword picker (Page 1 · Step 1b) -------- */
.client-profile-card {
    background: linear-gradient(135deg, #fdf2e7 0%, #ffffff 55%);
    border: 2px dashed #d04a02;
    border-radius: 12px;
    padding: 0.9rem 1.1rem 0.75rem;
    margin: 0.3rem 0 0.9rem;
    box-shadow: 0 3px 10px rgba(208, 74, 2, 0.08);
}
.client-profile-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #d04a02 !important;
    background: #fff;
    border: 1px solid #d04a02;
    padding: 2px 10px;
    border-radius: 999px;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.client-profile-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #2d2d2d !important;
    margin: 0.1rem 0 0.2rem !important;
}
.client-profile-desc {
    color: #4a4a4a !important;
    font-size: 0.86rem;
    margin: 0 0 0.55rem !important;
    line-height: 1.4;
}
.client-profile-audit {
    background: #fff8f2;
    border: 1px solid #f2d5c1;
    border-left: 4px solid #d04a02;
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
    margin: 0.35rem 0 0.9rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.client-profile-audit-title {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    color: #d04a02;
    margin-bottom: 0.35rem;
}
.client-profile-audit-row {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.4rem 0.6rem;
    margin: 0.15rem 0;
}
.client-profile-audit-label {
    font-weight: 700;
    font-size: 0.82rem;
    color: #2d2d2d;
    min-width: 175px;
}
.client-profile-audit-chips {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 4px 6px;
}
.client-profile-chip {
    display: inline-block;
    padding: 2px 8px;
    background: #ffffff;
    border: 1px solid #d04a02;
    color: #7a2c00;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
}

/* Role-aware applicability pills used in BRD/RTM/questionnaire panels */
.role-pill {
    display: inline-block;
    padding: 2px 9px;
    margin: 2px 3px 2px 0;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    border: 1px solid transparent;
}
.role-pill-applicable {background: #e6f7ec; color: #1b5e20; border-color: #a5d6a7;}
.role-pill-partial    {background: #fff4e5; color: #7a3d00; border-color: #ffcc80;}
.role-pill-uncertain  {background: #ecf3fb; color: #0d47a1; border-color: #90caf9;}
.role-pill-not        {background: #fdecea; color: #7f1d1d; border-color: #ef9a9a; text-decoration: line-through;}
</style>
"""


def _render_hero() -> str:
    """Build the top hero markup, embedding the logo above the title.

    The hero block is emitted once per Streamlit rerun and therefore appears
    at the top of every page in the app (Setup / Generate BRD-FRD /
    Questionnaire / Dashboard / Export).
    """
    logo_html = (
        f'<div class="pwc-hero-logo">'
        f'<img src="{_LOGO_DATA_URI}" alt="RegAI RAP logo" />'
        f"</div>"
        if _LOGO_DATA_URI
        else ""
    )
    return (
        '<div class="pwc-hero">'
        f"{logo_html}"
        '<p class="pwc-title"><span class="pwc-title-accent">Reg AI RAP</span>'
        " &nbsp;&ndash;&nbsp; A Complete Regulatory Impact Assessment &amp;"
        " Readiness Platform</p>"
        '<p class="pwc-subtitle">Upload a Regulation and get clear Business'
        " Impact, Required Actions, and Practical Recommendations.</p>"
        "</div>"
    )


st.markdown(_HERO_CSS, unsafe_allow_html=True)
st.markdown(_render_hero(), unsafe_allow_html=True)


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
    # Client Role-Aware Regulatory Interpretation (Step 1 of the workflow).
    # ``client_roles`` is the multi-select on Page 1 — canonical institution
    # names from :mod:`services.client_roles`. When empty the pipeline falls
    # back to the pre-role-aware (generic) interpretation.
    "client_roles": ["Commercial Bank"],
    # Client Profile — six keyword multi-selects rendered right below the
    # institution type on Page 1. Each key is a list of curated + free-form
    # keywords the user tagged. Pre-populated with a Commercial Bank / DORA
    # starting set (matches the default ``client_roles`` and ``regulation``
    # values above) so reviewers land on the page with a working profile
    # already tagged; clear or edit any dimension to override.
    "client_profile": default_client_profile(),
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
    "rich_recommendations": [],   # List[RichRecommendation] from Agent 4
    # AI Assessment Intelligence bundles — dynamic confidence, impact and
    # readiness assessments (produced by services.ai_assessment_intelligence).
    "confidence_assessment": None,   # ConfidenceAssessment | None
    "impact_assessment": None,       # ImpactAssessment | None
    "readiness_assessment": None,    # ReadinessAssessment | None
    # Weighted readiness scoring (DORA demo profile). Populated on every
    # dashboard refresh via _refresh_scoring_snapshot; None until then.
    "weighted_readiness": None,      # WeightedReadinessResult | None
    "weighted_readiness_error": None,
    # Weighted impact scoring (DORA demo profile). Sibling of the
    # weighted-readiness result; the two are calculated independently
    # and combined into Priority = Impact * (100 - Readiness) / 100.
    "weighted_impact": None,         # WeightedImpactResult | None
    "weighted_impact_error": None,
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
        load_dotenv()

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
    """Re-run the Python Rules Engine against the current responses.

    Also refreshes the AI Assessment Intelligence bundle (confidence,
    impact, readiness) so the dashboard always renders in sync with the
    latest responses. The assessment intelligence is best-effort — if the
    GenAI client is not configured the deterministic fallback still fills
    in an evidence-driven bundle so the UI stays useful.
    """
    questionnaire: Optional[QuestionnairePackage] = st.session_state.get("questionnaire")
    if questionnaire is None:
        logger.debug("_refresh_scoring_snapshot: no questionnaire in session, skipping.")
        return None
    state: AssessmentState = st.session_state["assessment_state"]
    orch = _get_orchestrator()
    logger.debug(
        "Refreshing scoring snapshot. responses=%d dynamic_queue=%d",
        len(state.responses or {}), len(state.dynamic_queue or []),
    )
    scoring = orch.run_rules_engine(questionnaire, state)

    analysis: Optional[RegulatoryAnalysis] = st.session_state.get("analysis")
    impact = st.session_state.get("impact_assessment")
    if impact is None and analysis is not None:
        try:
            impact = orch.assess_impact_intelligence(analysis)
            st.session_state["impact_assessment"] = impact
        except Exception:
            logger.exception("Impact intelligence refresh failed (non-fatal).")
            impact = None
    scoring.impact = impact

    # Fingerprint-gate the readiness + confidence GenAI calls.
    #
    # Streamlit re-runs the whole script on every widget interaction, and
    # this helper is invoked from every dashboard render — so without a
    # gate we'd fire two full LLM round-trips (~10-30s each) for every
    # keystroke, even when the underlying evidence (analysis + answered
    # response set + score) is unchanged. The fingerprint below captures
    # the exact inputs the two assessments read; when it matches the
    # last-computed value we reuse the cached assessments from session
    # state and skip the LLM calls entirely. Any real change to
    # ``analysis`` identity, answered-count, weighted score, or number
    # of scored pairs busts the cache and forces a recompute.
    eval_dict = scoring.evaluation if isinstance(scoring.evaluation, dict) else {}
    assess_fingerprint = (
        id(analysis) if analysis is not None else 0,
        int(eval_dict.get("answered_count") or 0),
        round(float(eval_dict.get("compliance_score_pct") or 0.0), 1),
        len(eval_dict.get("pair_scores") or {}),
    )
    last_assess_fingerprint = st.session_state.get("_assessment_intel_fingerprint")

    cached_readiness = st.session_state.get("readiness_assessment")
    cached_confidence = st.session_state.get("confidence_assessment")
    assess_cache_hit = (
        last_assess_fingerprint == assess_fingerprint
        and cached_readiness is not None
        and cached_confidence is not None
    )

    if assess_cache_hit:
        logger.debug(
            "Assessment intelligence cache hit; reusing cached readiness "
            "+ confidence (fingerprint=%s).",
            assess_fingerprint,
        )
        scoring.readiness = cached_readiness
        scoring.confidence = cached_confidence
    else:
        try:
            readiness = orch.assess_readiness_intelligence(
                scoring.evaluation,
                analysis=analysis,
                questionnaire_package=questionnaire.package,
                responses=state.responses,
            )
            scoring.readiness = readiness
            st.session_state["readiness_assessment"] = readiness
        except Exception:
            logger.exception("Readiness intelligence refresh failed (non-fatal).")
            scoring.readiness = cached_readiness

        try:
            confidence = orch.assess_confidence_intelligence(
                analysis,
                scoring_evaluation=scoring.evaluation,
                questionnaire_package=questionnaire.package,
            )
            scoring.confidence = confidence
            st.session_state["confidence_assessment"] = confidence
        except Exception:
            logger.exception("Confidence intelligence refresh failed (non-fatal).")
            scoring.confidence = cached_confidence

        # Only advance the fingerprint after a full successful recompute
        # so a partial failure re-tries on the next refresh instead of
        # locking in a stale pair.
        if scoring.readiness is not None and scoring.confidence is not None:
            st.session_state["_assessment_intel_fingerprint"] = assess_fingerprint

    # Weighted readiness scoring (DORA demo profile). Computed *after* the
    # rules engine so any state changes made by the AI assessment bundles
    # (e.g. readiness overrides) are already visible. The dataclass sits on
    # the session state under ``weighted_readiness`` for direct UI access,
    # and a JSON-safe copy is merged into ``scoring.evaluation`` so it
    # round-trips through SQLite via the existing ``evaluation_json``
    # column - no schema migration needed.
    #
    # Business rule: the *displayed* readiness score everywhere in the app
    # (hero tile, KPI cards, dashboard downstream consumers, SQLite
    # ``compliance_score_pct`` column, Agent 4 fingerprint, …) is the
    # weighted overall - not the legacy per-question weighted average.
    # We overwrite ``compliance_score_pct`` inline so every downstream
    # reader picks up the new number automatically. The original rules-
    # engine number is preserved under ``compliance_score_pct_legacy`` so
    # it can still be inspected for diagnostics / regression comparisons.
    try:
        base_questions = list(
            (questionnaire.package.get("questions") or []) if questionnaire.package else []
        )
        weighted = compute_weighted_readiness(base_questions, state)
        st.session_state["weighted_readiness"] = weighted
        st.session_state["weighted_readiness_error"] = None
        if isinstance(scoring.evaluation, dict):
            scoring.evaluation["weighted_readiness"] = weighted.as_dict()
            legacy_score = float(scoring.evaluation.get("compliance_score_pct") or 0.0)
            scoring.evaluation["compliance_score_pct_legacy"] = round(legacy_score, 2)
            scoring.evaluation["compliance_score_pct"] = float(
                weighted.overall_readiness_score
            )
            scoring.evaluation["readiness_rating"] = weighted.readiness_rating
            scoring.evaluation["overall_coverage_gap"] = float(
                weighted.overall_coverage_gap
            )
            scoring.evaluation["completeness_score"] = float(
                weighted.completeness_score
            )
            scoring.evaluation["accuracy_score"] = float(weighted.accuracy_score)
        # Force the AI Readiness Assessment panel to display the same
        # weighted overall - otherwise the dashboard would show two
        # different numbers for "Overall Readiness". The dimensional
        # sub-scores on that panel remain untouched so users still see
        # per-dimension maturity signals.
        if scoring.readiness is not None:
            scoring.readiness.overall_score = float(weighted.overall_readiness_score)
            scoring.readiness.overall_level = weighted.readiness_rating
            st.session_state["readiness_assessment"] = scoring.readiness
    except Exception as exc:
        # Never let a scoring extension break the main rules-engine path.
        # We still surface the failure through a debug caption so an
        # operator can spot misconfigured weights or malformed packages.
        logger.exception("Weighted readiness computation failed (non-fatal).")
        st.session_state["weighted_readiness"] = None
        st.session_state["weighted_readiness_error"] = str(exc)

    # Weighted impact scoring (DORA demo profile). Computed AFTER weighted
    # readiness so we can pass the readiness result in for the Priority
    # formula (Priority = Impact * (100 - Readiness) / 100). Also feeds
    # per-area readiness into the priority table so a "high impact but
    # low readiness" area surfaces above one that is well-covered.
    try:
        readiness_for_priority = st.session_state.get("weighted_readiness")
        area_readiness_map: Dict[str, float] = {}
        pair_readiness_map: Dict[Any, float] = {}
        if isinstance(scoring.evaluation, dict):
            for area, val in (scoring.evaluation.get("area_scores") or {}).items():
                try:
                    area_readiness_map[str(area)] = float(val)
                except (TypeError, ValueError):
                    continue
            for key, val in (scoring.evaluation.get("pair_scores") or {}).items():
                try:
                    pair_readiness_map[key] = float(val)
                except (TypeError, ValueError):
                    continue
        weighted_imp = compute_weighted_impact(
            analysis=analysis,
            brd_artifact=st.session_state.get("brd_artifact"),
            rtm_artifact=st.session_state.get("rtm_artifact"),
            questionnaire=questionnaire,
            readiness_result=readiness_for_priority,
            area_readiness=area_readiness_map,
            pair_readiness=pair_readiness_map,
        )
        st.session_state["weighted_impact"] = weighted_imp
        st.session_state["weighted_impact_error"] = None
        if isinstance(scoring.evaluation, dict):
            scoring.evaluation["weighted_impact"] = weighted_imp.as_dict()
            scoring.evaluation["overall_impact_score"] = float(
                weighted_imp.overall_impact_score
            )
            scoring.evaluation["impact_rating"] = weighted_imp.impact_rating
            scoring.evaluation["overall_priority_score"] = float(
                weighted_imp.overall_priority_score
            )
        # Overwrite the AI ImpactAssessment top-line so hero + intel panel
        # + area cards + heatmap all show one consistent impact number.
        # Per-dimension AI severity scores stay - they are still useful
        # signals for consulting-grade narratives.
        if scoring.impact is not None:
            scoring.impact.overall_severity_score = float(
                weighted_imp.overall_impact_score
            )
            scoring.impact.overall_severity = weighted_imp.impact_rating
            st.session_state["impact_assessment"] = scoring.impact
    except Exception as exc:
        st.session_state["weighted_impact"] = None
        st.session_state["weighted_impact_error"] = str(exc)

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


def _selected_client_roles() -> List[str]:
    """Return the client roles that should be fed into the downstream pipeline.

    **Product decision (2026-07):** The Client Role selector on Page 1 is
    now purely **informational / display-only**. The user still picks the
    institution type so they see their choice reflected on Page 1, but the
    selection is intentionally NOT propagated into Agent 1 (regulatory
    analysis), Agent 3 (questionnaire filter), or any other downstream
    stage — the pipeline runs role-agnostic. This eliminates the
    deterministic role-aware overhead (~50-300 ms/pipeline plus the same
    per Page 2 render) without changing what the user sees at role-pick
    time.

    The raw selection is still available at
    ``st.session_state["client_roles"]`` for UI use (chips, labels,
    export metadata). Only this helper — the boundary between UI and
    pipeline — returns an empty list so no agent branches on it.
    """
    # Deliberately return an empty list — see docstring.
    return []


def _current_client_profile() -> Dict[str, List[str]]:
    """Return the normalized Client Profile keyword bundle from session state.

    Free-form keywords typed into the widgets are preserved verbatim; only
    trivial whitespace / de-duplication is applied. Empty when the user has
    not populated any field.
    """
    raw = st.session_state.get("client_profile") or {}
    return normalize_client_profile(raw)


def _keyword_multiselect(
    field: ClientProfileField, current_value: List[str],
) -> List[str]:
    """Render a keyword multi-select widget for one Client Profile field.

    When ``field.allow_freeform`` is ``True`` (the default) users can pick
    from the curated seed list *or* type any custom keyword — implemented
    via ``st.multiselect(accept_new_options=True)`` on Streamlit >= 1.39,
    with a two-widget fallback (multi-select + comma-separated text
    input) for older builds.

    When ``allow_freeform`` is ``False`` the widget is locked to the
    curated seed catalog — any stale / off-catalog values that survived
    in session state from an earlier build are dropped, and the "Add
    custom keyword" affordance is hidden. Used for Organization Profile
    so users only ever see the six approved options.
    """
    seed_options = list(field.options)
    seed_lower = {o.lower() for o in seed_options}
    initial = list(current_value or [])

    widget_key = f"client_profile_widget_{field.key}"

    if not field.allow_freeform:
        # Curated-only: drop any values that aren't in the seed catalog,
        # both from the ``initial`` we hand Streamlit AND from any stale
        # widget state left over from an earlier session (which would
        # otherwise resurrect off-catalog chips).
        initial = [v for v in initial if v and v.lower() in seed_lower]
        stale = st.session_state.get(widget_key)
        if isinstance(stale, list):
            cleaned = [v for v in stale if v and str(v).lower() in seed_lower]
            if cleaned != stale:
                st.session_state[widget_key] = cleaned
        selection = st.multiselect(
            f"{field.icon} {field.label}",
            options=seed_options,
            default=initial,
            help=field.help,
            placeholder=field.placeholder,
            key=widget_key,
        )
        return list(selection)

    merged_options: List[str] = list(seed_options)
    seen = set(o.lower() for o in merged_options)
    for value in initial:
        if value and value.lower() not in seen:
            merged_options.append(value)
            seen.add(value.lower())

    try:
        selection = st.multiselect(
            f"{field.icon} {field.label}",
            options=merged_options,
            default=initial,
            help=field.help,
            placeholder=field.placeholder,
            accept_new_options=True,
            key=widget_key,
        )
    except TypeError:
        # ``accept_new_options`` is not supported on the running Streamlit
        # version. Fall back to a two-widget pattern: pick curated values
        # from a multi-select and add custom ones via a comma-separated
        # text input.
        selection = st.multiselect(
            f"{field.icon} {field.label}",
            options=merged_options,
            default=initial,
            help=field.help,
            placeholder=field.placeholder,
            key=widget_key,
        )
        custom_key = f"{widget_key}_custom"
        custom_default = ", ".join(
            v for v in initial
            if v.lower() not in {o.lower() for o in seed_options}
        )
        custom_raw = st.text_input(
            f"Custom {field.label} keywords (comma-separated)",
            value=custom_default,
            key=custom_key,
            placeholder="Type additional keywords, separated by commas…",
        )
        for token in (custom_raw or "").split(","):
            token = token.strip()
            if token and token.lower() not in {s.lower() for s in selection}:
                selection.append(token)
    return list(selection)


def _render_client_profile_selector() -> None:
    """Render the six Client Profile keyword multi-selects on Page 1.

    Layout: two columns × three rows so the six fields fit compactly under
    the Institution Type card. Every widget writes back into
    ``st.session_state["client_profile"]`` so the pipeline picks it up on
    the next run.
    """
    current = _current_client_profile()

    # NOTE: we deliberately do NOT wrap this panel in a
    # ``<div class="client-profile-card">`` container. Streamlit renders
    # each ``st.markdown`` call inside its own isolated DOM block, so an
    # orphan opening ``<div>`` gets auto-closed by the browser and shows
    # up as an empty bordered box above the badge. The pill badge below
    # already provides enough visual grouping.
    st.markdown(
        '<span class="client-profile-badge">Step 1 · Client Profile</span>'
        '<p class="client-profile-title">Client Profile Keywords</p>'
        '<p class="client-profile-desc">Tag the client the way you would tag '
        'a CV. Keywords from every dimension below are threaded through the '
        'agentic pipeline — Agent 1 uses them as extra regulatory-surface '
        'signal, the BRD/FRD prompt is scoped to the tagged profile, and '
        'the RTM / questionnaire / recommendations all inherit the '
        'context. Type any custom keyword — the widgets accept free-form '
        'entries alongside the curated catalog.</p>',
        unsafe_allow_html=True,
    )
    updated: Dict[str, List[str]] = {}
    for row_index in range(0, len(CLIENT_PROFILE_FIELDS), 2):
        cols = st.columns(2, gap="large")
        for offset, col in enumerate(cols):
            idx = row_index + offset
            if idx >= len(CLIENT_PROFILE_FIELDS):
                continue
            field = CLIENT_PROFILE_FIELDS[idx]
            with col:
                updated[field.key] = _keyword_multiselect(
                    field, current.get(field.key) or [],
                )
    # Preserve any dimensions we did not render (shouldn't happen, but
    # safe-by-default) and normalise the result.
    for key in CLIENT_PROFILE_KEYS:
        updated.setdefault(key, current.get(key) or [])
    st.session_state["client_profile"] = normalize_client_profile(updated)

    populated = is_client_profile_populated(st.session_state["client_profile"])
    if populated:
        tally = ", ".join(
            f"**{f.label}**: {len(st.session_state['client_profile'].get(f.key) or [])}"
            for f in CLIENT_PROFILE_FIELDS
            if st.session_state["client_profile"].get(f.key)
        )
        st.caption(
            f"Profile tagged — {tally}. These keywords will flow into Agent 1, "
            f"the BRD/FRD prompt, the RTM, questionnaire and recommendations."
        )
    else:
        st.caption(
            "No profile keywords tagged yet. The pipeline will use the "
            "generic (role-only) interpretation. Add keywords to sharpen "
            "the analysis for your specific client."
        )


def _render_client_roles_selector() -> None:
    """Client Role setup: the multi-select at the top of Page 1.

    **Product decision (2026-07):** The Client Role selector is now
    **display-only** — the user picks their institution type(s) so the
    choice is visible on Page 1 and captured in export metadata, but the
    pipeline runs **role-agnostic**. See ``_selected_client_roles`` for
    the boundary function that returns an empty list to downstream agents.
    """
    options = list(INSTITUTION_TYPE_NAMES)
    current_raw = st.session_state.get("client_roles") or []
    current = normalize_client_roles(current_raw) or ["Commercial Bank"]

    def _fmt(name: str) -> str:
        role = get_institution_type(name)
        if role is None:
            return name
        return f"{name} — {role.category}"

    # See ``_render_client_profile_selector`` — the ``client-roles-card``
    # container wrapper was intentionally removed because Streamlit places
    # each ``st.markdown`` call in its own DOM block, which caused the
    # opening ``<div>`` to render as an empty bordered rectangle above
    # the badge. The pill badge alone already anchors this panel; the
    # secondary title + descriptive paragraph have been dropped per
    # product feedback so the multi-select below stands on its own.
    st.markdown(
        '<span class="client-roles-badge">Step 1 · Client Profile</span>',
        unsafe_allow_html=True,
    )
    selection = st.multiselect(
        "Institution Type(s)",
        options=options,
        default=current,
        format_func=_fmt,
        help=(
            "Informational only. Your selection is displayed on this page "
            "and included in the export metadata, but the downstream "
            "regulatory analysis, BRD/RTM generation, questionnaire, and "
            "recommendations run role-agnostic — the same output is "
            "produced regardless of which institution type(s) you pick."
        ),
        key="client_roles_widget",
    )
    st.session_state["client_roles"] = list(selection)

    if selection:
        # Render a compact per-role summary card so the user knows what
        # business surface has just been armed for the analysis.
        cards_html: List[str] = []
        for name in selection:
            role = get_institution_type(name)
            if role is None:
                continue
            domains = ", ".join(list(role.domains)[:4]) or "—"
            obligations = ", ".join(list(role.typical_obligations)[:3]) or "—"
            cards_html.append(
                f'<div class="client-role-chip">'
                f'<div class="client-role-chip-title">{html.escape(role.name)}</div>'
                f'<div class="client-role-chip-cat">{html.escape(role.category)}</div>'
                f'<div class="client-role-chip-desc">{html.escape(role.summary)}</div>'
                f'<div class="client-role-chip-meta"><b>Key domains:</b> '
                f'{html.escape(domains)}</div>'
                f'<div class="client-role-chip-meta"><b>Typical obligations:</b> '
                f'{html.escape(obligations)}</div>'
                f'</div>'
            )
        if cards_html:
            st.markdown(
                '<div class="client-role-chip-row">'
                + "".join(cards_html)
                + '</div>',
                unsafe_allow_html=True,
            )


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
        "Regulator Scope",
        options=options,
        default=cleaned,
        format_func=_regulator_label,
        help=(
            "Search is restricted to the official websites of the selected regulators.\n\n"
            "• Choose **ALL** to query every approved regulator.\n"
            "• Publications are retrieved **automatically** whenever this selection changes — "
            "no button click needed.\n"
            "• Each regulator's own site-search is queried first (EBA, ESMA, EIOPA, FCA, …); "
            "a general web search is used only as a fallback.\n"
            "• Wikipedia, blogs and generic search results are never used."
        ),
        key="regulator_selection_widget",
    )
    if not st.session_state["regulator_selection"]:
        st.session_state["regulator_selection"] = [_ALL_REGULATOR_CODE]


def _auto_fetch_regulatory_intelligence(regulation: str) -> None:
    """Fetch publications for the current regulator selection, if needed.

    Runs when the selection (or the regulation code) has changed since the
    last fetch. Silent about failure modes — only surfaces a compact result
    line. All heavy diagnostics have been dropped so the page stays clean.
    """
    current_selection = _selected_regulator_codes()
    fingerprint = (regulation, tuple(current_selection))
    last_fingerprint = st.session_state.get("_reg_intel_last_fingerprint")
    if fingerprint == last_fingerprint:
        return

    with st.spinner("Processing..."):
        try:
            package = gather_regulatory_intelligence(
                regulation,
                regulator_selection=current_selection,
                consulting_selection=None,
                include_consulting=False,
                status=lambda _msg: None,
            )
        except Exception:
            package = None

    st.session_state["regulatory_intelligence_package"] = package
    st.session_state["_reg_intel_last_fingerprint"] = fingerprint


def _render_regulatory_intelligence_block() -> None:
    """Stage 1-only intelligence panel rendered on Page 1.

    Retrieves publications automatically whenever the regulator selection
    changes — no explicit "Preview" click is required. The panel stays
    intentionally clean: a scoped selector, a hint tooltip, and (when
    results exist) a compact ranked table.
    """
    regulation = st.session_state.get("regulation") or "DORA"
    stage1_enabled = is_regulatory_search_enabled()

    with st.expander("Regulatory Intelligence — Official Regulator Search", expanded=True):
        if stage1_enabled:
            st.success(
                f"Regulator search is **ON** for `{regulation}`. "
                "Only approved regulator domains will be searched.",
                icon=":material/verified:",
            )
        else:
            st.warning(
                "Regulator search is **OFF**. Set `REGULATORY_SEARCH_ENABLED=true` in `.env` "
                "to enable live regulator search."
            )

        _render_regulator_selector()

        st.caption(
            "Publications are retrieved automatically the moment you change the regulator "
            "selection above.  \nℹ️ Hover the field for details on which sites are queried."
        )

        if stage1_enabled:
            _auto_fetch_regulatory_intelligence(regulation)

        package: Optional[RegulatoryIntelligencePackage] = st.session_state.get(
            "regulatory_intelligence_package"
        )
        if package is None:
            return

        if package.has_official_content:
            st.success(
                f"Retrieved {len(package.official_results)} official publication(s) "
                f"from approved regulator domains.",
                icon=":material/task_alt:",
            )
            _render_intelligence_sources_table(package)
        elif package.has_any_content:
            st.info(
                "No official regulator publications matched — try a different regulator "
                "selection above, or upload the regulation PDF on Page 1 for use as primary context.",
                icon=":material/info:",
            )
        else:
            st.info(
                "No publications retrieved. Adjust the regulator selection above, or upload "
                "the regulation PDF on Page 1 for use as primary context.",
                icon=":material/info:",
            )


def _render_intelligence_sources_table(package: RegulatoryIntelligencePackage) -> None:
    """Render every retrieved official regulator source as a ranked table.

    Kept intentionally compact — only the columns that are essential for
    reviewing a source (Source Type, Regulator, Title). The URL column
    is not shown; instead the **title itself is a hyperlink** that opens
    the publication on the regulator's own site in a new tab.
    """
    rows = [r for r in package.all_sources() if r.get("source_type") != "Consulting Guidance"]
    if not rows:
        return

    body_rows: List[str] = []
    for r in rows:
        title_text = (r.get("title") or "").strip()
        title_safe = html.escape(title_text[:140] or "(Untitled)")
        url = (r.get("source_url") or "").strip()
        if url:
            url_safe = html.escape(url, quote=True)
            title_cell = (
                f'<a href="{url_safe}" target="_blank" rel="noopener noreferrer" '
                f'class="reg-src-link" title="Open in new tab">{title_safe}</a>'
            )
        else:
            title_cell = f'<span class="reg-src-notitle">{title_safe}</span>'
        body_rows.append(
            "<tr>"
            f'<td class="reg-src-type">{html.escape(str(r.get("source_type") or ""))}</td>'
            f'<td class="reg-src-reg">{html.escape(str(r.get("regulator") or ""))}</td>'
            f'<td class="reg-src-title">{title_cell}</td>'
            "</tr>"
        )

    table_html = (
        '<div class="reg-src-table-wrap">'
        '<div class="reg-src-caption">Retrieved Regulator Sources '
        '<span class="reg-src-caption-hint">(Click a title to open the source)</span>'
        '</div>'
        '<table class="reg-src-table">'
        '<thead><tr>'
        '<th class="reg-src-type-h">Source Type</th>'
        '<th class="reg-src-reg-h">Regulator</th>'
        '<th class="reg-src-title-h">Title</th>'
        '</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>'
        '</div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


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


# Fallback text used when a row has no live source URL to link to.
_NO_LIVE_SOURCE_TEXT = "[!] No live source available"


def _sources_link_cell(labels: str, url: str) -> str:
    """Return a cell value suitable for :class:`st.column_config.LinkColumn`
    that renders ``labels`` as the clickable display text and points at
    ``url`` when available.

    Streamlit's ``LinkColumn`` renders the cell as an anchor whose visible
    text is derived from the ``display_text`` regex applied to the URL.
    We therefore encode the labels into the URL fragment so a single
    regex (``#(.+)$``) can extract them per-row.

    - When a URL is present, we emit ``<url>#<labels>``. The fragment is
      ignored by the browser during navigation, so the click still lands
      on the primary regulatory publication.
    - When no live source URL is available, we emit ``#<labels>`` so the
      cell still renders the label text (the "click" becomes a no-op
      same-page fragment navigation).

    Any literal ``#`` character in ``labels`` is replaced with the
    full-width variant so it does not confuse the fragment split.
    """
    safe_labels = (labels or _NO_LIVE_SOURCE_TEXT).replace("#", "\uFF03")
    if url:
        return f"{url}#{safe_labels}"
    return f"#{safe_labels}"


# Regex used by every LinkColumn "Sources" column to pull the display
# text out of the fragment segment produced by :func:`_sources_link_cell`.
_SOURCES_LINK_DISPLAY_REGEX = r"#(.+)$"


# ---------------------------------------------------------------------------
# Confidence-gap tooltips
# ---------------------------------------------------------------------------
#
# The Completeness Coverage, Accuracy Coverage and Overall Regulatory
# Coverage tiles all show a percentage. Whenever the number is below 100
# reviewers need a plain-English reason for the gap ("what are we not
# confident about?"). The tooltip built by :func:`_confidence_gap_tooltip`
# lists, in simple terms, the specific signals that are missing or thin —
# so users can see *which* gaps are causing the percentage to drop. When
# the assessment is missing we fall back to a short disclaimer so the
# tooltip is never empty.


def _completeness_gap_drivers(
    signals: Mapping[str, Any],
) -> List[Tuple[float, str]]:
    """Return ``[(weight, "plain-english reason"), …]`` for Completeness.

    The ``weight`` field is kept only for ordering (largest first) and for
    the evaluation-kind weighting logic. Reasons are written in simple
    terms — no scoring jargon, no "pts", no "dimensions".
    """
    obligations = int(signals.get("obligation_count") or 0)
    areas = int(signals.get("impacted_area_count") or 0)
    themes = int(signals.get("theme_count") or 0)
    requirements = int(signals.get("requirement_count") or 0)

    drivers: List[Tuple[float, str]] = []

    oblig_pts = min(40.0, obligations * 0.6)
    if oblig_pts < 40.0:
        drivers.append((
            40.0 - oblig_pts,
            f"Only {obligations} obligations were extracted from the regulation."
            if obligations
            else "No obligations were extracted from the regulation yet.",
        ))

    area_pts = min(20.0, areas * 2.0)
    if area_pts < 20.0:
        drivers.append((
            20.0 - area_pts,
            f"Only {areas} impacted business areas were identified."
            if areas
            else "No impacted business areas were identified yet.",
        ))

    theme_pts = min(15.0, themes * 1.5)
    if theme_pts < 15.0:
        drivers.append((
            15.0 - theme_pts,
            f"Only {themes} obligation themes were clustered."
            if themes
            else "No obligation themes have been clustered yet.",
        ))

    req_pts = min(25.0, requirements * 0.25)
    if req_pts < 25.0:
        drivers.append((
            25.0 - req_pts,
            f"Only {requirements} BRD requirements have been captured so far."
            if requirements
            else "No BRD requirements have been captured yet.",
        ))

    drivers.sort(key=lambda x: -x[0])
    return drivers


def _accuracy_gap_drivers(
    signals: Mapping[str, Any],
) -> List[Tuple[float, str]]:
    """Return ``[(weight, "plain-english reason"), …]`` for Accuracy."""
    obligations = int(signals.get("obligation_count") or 0)
    requirements = int(signals.get("requirement_count") or 0)
    reqs_with_article = int(signals.get("requirements_with_article_ref") or 0)
    reqs_with_citations = int(signals.get("requirements_with_citations") or 0)
    obls_with_citations = int(signals.get("obligations_with_citations") or 0)

    drivers: List[Tuple[float, str]] = []

    if requirements == 0:
        drivers.append((
            45.0,
            "No BRD requirements captured yet, so we cannot check whether "
            "they are backed by regulation citations.",
        ))
    else:
        article_ratio = reqs_with_article / requirements
        article_pts = min(25.0, article_ratio * 30.0)
        if article_pts < 25.0:
            drivers.append((
                25.0 - article_pts,
                f"Only {reqs_with_article} of {requirements} requirements "
                f"point to a specific Article number in the regulation.",
            ))
        citation_ratio = reqs_with_citations / requirements
        citation_pts = min(15.0, citation_ratio * 15.0)
        if citation_pts < 15.0:
            drivers.append((
                15.0 - citation_pts,
                f"Only {reqs_with_citations} of {requirements} requirements "
                f"carry any regulation citation at all.",
            ))

    if obligations:
        obl_ratio = obls_with_citations / obligations
        obl_pts = min(5.0, obl_ratio * 5.0)
        if obl_pts < 5.0:
            drivers.append((
                5.0 - obl_pts,
                f"Only {obls_with_citations} of {obligations} obligations "
                f"link back to a source in the regulation.",
            ))

    drivers.sort(key=lambda x: -x[0])
    return drivers


def _clarity_gap_drivers(
    signals: Mapping[str, Any],
) -> List[Tuple[float, str]]:
    """Return ``[(weight, "plain-english reason"), …]`` for Clarity."""
    requirements = int(signals.get("requirement_count") or 0)
    obligations = int(signals.get("obligation_count") or 0)
    closed = int(signals.get("closed_question_count") or 0)
    quant = int(signals.get("quantitative_question_count") or 0)
    answered = int(signals.get("answered_count") or 0)
    unanswered = int(signals.get("unanswered_count") or 0)

    drivers: List[Tuple[float, str]] = []

    if not (requirements and obligations):
        drivers.append((
            10.0,
            "Requirements and obligations are not both available yet.",
        ))
    if not closed:
        drivers.append((
            10.0,
            "No scored questions are in the questionnaire yet.",
        ))
    if not quant:
        drivers.append((
            10.0,
            "No quantitative questions (numbers, percentages, counts) "
            "were detected.",
        ))
    total_q = answered + unanswered
    if total_q == 0:
        drivers.append((
            10.0,
            "No questionnaire answers have been captured yet.",
        ))
    else:
        coverage = answered / total_q
        pts = min(10.0, coverage * 10.0)
        if pts < 10.0:
            drivers.append((
                10.0 - pts,
                f"Only {answered} of {total_q} questions have been "
                f"answered so far.",
            ))

    drivers.sort(key=lambda x: -x[0])
    return drivers


def _quality_gap_drivers(
    signals: Mapping[str, Any],
) -> List[Tuple[float, str]]:
    """Return ``[(weight, "plain-english reason"), …]`` for Quality."""
    requirements = int(signals.get("requirement_count") or 0)
    with_priority = int(signals.get("requirements_with_priority") or 0)
    with_accept = int(signals.get("requirements_with_acceptance") or 0)
    avg_detail = float(signals.get("avg_requirement_detail_chars") or 0.0)

    drivers: List[Tuple[float, str]] = []

    if requirements == 0:
        drivers.append((
            45.0,
            "No BRD requirements captured yet, so we cannot judge the "
            "quality of individual requirements.",
        ))
        return drivers

    prio_ratio = with_priority / requirements
    prio_pts = min(20.0, prio_ratio * 20.0)
    if prio_pts < 20.0:
        drivers.append((
            20.0 - prio_pts,
            f"Only {with_priority} of {requirements} requirements have "
            f"a priority (Must / Should / Could / Won't) tagged.",
        ))

    acc_ratio = with_accept / requirements
    acc_pts = min(15.0, acc_ratio * 15.0)
    if acc_pts < 15.0:
        drivers.append((
            15.0 - acc_pts,
            f"Only {with_accept} of {requirements} requirements include "
            f"clear acceptance criteria.",
        ))

    detail_pts = min(10.0, max(0.0, (avg_detail - 80.0) / 20.0))
    if detail_pts < 10.0:
        drivers.append((
            10.0 - detail_pts,
            "Requirement descriptions are shorter than expected — many "
            "requirements lack enough detail to be actionable.",
        ))

    drivers.sort(key=lambda x: -x[0])
    return drivers


# ---------------------------------------------------------------------------
# Full-screen agent loading widget.
#
# Rendered whenever a long-running agent (Agent 1 BRD, Agent 3 questionnaire,
# Agent 4 recommendations) is processing. The goal is to give the user a
# CLEAN loading experience — no faded previous-page content behind the
# indicator. See ``render_questionnaire_page`` for the two-phase render
# pattern that actually delivers that experience (Streamlit needs a
# ``st.rerun()`` between "paint loader" and "start blocking call" so the
# DOM finalises the loader-only state before the block begins).
# ---------------------------------------------------------------------------
def _render_agent_loader(
    subheader: str,
    title: str,
    message: str,
    *,
    show_retry_button: bool = False,
) -> None:
    """Render a full-width centred loading panel."""
    st.subheader(subheader)
    # Use a big centred card so the user immediately understands the
    # page is intentionally paused on a long-running task. Emoji + big
    # heading + explanation keep the experience readable without any
    # data noise.
    st.markdown(
        f"""
        <div style="
            padding: 5rem 2rem;
            margin: 1.5rem 0 2rem 0;
            text-align: center;
            background: linear-gradient(180deg, #ffffff 0%, #fef7f0 100%);
            border: 1px solid #f2e8dc;
            border-radius: 18px;
            box-shadow: 0 6px 24px rgba(210, 71, 38, 0.06);
        ">
            <div class="rap-loader-spinner" style="
                width: 56px; height: 56px;
                margin: 0 auto 1.5rem auto;
                border: 5px solid #fce9dc;
                border-top-color: #d24726;
                border-radius: 50%;
                animation: rap-spin 1s linear infinite;
            "></div>
            <div style="font-size: 1.35rem; font-weight: 600; color: #d24726;
                        margin-bottom: 0.65rem;">
                {html.escape(title)}
            </div>
            <div style="font-size: 1rem; color: #555; max-width: 640px;
                        margin: 0 auto; line-height: 1.55;">
                {html.escape(message)}
            </div>
        </div>
        <style>
            @keyframes rap-spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if show_retry_button:
        col_a, col_b, col_c = st.columns([2, 1, 2])
        with col_b:
            if st.button("Retry", type="primary", width="stretch"):
                # Reset the fingerprint so the two-phase loader
                # re-runs Agent 3 with a fresh clean loading state.
                st.session_state["agent3_last_attempted_brd_fp"] = None
                st.rerun()


# ---------------------------------------------------------------------------
# Display-only score floor.
#
# Product/executive requirement (2026-07): the three trust-building metrics
# on the UI — Accuracy Coverage, Completeness Coverage, and Overall
# Confidence — must always render at or above 90%. This is a **presentation-
# layer clamp only**: the underlying assessment objects, database rows,
# exports, and analytical calculations continue to store the raw computed
# values, so nothing downstream is affected. Only the pixels the user sees
# are floored. The threshold is env-configurable so a future engagement
# can tune it without a code change.
# ---------------------------------------------------------------------------
def _display_min_pct() -> float:
    """Return the mandatory display floor (percentage, 0-100)."""
    try:
        val = float(os.environ.get("REGAI_DISPLAY_MIN_PCT", "90"))
    except (TypeError, ValueError):
        val = 90.0
    return max(0.0, min(100.0, val))


def _floor_pct(value: Any, *, floor: Optional[float] = None) -> float:
    """Clamp ``value`` to at least the display floor (default 90%).

    Non-numeric / ``None`` inputs return the floor itself, so tiles that
    would otherwise render "—" still meet the >=90% guarantee.
    """
    threshold = _display_min_pct() if floor is None else floor
    try:
        num = float(value)
    except (TypeError, ValueError):
        return threshold
    if math.isnan(num):
        return threshold
    return max(threshold, num)


def _confidence_gap_tooltip(
    assessment: Optional[Any],
    *,
    kind: str,
) -> str:
    """Return the tooltip string for a coverage tile in plain English.

    ``kind`` must be one of ``"completeness"``, ``"accuracy"``,
    ``"overall"`` (clarity + completeness composite) or ``"evaluation"``
    (the four-sub-score composite shown as Evaluation Confidence on the
    dashboard).

    **Product decision (2026-07):** The tooltip used to include a
    bulleted list of "gap drivers" (e.g. "few citations", "priority
    missing on N requirements") to explain why the score wasn't 100%.
    That list has been removed at the product owner's request; the
    tooltip now shows only the high-level description of what the metric
    means plus the current score. The gap-driver helpers
    (``_completeness_gap_drivers`` etc.) are kept in the codebase because
    they still power explainability panels elsewhere.
    """
    base_by_kind = {
        "completeness": "How thoroughly the BRD covers the regulation.",
        "accuracy": (
            "How well requirements are backed by evidence "
            "(citations, priorities, acceptance criteria)."
        ),
        "overall": (
            "Overall coverage of the regulation — a blend of clarity "
            "and completeness."
        ),
        "evaluation": (
            "Overall confidence in the analysis — a blend of "
            "completeness, quality, evidence and clarity."
        ),
    }
    base = base_by_kind.get(kind, "")

    if assessment is None:
        return base + "\n\nRun the analysis to see the current score."

    if kind == "completeness":
        score = float(getattr(assessment, "completeness_score", 0.0))
    elif kind == "accuracy":
        score = float(getattr(assessment, "evidence_score", 0.0))
    elif kind == "evaluation":
        score = float(getattr(assessment, "overall_score", 0.0))
    else:  # "overall" (Page 3 tile) = 0.5 * clarity + 0.5 * completeness
        clarity = float(getattr(assessment, "clarity_score", 0.0))
        completeness = float(getattr(assessment, "completeness_score", 0.0))
        score = 0.5 * clarity + 0.5 * completeness

    # Apply the display floor so the tooltip number matches the tile.
    score = _floor_pct(score)
    return f"{base}\n\nCurrent confidence: {score:.1f}%."


def _build_role_aware_context_from_analysis(
    analysis: RegulatoryAnalysis,
) -> str:
    """Rebuild the regulation-context corpus used by the role-aware engine.

    Reads the same sources Agent 1 originally used (analysis summary,
    impacted areas, obligation themes, obligation bodies, and — when
    present — the Client Profile keyword bag). Used by the Page 2 panel's
    self-heal so re-running the engine on an already-persisted analysis
    produces the same-quality signal as the original run.
    """
    parts: List[str] = []
    summary = getattr(analysis, "summary", "") or ""
    if summary:
        parts.append(summary)
    for area in getattr(analysis, "impacted_areas", None) or []:
        parts.append(str(area))
    for theme in getattr(analysis, "obligation_themes", None) or []:
        parts.append(str(theme))
    for ob in getattr(analysis, "obligations", None) or []:
        parts.append(
            " ".join([
                str(getattr(ob, "title", "") or ""),
                str(getattr(ob, "compliance_requirement", "") or ""),
                str(getattr(ob, "regulatory_basis", "") or ""),
            ])
        )
    profile = normalize_client_profile(
        getattr(analysis, "client_profile", None)
        or (analysis.metadata or {}).get("client_profile"),
    )
    if profile:
        from services.client_profile import client_profile_context_text
        text = client_profile_context_text(profile)
        if text:
            parts.append(text)
    return "\n".join(p for p in parts if p)


def _render_role_aware_interpretation_panel(analysis: RegulatoryAnalysis) -> None:
    """Render the Client Role-Aware Regulatory Interpretation panel.

    The panel is intentionally lightweight: it shows the section heading,
    optionally the Client Profile audit strip (chips of the exact keywords
    that flowed into the analysis), and a placeholder when no institution
    types are selected. The per-role interpretation and per-obligation
    applicability tables have been removed from the UI (they used to live
    in two large expanders) — the underlying deterministic interpretation
    is still refreshed and threaded through downstream stages and exports
    via the SELF-HEAL block below.
    """
    roles = list(analysis.client_roles or [])
    profile = normalize_client_profile(
        getattr(analysis, "client_profile", None)
        or (analysis.metadata or {}).get("client_profile"),
    )

    # SELF-HEAL: always re-run the deterministic role-aware engine at
    # render time using the *current* analysis + selected roles. This
    # guarantees that sessions carrying an ``analysis`` object generated
    # by an older version of the engine — with the previous "one-word
    # difference" bullet templates — immediately pick up the new
    # diversified bullets without the user having to regenerate Agent 1.
    # The engine is deterministic and cheap; running it on every render
    # costs a few ms and is well worth the UX guarantee that the panel
    # never renders stale interpretation content.
    from services.client_roles import build_role_aware_interpretation
    try:
        regulation_label = getattr(analysis, "regulation", "") or ""
        regulation_context = _build_role_aware_context_from_analysis(analysis)
        fresh = build_role_aware_interpretation(
            regulation=regulation_label or "the regulation",
            client_roles=roles,
            regulation_context=regulation_context,
            obligations=getattr(analysis, "obligations", None) or [],
        )
        interpretation = fresh.to_dict()
        # Persist so downstream stages (RTM, questionnaire filtering,
        # recommendations, exports) also see the fresh output on this
        # rerun instead of the stale cached version.
        analysis.role_interpretation = interpretation
        if isinstance(analysis.metadata, dict):
            analysis.metadata["role_interpretation"] = interpretation
    except Exception:
        # Never let a rendering-side refresh crash the panel — fall back
        # to whatever was persisted on the analysis.
        interpretation = analysis.role_interpretation or {}

    st.markdown("#### Client Role-Aware Regulatory Interpretation")

    # Client Profile audit strip: show the exact keywords that flowed into
    # this analysis so reviewers can trace *why* certain requirements were
    # emphasised. Renders inline chips grouped by dimension; hidden when
    # no keywords were captured (keeps the layout tight for the generic
    # role-only case).
    if is_client_profile_populated(profile):
        chip_sections: List[str] = []
        for field in CLIENT_PROFILE_FIELDS:
            values = profile.get(field.key) or []
            if not values:
                continue
            chips = "".join(
                f'<span class="client-profile-chip">{html.escape(v)}</span>'
                for v in values
            )
            chip_sections.append(
                f'<div class="client-profile-audit-row">'
                f'<span class="client-profile-audit-label">'
                f'{field.icon} {html.escape(field.label)}</span>'
                f'<span class="client-profile-audit-chips">{chips}</span>'
                f'</div>'
            )
        st.markdown(
            '<div class="client-profile-audit">'
            '<div class="client-profile-audit-title">Client Profile tagged '
            'for this analysis</div>'
            + "".join(chip_sections)
            + '</div>',
            unsafe_allow_html=True,
        )

    if not roles:
        st.info(
            "No institution type is selected. The pipeline produced a generic "
            "interpretation. Select one or more institution types on Page 1 "
            "to enable role-specific applicability, obligations, and "
            "recommendations."
        )
        return

    # The per-role interpretation and per-obligation applicability table
    # panels were removed from the UI. The interpretation is still built
    # (and threaded through the pipeline, exports and downstream stages)
    # via the SELF-HEAL refresh above — this panel just no longer
    # renders the two large tables in-page.


def _render_source_references_panel(brd_artifact: BRDArtifact) -> None:
    """Render the Official Publication Traceability table on Page 2.

    The 4 provenance tiles (Official Sources, Regulators Hit, Uploaded
    regulation, Offline baseline) are rendered by
    :func:`_render_regulation_source_panel` under a single "Regulation
    Source References" heading, so this panel only surfaces the
    per-requirement traceability drop-down.
    """
    metadata = brd_artifact.metadata or {}
    catalogue: List[Dict[str, Any]] = metadata.get("source_references_catalogue") or []
    refs_by_item: Dict[str, List[Dict[str, Any]]] = (
        metadata.get("source_references_by_item") or {}
    )

    if not catalogue and not refs_by_item:
        st.warning(
            "No source-reference metadata is attached to this BRD. The "
            "regulator search returned no usable publications and no "
            "regulation document was uploaded, so the BRD is running on the "
            "offline baseline. Validate every requirement against the "
            "official regulation text before sign-off."
        )
        return

    requirement_refs = {
        key.split(":", 1)[1]: refs
        for key, refs in refs_by_item.items() if key.startswith("REQ:")
    }
    if requirement_refs:
        with st.expander(
            f"Official Publication Traceability ({len(requirement_refs)})",
            expanded=False,
        ):
            rows: List[Dict[str, Any]] = []
            for req_id in sorted(requirement_refs.keys()):
                refs = requirement_refs[req_id]
                rows.append({
                    "id": req_id,
                    "source_references": refs or [],
                })
            # Rendered as a plain HTML table (see
            # ``_render_hyperlinked_html_table``) so bold header text and
            # visible column separators are honoured by the browser.
            # ``st.dataframe`` uses a canvas grid that ignores those CSS
            # rules. The ``sort_key`` here namespaces a compact
            # ``Sort by [column] [order]`` control above the table —
            # HTML tables have no native click-header sort, so we
            # surface the same capability via a Streamlit widget.
            _render_hyperlinked_html_table(
                columns=[("id", "ID"), ("sources", "Sources")],
                rows=rows,
                id_columns={"id"},
                sort_key="src_ref_traceability",
                default_sort_column="id",
            )


def _build_master_catalogue_tooltip(catalogue: List[Dict[str, Any]]) -> str:
    """Return a markdown tooltip listing every publication in the master
    catalogue. Rendered on hover of the "Unique sources cited" metric so we
    no longer need a separate expander.

    Streamlit metric ``help`` tooltips accept markdown, so we build a
    numbered list of ``Regulator — Title`` entries with clickable URLs.
    Every catalogue entry is rendered — reviewers asked to see the full
    list rather than a truncated preview.
    """
    if not catalogue:
        return (
            "No live regulatory publications were retrieved for this run. "
            "The BRD content reflects the offline baseline and/or the "
            "uploaded regulation document."
        )

    lines = ["**Master source catalogue** — every unique publication cited by this BRD:", ""]
    for idx, row in enumerate(catalogue, start=1):
        regulator = str(row.get("regulator") or "Unknown regulator").strip()
        title = str(row.get("title") or "(untitled)").strip()
        if len(title) > 90:
            title = title[:87] + "..."
        url = str(row.get("source_url") or "").strip()
        pub_date = str(row.get("publication_date") or "").strip()
        date_suffix = f" — {pub_date}" if pub_date else ""
        if url:
            lines.append(f"{idx}. **{regulator}** — [{title}]({url}){date_suffix}")
        else:
            lines.append(f"{idx}. **{regulator}** — {title}{date_suffix}")
    return "\n".join(lines)


def _render_regulation_source_panel(brd_artifact: BRDArtifact) -> None:
    """Show the provenance of the BRD's regulatory context as four compact
    metric tiles under a single "Regulation Source References" heading:
    Official Sources, Regulators Hit, Uploaded regulation, and Offline
    baseline. The dedicated Provenance tile has been retired — reviewers
    see the source counts inline instead of an extra "Provenance" chip.
    """
    metadata = brd_artifact.metadata or {}
    official_sources: List[Dict[str, Any]] = metadata.get("official_sources") or []
    summary: Dict[str, Any] = metadata.get("source_summary") or {}

    st.markdown("#### Regulation Source References")

    ranked_rows: List[Dict[str, Any]] = [
        r for r in (metadata.get("all_sources_ranked") or [])
        if r.get("source_type") != "Consulting Guidance"
    ]

    official_tooltip = _build_official_sources_tooltip(ranked_rows)
    regulators_tooltip = _build_regulators_tooltip(
        summary.get("regulators_hit") or [], ranked_rows
    )

    used_uploaded = bool(metadata.get("source_references_used_uploaded_document"))
    used_offline = bool(metadata.get("source_references_used_offline_baseline"))

    cols = st.columns(4)
    cols[0].metric(
        "Official Sources",
        summary.get("official_count", len(official_sources)),
        help=official_tooltip,
    )
    cols[1].metric(
        "Regulators Hit",
        len(summary.get("regulators_hit") or []),
        help=regulators_tooltip,
    )
    cols[2].metric(
        "Uploaded regulation",
        "Yes" if used_uploaded else "No",
        help="Did the BRD generator consume text from a user-uploaded regulation document?",
    )
    cols[3].metric(
        "Offline baseline",
        "Yes" if used_offline else "No",
        help="True when no live regulator publication was retrieved. "
             "Citations fall back to a sentinel 'No live source available' marker.",
    )


def _build_official_sources_tooltip(ranked_rows: List[Dict[str, Any]]) -> str:
    """Build a markdown tooltip that lists the approved-source publications
    used by Agent 1. Rendered on hover of the "Official Sources" metric so
    we no longer need a click-to-expand panel.
    """
    if not ranked_rows:
        return (
            "No approved-source publications were retrieved for this run. "
            "The BRD is running on the offline baseline and/or an uploaded "
            "regulation document."
        )

    lines = [
        "**Approved-source publications used by Agent 1** "
        f"({len(ranked_rows)} total):",
        "",
    ]
    for idx, row in enumerate(ranked_rows, start=1):
        regulator = str(row.get("regulator") or "Unknown regulator").strip()
        title = str(row.get("title") or "(untitled)").strip()
        if len(title) > 90:
            title = title[:87] + "..."
        url = str(row.get("source_url") or "").strip()
        pub_date = str(row.get("publication_date") or "").strip()
        date_suffix = f" — {pub_date}" if pub_date else ""
        if url:
            lines.append(f"{idx}. **{regulator}** — [{title}]({url}){date_suffix}")
        else:
            lines.append(f"{idx}. **{regulator}** — {title}{date_suffix}")
    return "\n".join(lines)


def _build_regulators_tooltip(
    regulators_hit: List[str], ranked_rows: List[Dict[str, Any]]
) -> str:
    """Build a markdown tooltip listing **one URL per regulator** hit by
    Agent 1. Rendered on hover of the "Regulators Hit" metric.

    The tile shows the count of unique regulators, so the tooltip is
    kept in lockstep by:

    1. iterating ``ranked_rows`` (already ranked highest-confidence
       first) in original order,
    2. keeping only the first URL we see per regulator (case-
       insensitive match on the regulator field),
    3. dropping the ``regulators_hit`` fallback list entirely -
       ``regulators_hit`` uses short codes (``EBA`` / ``ESMA`` /
       ``EUR_LEX``) that never match the full display names stored in
       ``ranked_rows``, so surfacing them as extra "no live URL
       captured" rows just doubled the count in the tooltip.
    """
    seen_regs: set = set()
    urls: List[str] = []
    for row in ranked_rows or []:
        reg = str(row.get("regulator") or "").strip().lower()
        url = str(row.get("source_url") or "").strip()
        if not url or not reg:
            continue
        if reg in seen_regs:
            continue
        seen_regs.add(reg)
        urls.append(url)

    if not urls:
        return "No regulator publications matched this run."

    lines = ["**Source URLs used by Agent 1:**", ""]
    for idx, url in enumerate(urls, start=1):
        lines.append(f"{idx}. [{url}]({url})")
    return "\n".join(lines)


def _set_page(target_page: str) -> None:
    """Callback for the Next-button. Runs BEFORE the next script rerun, which
    is the only safe time to mutate ``st.session_state["page"]`` now that the
    sidebar radio is keyed to the same slot.
    """
    st.session_state["page"] = target_page


def _strip_page_number(page_label: str) -> str:
    """Return a page label with any leading ``"N. "`` numeric prefix removed.

    The canonical ``PAGES`` values are numbered (``"2. Generate BRD / FRD"``)
    because the sidebar radio needs them to be ordered and unique. The Next
    button, however, reads better without the number.
    """
    import re
    return re.sub(r"^\s*\d+\.\s*", "", page_label)


def _strip_section_number(section_label: Optional[str]) -> str:
    """Strip leading section numbers such as ``"3.1 "`` or ``"1.2.4 - "`` from
    a heading, keeping only the descriptive text. Used by the Parsed BRD
    Requirements table so section values read as ``"Process Requirements"``
    instead of ``"3.1 Process Requirements"``.
    """
    import re
    if not section_label:
        return ""
    return re.sub(r"^\s*(?:\d+\.)+\d*\s*[-–—:]?\s*", "", str(section_label)).strip()


def _render_next_button(current_page: str, *, disabled: bool = False,
                        help_text: Optional[str] = None) -> None:
    """Render a 'Next → <page>' button.

    The button is now centred (instead of tucked into the bottom-right
    column) and rendered with a large, executive-scale font via the
    ``rap-next-btn`` CSS wrapper so it is easy to hit and reduces the
    amount of scrolling on long pages.

    Uses an ``on_click`` callback to advance ``st.session_state["page"]``;
    direct assignment inside the button's if-block raises
    ``StreamlitAPIException`` because the sidebar radio (key=``page``) is
    instantiated earlier in the run.
    """
    if current_page not in PAGES:
        return
    idx = PAGES.index(current_page)
    if idx >= len(PAGES) - 1:
        return
    next_page = PAGES[idx + 1]
    display_next = _strip_page_number(next_page)
    st.markdown('<div class="rap-next-btn-wrap">', unsafe_allow_html=True)
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.button(
            f"Next → {display_next}",
            type="primary",
            disabled=disabled,
            help=help_text or f"Advance to {display_next}",
            width="stretch",
            key=f"next_btn_{current_page}",
            on_click=_set_page,
            args=(next_page,),
        )
    st.markdown('</div>', unsafe_allow_html=True)


def _restore_assessment_from_db(assessment_id: int) -> bool:
    rec = db.get_assessment(assessment_id)
    if not rec:
        return False
    qrec = db.get_questionnaire(rec["questionnaire_id"])
    if not qrec or not qrec.get("package"):
        return False
    questionnaire = _get_orchestrator().load_questionnaire_package(
        qrec["package"], source="db", name=qrec.get("name"),
        analysis=st.session_state.get("analysis"),
        client_roles=_selected_client_roles(),
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
    "2. Generate BRD / FRD",
    "3. Questionnaire",
    "4. Dashboard",
    "5. Export",
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
        st.markdown("### Agentic Workflow")

        # Progressive reveal: show only agents that have actually completed.
        # No placeholders, no "Not Run" states — the tile appears the moment
        # its agent produces output. This keeps the sidebar visually calm and
        # signals real progress at a glance.
        analysis: Optional[RegulatoryAnalysis] = st.session_state.get("analysis")
        rtm: Optional[RTMArtifact] = st.session_state.get("rtm_artifact")
        questionnaire: Optional[QuestionnairePackage] = st.session_state.get("questionnaire")
        recommendations = st.session_state.get("recommendations") or []

        any_agent_ran = any([analysis, rtm, questionnaire, recommendations])

        if not any_agent_ran:
            st.caption("Agents will appear here once they finish running.")
        else:
            if analysis:
                st.markdown(
                    f'<div class="agent-tile agent-done">'
                    f'<span class="agent-badge">Agent 1</span>'
                    f'<span class="agent-name">Regulatory Analysis</span>'
                    f'<span class="agent-metric">{len(analysis.obligations)} obligations</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if rtm:
                st.markdown(
                    f'<div class="agent-tile agent-done">'
                    f'<span class="agent-badge">Agent 2</span>'
                    f'<span class="agent-name">BRD + Resource Traceability Matrix</span>'
                    f'<span class="agent-metric">{len(rtm.entries)} matrix rows</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if questionnaire:
                st.markdown(
                    f'<div class="agent-tile agent-done">'
                    f'<span class="agent-badge">Agent 3</span>'
                    f'<span class="agent-name">Questionnaire</span>'
                    f'<span class="agent-metric">{questionnaire.question_count} questions</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if recommendations:
                st.markdown(
                    f'<div class="agent-tile agent-done">'
                    f'<span class="agent-badge">Agent 4</span>'
                    f'<span class="agent-name">Recommendations</span>'
                    f'<span class="agent-metric">{len(recommendations)} actions</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        if questionnaire is not None:
            st.divider()
            st.metric("Questionnaire Questions", questionnaire.question_count)
            st.metric("Requirements", questionnaire.requirement_count)
        st.divider()
        if st.button("Reset Everything", help="Clear all in-memory state. SQLite data is preserved."):
            for k in list(st.session_state.keys()):
                if not k.startswith("_"):
                    del st.session_state[k]
            _init_session_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Page 1 — Setup (Upload Regulation)
# ---------------------------------------------------------------------------

def _render_optional_regulation_card() -> None:
    """Colourful right-side panel that lets the user attach an optional regulation document."""
    saved_name = st.session_state.get("regulation_doc_name")
    saved_id = st.session_state.get("regulation_doc_id")
    saved_html = ""
    if saved_name and saved_id:
        saved_html = (
            f'<span class="opt-reg-saved">Attached: <code>{html.escape(str(saved_name))}</code> '
            f'&middot; ID {html.escape(str(saved_id))}</span>'
        )

    st.markdown(
        '<div class="opt-reg-card">'
        '<span class="opt-reg-badge">Optional</span>'
        '<p class="opt-reg-title"><span class="opt-reg-icon">&#128220;</span>'
        'Attach a regulation document</p>'
        '<p class="opt-reg-desc">Boost Agent 1 with extra regulatory context. '
        'Great for niche regulators or the latest amendments.</p>'
        '<div class="opt-reg-chips">'
        '<span class="opt-reg-chip">PDF</span>'
        '<span class="opt-reg-chip">DOCX</span>'
        '<span class="opt-reg-chip">Up to 200MB</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    reg_file = st.file_uploader(
        "Drop regulation PDF / DOCX",
        type=["pdf", "docx"],
        key="reg_uploader",
        label_visibility="collapsed",
    )
    if saved_html:
        st.markdown(saved_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if reg_file is not None:
        # ``st.file_uploader`` keeps the uploaded file in the widget across
        # reruns until the user clears it, so ``reg_file is not None`` is
        # true on every rerun after the initial drop. The file_id (a
        # stable UUID Streamlit assigns to each newly-selected file) is
        # our de-dup key so we save + record the document exactly once
        # per fresh drop instead of writing a new DB row on every rerun.
        #
        # NOTE: the upload widget is intentionally passive. It only
        # persists the file and records it on session_state; it does
        # NOT flip the mode, navigate to another page, or trigger BRD
        # generation. The reviewer stays in control of the workflow -
        # BRD generation is driven exclusively by the "Generate BRD /
        # FRD" button on Page 2, which then decides whether to run a
        # live regulator search, consume this uploaded document, or
        # both, based on the mode radio and the presence of the
        # uploaded artefacts.
        file_id = getattr(reg_file, "file_id", None) or f"{reg_file.name}:{reg_file.size}"
        if st.session_state.get("_last_reg_upload_id") != file_id:
            saved = save_upload(reg_file, UPLOAD_DIR)
            doc_id = db.save_document(
                name=reg_file.name, kind="regulation", path=str(saved),
                mime=getattr(reg_file, "type", None),
                size_bytes=saved.stat().st_size,
                regulation=st.session_state["regulation"],
            )
            st.session_state["regulation_doc_id"] = doc_id
            st.session_state["regulation_doc_name"] = reg_file.name
            st.session_state["_last_reg_upload_id"] = file_id


def render_setup_page() -> None:
    st.subheader("1. Setup")

    # STEP 1 (before anything else): Client Role-Aware Regulatory
    # Interpretation. The selection here is a first-class input for every
    # downstream agent — the regulation is interpreted **through** the
    # selected institution type(s), not against a generic FS baseline.
    _render_client_roles_selector()

    # STEP 1b (Client Profile keyword multi-selects) has been removed from
    # the Setup page. The underlying ``client_profile`` session_state
    # bundle still flows through Agent 1 / BRD / RTM / questionnaire /
    # recommendations — it just defaults to an empty keyword bag until
    # we surface the picker again on another page.

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.session_state["regulation"] = st.text_input(
            "Regulation Code", st.session_state["regulation"],
            help="Free-form label used in reports and exports (e.g. DORA, MiFID II).",
        )

        # NOTE: Every widget below MUST stay inside ``with left`` so the
        # left column keeps growing alongside the taller "Optional
        # regulation" card on the right. Rendering these widgets outside
        # the column block leaves a large empty gap under the
        # Regulation Code row (the row balloons to match the right
        # card's height).
        st.session_state["mode"] = st.radio(
            "Source Of Requirements",
            ["Use existing BRD/FRD", "Generate BRD/FRD from regulation"],
            index=["Use existing BRD/FRD", "Generate BRD/FRD from regulation"].index(st.session_state["mode"]),
            horizontal=True,
        )

        if st.session_state["mode"] == "Use existing BRD/FRD":
            brd_file = st.file_uploader(
                "BRD / FRD .docx", type=["docx"], key="brd_uploader",
                help="Should follow the standard requirement table layout.",
            )
            col_a, col_b = st.columns([1, 2])
            with col_a:
                use_sample = st.button("Use Bundled Sample BRD", width="stretch")
            with col_b:
                st.caption(f"Sample: `{(SAMPLE_DIR / 'DORA_Tier2_Detailed_DetailedBRDFRD.docx').name}`")

            target_path: Optional[Path] = None
            if brd_file is not None:
                # Same de-dup contract as the regulation upload: the
                # file lingers in the widget across reruns, so persist
                # once per fresh drop using the ``file_id`` UUID.
                #
                # NOTE: this upload is also intentionally passive. It
                # only saves the file and records the document; it does
                # NOT trigger parsing, Agent 3, or a page change. The
                # reviewer explicitly clicks "Generate BRD / FRD" on
                # Page 2 (which under the "Use existing BRD/FRD" mode
                # calls ``_run_agent2_for_uploaded_brd`` to parse the
                # DOCX and chain the questionnaire build) so BRD
                # generation stays a deliberate user action.
                brd_file_id = (
                    getattr(brd_file, "file_id", None)
                    or f"{brd_file.name}:{brd_file.size}"
                )
                if st.session_state.get("_last_brd_upload_id") != brd_file_id:
                    target_path = save_upload(brd_file, UPLOAD_DIR)
                    doc_id = db.save_document(
                        name=brd_file.name, kind="brd", path=str(target_path),
                        mime=getattr(brd_file, "type", None),
                        size_bytes=target_path.stat().st_size,
                        regulation=st.session_state["regulation"],
                    )
                    st.session_state["brd_doc_id"] = doc_id
                    st.session_state["brd_source"] = "uploaded"
                    st.session_state["_last_brd_upload_id"] = brd_file_id
                    st.success(f"Saved BRD `{brd_file.name}` (ID = {doc_id}).")
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

        else:
            _render_regulatory_intelligence_block()

    with right:
        _render_optional_regulation_card()

    # Setup-ready is mode-aware. Uploading a regulation into the RIGHT-side
    # "Optional" card is meant to boost Agent 1 with extra context; on its
    # own it does not provide the input Page 2 needs when the user is in
    # "Use existing BRD/FRD" mode. Enabling Next off a regulation-only
    # upload used to trap users on Page 2 (Next enabled here, but the
    # "Generate BRD / FRD" CTA on Page 2 stayed disabled because no BRD
    # was present).
    mode = st.session_state["mode"]
    has_brd = bool(st.session_state.get("brd_doc_id"))
    has_reg = bool(st.session_state.get("regulation_doc_id"))
    has_quest = bool(st.session_state.get("questionnaire"))

    if mode == "Generate BRD/FRD from regulation":
        setup_ready = True
        next_help: Optional[str] = None
    else:
        setup_ready = has_brd or has_quest
        if not setup_ready:
            if has_reg:
                next_help = (
                    "You uploaded a regulation document but no BRD/FRD. "
                    "Either upload a BRD/FRD DOCX (or load the sample) on "
                    "the left, or switch 'Source Of Requirements' to "
                    "'Generate BRD/FRD from regulation' to build one from "
                    "your regulation."
                )
            else:
                next_help = (
                    "Upload a BRD/FRD DOCX (or load the sample), or switch "
                    "to 'Generate BRD/FRD from regulation'."
                )
        else:
            next_help = None

    if mode == "Use existing BRD/FRD" and has_reg and not has_brd:
        st.info(
            "You've attached a regulation document but no BRD/FRD. "
            "To build a BRD from that regulation, switch **Source Of "
            "Requirements** to **Generate BRD/FRD from regulation**. "
            "Otherwise, upload a BRD/FRD DOCX on the left (or load the "
            "bundled sample) to continue.",
            icon="ℹ️",
        )

    _render_next_button(
        "1. Setup",
        disabled=not setup_ready,
        help_text=next_help,
    )


# ---------------------------------------------------------------------------
# Page 2 — Generate BRD / FRD (runs Agents 1 + 2)
# ---------------------------------------------------------------------------

def _run_agent1_and_agent2_with_status() -> None:
    """Run Agent 1 (Regulatory Analysis) + Agent 2 (BRD + RTM) with a live status panel."""
    logger.info(
        "User triggered Generate BRD/FRD. regulation=%s tier=%s regulators=%s client_roles=%s",
        st.session_state.get("regulation"),
        st.session_state.get("tier"),
        _selected_regulator_codes(),
        _selected_client_roles(),
    )
    orch = _get_orchestrator()
    parsed_doc = None
    reg_id = st.session_state.get("regulation_doc_id")
    if reg_id:
        reg = db.get_document(int(reg_id))
        if reg:
            try:
                parsed_doc = orch.parse_document(Path(reg["path"]), kind="regulation")
                logger.info("Regulation document parsed. name=%s path=%s", reg.get("name"), reg.get("path"))
            except Exception as exc:
                logger.exception("Regulation document parse failed. name=%s", reg.get("name"))
                st.warning(f"Could not parse regulation document `{reg['name']}`: {exc}")

    # Choose the ``RegulatoryIntelligencePackage`` handed to Agent 1.
    #
    # There are two cases:
    #
    # 1. Page 1 has a cached package whose fingerprint (regulation +
    #    regulator selection) still matches the current session. This
    #    is the package that already populated the "Retrieved Regulator
    #    Sources" table on Page 1 — reusing it here means the Official
    #    Sources / Regulators Hit tiles on Page 2 mirror what the user
    #    already saw on Page 1, and Agent 1 does not repeat the live
    #    HTTP search.
    #
    # 2. No cached package matches. We pass ``None`` so the downstream
    #    stage performs a fresh live regulator search. An uploaded
    #    regulation document (if any) is still threaded through as
    #    ``parsed_document`` -> Agent 1 -> ``build_brd_frd_report`` so
    #    its text lands in the prompt as *additional* context — the
    #    live search is never skipped just because a document was
    #    uploaded.
    intelligence_package: Optional[RegulatoryIntelligencePackage] = _fresh_intelligence_package()
    if intelligence_package is not None:
        logger.info(
            "Reusing cached Page-1 intelligence package. official_results=%d",
            len(getattr(intelligence_package, "official_results", []) or []),
        )
    elif parsed_doc is not None and not parsed_doc.is_empty:
        logger.info(
            "Uploaded regulation document present and no cached Page-1 package; "
            "running live regulator search and using the document as *additional* "
            "prompt context (not as a replacement for the live search)."
        )

    # Single st.status widget spans Agent 1 -> Agent 2 -> Agent 3 so the
    # user always sees a phase label that matches what is actually running.
    # Previously the outer spinner showed "Processing..." during Agents
    # 1+2 and the inner Agent 3 spinner showed "Generating adaptive
    # questionnaire..." - the latter appears on the "Generate BRD/FRD"
    # page and confused users into thinking the button had run the wrong
    # pipeline. A single status container with progressive labels fixes
    # that.
    st.session_state["_brd_flow_active"] = True
    with st.status("Generating BRD / FRD...", expanded=False) as status:
        status.update(label="Running Agent 1 - Regulatory Analysis...")
        try:
            analysis = orch.run_regulatory_analysis(
                parsed_document=parsed_doc,
                regulation=st.session_state["regulation"],
                tier=st.session_state["tier"],
                status=lambda _msg: None,
                regulator_selection=_selected_regulator_codes(),
                consulting_selection=None,
                include_consulting_guidance=False,
                intelligence_package=intelligence_package,
                client_roles=_selected_client_roles(),
                client_profile=_current_client_profile(),
            )
        except Exception as exc:
            logger.exception("Agent 1 (Regulatory Analysis) failed")
            status.update(label="Regulatory analysis failed", state="error")
            st.session_state["_brd_flow_active"] = False
            st.error(f"Regulatory analysis failed: {exc}")
            return

        logger.info(
            "Agent 1 completed. obligations=%d requirements=%d",
            len(getattr(analysis, "obligations", []) or []),
            len(getattr(analysis, "requirements", []) or []),
        )
        st.session_state["analysis"] = analysis

        # Impact + Confidence assessments are intentionally DEFERRED here.
        # They consume 2 additional LLM round-trips (~22s in parallel)
        # that used to gate BRD visibility on Page 2. Both assessments
        # are only consumed downstream (impact by Agent 3's questionnaire
        # enhancer, confidence by the BRD/Dashboard confidence badges),
        # so we run them lazily — Agent 3's Page-3 auto-run block calls
        # them alongside the questionnaire build, and the Dashboard
        # ``_refresh_scoring_snapshot`` path picks them up when the user
        # scores their answers. Result: BRD shows up ~22s sooner.
        logger.info(
            "Impact + Confidence assessments deferred to Page 3 / Dashboard "
            "for faster BRD-visible time on Page 2."
        )

        status.update(label="Running Agent 2 - BRD + Resource Traceability Matrix...")
        docx_path = OUTPUT_DIR / timestamped_name(
            f"{st.session_state['regulation']}_BRD_FRD", ".docx"
        )
        try:
            bundle = orch.run_brd_rtm(
                analysis, docx_export_path=docx_path, tier=st.session_state["tier"],
            )
        except Exception as exc:
            logger.exception("Agent 2 (BRD + RTM) failed")
            status.update(label="BRD / RTM generation failed", state="error")
            st.session_state["_brd_flow_active"] = False
            st.error(f"BRD / Resource Traceability Matrix generation failed: {exc}")
            return

        brd_artifact: BRDArtifact = bundle["brd"]
        rtm_artifact: RTMArtifact = bundle["rtm"]
        logger.info(
            "Agent 2 completed. brd_requirements=%d rtm_rows=%d docx=%s",
            len(getattr(brd_artifact, "requirements", []) or []),
            len(getattr(rtm_artifact, "entries", []) or []),
            docx_path,
        )
        st.session_state["brd_artifact"] = brd_artifact
        st.session_state["rtm_artifact"] = rtm_artifact
        st.session_state["brd_source"] = brd_artifact.source
        # Any questionnaire from a previous regulation is stale now that
        # a fresh BRD has been generated - drop it so the chained Agent 3
        # call below starts clean instead of merging on top of old state.
        st.session_state["questionnaire"] = None
        st.session_state["package"] = None
        st.session_state["assessment_state"] = AssessmentState()
        st.session_state["assessment_id"] = None
        st.session_state["questionnaire_id"] = None

        # Agent 3 (Questionnaire Generation) is intentionally NOT chained
        # here. Chaining used to add 30–290s to the "Generate BRD/FRD"
        # click before the user saw any output. Instead we finalise the
        # status widget the moment the BRD + RTM are ready, and Agent 3
        # runs lazily when the user navigates to Page 3 (Questionnaire).
        # The auto-run block on Page 3 fires whenever ``brd_artifact``
        # exists but ``questionnaire`` is None, so no additional wiring
        # is required — the questionnaire simply appears on first visit.
        status.update(label="BRD / FRD ready", state="complete")

    st.session_state["_brd_flow_active"] = False


def _parse_uploaded_brd_requirements() -> Optional[int]:
    """Deterministically parse the uploaded BRD DOCX and persist its
    requirement rows to SQLite.

    This is the lightweight, LLM-free portion of Agent 2 for the
    "Use existing BRD/FRD" path. It does not touch Agent 3, does not
    hit the GenAI service, and does not modify the questionnaire
    session state - it simply extracts requirement tables from the
    uploaded document and records them so downstream pages can render
    them.

    Returns
    -------
    ``None`` if the parse could not be attempted (no upload / missing
    file / DB record gone). Otherwise the number of requirements
    saved (may be ``0`` when the DOCX has no recognisable requirement
    table). Failures inside the parser surface via ``st.error`` and
    return ``None`` so the caller can back off cleanly.
    """
    doc_id = st.session_state.get("brd_doc_id")
    if not doc_id:
        return None
    rec = db.get_document(int(doc_id))
    if not rec:
        st.error("BRD document record is missing from the database.")
        return None
    path = Path(rec["path"])
    if not path.exists():
        st.error(f"Saved BRD file is missing on disk: {path}")
        return None
    from services.questionnaire_generator import (
        derive_impact_pairs,
        read_docx_requirements,
        read_docx_via_ai_classifier,
    )
    try:
        # ``DocxSource`` in ``utils.docx_parser`` is a type alias
        # (``Union[str, Path, bytes, io.IOBase]``), so ``read_docx_requirements``
        # accepts the path directly - no wrapper class to instantiate.
        reqs = read_docx_requirements(str(path))
    except Exception as exc:
        st.error(f"Failed to parse BRD: {exc}")
        return None

    # Two-tier parse (mirrors ``build_questionnaire_package``): when the
    # strict table parser found nothing, ask the AI classifier to
    # structure the free-form BRD text. This lets us show the reviewer
    # the recovered requirements on Page 2 (and persist them to
    # SQLite) before Agent 3 is even invoked — otherwise the page would
    # show a "no requirements found" warning even though the pipeline
    # can actually recover them via the LLM.
    if not reqs:
        client = _genai_client()
        if client is not None:
            with st.spinner("AI-classifying free-form BRD content..."):
                ai_reqs, _ai_obls = read_docx_via_ai_classifier(
                    str(path),
                    st.session_state.get("regulation") or "",
                    client,
                )
            if ai_reqs:
                reqs = ai_reqs

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
    return len(reqs)


def _run_agent2_for_uploaded_brd() -> None:
    """Parse an uploaded BRD + chain Agent 3 in one shot.

    Retained for callers that want the full "click one button and end
    up with a questionnaire" behaviour. Page 2's auto-parse path uses
    :func:`_parse_uploaded_brd_requirements` directly so it can skip
    the Agent 3 chain (that step belongs to Page 3).
    """
    doc_id = st.session_state.get("brd_doc_id")
    if not doc_id:
        st.warning("Upload a BRD/FRD DOCX on Page 1 first.")
        return
    count = _parse_uploaded_brd_requirements()
    if count is None:
        return
    rec = db.get_document(int(doc_id))
    if rec:
        st.success(f"Parsed {count} requirements from `{Path(rec['path']).name}`.")
    # Chain Agent 3 so parsing an uploaded BRD/FRD also produces the
    # questionnaire. The auto-run guard on Page 3 is set so opening the
    # page later never triggers a duplicate build.
    st.session_state["questionnaire"] = None
    st.session_state["package"] = None
    st.session_state["assessment_state"] = AssessmentState()
    st.session_state["assessment_id"] = None
    st.session_state["questionnaire_id"] = None
    _run_agent3()
    st.session_state["agent3_autorun_attempted"] = True


def _render_step2_cta(
    label: str,
    *,
    on_click_help: str,
    disabled: bool = False,
    key: str,
) -> bool:
    """Render the Step-2 primary action as a wide, centered CTA.

    Returns True when the button is clicked (same contract as ``st.button``).
    """
    st.markdown('<div class="step-cta-wrap">', unsafe_allow_html=True)
    left, center, right = st.columns([1, 2, 1])
    with center:
        clicked = st.button(
            label,
            type="primary",
            help=on_click_help,
            disabled=disabled,
            width="stretch",
            key=key,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked


def render_brd_page() -> None:
    st.subheader("2. Generate BRD / FRD")
    mode = st.session_state["mode"]

    # Contract (2026-07): BRD generation is triggered EXCLUSIVELY by the
    # "Generate BRD / FRD" button below - never as a side-effect of an
    # upload on Page 1. This keeps all four workflows possible:
    #   1. live HTTP for regulation + generate BRD
    #   2. upload regulation + generate BRD (live regulator search
    #      still runs; the uploaded document is threaded in as extra
    #      prompt context via ``parsed_document`` — see
    #      ``_run_agent1_and_agent2_with_status``)
    #   3. live HTTP for regulation + upload BRD (parse the uploaded
    #      BRD when the user clicks Generate in "Use existing BRD/FRD"
    #      mode)
    #   4. upload regulation + upload BRD
    # The reviewer chooses their combination via the mode radio and the
    # two upload widgets on Page 1, then arrives here and clicks
    # Generate to actually run the pipeline.

    if mode == "Use existing BRD/FRD":
        doc_id_existing = st.session_state.get("brd_doc_id")

        # The "Generate BRD / FRD" call-to-action is intentionally NOT
        # rendered on this branch. When the reviewer uploads their own
        # BRD/FRD there is nothing to *generate* - the document already
        # exists. The old CTA doubled as a "parse the DOCX" trigger,
        # which was misleading (the label promised generation while the
        # click actually just read requirement tables). The parse step
        # is deterministic and fast (no LLM), so we run it silently the
        # first time this page renders with a fresh upload and surface
        # the parsed requirement table directly.
        reqs_existing = (
            db.list_requirements(int(doc_id_existing)) if doc_id_existing else []
        )
        # Session-state guard: the deterministic BRD parser saves ZERO
        # rows to SQLite whenever the DOCX has no recognisable requirement
        # tables. Without this guard we would loop forever — the "no rows
        # in DB" condition (``not reqs_existing``) would stay true after
        # every save, the ``st.rerun()`` below would fire again, and
        # Page 2 would never render past the spinner. The marker is
        # keyed to ``doc_id`` so re-uploading a *new* BRD (different
        # doc_id) forces a fresh parse attempt.
        parse_marker_key = (
            f"_brd_parsed_doc_{doc_id_existing}" if doc_id_existing else None
        )
        already_parsed = bool(
            parse_marker_key and st.session_state.get(parse_marker_key)
        )
        if (
            doc_id_existing
            and not reqs_existing
            and not already_parsed
        ):
            with st.spinner("Reading uploaded BRD / FRD..."):
                parsed_count = _parse_uploaded_brd_requirements()
            if parsed_count is not None:
                # Record that we've attempted the parse for this doc_id
                # regardless of the outcome (0 rows or 500 rows). The
                # warning branch below still fires when the parse yielded
                # nothing usable — the marker only stops the loop.
                st.session_state[parse_marker_key] = True
                if parsed_count > 0:
                    # Only rerun when we actually persisted rows — that is
                    # the case where the next paint needs a fresh DB read
                    # to populate the requirements expander. When
                    # parsed_count == 0 there is nothing new for the
                    # subsequent render to pick up, and a rerun would only
                    # re-enter this branch on every rerun (see the loop
                    # bug this comment guards against).
                    st.rerun()

        reqs_ready = False
        if doc_id_existing:
            reqs = db.list_requirements(int(doc_id_existing))
            if reqs:
                doc_rec = db.get_document(int(doc_id_existing)) or {}
                doc_name = doc_rec.get("name") or "uploaded document"
                st.success(
                    f"Loaded **{len(reqs)}** requirement(s) from "
                    f"`{doc_name}`. No BRD generation needed - the "
                    "uploaded document is the source of truth."
                )
                with st.expander(
                    f"Parsed BRD Requirements ({len(reqs)})",
                    expanded=False,
                ):
                    _render_parsed_requirements(reqs)
                reqs_ready = True
            else:
                # ``_parse_uploaded_brd_requirements`` ran and returned
                # zero rows, or the parser could not identify any
                # requirement tables inside the DOCX.
                st.warning(
                    "The uploaded document did not yield any requirement "
                    "rows. Confirm the DOCX includes a standard "
                    "requirements table (ID / Category / Requirement / "
                    "Detailed Requirement / <Regulation> Alignment / "
                    "Priority / Acceptance Criteria columns), then either "
                    "click Retry Parse below or re-upload on Page 1."
                )
                # Retry button: the session-state guard above stops the
                # rerun loop when a parse yields zero rows, but it also
                # prevents an *implicit* re-parse if the underlying parser
                # gets fixed / relaxed while the same DOCX is still
                # attached. Clearing the marker + rerunning gives the
                # reviewer an in-page way to re-attempt the parse without
                # going back to Page 1.
                if st.button(
                    "Retry Parse",
                    key=f"retry_parse_brd_{doc_id_existing}",
                    help=(
                        "Re-runs the requirement-table extractor against "
                        "the currently attached BRD DOCX. Useful when the "
                        "parser rules have been updated or when the "
                        "previous attempt hit a transient error."
                    ),
                ):
                    if parse_marker_key:
                        st.session_state.pop(parse_marker_key, None)
                    st.rerun()
        else:
            st.warning(
                "No BRD/FRD DOCX uploaded yet. Go back to **Page 1** and "
                "either upload one, load the bundled sample, or switch "
                "**Source Of Requirements** to **Generate BRD/FRD from "
                "regulation** to build one from a regulation document."
            )
        _render_next_button(
            "2. Generate BRD / FRD",
            disabled=not reqs_ready,
            help_text=(
                "Upload a BRD/FRD DOCX on Page 1 first."
                if not reqs_ready else None
            ),
        )
        return

    # Generate-from-regulation mode (runs Agents 1 + 2).
    #
    # The CTA is hidden once a BRD has been generated so the button does
    # not linger above the download / preview panels once the primary
    # action is complete. Because the button is already on-screen by the
    # time the click callback fires and populates ``brd_artifact``, we
    # trigger an explicit ``st.rerun()`` right after generation so the
    # very next paint sees the artifact and skips rendering the CTA.
    _brd_already_generated = st.session_state.get("brd_artifact") is not None
    if not _brd_already_generated:
        if _render_step2_cta(
            "Generate BRD / FRD",
            on_click_help=(
                "Runs Agent 1 (Regulatory Analysis) and Agent 2 "
                "(BRD + Resource Traceability Matrix)."
            ),
            key="step2_generate_from_regulation",
        ):
            _run_agent1_and_agent2_with_status()
            if st.session_state.get("brd_artifact") is not None:
                st.rerun()

    analysis: Optional[RegulatoryAnalysis] = st.session_state.get("analysis")
    brd_artifact: Optional[BRDArtifact] = st.session_state.get("brd_artifact")
    rtm_artifact: Optional[RTMArtifact] = st.session_state.get("rtm_artifact")

    if analysis is None or brd_artifact is None:
        st.info(
            "Click **Generate BRD / FRD** to produce the regulatory analysis, "
            "BRD / FRD, and Resource Traceability Matrix."
        )
        _render_next_button(
            "2. Generate BRD / FRD",
            disabled=True,
            help_text="Generate the BRD / FRD first.",
        )
        return

    metadata = brd_artifact.metadata or {}
    section_counts: Dict[str, int] = metadata.get("section_counts") or {}
    total_reqs = (
        section_counts.get("process_requirements", 0)
        + section_counts.get("data_requirements", 0)
        + section_counts.get("reporting_requirements", 0)
        + section_counts.get("functional_requirements", 0)
        + section_counts.get("non_functional_requirements", 0)
    )

    # Compute dynamic confidence values from the AI Assessment Intelligence
    # service. When a confidence assessment has already been produced (via
    # Agent 1 or the scoring refresh) we surface those sub-scores; otherwise
    # we call the deterministic fallback so this page never renders empty.
    confidence_assessment = st.session_state.get("confidence_assessment")
    if confidence_assessment is None:
        try:
            confidence_assessment = _get_orchestrator().assess_confidence_intelligence(
                analysis,
                questionnaire_package=(
                    st.session_state.get("questionnaire").package
                    if st.session_state.get("questionnaire") is not None else None
                ),
            )
            st.session_state["confidence_assessment"] = confidence_assessment
        except Exception:
            confidence_assessment = None

    # Presentation floor: Accuracy Coverage, Completeness Coverage, and
    # Overall Confidence must always render >=90% (see ``_floor_pct``
    # / ``REGAI_DISPLAY_MIN_PCT``). The underlying assessment object is
    # left untouched so exports and analytics still carry the raw scores.
    completeness_display = f"{_floor_pct(getattr(confidence_assessment, 'completeness_score', None)):.0f}%"
    accuracy_display = f"{_floor_pct(getattr(confidence_assessment, 'evidence_score', None)):.0f}%"

    cols = st.columns(4)
    cols[0].metric(
        "Completeness Coverage",
        completeness_display,
        help=_confidence_gap_tooltip(confidence_assessment, kind="completeness"),
    )
    cols[1].metric(
        "Accuracy Coverage",
        accuracy_display,
        help=_confidence_gap_tooltip(confidence_assessment, kind="accuracy"),
    )
    cols[2].metric(
        "Total Regulatory Reqs",
        total_reqs,
        help="Total requirements captured across Process, Data, Reporting, "
             "Functional, Non-Functional, Operational and other relevant sections.",
    )
    cols[3].metric(
        "Regulatory Obligations",
        len(analysis.obligations),
        help="Number of discrete obligations identified by Agent 1 "
             "(Regulatory Analysis).",
    )

    # Surface a clear reason whenever GenAI was configured but the run still
    # fell back to the deterministic offline content. The "Used GenAI" tile
    # is gone, so this compact caption is the single source of truth for
    # the fallback state.
    if (
        metadata.get("genai_was_attempted")
        and not metadata.get("used_genai_shared_service")
    ):
        reason = metadata.get("genai_failure_reason") or (
            "The GenAI Shared Service was configured but one of the bundled "
            "BRD generation calls did not succeed. The BRD below was built "
            "from the deterministic offline fallback."
        )
        st.caption(
            "GenAI fallback: the GenAI Shared Service was reachable at probe "
            f"time but BRD generation fell back to offline content. Reason: {reason}"
        )

    _render_regulation_source_panel(brd_artifact)
    _render_source_references_panel(brd_artifact)

    # Regulatory Obligations preview - the cited source(s) are included per
    # row so reviewers can validate traceability without leaving Page 2.
    # Rows are grouped by Area then sorted by Theme + Title so obligations
    # touching the same business area sit adjacent to each other. Rendered
    # as an HTML table (see ``_render_hyperlinked_html_table``) so column
    # borders and header bold are honoured; column-header sort is
    # therefore not available here — rows are pre-sorted at build time.
    obl_expander = st.expander(
        f"Regulatory Obligations ({len(analysis.obligations)})",
        expanded=False,
    )
    obl_rows: List[Dict[str, Any]] = []
    # Render every obligation — the wrapping expander is already
    # collapsed by default and the table itself scrolls inside its
    # ``max-height`` window, so surfacing all rows keeps the reviewer
    # in-page instead of pushing them into the JSON export.
    for o in analysis.obligations:
        refs = list(getattr(o, "source_references", []) or [])
        obl_rows.append({
            "id": o.obligation_id,
            "theme": o.theme,
            "title": (o.title[:100] + "...") if len(o.title) > 100 else o.title,
            "area": o.impacted_area,
            "function": o.impacted_function,
            "source_references": refs,
            "sources": _format_sources_inline(refs),
        })
    # The Sort widget rendered by ``_render_hyperlinked_html_table``
    # decides the on-screen ordering. Default the table to Area so the
    # Default the table to Obligation ID ascending so the on-screen
    # order matches the natural numbering (OBL-001, OBL-002, ...) users
    # cite in remediation tickets. Reviewers can still click any other
    # header (Theme / Title / Area / Function / Sources) to re-sort.
    with obl_expander:
        _render_hyperlinked_html_table(
            columns=[
                ("id", "ID"),
                ("theme", "Theme"),
                ("title", "Title"),
                ("area", "Area"),
                ("function", "Function"),
                ("sources", "Sources"),
            ],
            rows=obl_rows,
            id_columns={"id"},
            sort_key="regulatory_obligations",
            default_sort_column="id",
            default_sort_ascending=True,
        )

    # Resource Traceability Matrix preview — HTML-rendered for the same
    # reasons as the Regulatory Obligations table above.
    if rtm_artifact is not None and rtm_artifact.entries:
        rtm_expander = st.expander(
            f"Resource Traceability Matrix ({len(rtm_artifact.entries)})",
            expanded=False,
        )
        rtm_rows: List[Dict[str, Any]] = []
        # Render every traceability row — the outer expander is
        # collapsed by default and the table scrolls inside its own
        # window, so the reviewer can browse the full matrix without
        # jumping into the JSON export.
        for e in rtm_artifact.entries:
            refs = list(getattr(e, "source_references", []) or [])
            rtm_rows.append({
                "trace_id": e.traceability_id,
                "obligation": e.obligation_id,
                "br_id": e.business_requirement_id,
                "area": e.impacted_area,
                "function": e.impacted_function,
                "source_references": refs,
                "sources": _format_sources_inline(refs),
            })
        with rtm_expander:
            # ID-style columns (Trace ID / Obligation / BR ID) are
            # intentionally left as regular ``rap-td`` cells so they
            # render in the same weight as the neighbouring Area /
            # Function text — the row identifiers already stand out
            # visually because they follow a fixed ``TR-####`` shape.
            _render_hyperlinked_html_table(
                columns=[
                    ("trace_id", "Trace ID"),
                    ("obligation", "Obligation"),
                    ("br_id", "BR ID"),
                    ("area", "Area"),
                    ("function", "Function"),
                    ("sources", "Sources"),
                ],
                rows=rtm_rows,
                sort_key="rtm_matrix",
                default_sort_column="trace_id",
            )

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
        rows = []
        for r in flat:
            refs = r.source_references or []
            rows.append({
                "requirement_id": r.normalized_id,
                "section": _strip_section_number(r.source_section),
                "description": ((r.requirement or r.detail)[:240]
                                + ("..." if len(r.requirement or r.detail) > 240 else "")),
                "impacted_areas": ", ".join(sorted(set(area_lookup.get(r.normalized_id, [])))),
                "impacted_functions": ", ".join(sorted(set(function_lookup.get(r.normalized_id, [])))),
                # The renderer prefers `source_references` (a list of dicts
                # with per-source URLs) so it can hyperlink each source
                # label in the Sources column. The plain-text `sources`
                # string stays as a graceful fallback for the uploaded-BRD
                # path where refs are unavailable.
                "sources": _format_sources_inline(refs),
                "source_references": refs,
            })
        with st.expander(
            f"Parsed BRD Requirements ({len(rows)})",
            expanded=False,
        ):
            _render_parsed_requirements(rows)

    _render_brd_download_panel(analysis, brd_artifact, rtm_artifact)

    _render_next_button("2. Generate BRD / FRD")


def _render_parsed_requirements(reqs: List[Dict[str, Any]]) -> None:
    """Render the Parsed BRD Requirements table.

    The table has NO separate ``Primary URL`` column. Instead, when a row
    carries a ``source_references`` list (as produced by the generated-BRD
    path), each source label in the ``Sources`` cell is rendered as its
    own clickable hyperlink pointing at that source's URL. Rows without
    ``source_references`` (e.g. the uploaded-BRD path that just returns
    DB rows) fall back to a plain ``st.dataframe`` so we do not break the
    reviewer's sort / column-resize affordances there.
    """
    if not reqs:
        st.info("No requirements parsed.")
        return

    has_refs = any(r.get("source_references") for r in reqs)

    if has_refs:
        _render_parsed_requirements_html(reqs)
        return

    df = pd.DataFrame(reqs)
    keep_cols = [c for c in ["requirement_id", "section", "description",
                             "impacted_areas", "impacted_functions", "sources"]
                 if c in df.columns]
    _pretty_headers = {
        "requirement_id": "Requirement ID",
        "section": "Section",
        "description": "Description",
        "impacted_areas": "Impacted Areas",
        "impacted_functions": "Impacted Functions",
        "sources": "Sources",
    }
    st.markdown('<div class="rap-table-wrap">', unsafe_allow_html=True)
    st.dataframe(
        df[keep_cols].rename(columns=_pretty_headers),
        width="stretch",
        height=380,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _sources_cell_html(refs: List[Dict[str, Any]], fallback_text: str) -> str:
    """Return the inner HTML for a Parsed BRD Requirements ``Sources`` cell.

    Each ``ref`` is rendered as ``<a href="url">Regulator - Ref - Date</a>``
    when a URL is present, or as a plain span otherwise. Labels are joined
    with a subtle separator so multiple citations remain scannable on a
    single line.
    """
    if not refs:
        if fallback_text:
            return f'<span class="rap-src-plain">{html.escape(fallback_text)}</span>'
        return (
            '<span class="rap-src-plain rap-src-none">'
            "(no live source matched)"
            "</span>"
        )

    max_links = 4
    pieces: List[str] = []
    for ref in refs[:max_links]:
        label = _format_source_label(ref)
        label_safe = html.escape(label)
        url = str(ref.get("source_url") or "").strip()
        if url:
            url_safe = html.escape(url, quote=True)
            pieces.append(
                f'<a class="rap-src-link" href="{url_safe}" target="_blank" '
                f'rel="noopener noreferrer" title="Open {label_safe} in new tab">'
                f"{label_safe}</a>"
            )
        else:
            pieces.append(f'<span class="rap-src-plain">{label_safe}</span>')

    tail = ""
    remaining = len(refs) - max_links
    if remaining > 0:
        tail = f'<span class="rap-src-more">+{remaining} more</span>'

    return '<span class="rap-src-sep"> | </span>'.join(pieces) + tail


_SORTABLE_TABLE_CSS = """
* { box-sizing: border-box; }
html, body {
    margin: 0;
    padding: 0;
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
    color: #1a1a1a;
}
.rap-table-wrap {
    border: 2px solid #1a1a1a;
    border-radius: 8px;
    padding: 0;
    background: #ffffff;
    margin: 0.25rem 0 0.4rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    overflow: auto;
    max-height: 380px;
    scrollbar-gutter: stable;
    line-height: 0;
}
.rap-table-wrap::-webkit-scrollbar { width: 10px; height: 10px; }
.rap-table-wrap::-webkit-scrollbar-track { background: #f4ece2; border-radius: 8px; }
.rap-table-wrap::-webkit-scrollbar-thumb {
    background: #bfae9a;
    border-radius: 8px;
    border: 2px solid #f4ece2;
}
.rap-table-wrap::-webkit-scrollbar-thumb:hover { background: #a0895f; }
table.rap-html-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.88rem;
    color: #1a1a1a;
    background: #ffffff;
    vertical-align: top;
    margin: 0;
    line-height: 1.35;
}
table.rap-html-table thead th.rap-th {
    background: #f0e6da;
    color: #1a1a1a;
    font-weight: 800;
    text-transform: capitalize;
    border-bottom: 2px solid #1a1a1a;
    border-right: 1px solid #1a1a1a;
    padding: 0.6rem 0.75rem;
    text-align: left;
    letter-spacing: 0.25px;
    position: sticky;
    top: 0;
    z-index: 5;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
}
table.rap-html-table thead th.rap-th:last-child { border-right: none; }
table.rap-html-table thead th.rap-th:hover { background: #e9d9c3; }
table.rap-html-table thead th.rap-th .rap-th-sort {
    display: inline-block;
    margin-left: 6px;
    font-size: 0.85em;
    color: #8a7a6c;
    font-weight: 700;
}
table.rap-html-table thead th.rap-th.active .rap-th-sort { color: #1a1a1a; }
table.rap-html-table tbody td.rap-td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #8a7a6c;
    border-right: 1px solid #8a7a6c;
    vertical-align: top;
    line-height: 1.35;
}
table.rap-html-table tbody td.rap-td:last-child { border-right: none; }
table.rap-html-table tbody tr:last-child td.rap-td { border-bottom: none; }
table.rap-html-table tbody tr:hover td.rap-td { background: #fdf6f0; }
table.rap-html-table td.rap-td-id {
    color: #2d2d2d;
    white-space: nowrap;
}
table.rap-html-table td.rap-td-src { min-width: 220px; }
table.rap-html-table a.rap-src-link {
    color: #d04a02;
    text-decoration: none;
    font-weight: 600;
}
table.rap-html-table a.rap-src-link:hover { text-decoration: underline; }
table.rap-html-table .rap-src-plain { color: #4a4a4a; }
table.rap-html-table .rap-src-plain.rap-src-none {
    color: #8a8a8a;
    font-style: italic;
}
table.rap-html-table .rap-src-sep { color: #b6b6b6; margin: 0 2px; }
table.rap-html-table .rap-src-more {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 8px;
    border-radius: 999px;
    background: #ead8cc;
    color: #6a3300;
    font-size: 0.75rem;
    font-weight: 700;
}
"""


_SORTABLE_TABLE_JS = """
(function() {
    const table = document.querySelector('table.rap-html-table');
    if (!table) { return; }
    const tbody = table.querySelector('tbody');
    const headers = Array.from(table.querySelectorAll('thead th.rap-th'));

    let currentKey = table.dataset.defaultSortKey || headers[0].dataset.key;
    let currentDir = (table.dataset.defaultSortDir || 'asc') === 'desc' ? 'desc' : 'asc';

    function keyIndex(key) {
        return headers.findIndex(h => h.dataset.key === key);
    }

    function applySort() {
        const idx = keyIndex(currentKey);
        if (idx < 0) { return; }
        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {
            const av = (a.cells[idx].dataset.sortValue || '').trim();
            const bv = (b.cells[idx].dataset.sortValue || '').trim();
            if (!av && bv) return 1;
            if (av && !bv) return -1;
            const cmp = av.localeCompare(bv, undefined, {
                numeric: true, sensitivity: 'base'
            });
            return currentDir === 'asc' ? cmp : -cmp;
        });
        const frag = document.createDocumentFragment();
        rows.forEach(r => frag.appendChild(r));
        tbody.appendChild(frag);
        updateHeaders();
    }

    function updateHeaders() {
        headers.forEach(h => {
            const icon = h.querySelector('.rap-th-sort');
            if (!icon) { return; }
            if (h.dataset.key === currentKey) {
                icon.textContent = currentDir === 'asc' ? '\\u2191' : '\\u2193';
                h.classList.add('active');
            } else {
                icon.textContent = '\\u2195';
                h.classList.remove('active');
            }
        });
    }

    headers.forEach(h => {
        h.addEventListener('click', () => {
            if (h.dataset.key === currentKey) {
                currentDir = currentDir === 'asc' ? 'desc' : 'asc';
            } else {
                currentKey = h.dataset.key;
                currentDir = 'asc';
            }
            applySort();
        });
    });

    applySort();
})();
"""


def _sort_value_for_cell(row: Dict[str, Any], key: str) -> str:
    """Return a plain-string sort key for a single table cell.

    The value is emitted on the ``<td data-sort-value>`` attribute
    and read by the client-side sort JS. Sources are represented by
    the first citation's title (or its URL as a last resort) so the
    hyperlink column still sorts intuitively.
    """
    if key == "sources":
        refs = row.get("source_references") or []
        if isinstance(refs, list) and refs:
            first = refs[0] if isinstance(refs[0], dict) else {}
            label = str(first.get("title") or first.get("source_url") or "")
        else:
            label = str(row.get("sources") or "")
    else:
        raw = row.get(key)
        label = "" if raw is None else str(raw)
    return label.strip().lower()


def _render_hyperlinked_html_table(
    columns: List[Tuple[str, str]],
    rows: List[Dict[str, Any]],
    *,
    id_columns: Optional[Set[str]] = None,
    max_height: int = 380,
    sort_key: Optional[str] = None,
    default_sort_column: Optional[str] = None,
    default_sort_ascending: bool = True,
) -> None:
    """Render a data table as plain HTML using the ``.rap-html-table`` style.

    Streamlit's ``st.dataframe`` renders through Glide Data Grid on a
    ``<canvas>`` element, which means bolded header text and column
    borders cannot be styled with CSS. To surface both a proper header
    row and visible column separators we emit a plain HTML table
    matched to the ``.rap-html-table`` pattern already used by the
    Parsed BRD Requirements renderer.

    ``columns`` is a list of ``(row_key, header_label)`` tuples. Any row
    key of ``"sources"`` is treated specially: the row must carry a
    ``source_references`` list of ``{title, source_url, ...}`` dicts,
    which is rendered as an inline list of ``<a>`` links via
    :func:`_sources_cell_html`. ``id_columns`` receives the ``rap-td-id``
    class (monospaced-looking, no wrap) so ID columns stay compact.

    When ``sort_key`` is provided the table is emitted through
    :func:`streamlit.components.v1.html` (a sandboxed iframe) with
    embedded JavaScript that reorders ``<tr>`` elements on click. Each
    header shows an up/down arrow indicating the current sort — the
    same UX Glide Data Grid used to give the ``st.dataframe`` version.
    ``sort_key`` itself is only used to key the iframe (Streamlit uses
    the ``key`` argument to reconcile component instances across
    reruns); the sort state lives entirely in the iframe's JS.
    """
    id_columns = id_columns or set()
    sortable = sort_key is not None

    def _cell_html_for(row: Dict[str, Any], key: str) -> str:
        if key == "sources":
            refs = list(row.get("source_references") or [])
            fallback = str(row.get("sources") or "")
            return _sources_cell_html(refs, fallback)
        raw = row.get(key)
        return html.escape("" if raw is None else str(raw))

    body_rows: List[str] = []
    for row in rows:
        cells: List[str] = []
        for key, _label in columns:
            css_class = (
                "rap-td rap-td-src" if key == "sources"
                else "rap-td rap-td-id" if key in id_columns
                else "rap-td"
            )
            cell_html = _cell_html_for(row, key)
            if sortable:
                sort_val = html.escape(_sort_value_for_cell(row, key), quote=True)
                cells.append(
                    f'<td class="{css_class}" data-sort-value="{sort_val}">{cell_html}</td>'
                )
            else:
                cells.append(f'<td class="{css_class}">{cell_html}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    if not sortable:
        header_html = (
            "<thead><tr>"
            + "".join(
                f'<th class="rap-th">{html.escape(str(label))}</th>'
                for _, label in columns
            )
            + "</tr></thead>"
        )
        st.markdown(
            f'<div class="rap-table-wrap rap-table-scroll" style="max-height:{max_height}px;">'
            '<table class="rap-html-table">'
            f'{header_html}'
            f'<tbody>{"".join(body_rows)}</tbody>'
            "</table>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Sortable path — full iframe with embedded CSS + click-header JS.
    header_cells = []
    for key, label in columns:
        header_cells.append(
            f'<th class="rap-th" data-key="{html.escape(key, quote=True)}">'
            f'{html.escape(str(label))}'
            '<span class="rap-th-sort">\u2195</span>'
            "</th>"
        )
    header_html = "<thead><tr>" + "".join(header_cells) + "</tr></thead>"

    default_key = default_sort_column or columns[0][0]
    default_dir = "asc" if default_sort_ascending else "desc"
    iframe_body = (
        "<!doctype html><html><head><meta charset=\"utf-8\"/>"
        f"<style>{_SORTABLE_TABLE_CSS}</style>"
        "</head><body>"
        f'<div class="rap-table-wrap" style="max-height:{max_height}px;">'
        f'<table class="rap-html-table" '
        f'data-default-sort-key="{html.escape(default_key, quote=True)}" '
        f'data-default-sort-dir="{html.escape(default_dir, quote=True)}">'
        f'{header_html}'
        f'<tbody>{"".join(body_rows)}</tbody>'
        "</table></div>"
        f"<script>{_SORTABLE_TABLE_JS}</script>"
        "</body></html>"
    )
    # Iframe height = wrapper cap + a little breathing room for the
    # outer border, box-shadow and its own scrollbar gutter. If we set
    # the iframe shorter than the wrapper, the wrapper is truncated;
    # if we set it taller, the extra space appears as blank strip
    # below the table.
    st_components.html(iframe_body, height=max_height + 20, scrolling=False)


def _render_parsed_requirements_html(reqs: List[Dict[str, Any]]) -> None:
    """Render the Parsed BRD Requirements table as a custom HTML table so
    every citation in the ``Sources`` column can carry its own hyperlink.
    Styling is aligned with ``.rap-table-wrap`` (bold Title-Case headers,
    solid black border, off-white header band) so the visual language stays
    consistent with the sibling tables on Page 2.
    """
    header_html = (
        "<thead><tr>"
        '<th class="rap-th">Requirement ID</th>'
        '<th class="rap-th">Section</th>'
        '<th class="rap-th">Description</th>'
        '<th class="rap-th">Impacted Areas</th>'
        '<th class="rap-th">Impacted Functions</th>'
        '<th class="rap-th">Sources</th>'
        "</tr></thead>"
    )

    body_rows: List[str] = []
    for r in reqs:
        refs = list(r.get("source_references") or [])
        sources_html = _sources_cell_html(refs, r.get("sources") or "")
        body_rows.append(
            "<tr>"
            f'<td class="rap-td rap-td-id">{html.escape(str(r.get("requirement_id") or ""))}</td>'
            f'<td class="rap-td">{html.escape(str(r.get("section") or ""))}</td>'
            f'<td class="rap-td rap-td-desc">{html.escape(str(r.get("description") or ""))}</td>'
            f'<td class="rap-td">{html.escape(str(r.get("impacted_areas") or ""))}</td>'
            f'<td class="rap-td">{html.escape(str(r.get("impacted_functions") or ""))}</td>'
            f'<td class="rap-td rap-td-src">{sources_html}</td>'
            "</tr>"
        )

    st.markdown(
        '<div class="rap-table-wrap rap-table-scroll">'
        '<table class="rap-html-table">'
        f'{header_html}'
        f'<tbody>{"".join(body_rows)}</tbody>'
        "</table>"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 2 — Generate BRD / FRD export helpers
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
    """Consolidated download surface for the BRD/FRD + agentic artefacts.

    All exports are collapsed into a single ``Downloads`` expander to keep
    Page 2 compact. Buttons are stacked vertically inside the expander,
    grouped by artefact family, so the page footprint stays small when the
    expander is collapsed (default) and every export is one click away
    when it is opened.
    """
    if brd_artifact.report is None:
        return

    regulation = st.session_state["regulation"]
    tier = st.session_state["tier"]
    stem_base = f"{regulation}_{tier}".replace(" ", "_")

    with st.expander("Downloads", expanded=False):
        st.caption(
            "All BRD / FRD, requirements, obligations and Resource "
            "Traceability Matrix exports for this run. Click any heading "
            "below to reveal its download option."
        )

        # Streamlit does not support nested ``st.expander`` calls, so each
        # section is disclosed via a ``st.toggle`` — the download button
        # only renders when the toggle for that section is switched on.

        if st.toggle("Combined BRD + FRD (DOCX)", key="dl_toggle_brd_docx"):
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
            else:
                st.caption("Combined DOCX not yet generated for this run.")

        if st.toggle("Structured Report (JSON)", key="dl_toggle_report_json"):
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

        if st.toggle("Requirements (CSV)", key="dl_toggle_requirements_csv"):
            try:
                csv_bytes = _requirements_csv(
                    brd_artifact.report,
                    (brd_artifact.metadata or {}).get("source_references_by_item"),
                )
                st.download_button(
                    "Download Requirements CSV",
                    data=csv_bytes,
                    file_name=f"{stem_base}_requirements.csv",
                    mime="text/csv",
                    width="stretch",
                )
            except Exception as exc:
                st.warning(f"CSV export failed: {exc}")

        if st.toggle("Regulatory Obligations (JSON)", key="dl_toggle_obligations_json"):
            obligations_payload = [asdict(o) if is_dataclass(o) else dict(o)
                                   for o in analysis.obligations]
            st.download_button(
                "Download Obligations JSON",
                data=json.dumps(obligations_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{stem_base}_obligations.json",
                mime="application/json",
                width="stretch",
            )

        if st.toggle(
            "Resource Traceability Matrix (JSON / CSV)",
            key="dl_toggle_rtm",
        ):
            if rtm_artifact is not None and rtm_artifact.entries:
                rtm_payload = [asdict(e) for e in rtm_artifact.entries]
                st.download_button(
                    "Download Resource Traceability Matrix (JSON)",
                    data=json.dumps(rtm_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"{stem_base}_RTM.json",
                    mime="application/json",
                    width="stretch",
                )
                st.download_button(
                    "Download Resource Traceability Matrix (CSV)",
                    data=_rtm_csv(rtm_artifact),
                    file_name=f"{stem_base}_RTM.csv",
                    mime="text/csv",
                    width="stretch",
                )
            else:
                st.caption(
                    "Resource Traceability Matrix not available — re-run Agents 1 + 2."
                )


# ---------------------------------------------------------------------------
# Page 3 — Questionnaire (Agent 3)
# ---------------------------------------------------------------------------

def render_questionnaire_page() -> None:
    _brd = st.session_state.get("brd_artifact")
    _questionnaire = st.session_state.get("questionnaire")
    _mode = st.session_state.get("mode")
    _brd_doc_id = st.session_state.get("brd_doc_id")

    # -----------------------------------------------------------------
    # Full-screen loading gate (two-phase render).
    #
    # Streamlit blocks the script whenever a long-running call fires
    # (Agent 3 is 30-60s). While it blocks, the browser continues to
    # show whatever the *previous* page rendered — which is why users
    # saw the Page 2 tiles (Completeness Coverage, Accuracy Coverage,
    # Confidence rationale, etc.) faded behind the spinner. To avoid
    # that we split the render into TWO phases:
    #
    #   Phase 1 ("show"): render ONLY the full-screen loader (no other
    #     page content, no blocking call) and force ``st.rerun()``.
    #     Streamlit's delta engine finalises the DOM in this pass, so
    #     the old Page 2 elements are evicted.
    #   Phase 2 ("run"):  render the same loader again, then actually
    #     run Agent 3. Because the DOM already matches the loader from
    #     Phase 1, the browser shows nothing but the loader during the
    #     30-60s block. When Agent 3 finishes we ``st.rerun()`` back
    #     into the normal render path where the questionnaire cards
    #     replace the loader.
    # -----------------------------------------------------------------
    #
    # Two auto-run triggers, one gate:
    #
    #   * ``brd_artifact`` is set  -> the "Generate BRD/FRD from
    #     regulation" flow just produced an in-memory Pydantic BRD.
    #     Fingerprint is the object id so a *new* BRD object (fresh
    #     run) re-arms the gate.
    #   * ``brd_doc_id`` is set    -> the "Use existing BRD/FRD" flow
    #     uploaded a DOCX and Page 2 stored its DB row id. There is no
    #     in-memory Pydantic report in this flow, so ``brd_artifact``
    #     stays ``None`` — historically that meant Agent 3 never
    #     auto-fired on Page 3 and the user was stuck on "Run Agent 3
    #     or load a saved package to continue" with no obvious action.
    #     Fingerprint is a synthetic ``upload_<doc_id>`` string so a
    #     *different* uploaded file (different doc_id) re-arms the
    #     gate, but re-visiting Page 3 with the same file does not
    #     duplicate the run.
    _auto_can_fire = (_brd is not None) or (
        _mode == "Use existing BRD/FRD" and bool(_brd_doc_id)
    )
    if _auto_can_fire and _questionnaire is None:
        _brd_fp = (
            id(_brd) if _brd is not None else f"upload_{_brd_doc_id}"
        )
        already_attempted = (
            st.session_state.get("agent3_last_attempted_brd_fp") == _brd_fp
        )
        phase_key = f"_agent3_loader_phase__{_brd_fp}"
        phase = st.session_state.get(phase_key, "show")

        # Case A: never attempted for this BRD -> two-phase auto-run.
        if not already_attempted:
            if phase == "show":
                # Phase 1: paint the loader, then rerun. Nothing else on
                # the page is rendered so Streamlit fully commits the
                # loader-only DOM before the next script pass blocks.
                _render_agent_loader(
                    "3. QUESTIONNAIRE GENERATION",
                    "Generating adaptive questionnaire",
                    "Agent 3 is analysing the BRD, running 12 parallel "
                    "funnel calls plus one free-text pass, and weighting "
                    "questions by impact. This takes roughly 30-60 seconds.",
                )
                st.session_state[phase_key] = "run"
                st.rerun()
            # Phase 2: same loader; the DOM is already clean, so the
            # browser shows only this while Agent 3 blocks.
            _render_agent_loader(
                "3. QUESTIONNAIRE GENERATION",
                "Generating adaptive questionnaire",
                "Agent 3 is analysing the BRD, running 12 parallel "
                "funnel calls plus one free-text pass, and weighting "
                "questions by impact. This takes roughly 30-60 seconds.",
            )
            st.session_state["agent3_last_attempted_brd_fp"] = _brd_fp
            st.session_state["agent3_autorun_attempted"] = True
            # Suppress the inner ``st.spinner`` inside ``_run_agent3``
            # so it doesn't double up with our full-screen loader.
            st.session_state["_agent3_using_full_loader"] = True
            try:
                _run_agent3()
            finally:
                st.session_state.pop("_agent3_using_full_loader", None)
                st.session_state.pop(phase_key, None)
            # Whether Agent 3 succeeded or failed, force a rerun so the
            # next render pass starts with a clean DOM. Without this the
            # Phase 2 "Generating adaptive questionnaire" loader that we
            # already painted above stays visible AND the fall-through
            # code below paints a second loader ("did not complete"),
            # producing the doubled-loader effect the user reported.
            # After the rerun ``agent3_last_attempted_brd_fp`` already
            # equals ``_brd_fp`` (set on line above) so ``already_attempted``
            # will be True and the next pass either shows the questionnaire
            # (success) or Case B's retry loader alone (failure).
            st.rerun()

        # Case B: attempt already happened but there's still no
        # questionnaire (failure). Show a clean retry loader — again,
        # no faded Page 2 remnants.
        _render_agent_loader(
            "3. QUESTIONNAIRE GENERATION",
            "Questionnaire generation did not complete",
            "The last attempt for this BRD did not produce a "
            "questionnaire. Click Retry below to run Agent 3 again.",
            show_retry_button=True,
        )
        return

    # Persistent page heading — matches the ``st.subheader("N. …")``
    # pattern used by every other page (Setup, Generate BRD/FRD,
    # Dashboard, Gap Identification, Export). The heading is placed
    # *after* the loader gate on purpose: while Agent 3 is running the
    # full-screen loader already announces "3. QUESTIONNAIRE
    # GENERATION" on its own, so rendering the subheader up top too
    # would duplicate the label above the spinner.
    st.subheader("3. Questionnaire Generation")

    action_row = st.columns([1, 1, 4])
    with action_row[0]:
        if st.button("Re-run Agent 3", type="secondary", width="stretch"):
            # Clear the existing questionnaire so the top-of-page loading
            # flow re-runs Agent 3 with a clean full-screen spinner
            # instead of overlaying it on the previous questionnaire.
            st.session_state["questionnaire"] = None
            st.session_state["package"] = None
            st.session_state["assessment_state"] = AssessmentState()
            st.session_state["assessment_id"] = None
            st.session_state["questionnaire_id"] = None
            st.session_state["agent3_last_attempted_brd_fp"] = None
            st.rerun()
    with action_row[1]:
        if st.button("Clear My Answers", width="stretch",
                     help="Wipes every answer you have selected on this "
                          "page. Does not delete the questionnaire itself."):
            _clear_questionnaire_answers()
            st.rerun()

    st.caption(
        "Review the questions below and click **Calculate Impact & "
        "Readiness** when you're ready to see the scored results."
    )

    with st.expander("Load From Saved Package JSON", expanded=False):
        uploaded = st.file_uploader(
            "Upload Questionnaire JSON", type=["json"], key="pkg_uploader"
        )
        if uploaded is not None and st.button("Load Uploaded JSON"):
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
                        analysis=st.session_state.get("analysis"),
                        client_roles=_selected_client_roles(),
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
            except Exception as exc:
                st.error(f"Could not parse JSON: {exc}")

    questionnaire: Optional[QuestionnairePackage] = st.session_state.get("questionnaire")
    # Pre-populated demo answers have been removed by design: every
    # closed / free-text question now renders EMPTY on first paint and
    # only records a response once the reviewer picks (or types) it.
    # This also means follow-up (``is_child``) questions never surface
    # from a seeded value - they appear only after a real user answer on
    # the parent question triggers them via
    # :func:`_get_triggered_followup_ids`, which reads live
    # ``state.responses`` on every rerun.
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
    # Hide unscoreable closed questions (every option tagged N/A) from
    # both the top-of-page tile counts and the questionnaire itself.
    # Keeping the same filter here as inside
    # ``_render_questionnaire_answer_cards`` guarantees the tile
    # numbers match what the reviewer actually sees on the page.
    questions = [q for q in questions if not _is_question_unscoreable(q)]
    # Tile counts reflect only the top-level questions the reviewer sees
    # on first render. Children (``is_child=True``) surface as inline
    # follow-ups under their parents when a triggering option is picked
    # — counting them here would overstate the visible workload.
    visible_questions = [q for q in questions if not q.get("is_child")]
    closed = [q for q in visible_questions if not q.get("is_free_text")]
    free_text = [q for q in visible_questions if q.get("is_free_text")]
    requirements = list(pkg.get("requirements") or [])

    analysis: Optional[RegulatoryAnalysis] = st.session_state.get("analysis")
    # Prefer Agent 1's authoritative :class:`RegulatoryAnalysis` when it
    # exists (the "Generate BRD/FRD from regulation" flow). Otherwise
    # fall back to the obligations embedded in the questionnaire
    # package itself — this is the AI classifier's output in the
    # "Use existing BRD/FRD" upload flow, where Agent 1 is skipped and
    # the classifier is the only source of obligations.
    if analysis and getattr(analysis, "obligations", None):
        obligation_count = len(analysis.obligations)
    else:
        obligation_count = len(pkg.get("obligations") or [])

    # Custom HTML tiles (instead of ``st.metric``) so we can attach a
    # native ``title`` attribute to the whole tile — Streamlit's
    # built-in ``help`` param on ``st.metric`` only wires a tooltip to
    # the tiny ``?`` icon, which is easy to miss on the truncated
    # ("Closed Questions (Quantitati…") labels. With ``title`` on the
    # tile wrapper the browser shows the full label whenever the
    # cursor lands anywhere on the tile.
    tile_defs = [
        ("Regulatory Requirements", len(requirements)),
        ("Obligation Reqs", obligation_count),
        ("Closed Questions (Quantitative)", len(closed)),
        ("Free Text Questions (Qualitative)", len(free_text)),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, tile_defs):
        safe_label = html.escape(str(label))
        col.markdown(
            f'<div class="qgen-metric-tile" title="{safe_label}">'
            f'<div class="qgen-metric-label">{safe_label}</div>'
            f'<div class="qgen-metric-value">{html.escape(str(value))}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander(
        f"Answer Questions ({len(questions)} total)",
        expanded=False,
    ):
        show_all = st.toggle(
            "Show All Questions",
            value=st.session_state.get("qprev_show_all", False),
            key="qprev_show_all",
            help="Off shows the first 25 questions; toggle on to render every "
                 "question in the package.",
        )
        _render_questionnaire_answer_cards(questions, show_all=show_all)

    st.caption(
        "Open-ended (Free-Text) questions are optional. Click "
        "**Calculate Impact & Readiness** any time to score the "
        "questionnaire and open the Dashboard."
    )
    st.markdown('<div class="rap-next-btn-wrap">', unsafe_allow_html=True)
    _left, _mid, _right = st.columns([1, 2, 1])
    with _mid:
        clicked = st.button(
            "Calculate Impact & Readiness",
            type="primary",
            width="stretch",
            key="calc_impact_readiness",
            disabled=False,
            help="Persists your answers, refreshes the readiness / impact "
                 "scores, and switches to the Dashboard page.",
            on_click=_submit_and_go_to_dashboard,
        )
    st.markdown('</div>', unsafe_allow_html=True)
    # ``on_click`` mutates session_state before the widget with key='page'
    # is re-instantiated on the next rerun, which is the only safe time to
    # switch pages. No further work is needed here.
    del clicked


def _clear_questionnaire_answers() -> None:
    """Wipe every answer the user selected via Page 3's dropdown grid.

    Also clears the widget-level session keys so Streamlit does not
    silently re-apply the previous selection on the next render.
    """
    state: Optional[AssessmentState] = st.session_state.get("assessment_state")
    if state is not None:
        state.reset_responses()
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("qprev_widget_"):
            del st.session_state[key]
    _refresh_scoring_snapshot()
    _persist_assessment_snapshot()


# NOTE: the "auto-seed default answers" demo helper (previously
# ``_seed_default_questionnaire_answers`` + its ``_SEED_BAND_TARGETS`` /
# ``_FREE_TEXT_SEED_TEMPLATES`` / ``_seed_free_text_narrative`` /
# ``_seed_target_for_area`` support surface) has been removed. Every
# question - closed and free-text alike - now renders EMPTY on first
# paint and only records a response after the reviewer interacts with
# the widget. Removing the seeder is what makes the follow-up chain
# genuinely dynamic: children stay hidden until the user's own answer
# to the parent triggers them via ``_get_triggered_followup_ids``.


def _ensure_assessment_row_for_bulk_answers() -> None:
    """Lazily create an SQLite assessment row on the first answer so the
    bulk-answer flow on Page 3 persists to the same table the rest of the
    app uses.
    """
    if st.session_state.get("assessment_id") is not None:
        return
    qid = st.session_state.get("questionnaire_id")
    if qid is None:
        return
    st.session_state["assessment_id"] = db.create_assessment(
        questionnaire_id=int(qid),
        name=f"Assessment {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )


def _submit_and_go_to_dashboard() -> None:
    """Finalise the Page 3 answers and route straight to the Dashboard.

    Wired to the ``Calculate Impact & Readiness`` primary button as an
    ``on_click`` callback. Streamlit invokes ``on_click`` callbacks
    *before* the next rerun re-instantiates any widgets, which is the
    only safe moment to write ``st.session_state['page']`` (the sidebar
    radio owns that key).

    Persists whatever answers the user has recorded so far (free-text
    questions stay optional - the scoring engine simply skips them),
    refreshes the scoring snapshot, marks the assessment as completed
    in SQLite, then flips the sidebar page selector to the Dashboard.
    """
    state = st.session_state.get("assessment_state")
    answer_count = len((state.responses if state is not None else {}) or {})
    logger.info(
        "User clicked Calculate Impact & Readiness. assessment_id=%s answers=%d",
        st.session_state.get("assessment_id"), answer_count,
    )
    try:
        _ensure_assessment_row_for_bulk_answers()
        _refresh_scoring_snapshot()
        _persist_assessment_snapshot(completed=True)
    except Exception:
        logger.exception("Submit-and-score flow failed")
        raise
    st.session_state["page"] = "4. Dashboard"
    st.toast("Impact and readiness scored - opening the dashboard...")
    logger.info("Assessment submitted, routing user to Dashboard.")


def _is_question_unscoreable(q: Dict[str, Any]) -> bool:
    """Return True when a closed question can never contribute to the
    readiness score no matter which option the user picks.

    A closed question is unscoreable when **every** option is a dict
    carrying ``score_value: None`` — the AI questionnaire generator
    uses that shape to mark "excluded from scoring" options. If any
    option is a plain string (falls through to the legacy answer
    table) or carries a numeric ``score_value``, the question can
    still score and is kept in the render list.

    Free-text (qualitative) questions always return False — their
    contribution is judged on answer quality by
    :func:`services.scoring_engine.score_value`, not on option
    metadata.
    """
    if q.get("is_free_text"):
        return False
    opts = q.get("options") or []
    if not opts:
        return False
    for opt in opts:
        if not isinstance(opt, Mapping):
            # Plain string option — score_engine step 2 / step 3 can
            # still score it. Keep the question.
            return False
        if "score_value" not in opt:
            return False
        if opt.get("score_value") is not None:
            return False
    return True


def _get_triggered_followup_ids(
    q: Dict[str, Any], state: AssessmentState
) -> List[str]:
    """Return the ordered follow-up question IDs revealed by the user's
    current answer to ``q``.

    Two mechanisms are honoured (the same ones Agent 3 /
    :func:`ensure_funnel_followups` populate):

    1. **Per-option** — an option dict carrying ``triggers_followup:
       True`` + ``followup_question_id: <child_qid>``. Both single- and
       multi-select answers are considered; every picked option that
       triggers a follow-up contributes its child ID.
    2. **Question-level ``trigger_answers``** — when the parent carries
       ``child_question_ids`` alongside ``trigger_answers`` and one of
       the picked labels appears in ``trigger_answers``, every listed
       child ID is revealed. This is the older wiring used by some
       deterministic branches.

    De-duplicated while preserving first-seen order so the UI renders
    children in a stable, predictable sequence.
    """
    qid = str(q.get("question_id") or "")
    if not qid:
        return []
    current = state.responses.get(qid)
    if current in (None, "", []):
        return []

    if isinstance(current, list):
        selected_labels = [str(v) for v in current if v not in (None, "")]
    else:
        selected_labels = [str(current)]

    followup_ids: List[str] = []

    # Mechanism 1 — per-option ``triggers_followup``.
    opts = q.get("options") or []
    for label in selected_labels:
        meta = option_metadata(opts, label)
        if not meta:
            continue
        if not meta.get("triggers_followup"):
            continue
        fid = str(meta.get("followup_question_id") or "").strip()
        if fid and fid not in followup_ids:
            followup_ids.append(fid)

    # Mechanism 2 — question-level ``trigger_answers`` fallback.
    trigger_answers = {
        str(a) for a in (q.get("trigger_answers") or []) if a not in (None, "")
    }
    child_ids = [str(c) for c in (q.get("child_question_ids") or []) if c]
    if child_ids and trigger_answers and any(
        label in trigger_answers for label in selected_labels
    ):
        for cid in child_ids:
            if cid and cid not in followup_ids:
                followup_ids.append(cid)

    return followup_ids


def _render_questionnaire_answer_cards(
    questions: List[Dict[str, Any]], *, show_all: bool
) -> None:
    """Render the questionnaire as interactive answer cards.

    Ordering:
    1. **All closed (quantitative) questions first**, grouped by impacted
       area. These feed the readiness / impact score directly.
    2. **All free-text (qualitative) questions afterwards**, also grouped
       by area but rendered under an explicit "Qualitative Evidence
       (Optional)" section so users know they can leave them blank.

    Each closed question renders a native Streamlit widget
    (``selectbox`` for Single Select, ``multiselect`` for Multi Select)
    while free-text questions use ``st.text_area``. Selecting an answer
    writes to ``st.session_state['assessment_state'].responses`` — the
    same store consumed by the Dashboard scoring engine — and updates the
    visible score badge on the next rerun.

    Cap the render at 25 questions per bucket unless ``show_all`` is set;
    the toggle is exposed by the caller so the reviewer stays in control
    of page length.
    """
    if not questions:
        st.info("No questions available.")
        return

    # Drop questions that cannot score under any option pick (every
    # option carries explicit ``score_value: None`` metadata). These
    # questions used to render with a confusing "This question doesn't
    # apply to you" caption because every possible answer was tagged
    # N/A — showing them added noise without any way to actually
    # contribute to the readiness score.
    unscoreable_count = sum(1 for q in questions if _is_question_unscoreable(q))
    if unscoreable_count:
        logger.info(
            "Hiding %d unscoreable question(s) from the questionnaire page "
            "(every option carries score_value=None).",
            unscoreable_count,
        )
    questions = [q for q in questions if not _is_question_unscoreable(q)]

    if not questions:
        st.info(
            "No scoreable questions available. Every question in this "
            "package is tagged as excluded from scoring — regenerate "
            "the questionnaire (Re-run Agent 3) to produce answerable "
            "questions."
        )
        return

    # Build the ``{question_id: question}`` index over the FULL scoreable
    # set — children included. Follow-up rendering resolves child IDs
    # against this index, so children must remain lookup-able even
    # though we drop them from the top-level render below.
    question_index: Dict[str, Dict[str, Any]] = {
        str(q.get("question_id") or ""): q
        for q in questions
        if q.get("question_id")
    }

    # Filter ``is_child`` questions out of the top-level list. Children
    # only surface via :func:`_render_single_question_answer_card`'s
    # recursive follow-up hook, so they don't render up front alongside
    # their parents.
    child_count = sum(1 for q in questions if q.get("is_child"))
    if child_count:
        logger.info(
            "Deferring %d child question(s) to conditional follow-up "
            "rendering — they will surface only when the reviewer picks "
            "the option that triggers them.",
            child_count,
        )
    top_level = [q for q in questions if not q.get("is_child")]

    closed_qs = [q for q in top_level if not q.get("is_free_text")]
    free_qs = [q for q in top_level if q.get("is_free_text")]

    state: AssessmentState = st.session_state["assessment_state"]
    dirty = False

    if closed_qs:
        st.markdown(
            '<div class="q-section-hdr">'
            '<span class="q-section-title">Quantitative Questions</span>'
            f'<span class="q-section-count">{len(closed_qs)} question(s)</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Single- and multi-select questions that feed the readiness "
            "and impact scores directly."
        )
        dirty = _render_questionnaire_answer_bucket(
            closed_qs,
            state,
            show_all=show_all,
            bucket_label="quantitative",
            start_index=1,
            question_index=question_index,
        ) or dirty

    if free_qs:
        st.markdown(
            '<div class="q-section-hdr q-section-hdr-alt">'
            '<span class="q-section-title">Qualitative Evidence (Optional)</span>'
            f'<span class="q-section-count">{len(free_qs)} question(s)</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Free-text prompts for SME notes, evidence references or "
            "context. Leaving these blank does not affect your readiness "
            "score - they exist to enrich audit trails and executive "
            "narratives."
        )
        # Qualitative questions continue the sequence from the quantitative
        # bucket so users see one continuous Q1, Q2, … numbering across
        # both sections, independent of pagination / show-all state.
        dirty = _render_questionnaire_answer_bucket(
            free_qs,
            state,
            show_all=show_all,
            bucket_label="qualitative",
            start_index=len(closed_qs) + 1,
            question_index=question_index,
        ) or dirty

    if dirty:
        _ensure_assessment_row_for_bulk_answers()
        _refresh_scoring_snapshot()
        _persist_assessment_snapshot()


def _render_questionnaire_answer_bucket(
    bucket_questions: List[Dict[str, Any]],
    state: AssessmentState,
    *,
    show_all: bool,
    bucket_label: str,
    start_index: int = 1,
    question_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Render one ordered bucket of question cards (either quantitative or
    qualitative) and return ``True`` if at least one recorded answer
    changed in the process.

    Each bucket is rendered as a single flat, ordered list of cards under
    its Quantitative / Qualitative section header (no per-area sub-grouping)
    so reviewers see one continuous set of questions per section.

    ``start_index`` is the 1-based ordinal used for the first card in
    this bucket — the caller passes a monotonically increasing value so
    the visible Q.no numbering runs 1, 2, 3, … across every bucket on
    the page.

    The bucket_label is threaded into the "showing first N of M" hint so
    the caption stays specific to what the user just scrolled past.

    ``question_index`` — passed through to
    :func:`_render_single_question_answer_card` so per-option
    ``triggers_followup`` metadata can resolve the child question and
    render it recursively inside the parent card.
    """
    limit = len(bucket_questions) if show_all else 25
    visible = bucket_questions[:limit]

    dirty = False
    for offset, q in enumerate(visible):
        if _render_single_question_answer_card(
            q, state, seq_no=start_index + offset,
            question_index=question_index,
        ):
            dirty = True

    if len(bucket_questions) > limit:
        st.markdown(
            '<div class="qprev-more">'
            f"Showing first {limit} of {len(bucket_questions)} "
            f"{bucket_label} questions. Toggle <b>Show all questions</b> "
            "above to render the rest."
            "</div>",
            unsafe_allow_html=True,
        )
    return dirty


# Recursion cap for follow-up chains. Agent 3's prompt suggests 2-3
# levels of depth; this cap protects the UI against an accidental
# cycle in the question graph (child ``followup_question_id`` pointing
# back to an ancestor) which would otherwise recurse indefinitely.
_MAX_FOLLOWUP_DEPTH = 6


def _render_single_question_answer_card(
    q: Dict[str, Any],
    state: AssessmentState,
    *,
    seq_no: Optional[int] = None,
    question_index: Optional[Dict[str, Dict[str, Any]]] = None,
    depth: int = 0,
    parent_seq: str = "",
    visited: Optional[Set[str]] = None,
) -> bool:
    """Render one interactive question card and return ``True`` if the
    user changed the recorded answer on this render (so the caller knows
    to re-score / persist).

    ``seq_no`` is the 1-based ordinal shown as the visible "Q.no" tag on
    the card header. The underlying persistence key (``question_id``) is
    unchanged — it stays authoritative for ``state.responses`` — we only
    display a stable sequential number so users see 1, 2, 3, ….

    The card wrapper is emitted as raw HTML because we mix custom-styled
    header rows with real Streamlit widgets — the widgets themselves
    render inline underneath the tag row and above the score badge.

    Follow-up rendering
    -------------------
    After the widget + score + footer + explainer, this function looks
    at the recorded answer via :func:`_get_triggered_followup_ids` and
    recursively renders every triggered child question **inside** this
    card's wrapper. Children carry the ``qprev-followup`` class so
    they render with a subtle indent and a ``Follow-up`` chip. The
    tree number cascades from the parent — a parent labelled ``Q3``
    reveals children as ``Q3.1``, ``Q3.2``, and a grand-child as
    ``Q3.1.1``. ``depth`` is bounded by :data:`_MAX_FOLLOWUP_DEPTH` and
    ``visited`` guards against cycles in the question graph.
    """
    is_free_text = bool(q.get("is_free_text"))
    area = str(q.get("area") or "—")
    function = str(q.get("function") or "")
    qtype = str(q.get("question_type") or ("Free Text" if is_free_text else "—"))

    qtype_lower = qtype.lower()
    if is_free_text or "free" in qtype_lower:
        type_class = "type-free"
    elif "multi" in qtype_lower:
        type_class = "type-multi"
    elif "single" in qtype_lower or "select" in qtype_lower:
        type_class = "type-single"
    else:
        type_class = ""

    # Header tag row. Children reuse the parent-driven ``parent_seq``
    # (e.g. ``Q3.1``, ``Q3.1.2``) so the reviewer can locate them by
    # sight, while top-level questions keep the simple sequential
    # ``Q1``, ``Q2``, … labelling.
    if depth > 0 and parent_seq:
        qno_display = parent_seq
    else:
        qno_display = (
            f"Q{int(seq_no)}" if seq_no else str(q.get("question_id") or "")
        )
    tags: List[str] = []
    if depth > 0:
        tags.append('<span class="qprev-tag qprev-tag-followup">Follow-up</span>')
    tags.append(f'<span class="qprev-tag">{html.escape(qno_display)}</span>')
    tags.append(f'<span class="qprev-tag">Area: {html.escape(area)}</span>')
    if function:
        tags.append(f'<span class="qprev-tag">Function: {html.escape(function)}</span>')
    tags.append(f'<span class="qprev-tag {type_class}">{html.escape(qtype)}</span>')

    # Impact / severity chip is intentionally NOT shown here. The label
    # (CRITICAL / HIGH / MEDIUM / LOW) is derived from the user's actual
    # answer and rendered by ``_render_question_score_badge`` after they
    # answer the question — so criticality reflects the response, not a
    # pre-assigned tag.

    card_classes = ["qprev-card"]
    if is_free_text:
        card_classes.append("free-text")
    if depth > 0:
        # ``qprev-child`` (not ``qprev-followup``) — see the CSS block
        # comment: ``qprev-followup`` is already claimed by the free-
        # text brief-answer nudge, so children get their own class.
        card_classes.append("qprev-child")
        card_classes.append(f"qprev-child-depth-{min(depth, 3)}")
    card_class = " ".join(card_classes)
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="qprev-tag-row">{"".join(tags)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'**{html.escape(str(q.get("question") or "—"))}**')

    changed = _render_question_input_widget(q, state)
    _render_question_score_badge(q, state)
    _render_question_footer(q)
    _render_question_explainer(q)

    # Recursively render any follow-up children this parent's current
    # answer has triggered. The ``question_index`` is required — we
    # resolve child IDs against it rather than walking the flat list.
    # ``visited`` guards against cycles: the same question ID rendering
    # twice in a single chain would recurse indefinitely.
    followup_changed = False
    parent_qid = str(q.get("question_id") or "")
    if question_index is not None and parent_qid and depth < _MAX_FOLLOWUP_DEPTH:
        visited = set(visited) if visited else set()
        if parent_qid not in visited:
            visited.add(parent_qid)
            followup_ids = _get_triggered_followup_ids(q, state)
            for idx, fid in enumerate(followup_ids, start=1):
                if fid in visited:
                    continue
                child_q = question_index.get(fid)
                if not child_q:
                    continue
                if _is_question_unscoreable(child_q):
                    # Same policy as the top-level filter — hide
                    # follow-ups that can never contribute to the
                    # readiness score.
                    continue
                child_seq = f"{qno_display}.{idx}"
                if _render_single_question_answer_card(
                    child_q,
                    state,
                    question_index=question_index,
                    depth=depth + 1,
                    parent_seq=child_seq,
                    visited=visited,
                ):
                    followup_changed = True

    # AI-driven dynamic follow-up. Fires for free-text AND closed answers.
    # For closed questions the helper additionally short-circuits when the
    # selected option already surfaced a pre-generated child (see loop
    # above) so we never stack a dynamic follow-up on top of an existing
    # curated one — the two mechanisms complement each other rather than
    # duplicate. Suppressed at depth > 0 so a dynamic follow-up on the
    # parent doesn't cascade infinitely down its own children.
    if parent_qid and depth == 0:
        parent_seq_for_dyn = qno_display
        pregen_ids = _get_triggered_followup_ids(q, state) if question_index is not None else []
        if _render_dynamic_followup(
            q,
            state,
            parent_seq=parent_seq_for_dyn,
            pregen_followup_ids=pregen_ids,
        ):
            followup_changed = True

    st.markdown('</div>', unsafe_allow_html=True)
    return changed or followup_changed


# Minimum character count required on a FREE-TEXT answer before we spend
# an LLM roundtrip evaluating it. Below this length the answer is almost
# certainly too short to justify a nuanced "detailed enough?" verdict and
# we hide the dynamic follow-up card entirely (the deterministic
# ``_render_brief_answer_followup`` nudge above the widget already covers
# obvious short-answer cases). Closed-selection answers (single/multi-
# select from a pick list) skip this gate — any non-empty selection is
# a valid signal to evaluate.
_DYN_FOLLOWUP_MIN_CHARS = 20


def _dyn_followup_cache_key(qid: str, answer_hash: str) -> str:
    """Session-state key for the cached AI verdict on one (qid, answer)."""
    return f"dyn_followup:{qid}:{answer_hash}"


def _normalise_answer_for_hash(answer: Any) -> str:
    """Turn ``state.responses[qid]`` into a stable string for hashing.

    Multi-select answers arrive as a list — sort the labels so the same
    selection (in different UI orders) maps to the same hash. Single-
    select and free-text answers arrive as strings and pass through
    unchanged after trimming.
    """
    if answer in (None, "", []):
        return ""
    if isinstance(answer, list):
        parts = sorted(str(p).strip() for p in answer if str(p).strip())
        return " | ".join(parts)
    return str(answer).strip()


def _render_dynamic_followup(
    q: Dict[str, Any],
    state: AssessmentState,
    *,
    parent_seq: str,
    pregen_followup_ids: Optional[Sequence[str]] = None,
) -> bool:
    """Render an AI-generated adaptive follow-up under any parent question.

    Supports **both** answer types:

    * Free-text — always fires when the answer is at least
      :data:`_DYN_FOLLOWUP_MIN_CHARS` characters long.
    * Closed (single / multi-select) — fires only when the selected
      option(s) did NOT already trigger a pre-generated per-option
      child (see the ``pregen_followup_ids`` gate below). This keeps
      the two follow-up mechanisms complementary: curated children
      when they exist, AI-generated fills the gap otherwise.

    Cache & spinner behaviour, and the "narrative-only, not scored"
    storage contract for the follow-up's own answer, are identical to
    the earlier free-text-only version — see the module-level docstring
    on ``_dyn_followup_cache_key``.

    Returns ``True`` when the user changed the follow-up answer on this
    render so the caller can trigger a re-score / persist.
    """
    parent_qid = str(q.get("question_id") or "")
    if not parent_qid:
        return False

    is_free_text = bool(q.get("is_free_text"))
    qtype = str(q.get("question_type") or "").lower()
    if is_free_text or "free" in qtype:
        answer_kind = "free_text"
    elif "multi" in qtype:
        answer_kind = "multi_select"
    else:
        answer_kind = "single_select"

    current_answer = state.responses.get(parent_qid)
    answer_text = _normalise_answer_for_hash(current_answer)
    if not answer_text:
        return False

    # Free-text length gate — closed selections skip it because option
    # labels are typically short (< 20 chars) but still fully valid
    # signals.
    if answer_kind == "free_text" and len(answer_text) < _DYN_FOLLOWUP_MIN_CHARS:
        return False

    # Hybrid gate for closed questions: if the selected option(s) already
    # surfaced a pre-generated child, do not stack a second follow-up on
    # top. This is the "complement, not duplicate" contract the caller
    # documents.
    if answer_kind != "free_text" and pregen_followup_ids:
        return False

    # Assemble parent-option list for the LLM when the parent is closed.
    parent_options: List[str] = []
    if answer_kind != "free_text":
        raw_options = q.get("options") or []
        parent_options = option_labels(raw_options)

    answer_hash = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()[:16]
    cache_key = _dyn_followup_cache_key(parent_qid, answer_hash)

    if cache_key not in st.session_state:
        client = _genai_client()
        if client is None:
            st.session_state[cache_key] = None
            return False
        with st.spinner("Reviewing your answer for detail..."):
            try:
                refinement = evaluate_answer_and_generate_followup(
                    client,
                    regulation=str(st.session_state.get("regulation") or ""),
                    parent_question=str(q.get("question") or ""),
                    parent_explainer=str(q.get("plain_language_explainer") or ""),
                    parent_area=str(q.get("area") or ""),
                    parent_function=str(q.get("function") or ""),
                    user_answer=answer_text,
                    client_roles=_selected_client_roles(),
                    answer_kind=answer_kind,
                    parent_options=parent_options or None,
                )
            except Exception:
                logger.exception(
                    "Dynamic follow-up eval crashed. parent_qid=%s "
                    "answer_kind=%s answer_chars=%d",
                    parent_qid, answer_kind, len(answer_text),
                )
                refinement = None
        st.session_state[cache_key] = refinement

    refinement: Optional[AIAnswerRefinement] = st.session_state.get(cache_key)
    if refinement is None or not getattr(refinement, "needs_followup", False):
        return False

    followup_question = str(getattr(refinement, "followup_question", "") or "").strip()
    if not followup_question:
        return False
    followup_reason = str(getattr(refinement, "followup_reason", "") or "").strip()
    is_free_text_followup = bool(getattr(refinement, "followup_is_free_text", True))
    followup_options = [
        str(o).strip() for o in (getattr(refinement, "followup_options", []) or [])
        if str(o).strip()
    ]
    if not is_free_text_followup and not followup_options:
        is_free_text_followup = True

    dyn_answer_key = f"dyn_followup_answer:{parent_qid}:{answer_hash}"
    previous_answer = st.session_state.get(dyn_answer_key)

    st.markdown(
        '<div class="qprev-card qprev-child qprev-child-depth-1">'
        '<div class="qprev-tag-row">'
        f'<span class="qprev-tag qprev-tag-followup">AI-refined follow-up</span>'
        f'<span class="qprev-tag">{html.escape(parent_seq)}.AI</span>'
        f'<span class="qprev-tag type-free">Dynamic</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**{html.escape(followup_question)}**")
    if followup_reason:
        st.caption(f"Why this: {followup_reason}")

    widget_key = f"qprev_widget_dyn_{parent_qid}_{answer_hash}"
    changed = False
    if is_free_text_followup:
        if widget_key not in st.session_state:
            st.session_state[widget_key] = (
                "" if previous_answer in (None, "", []) else str(previous_answer)
            )
        new_value = st.text_area(
            "Follow-up answer",
            key=widget_key,
            label_visibility="collapsed",
            placeholder="Add the missing detail...",
        )
        if new_value != (previous_answer or ""):
            if new_value:
                st.session_state[dyn_answer_key] = new_value
            else:
                st.session_state.pop(dyn_answer_key, None)
            changed = True
    else:
        current_idx = 0
        if isinstance(previous_answer, str) and previous_answer in followup_options:
            current_idx = followup_options.index(previous_answer)
        if widget_key not in st.session_state:
            st.session_state[widget_key] = followup_options[current_idx]
        new_value = st.radio(
            "Follow-up answer",
            options=followup_options,
            key=widget_key,
            label_visibility="collapsed",
        )
        if new_value != previous_answer:
            st.session_state[dyn_answer_key] = new_value
            changed = True

    st.markdown('</div>', unsafe_allow_html=True)
    return changed


def _render_question_input_widget(
    q: Dict[str, Any], state: AssessmentState
) -> bool:
    """Render the widget appropriate for the question's type. Persists
    the current value directly to ``state.responses`` and returns
    ``True`` when the recorded answer changed compared to the previous
    render.
    """
    qid = str(q.get("question_id") or "")
    if not qid:
        st.caption("Question has no ID — cannot record answer.")
        return False

    is_free_text = bool(q.get("is_free_text"))
    qtype = str(q.get("question_type") or "").lower()
    raw_options = q.get("options") or []
    labels = option_labels(raw_options) if not is_free_text else []
    current = state.responses.get(qid)

    if is_free_text or "free" in qtype:
        current_text = "" if current in (None, "", []) else str(current)
        widget_key = f"qprev_widget_ft_{qid}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = current_text
        new_value = st.text_area(
            "Free-text answer",
            key=widget_key,
            label_visibility="collapsed",
            placeholder="Enter your answer or evidence notes...",
        )
        # Adaptive follow-up detection — surface a conversational prompt
        # when the user's answer is too brief, ambiguous, or contains only
        # short filler tokens like "yes" / "n/a". The prompt is
        # deterministic (no AI required) but context-aware.
        _render_brief_answer_followup(q, new_value)
        if new_value != current_text:
            if new_value:
                state.responses[qid] = new_value
            else:
                state.responses.pop(qid, None)
            return True
        return False

    if "multi" in qtype:
        current_list: List[str]
        if isinstance(current, list):
            current_list = [str(v) for v in current]
        elif current in (None, ""):
            current_list = []
        else:
            current_list = [str(current)]
        widget_key = f"qprev_widget_ms_{qid}"
        seeded_multi = [v for v in current_list if v in labels]
        stored_multi = st.session_state.get(widget_key)
        # Streamlit silently drops widget_key entries whose values are no
        # longer valid options. Re-sync from ``state.responses`` whenever
        # the seeded/state answer is a better match than what the widget
        # currently holds, otherwise the auto-filled dropdown would render
        # blank on the first paint of "Show all questions".
        if (
            widget_key not in st.session_state
            or (seeded_multi and not stored_multi)
            or (
                isinstance(stored_multi, list)
                and any(v not in labels for v in stored_multi)
            )
        ):
            st.session_state[widget_key] = seeded_multi
        new_value = st.multiselect(
            "Select all that apply",
            labels,
            key=widget_key,
            label_visibility="collapsed",
        )
        if new_value != current_list:
            if new_value:
                state.responses[qid] = new_value
            else:
                state.responses.pop(qid, None)
            return True
        return False

    placeholder = "— Select an answer —"
    display_options = [placeholder] + labels
    current_str = "" if current in (None, "", []) else str(current)
    default_idx = display_options.index(current_str) if current_str in display_options else 0
    widget_key = f"qprev_widget_sel_{qid}"
    stored_sel = st.session_state.get(widget_key)
    # Same defensive re-sync as above: whenever ``state.responses`` has a
    # concrete answer but the widget_key entry is missing / placeholder /
    # not part of the current option set, snap the widget back to the
    # authoritative value so auto-seeded dropdowns actually show.
    needs_resync = (
        widget_key not in st.session_state
        or stored_sel not in display_options
        or (
            current_str
            and current_str in display_options
            and stored_sel == placeholder
        )
    )
    if needs_resync:
        st.session_state[widget_key] = display_options[default_idx]
    new_value = st.selectbox(
        "Choose an answer",
        display_options,
        key=widget_key,
        label_visibility="collapsed",
    )
    recorded_value = "" if new_value == placeholder else new_value
    if recorded_value != current_str:
        if recorded_value:
            state.responses[qid] = recorded_value
        else:
            state.responses.pop(qid, None)
        return True
    return False


def _render_brief_answer_followup(q: Dict[str, Any], answer: str) -> None:
    """Show a conversational follow-up prompt when the user's free-text
    answer is too brief, ambiguous, or made of filler tokens.

    Detection is powered by
    :func:`services.ai_assessment_intelligence.detect_brief_answer` and is
    deterministic (no GenAI dependency). The follow-up prompt is context-
    aware — it references the current question and asks the user for more
    detail in a polite, professional voice.
    """
    if not answer or not str(answer).strip():
        return
    try:
        from services.ai_assessment_intelligence import detect_brief_answer

        needs_followup, prompt = detect_brief_answer(
            answer,
            question_context=str(q.get("question") or ""),
        )
    except Exception:
        return
    if not needs_followup:
        return
    st.markdown(
        '<div class="qprev-followup">'
        '<span class="qprev-followup-badge">Follow-up needed</span> '
        f'{html.escape(prompt)}'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_question_score_badge(q: Dict[str, Any], state: AssessmentState) -> None:
    """Emit a coloured CRITICAL / HIGH / MEDIUM / LOW pill under the widget.

    Every answered question (closed AND free-text) now shows an impact-
    ladder pill so users see how each answer lands on the readiness /
    impact axes. Free-text answers are scored via
    :func:`services.scoring_engine.score_free_text_answer` — a
    deterministic quality scorer that rewards length, concrete signals
    (policies, owners, evidence, cadences, metrics) and penalises
    vagueness ("tbd", "no owner", "not sure", …).
    """
    qid = str(q.get("question_id") or "")
    is_free_text = bool(q.get("is_free_text"))

    if is_free_text:
        text = str(state.responses.get(qid) or "").strip()
        if not text:
            st.markdown(
                '<div class="qprev-score unanswered">'
                'No answer entered yet — type your response to score this question.'
                '</div>',
                unsafe_allow_html=True,
            )
            return
        try:
            score = score_free_text_answer(
                text, question_text=str(q.get("question") or ""),
            )
        except Exception:
            score = None
        if score is None:
            st.markdown(
                '<div class="qprev-score unanswered">'
                'This answer is not applicable and is excluded from scoring.'
                '</div>',
                unsafe_allow_html=True,
            )
            return
        css = _severity_class(score)
        band = _severity_readiness_band(score)
        impact_word = _severity_impact_label(band) or "—"
        label = impact_word.upper()
        st.markdown(
            '<div class="qprev-score">'
            f'<span class="dash-pill {css}">{label}</span> '
            f'<b>{score:.0f}%</b> answer quality &nbsp;·&nbsp; '
            f'{len(text)} characters captured'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    if not answered(q, state.responses):
        st.markdown(
            '<div class="qprev-score unanswered">'
            'No answer selected yet — pick an option to score this question.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        score = score_value(state.responses.get(qid), q)
    except Exception:
        score = None
    if score is None:
        # ``score_value`` only returns None when the answer is genuinely
        # "Not applicable" — either the label is an N/A phrase or every
        # picked option carries explicit ``score_value=None`` metadata.
        raw_answer = state.responses.get(qid)
        answer_txt = ", ".join(
            str(v) for v in (raw_answer if isinstance(raw_answer, list) else [raw_answer])
            if v not in (None, "")
        )
        st.markdown(
            '<div class="qprev-score unanswered">'
            "This question doesn't apply to you, so it's not counted in your readiness score."
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Impact ladder (CRITICAL / HIGH / MEDIUM / LOW): worse readiness =
    # higher impact. A 0% readiness answer is CRITICAL, a 100% one is LOW.
    css = _severity_class(score)
    band = _severity_readiness_band(score)
    impact_word = _severity_impact_label(band) or "—"
    label = impact_word.upper()
    st.markdown(
        '<div class="qprev-score">'
        f'<span class="dash-pill {css}">{label}</span> '
        f'<b>{score:.0f}%</b> readiness contribution'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_question_footer(q: Dict[str, Any]) -> None:
    """Render the compact metadata footer at the bottom of an answer card.

    Only two signals are surfaced here — Mapped BRD and Obligations.
    Confidence / Team / Impact / Parent / Follow-up / Manual review /
    Theme / plain-English explainer / evidence-to-prepare have all been
    removed to keep the card tight; the "Why this question?" popover
    remains available for reviewers who want the full context.
    """
    footer_bits: List[str] = []
    mapped = q.get("mapped_requirement_ids") or []
    if mapped:
        preview_ids = ", ".join(html.escape(str(m)) for m in mapped[:3])
        extra = f" (+{len(mapped) - 3})" if len(mapped) > 3 else ""
        footer_bits.append(f"<b>Mapped BRD:</b> {preview_ids}{extra}")
    mapped_obl = q.get("mapped_obligation_ids") or []
    if mapped_obl:
        preview_obl = ", ".join(html.escape(str(m)) for m in mapped_obl[:2])
        extra = f" (+{len(mapped_obl) - 2})" if len(mapped_obl) > 2 else ""
        footer_bits.append(f"<b>Obligations:</b> {preview_obl}{extra}")
    if footer_bits:
        st.markdown(
            f'<div class="qprev-footer">{" &nbsp;·&nbsp; ".join(footer_bits)}</div>',
            unsafe_allow_html=True,
        )


def _render_question_explainer(q: Dict[str, Any]) -> None:
    """Render a per-question **Why this question?** affordance underneath
    each answer card.

    Restores the executive-brief explainability panel that used to live
    on the removed Assessment page. The panel is rendered as a compact
    `st.popover` when Streamlit exposes one (>=1.32) and gracefully
    falls back to an `st.expander` on older builds. Content is driven
    by the ``explainability`` bundle attached to every question by the
    v13 questionnaire generator: regulation & article, obligation ID,
    control objective, why the question exists, risk-if-negative
    narrative, and the underlying source references. Plain-English and
    evidence-expectation content are intentionally not rendered here so
    the popover matches the streamlined question card.
    """
    explain = q.get("explainability") or {}

    def _body() -> None:
        if not explain:
            rationale = str(q.get("rationale_text") or "").strip()
            if rationale:
                st.write(rationale)
            else:
                st.caption(
                    "No structured rationale is attached to this question. "
                    "It was generated from the mapped BRD requirement above."
                )
            return

        summary_parts: List[str] = []
        if explain.get("regulation"):
            summary_parts.append(f"**{html.escape(str(explain['regulation']))}**")
        if explain.get("article"):
            summary_parts.append(html.escape(str(explain["article"])))
        if explain.get("control_objective"):
            summary_parts.append(html.escape(str(explain["control_objective"])))
        if summary_parts:
            st.markdown(" \u00b7 ".join(summary_parts))
            st.markdown("---")

        purpose_raw = str(
            explain.get("question_purpose") or q.get("question_purpose") or ""
        ).lower()
        purpose_pretty = {
            "impact": "Impact probe (what would be affected / at risk)",
            "readiness": "Readiness probe (current state, controls, evidence)",
            "impact+readiness": "Impact + Readiness (both tested at once)",
        }.get(purpose_raw, purpose_raw.title() if purpose_raw else "")
        targets_impact = (
            explain.get("targets_impact_dimension") or q.get("targets_impact_dimension")
        )
        targets_readiness = (
            explain.get("targets_readiness_dimension") or q.get("targets_readiness_dimension")
        )
        two_col_rows = [
            ("Regulator", explain.get("regulator")),
            ("Article / clause", explain.get("article")),
            ("Obligation ID", explain.get("obligation_id")),
            ("Business function", explain.get("business_function")),
            ("Business area", explain.get("business_area")),
            ("Owning team",
             explain.get("owning_team") or q.get("owning_team")),
            ("Impact level",
             explain.get("impact_level") or q.get("impact_level")),
            ("Question purpose", purpose_pretty),
            ("Impact dimension tested", targets_impact),
            ("Readiness dimension tested", targets_readiness),
            ("Control objective", explain.get("control_objective")),
        ]
        col_a, col_b = st.columns(2)
        for idx, (key, value) in enumerate(two_col_rows):
            if not value:
                continue
            target = col_a if idx % 2 == 0 else col_b
            with target:
                    st.markdown(f"**{key}**  \n{value}")

        team_rationale = explain.get("team_rationale") or q.get("team_rationale")
        if team_rationale:
            st.markdown(f"**Why this team?** {team_rationale}")
        impact_reason = explain.get("impact_reason") or q.get("impact_reason")
        if impact_reason:
            st.markdown(f"**Why this impact level?** {impact_reason}")

        for key, items in [
            ("BRD requirement IDs", explain.get("brd_requirement_ids") or []),
            ("Resource Traceability Matrix trace IDs", explain.get("rtm_trace_ids") or []),
        ]:
            if items:
                st.markdown(f"**{key}:** {', '.join(str(i) for i in items)}")

        for key, value in [
            ("Why this question exists", explain.get("reason") or q.get("rationale")),
            ("Risk if answered negatively", explain.get("risk_if_negative")),
        ]:
            if value:
                st.markdown(f"**{key}**")
                st.write(value)

        if q.get("is_parent") and (q.get("child_question_ids") or []):
            child_ids = q.get("child_question_ids") or []
            st.info(
                f"This is a **parent question**. Depending on your answer, "
                f"up to {len(child_ids)} adaptive follow-up question(s) may be "
                f"surfaced next."
            )
        if q.get("is_child") or q.get("dynamic"):
            parent_id = q.get("funnel_parent_id") or q.get("source_parent_id")
            triggers = q.get("trigger_answers") or []
            if parent_id:
                trigger_str = ", ".join(str(t) for t in triggers) or "the previous response"
                st.info(
                    f"This is an **adaptive follow-up** to question "
                    f"**{parent_id}**, triggered because you answered "
                    f"'{trigger_str}'."
                )

        source_refs = explain.get("source_references") or []
        if source_refs:
            st.markdown("**Source references**")
            for ref in source_refs:
                label = _format_source_label(ref) if isinstance(ref, dict) else str(ref)
                url = ref.get("source_url") or "" if isinstance(ref, dict) else ""
                if url:
                    st.markdown(f"- {label}  \n  [{url}]({url})")
                else:
                    st.markdown(f"- {label}")

        if q.get("dynamic"):
            rule = str(q.get("branch_rule_id") or "generic")
            triggers = q.get("trigger_answers") or []
            trigger_str = ", ".join(str(t) for t in triggers) or "the prior response"
            st.caption(
                f"Adaptive follow-up - triggered by **{trigger_str}** on the previous "
                f"question. Branch rule: `{rule}`."
            )

    popover = getattr(st, "popover", None)
    if callable(popover):
        with popover("Why this question?", use_container_width=False):
            _body()
    else:
        with st.expander("Why this question?", expanded=False):
            _body()


def _run_agent3() -> None:
    mode = st.session_state["mode"]
    regulation = st.session_state["regulation"]
    orch = _get_orchestrator()
    logger.info("Agent 3 (Questionnaire) starting. mode=%s regulation=%s", mode, regulation)
    # When Agent 3 is chained inside the BRD/FRD status widget its own
    # spinner would confuse users into thinking a second unrelated
    # pipeline was running - we suppress it in that case and rely on the
    # outer widget's phase label instead.
    #
    # The same suppression applies when the caller has painted our
    # full-screen loader (``_agent3_using_full_loader`` is set by the
    # two-phase render in ``render_questionnaire_page``): the loader
    # already communicates progress, and an additional ``st.spinner``
    # would inject a small extra element into the DOM alongside it.
    if (
        st.session_state.get("_brd_flow_active")
        or st.session_state.get("_agent3_using_full_loader")
    ):
        spinner_ctx = contextlib.nullcontext()
    else:
        spinner_ctx = st.spinner("Generating adaptive questionnaire with the AI agent...")
    with spinner_ctx:
        try:
            readiness = st.session_state.get("readiness_assessment")
            rtm = st.session_state.get("rtm_artifact")
            client_profile = st.session_state.get("client_profile") or None
            analysis = st.session_state.get("analysis")
            roles = _selected_client_roles()

            # Kick off Impact + Confidence assessments in parallel with
            # Agent 3's questionnaire build (they were deferred from the
            # BRD-click flow to keep BRD visible fast on Page 2). All three
            # are independent LLM tasks so we run them concurrently under
            # the GenAI client's global semaphore. If impact completes in
            # time it feeds the questionnaire enhancer; otherwise the
            # enhancer falls back to default weights and the Dashboard's
            # ``_refresh_scoring_snapshot`` re-weights once impact lands.
            from concurrent.futures import ThreadPoolExecutor as _P3Pool
            impact = st.session_state.get("impact_assessment")
            confidence_needed = st.session_state.get("confidence_assessment") is None
            impact_needed = impact is None and analysis is not None

            assess_pool: Optional[Any] = None
            fut_impact = None
            fut_confidence = None
            if impact_needed or confidence_needed:
                assess_pool = _P3Pool(max_workers=2, thread_name_prefix="assess_p3")
                if impact_needed:
                    fut_impact = assess_pool.submit(
                        orch.assess_impact_intelligence, analysis,
                    )
                if confidence_needed and analysis is not None:
                    fut_confidence = assess_pool.submit(
                        orch.assess_confidence_intelligence, analysis,
                    )

            if mode == "Generate BRD/FRD from regulation":
                brd_artifact: Optional[BRDArtifact] = st.session_state.get("brd_artifact")
                if brd_artifact is None or brd_artifact.report is None:
                    st.error("Click **Generate BRD / FRD** on Page 2 before building the questionnaire.")
                    if assess_pool is not None:
                        assess_pool.shutdown(wait=False, cancel_futures=True)
                    return
                questionnaire = orch.run_questionnaire_from_report(
                    brd_artifact, regulation=regulation,
                    impact=impact, readiness=readiness,
                    analysis=analysis,
                    rtm=rtm,
                    client_roles=roles,
                    client_profile=client_profile,
                )
                source = "generated_brd"
                name = questionnaire.name
            else:
                doc_id = st.session_state.get("brd_doc_id")
                if not doc_id:
                    st.error("Upload a BRD on Page 1 before building the questionnaire.")
                    if assess_pool is not None:
                        assess_pool.shutdown(wait=False, cancel_futures=True)
                    return
                rec = db.get_document(int(doc_id))
                if not rec:
                    st.error("Saved BRD record is missing from the database.")
                    if assess_pool is not None:
                        assess_pool.shutdown(wait=False, cancel_futures=True)
                    return
                questionnaire = orch.run_questionnaire_from_docx(
                    Path(rec["path"]), regulation=regulation,
                    name=f"{regulation} — from {Path(rec['name']).stem}",
                    impact=impact, readiness=readiness,
                    analysis=analysis,
                    rtm=rtm,
                    client_roles=roles,
                    client_profile=client_profile,
                )
                source = "uploaded_brd"
                name = questionnaire.name

            # Harvest the parallel Impact + Confidence results (if any)
            # after Agent 3 completes so they're ready for the Dashboard.
            if assess_pool is not None:
                if fut_impact is not None:
                    try:
                        _impact_res = fut_impact.result(timeout=180)
                        if _impact_res is not None:
                            st.session_state["impact_assessment"] = _impact_res
                            logger.info("Impact intelligence assessed on Page 3 (parallel with Agent 3).")
                    except Exception:
                        logger.exception("Impact intelligence assessment on Page 3 failed (non-fatal).")
                if fut_confidence is not None:
                    try:
                        _conf_res = fut_confidence.result(timeout=180)
                        if _conf_res is not None:
                            st.session_state["confidence_assessment"] = _conf_res
                            logger.info("Confidence intelligence assessed on Page 3 (parallel with Agent 3).")
                    except Exception:
                        logger.exception("Confidence intelligence assessment on Page 3 failed (non-fatal).")
                assess_pool.shutdown(wait=True)
        except Exception as exc:
            logger.exception("Agent 3 (Questionnaire) failed. mode=%s regulation=%s", mode, regulation)
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
    logger.info(
        "Agent 3 completed. questionnaire_id=%s questions=%d source=%s",
        qid,
        len((questionnaire.package or {}).get("questions") or []),
        source,
    )


# ---------------------------------------------------------------------------
# Page 4 — Dashboard (Python Rules Engine + Agent 4)
# ---------------------------------------------------------------------------

def render_dashboard_page() -> None:
    """Rules-engine dashboard for Page 4.

    Layout follows the T+1 Rules Engine reference and the executive
    brief:
      1. **Overall Impact & Readiness** hero row (two big score tiles).
      2. **Readiness Overview By Area** (readiness cards per impacted area).
      3. **Impact Assessment By Area** (impact-severity cards - HIGH / MEDIUM
         / LOW - per impacted area, mirroring the executive heatmap).
      4. **Prioritized Remediation Recommendations** grouped per area, with
         3-4 executive-ready bullets each, filtered to Critical / At Risk.
      5. **Top gaps** and **Question-level scoring detail** for auditors.

    All underlying data still comes from ``services.scoring_engine`` —
    presentation is the only change.
    """
    st.subheader("4. Dashboard — Impact & Readiness")
    st.caption(
        "Readiness / impact scores derived from your Page 3 answers. "
        "Every area, function and Area \u00d7 Function pair gets a live "
        "severity classification on the same four-band ladder — "
        "Critical / At risk / Watch / Ready — matched with 3-4 concrete "
        "action bullets from Agent 4."
    )

    questionnaire: Optional[QuestionnairePackage] = st.session_state.get("questionnaire")
    if questionnaire is None:
        st.warning("No questionnaire loaded.")
        _render_next_button("4. Dashboard", disabled=True,
                            help_text="Load a questionnaire first.")
        return
    scoring = _refresh_scoring_snapshot()
    if scoring is None:
        st.warning("No evaluation available yet.")
        _render_next_button("4. Dashboard", disabled=True,
                            help_text="Answer some questions first.")
        return

    result = scoring.evaluation
    score = float(result.get("compliance_score_pct") or 0.0)
    # Prefer the AI Assessment Intelligence overall confidence (with reasoning)
    # over the legacy evaluation_confidence_pct clamp.
    confidence_assessment = getattr(scoring, "confidence", None) or st.session_state.get("confidence_assessment")
    if confidence_assessment is not None:
        eval_conf = float(confidence_assessment.overall_score)
    else:
        eval_conf = float(result.get("evaluation_confidence_pct") or 0.0)
    # Presentation floor: Confidence KPI on the dashboard hero must never
    # drop below the mandated display minimum (default 90%). The raw
    # value stays available on ``confidence_assessment`` / ``result`` for
    # downstream analytics.
    eval_conf = _floor_pct(eval_conf)
    answered = int(result.get("answered_count") or 0)
    unanswered = int(result.get("unanswered_count") or 0)
    total = answered + unanswered
    pair_scores: Dict[Any, float] = result.get("pair_scores") or {}
    area_summary: Dict[str, Dict[str, Any]] = result.get("area_summary") or {}
    function_summary: Dict[str, Dict[str, Any]] = result.get("function_summary") or {}

    # The weighted impact model (if available) is the single source of truth
    # for the "Overall Impact Score" tile - fall back to `100 - readiness`
    # only when the model has not yet been computed (e.g. before any
    # scoring refresh has run).
    weighted_impact: Optional[WeightedImpactResult] = st.session_state.get(
        "weighted_impact"
    )
    impact_hero_pct: Optional[float] = (
        float(weighted_impact.overall_impact_score) if weighted_impact is not None else None
    )
    _render_dashboard_hero(
        readiness_pct=score,
        confidence_pct=eval_conf,
        impact_pct=impact_hero_pct,
    )

    # Streamlined dashboard: hero + severity banner + area cards
    # (readiness + impact) + prioritized remediation recommendations +
    # three expanders below. The KPI row, AI intelligence panels,
    # weighted-readiness / weighted-impact detail panels, and the
    # Area × Function heatmap have all been intentionally removed - the
    # weighted calculations still run behind the scenes and drive the
    # numbers in the hero + area cards, but their detail views are no
    # longer rendered here.
    _render_dashboard_legend(
        area_summary=area_summary,
        function_summary=function_summary,
        pair_scores=pair_scores,
    )

    st.markdown(
        '<h4 class="rap-dash-hdr">Readiness Overview By Area</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Every impacted area ranked by its current readiness (higher is "
        "better). Colour follows the standard Critical / At Risk / Watch / "
        "Ready ladder."
    )
    _render_dashboard_readiness_cards(area_summary)

    st.markdown(
        '<h4 class="rap-dash-hdr">Impact Assessment By Area</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Impact severity uses the impact ladder (higher impact = worse): "
        "Critical (\u2265 75%), At Risk (50 - 75%), "
        "Watch (25 - 50%), Ready (< 25%). "
        "Readiness scores use the mirror ladder (higher readiness = better) "
        "so the two axes always agree."
    )
    _render_dashboard_impact_cards(area_summary)

    st.markdown(
        '<h4 class="rap-dash-hdr">Prioritized Remediation Recommendations '
        'For Critically Impacted Business Functions - Clear Business Outcomes'
        '</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Agent 4 groups every actionable gap by impacted area and readiness "
        "and expands each into 3-4 executive-ready bullets covering "
        "escalation, ownership, evidence and success criteria."
    )
    _autorun_recommendations_if_needed(questionnaire, scoring)
    recs = st.session_state.get("recommendations") or []
    rich_recs = st.session_state.get("rich_recommendations") or []
    # Prefer rich (consulting-grade) recommendations when available; fall
    # back to the legacy compact recommendations otherwise.
    if rich_recs:
        _render_rich_recommendations(rich_recs)
    else:
        _render_dashboard_area_recommendations(recs, area_summary)

    with st.expander("Advanced controls (regenerate recommendations)", expanded=False):
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
                value=bool(st.session_state.get("genai_available")),
                disabled=not st.session_state["genai_available"],
                help=(
                    "Rewrites each area's recommendation with the GenAI Shared "
                    "Service so wording is grounded in that area's specific gaps "
                    "and obligations. Disabled when the service is unavailable."
                ),
            )
        if st.button("Regenerate recommendations", type="secondary"):
            rec_state: AssessmentState = st.session_state["assessment_state"]
            recommendation_result = _get_orchestrator().run_recommendations(
                questionnaire,
                scoring,
                min_severity=min_sev,
                top_n_requirements=int(top_n),
                enrich_with_genai=bool(run_genai),
                branch_log=list(rec_state.branch_log),
                analysis=st.session_state.get("analysis"),
                client_roles=_selected_client_roles(),
            )
            if recommendation_result.used_genai:
                st.toast("Recommendations enriched via GenAI.")
            st.session_state["recommendations"] = recommendation_result.recommendations
            st.session_state["rich_recommendations"] = recommendation_result.rich_recommendations
            _persist_assessment_snapshot()
            st.rerun()

    with st.expander("Top gaps (lowest-scoring requirements)", expanded=False):
        _render_dashboard_top_gap_cards(scoring.top_gaps)

    with st.expander("Question-level scoring detail", expanded=False):
        _render_dashboard_question_scoring_table(questionnaire, scoring)

    _render_next_button("4. Dashboard")


def _autorun_recommendations_if_needed(
    questionnaire: QuestionnairePackage, scoring: Any
) -> None:
    """Run Agent 4 automatically the first time a user lands on the
    dashboard after answering questions.

    Uses a fingerprint of the scoring snapshot so the run is repeated
    exactly once per meaningful change in responses, not on every
    rerun. Advanced controls (below the card grid) still let the user
    force a manual regeneration.
    """
    if not st.session_state.get("assessment_state"):
        return
    result = scoring.evaluation or {}
    genai_on = bool(st.session_state.get("genai_available"))
    fingerprint = (
        round(float(result.get("compliance_score_pct") or 0.0), 1),
        int(result.get("answered_count") or 0),
        len(result.get("pair_scores") or {}),
        genai_on,
    )
    if st.session_state.get("dashboard_recs_fingerprint") == fingerprint:
        return
    try:
        rec_state: AssessmentState = st.session_state["assessment_state"]
        with st.spinner(
            "Generating area-specific consulting-grade recommendations via GenAI..."
            if genai_on
            else "Composing area-specific recommendations..."
        ):
            recommendation_result = _get_orchestrator().run_recommendations(
                questionnaire,
                scoring,
                min_severity="Watch",
                top_n_requirements=10,
                enrich_with_genai=genai_on,
                branch_log=list(rec_state.branch_log),
                analysis=st.session_state.get("analysis"),
                client_roles=_selected_client_roles(),
            )
        st.session_state["recommendations"] = recommendation_result.recommendations
        st.session_state["rich_recommendations"] = recommendation_result.rich_recommendations
        st.session_state["dashboard_recs_fingerprint"] = fingerprint
        _persist_assessment_snapshot()
    except Exception:
        # Recommendations are a UX enhancer, never a blocker. If Agent 4
        # errors, we still render the rest of the dashboard.
        logger.exception("Auto-run of Agent 4 (Recommendations) failed on dashboard load.")


# ---------------------------------------------------------------------------
# Page 4 — dashboard rendering helpers (all inspired by the T+1 reference)
# ---------------------------------------------------------------------------

def _severity_class(score: Optional[float]) -> str:
    """Map a **readiness / compliance** score to one of the four canonical
    severity CSS classes. Thin wrapper over
    :mod:`services.severity` so all bands / thresholds / class names stay
    in a single module.
    """
    return _severity_css_class(_severity_readiness_band(score))


def _impact_class(impact: Optional[float]) -> str:
    """Map an **impact %** (higher impact = worse) to a severity CSS class.

    Thin wrapper over :func:`services.severity.impact_band`.
    """
    return _severity_css_class(_severity_impact_band(impact))


def _severity_label_from_status(status: Optional[str]) -> str:
    """Return the CSS class for a CXO-status label (``Critical`` / ``At risk``
    / ``Watch`` / ``Ready``). Thin wrapper over :mod:`services.severity`.
    """
    return _severity_css_class(_severity_from_label(status))


def _dashboard_high_impact_area_count(area_summary: Dict[str, Dict[str, Any]]) -> int:
    """Count impacted areas classified as Critical or At risk. Surfaced in
    the KPI row so leadership sees the size of the remediation frontier at
    a glance.
    """
    if not area_summary:
        return 0
    count = 0
    for summary in area_summary.values():
        status = str(summary.get("CXO status") or "").strip().lower()
        if status in {"critical", "at risk"}:
            count += 1
    return count


def _readiness_severity_from_score(readiness: Optional[float]) -> Tuple[str, str]:
    """Bucket a **readiness / compliance** score into one of the four
    canonical severity labels + matching CSS class (higher readiness =
    better).

    Uses the same thresholds as :func:`_severity_class` so the
    "Readiness Overview By Area" tiles and area recommendation cards
    all share one four-band colour ladder:

      - readiness <  25%      -> CRITICAL   (crit  / red)
      - readiness 25 - 50%    -> AT RISK    (risk  / amber)
      - readiness 50 - 75%    -> WATCH      (watch / light green)
      - readiness >= 75%      -> READY      (ready / dark green)

    Returns ``(label, css_class)``.
    """
    if readiness is None:
        return ("N/A", "none")
    try:
        val = float(readiness)
    except (TypeError, ValueError):
        return ("N/A", "none")
    if val < 25.0:
        return ("Critical", "crit")
    if val < 50.0:
        return ("At Risk", "risk")
    if val < 75.0:
        return ("Watch", "watch")
    return ("Ready", "ready")


def _impact_severity_from_score(impact: Optional[float]) -> Tuple[str, str]:
    """Bucket an **impact %** (higher impact = worse) into one of the
    four canonical severity labels + matching CSS class.

    Impact bands (mirror of the readiness ladder):

      - impact >= 75%         -> CRITICAL   (crit  / red)
      - impact 50 - 75%       -> AT RISK    (risk  / amber)
      - impact 25 - 50%       -> WATCH      (watch / light green)
      - impact <  25%         -> READY      (ready / dark green)

    Returns ``(label, css_class)``. Because impact = 100 - readiness,
    calling this with impact and calling :func:`_readiness_severity_from_score`
    with the matching readiness value always return the same label.
    """
    if impact is None:
        return ("N/A", "none")
    try:
        val = float(impact)
    except (TypeError, ValueError):
        return ("N/A", "none")
    if val >= 75.0:
        return ("Critical", "crit")
    if val >= 50.0:
        return ("At Risk", "risk")
    if val >= 25.0:
        return ("Watch", "watch")
    return ("Ready", "ready")


def _render_rich_recommendations(recs: List[Any]) -> None:
    """Render the consulting-grade rich recommendations as tall stacked cards.

    Each card carries what / why / how / expected outcome / dependencies
    plus implementation steps, success metrics, mapped requirements and
    obligations, and the accountable owner. The short-term / long-term /
    quick-wins timelines are computed on the dataclass but intentionally
    not rendered in the card - product feedback preferred the shorter
    surface.

    Only recommendations for **Critical** and **At Risk** areas are
    surfaced; Watch / Ready areas are considered under control and their
    recommendations (if any) are filtered out at render time.
    """
    if not recs:
        st.caption("No consulting-grade recommendations yet.")
        return

    def _get(r: Any, key: str, default: Any = None) -> Any:
        if isinstance(r, dict):
            return r.get(key, default)
        return getattr(r, key, default)

    # Sort High > Medium > Low
    def _rank(pr: str) -> int:
        return {"high": 0, "medium": 1, "low": 2}.get((pr or "").strip().lower(), 3)

    # Keep only recommendations whose severity is Critical or At Risk so
    # the dashboard stays focused on the areas that actually need
    # remediation attention.
    _CRITICAL_SEVERITIES = {"critical", "at risk"}
    recs_filtered = [
        r for r in recs
        if str(_get(r, "severity", "") or "").strip().lower() in _CRITICAL_SEVERITIES
    ]

    if not recs_filtered:
        st.caption(
            "No Critical or At Risk areas — no remediation recommendations "
            "to surface for this run."
        )
        return

    recs_sorted = sorted(
        recs_filtered, key=lambda r: _rank(str(_get(r, "priority", "Medium"))),
    )

    st.markdown('<div class="dash-rich-rec-grid">', unsafe_allow_html=True)
    for r in recs_sorted:
        area = str(_get(r, "area", "") or "Unmapped")
        title = str(_get(r, "title", "") or f"Recommendation for {area}")
        priority = str(_get(r, "priority", "Medium") or "Medium")
        severity = str(_get(r, "severity", "Watch") or "Watch")
        # Horizon (e.g. "Short-term (30-90 days)") intentionally NOT
        # displayed in the Dashboard rich-recommendation card. Product
        # feedback: the fixed-window timeline was noise for consumers
        # who read these cards for the *action* + *priority*, not for
        # an arbitrary calendar horizon.
        function = str(_get(r, "function", "") or "")
        owner = str(_get(r, "owner", "") or "")
        what = str(_get(r, "what", "") or "")
        why = str(_get(r, "why", "") or "")
        how = str(_get(r, "how", "") or "")
        expected = str(_get(r, "expected_outcome", "") or "")
        deps = list(_get(r, "dependencies", []) or [])
        steps = list(_get(r, "implementation_steps", []) or [])
        metrics = list(_get(r, "success_metrics", []) or [])
        req_ids = list(_get(r, "mapped_requirement_ids", []) or [])
        obl_ids = list(_get(r, "mapped_obligation_ids", []) or [])
        by_ai = bool(_get(r, "generated_by_ai", False))
        identified_gap = str(_get(r, "identified_gap", "") or "")

        pill_css = _severity_pill_for_severity(severity) if severity else "watch"
        # Priority CSS class is still used for the card border colour so
        # High-priority items pop visually, but the "Priority: <label>"
        # pill itself is no longer rendered — product feedback: the
        # severity pill already conveys the same signal.
        priority_css = {"high": "crit", "medium": "watch", "low": "ready"}.get(
            priority.lower(), "watch"
        )

        badges = (
            f'<span class="dash-pill {pill_css}">{html.escape(severity)}</span>'
        )
        if by_ai:
            badges += '<span class="dash-pill">AI-refined</span>'

        area_line = html.escape(area) + (
            f" &nbsp;·&nbsp; {html.escape(function)}" if function else ""
        )
        if owner:
            area_line += f" &nbsp;·&nbsp; Owner: {html.escape(owner)}"

        def _list_block(label: str, items: List[str]) -> str:
            if not items:
                return ""
            li = "".join(f"<li>{html.escape(str(i))}</li>" for i in items[:8])
            return (
                f'<div class="dash-rich-rec-section">'
                f'<b>{label}</b>'
                f'<ul class="dash-rich-rec-list">{li}</ul>'
                f'</div>'
            )

        card_html = (
            f'<div class="dash-rich-rec-card {priority_css}">'
            f'<div class="dash-rich-rec-header">'
            f'<div class="dash-rich-rec-title">{html.escape(title)}</div>'
            f'<div class="dash-rich-rec-badges">{badges}</div>'
            f'</div>'
            f'<div class="dash-rich-rec-meta">'
            f'<div>{area_line}</div>'
            f'</div>'
        )
        if identified_gap:
            card_html += (
                f'<div class="dash-rich-rec-section">'
                f'<b>Identified gap.</b> {html.escape(identified_gap)}'
                f'</div>'
            )
        if what:
            card_html += (
                f'<div class="dash-rich-rec-section">'
                f'<b>What needs to be done.</b> {html.escape(what)}'
                f'</div>'
            )
        if why:
            card_html += (
                f'<div class="dash-rich-rec-section">'
                f'<b>Why it is important.</b> {html.escape(why)}'
                f'</div>'
            )
        if how:
            card_html += (
                f'<div class="dash-rich-rec-section">'
                f'<b>How to implement it.</b> {html.escape(how)}'
                f'</div>'
            )
        if expected:
            card_html += (
                f'<div class="dash-rich-rec-section">'
                f'<b>Expected outcome.</b> {html.escape(expected)}'
                f'</div>'
            )
        card_html += _list_block("Dependencies / prerequisites", deps)
        card_html += _list_block("Implementation steps", steps)
        card_html += _list_block("Success metrics", metrics)
        if req_ids or obl_ids:
            ref_parts = []
            if req_ids:
                ref_parts.append(
                    f"<b>Requirements:</b> {html.escape(', '.join(req_ids[:8]))}"
                )
            if obl_ids:
                ref_parts.append(
                    f"<b>Obligations:</b> {html.escape(', '.join(obl_ids[:8]))}"
                )
            card_html += (
                f'<div class="dash-rich-rec-section">'
                f'{" &nbsp;·&nbsp; ".join(ref_parts)}'
                f'</div>'
            )
        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_impact_intelligence_panel(impact: Any) -> None:
    """Render the AI-driven impact assessment as a structured panel.

    Shows the executive summary + one card per impact dimension
    (business functions, processes, systems, data, controls, stakeholders)
    with severity, item list, rationale and evidence.
    """
    st.markdown(
        '<h4 class="rap-dash-hdr">Regulatory Impact Assessment</h4>',
        unsafe_allow_html=True,
    )
    source = "AI-generated" if getattr(impact, "generated_by_ai", False) else "evidence-driven"
    header_bits = [
        f'<div class="dash-impact-header">'
        f'<div class="dash-impact-cap">Overall Impact</div>'
        f'<div class="dash-impact-value">{impact.overall_severity_score:.0f}/100</div>'
        f'<div class="dash-impact-sub"><span class="dash-pill {_severity_pill_for_severity(impact.overall_severity)}">'
        f'{html.escape(str(impact.overall_severity))}</span></div>'
        f'<div class="dash-impact-src">Source: {source}</div>'
        f'</div>'
    ]
    st.markdown("".join(header_bits), unsafe_allow_html=True)
    if impact.executive_summary:
        st.caption(impact.executive_summary)

    cards: List[str] = ['<div class="dash-cards impact-int-grid">']
    dim_labels = {
        "business_functions": "Business Functions",
        "processes": "Processes",
        "systems": "Systems & Applications",
        "data": "Data",
        "controls": "Controls",
        "stakeholders": "Stakeholders",
    }
    for dim in impact.dimensions():
        label = dim_labels.get(dim.dimension, dim.dimension.replace("_", " ").title())
        css = _severity_pill_for_severity(dim.severity)
        items_html = "".join(f"<li>{html.escape(str(i))}</li>" for i in (dim.items or [])[:8])
        evidence_html = (
            "".join(f"<li>{html.escape(str(e))}</li>" for e in (dim.evidence or [])[:3])
        )
        evidence_block = (
            f'<div class="dash-card-body"><b>Evidence:</b><ul>{evidence_html}</ul></div>'
            if evidence_html else ""
        )
        cards.append(
            f'<div class="dash-card {css}">'
            f'<div class="dash-card-title">{html.escape(label)}</div>'
            f'<div class="dash-card-meta">'
            f'<span class="dash-pill {css}">{html.escape(str(dim.severity))}</span> '
            f'&nbsp;<b>{dim.severity_score:.0f}/100</b> impact severity'
            f'</div>'
            f'<div class="dash-card-body">'
            f'<b>Affected items:</b><ul>{items_html or "<li>—</li>"}</ul>'
            f'<b>Why this area is impacted:</b><br>{html.escape(dim.rationale)}'
            f'</div>'
            f'{evidence_block}'
            f'</div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def _render_readiness_intelligence_panel(readiness: Any) -> None:
    """Render the AI-driven readiness assessment across seven consulting
    dimensions: existing controls, process maturity, policy coverage,
    technology readiness, documentation completeness, implementation gaps,
    and organizational preparedness.
    """
    st.markdown(
        '<h4 class="rap-dash-hdr">Regulatory Readiness Assessment</h4>',
        unsafe_allow_html=True,
    )
    source = "AI-generated" if getattr(readiness, "generated_by_ai", False) else "evidence-driven"
    css = _severity_class(readiness.overall_score)
    st.markdown(
        f'<div class="dash-readiness-header">'
        f'<div class="dash-impact-cap">Overall Readiness</div>'
        f'<div class="dash-impact-value">{readiness.overall_score:.1f}%</div>'
        f'<div class="dash-impact-sub"><span class="dash-pill {css}">'
        f'{html.escape(str(readiness.overall_level))}</span></div>'
        f'<div class="dash-impact-src">Source: {source}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if readiness.executive_summary:
        st.caption(readiness.executive_summary)

    dim_labels = {
        "existing_controls": "Existing Controls",
        "process_maturity": "Process Maturity",
        "policy_coverage": "Policy Coverage",
        "technology_readiness": "Technology Readiness",
        "documentation_completeness": "Documentation Completeness",
        "implementation_gaps": "Implementation Gaps",
        "organizational_preparedness": "Organizational Preparedness",
    }
    cards: List[str] = ['<div class="dash-cards readiness-int-grid">']
    for dim in readiness.dimensions():
        label = dim_labels.get(dim.dimension, dim.dimension.replace("_", " ").title())
        css_d = _severity_class(dim.score)
        strengths_html = (
            "".join(f"<li>{html.escape(str(s))}</li>" for s in (dim.strengths or [])[:4])
        )
        gaps_html = (
            "".join(f"<li>{html.escape(str(g))}</li>" for g in (dim.gaps or [])[:4])
        )
        strengths_block = (
            f'<div class="dash-card-body"><b>Strengths:</b><ul>{strengths_html}</ul></div>'
            if strengths_html else ""
        )
        gaps_block = (
            f'<div class="dash-card-body"><b>Gaps:</b><ul>{gaps_html}</ul></div>'
            if gaps_html else ""
        )
        cards.append(
            f'<div class="dash-card {css_d}">'
            f'<div class="dash-card-title">{html.escape(label)}</div>'
            f'<div class="dash-card-meta">'
            f'<span class="dash-pill {css_d}">{html.escape(str(dim.maturity_level))}</span> '
            f'&nbsp;<b>{dim.score:.0f}/100</b> maturity'
            f'</div>'
            f'<div class="dash-card-bar {css_d}"><span style="width:{max(0.0, min(100.0, dim.score)):.1f}%"></span></div>'
            f'<div class="dash-card-body">{html.escape(dim.rationale)}</div>'
            f'{strengths_block}'
            f'{gaps_block}'
            f'</div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def _severity_pill_for_severity(severity: str) -> str:
    s = (severity or "").strip().lower()
    return {
        "critical": "crit",
        "high": "risk",
        "at risk": "risk",
        "medium": "watch",
        "low": "ready",
    }.get(s, "watch")


# ---------------------------------------------------------------------------
# Weighted readiness panel (DORA demo profile)
# ---------------------------------------------------------------------------
#
# All rendering for the new weighted scoring model lives in this section.
# The panel is populated from ``st.session_state["weighted_readiness"]``
# which is a :class:`services.readiness_score.WeightedReadinessResult`
# instance refreshed by :func:`_refresh_scoring_snapshot` on every
# dashboard paint. Nothing in here talks to the scoring engine directly,
# so this section can be swapped out or hidden without touching the core
# rules-engine pipeline.


def _severity_class_for_gap(severity_label: str) -> str:
    """Map the gap severity vocabulary to the existing severity CSS classes.

    ``Low`` → ready, ``Medium`` → watch, ``High`` → risk,
    ``Critical`` → crit. The CSS palette is shared with the impact /
    readiness cards so all gap indicators use the same colour ladder.
    """
    return {
        "low": "ready",
        "medium": "watch",
        "high": "risk",
        "critical": "crit",
    }.get(str(severity_label).lower(), "watch")


def _render_weighted_readiness_panel(result: WeightedReadinessResult) -> None:
    """Render the full weighted readiness section for the Dashboard page.

    Layout:

    1. Section header + rating pill.
    2. Five KPI cards (Overall Readiness, Rating, Accuracy, Completeness,
       Overall Coverage Gap).
    3. Weighted scoring table (Area, Weight, #Q, Score, Weighted, Gap,
       Severity) - a plain ``st.dataframe`` so users can sort/copy.
    4. Top 5 gap areas as colour-coded chips.
    5. Nine gap-category rollup as compact cards.
    6. Accuracy breakdown expander showing the three sub-scores that
       feed the composite accuracy metric.
    7. Missing-evidence / low-mapping status caption.

    Every card uses the existing ``dash-*`` CSS palette so the styling
    matches the rest of the dashboard.
    """
    st.markdown(
        '<h4 class="rap-dash-hdr">Weighted Readiness (DORA)</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Overall Readiness Index is the weighted average of the seven "
        "DORA assessment areas. Weights sum to 100 and are validated at "
        "startup - swap the profile via `services.readiness_score.DORA_AREA_WEIGHTS`."
    )

    overall = float(result.overall_readiness_score)
    overall_css = _severity_class(overall)
    rating = result.readiness_rating
    # Underlying raw values (used by tables, exports, breakdown expander).
    accuracy_raw = float(result.accuracy_score)
    completeness_raw = float(result.completeness_score)
    overall_gap = float(result.overall_coverage_gap)
    # Presentation-floored values used ONLY inside the KPI cards below so
    # Accuracy Score / Completeness Score always render >=90% per the
    # product requirement. The Accuracy breakdown expander further down
    # continues to show the raw sub-scores (evidence, consistency,
    # mapping) so evaluators can still audit the underlying composition.
    accuracy = _floor_pct(accuracy_raw)
    completeness = _floor_pct(completeness_raw)

    # --- KPI cards ---------------------------------------------------------
    kpi_html = (
        '<div class="dash-kpis">'
        f'<div class="dash-kpi">'
        f'<div class="dash-kpi-label">Overall Readiness Index</div>'
        f'<div class="dash-kpi-value">{overall:.1f} / 100</div>'
        f'<div class="dash-kpi-bar {overall_css}"><span style="width:{overall:.1f}%"></span></div>'
        f'</div>'
        f'<div class="dash-kpi">'
        f'<div class="dash-kpi-label">Readiness Rating</div>'
        f'<div class="dash-kpi-value" style="font-size:1.05rem;">{html.escape(rating)}</div>'
        f'<div class="dash-kpi-bar {overall_css}"><span style="width:{overall:.1f}%"></span></div>'
        f'</div>'
        f'<div class="dash-kpi">'
        f'<div class="dash-kpi-label">Accuracy Score</div>'
        f'<div class="dash-kpi-value">{accuracy:.1f}%</div>'
        f'<div class="dash-kpi-bar {_severity_class(accuracy)}"><span style="width:{accuracy:.1f}%"></span></div>'
        f'</div>'
        f'<div class="dash-kpi">'
        f'<div class="dash-kpi-label">Completeness Score</div>'
        f'<div class="dash-kpi-value">{completeness:.1f}%</div>'
        f'<div class="dash-kpi-bar {_severity_class(completeness)}"><span style="width:{completeness:.1f}%"></span></div>'
        f'</div>'
        f'<div class="dash-kpi">'
        f'<div class="dash-kpi-label">Overall Coverage Gap</div>'
        f'<div class="dash-kpi-value">{overall_gap:.1f}%</div>'
        f'<div class="dash-kpi-bar {_severity_class(100 - overall_gap)}"><span style="width:{overall_gap:.1f}%"></span></div>'
        f'</div>'
        '</div>'
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

    # --- Weighted scoring table ------------------------------------------
    st.markdown("**Weighted Scoring Table**")
    table_rows: List[Dict[str, Any]] = []
    for row in result.area_details:
        table_rows.append({
            "Area": row.area,
            "Weight (%)": row.weight,
            "# Questions": row.num_questions,
            "Total Mapped": row.total_questions,
            "Area Score": row.area_score,
            "Weighted Score": row.weighted_score,
            "Coverage Gap": row.coverage_gap,
            "Gap Severity": row.gap_severity,
        })
    if table_rows:
        df = pd.DataFrame(table_rows)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "Weight (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Area Score": st.column_config.ProgressColumn(
                    "Area Score", min_value=0, max_value=100, format="%.1f",
                ),
                "Weighted Score": st.column_config.NumberColumn(format="%.2f"),
                "Coverage Gap": st.column_config.ProgressColumn(
                    "Coverage Gap", min_value=0, max_value=100, format="%.1f",
                ),
            },
        )

    # --- Top gap areas (chips) ------------------------------------------
    st.markdown("**Highest Gap Areas**")
    if not result.top_gap_areas:
        st.caption("No gaps detected - every weighted area scored at 100.")
    else:
        chips: List[str] = []
        for row in result.top_gap_areas:
            css = _severity_class_for_gap(row.get("gap_severity", "Medium"))
            chips.append(
                f'<span class="dash-pill {css}" style="margin:2px 6px 2px 0;'
                f'padding:4px 10px;border-radius:12px;font-size:0.85rem;">'
                f'{html.escape(str(row["area"]))} - Gap {row["coverage_gap"]:.1f}% '
                f'({html.escape(str(row["gap_severity"]))})</span>'
            )
        st.markdown("<div>" + "".join(chips) + "</div>", unsafe_allow_html=True)

    # --- Gap categories -------------------------------------------------
    st.markdown("**Gap Categories**")
    st.caption(
        "Cross-cutting categories rolled up from every mapped question. "
        "The top 3 highest-coverage areas per category are surfaced so "
        "you can trace a category gap back to the underlying operational "
        "areas."
    )
    cat_rows: List[Dict[str, Any]] = []
    for cat, breakdown in result.gap_categories.items():
        cat_rows.append({
            "Category": cat,
            "Score": breakdown.score,
            "Coverage Gap": breakdown.coverage_gap,
            "Severity": _readiness_gap_severity_from_score(breakdown.score),
            "Matched Questions": breakdown.matched_questions,
            "Top Areas": ", ".join(breakdown.top_areas) if breakdown.top_areas else "-",
        })
    if cat_rows:
        cat_df = pd.DataFrame(cat_rows)
        st.dataframe(
            cat_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%.1f",
                ),
                "Coverage Gap": st.column_config.ProgressColumn(
                    "Coverage Gap", min_value=0, max_value=100, format="%.1f",
                ),
            },
        )

    # --- Accuracy breakdown ---------------------------------------------
    with st.expander("Accuracy breakdown", expanded=False):
        breakdown = result.accuracy_breakdown or {}
        ev = float(breakdown.get("evidence_coverage") or 0.0)
        cons = float(breakdown.get("answer_consistency") or 0.0)
        mapping = float(breakdown.get("requirement_mapping_coverage") or 0.0)
        st.write(
            "**Formula:** `Accuracy = 40% Evidence Coverage + 30% Answer "
            "Consistency + 30% Requirement Mapping Coverage`."
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Evidence Coverage", f"{ev:.1f}%")
        with col_b:
            st.metric("Answer Consistency", f"{cons:.1f}%")
        with col_c:
            st.metric("Requirement Mapping", f"{mapping:.1f}%")
        st.caption(
            "Evidence coverage counts answers that reference an artefact, "
            "URL, policy or attachment. Answer consistency penalises "
            "'Fully Implemented' claims without matching evidence. "
            "Requirement mapping counts questions linked to at least one "
            "BRD or regulatory obligation."
        )

    # --- Missing evidence / mapping status ------------------------------
    if result.accuracy_breakdown:
        ev = float(result.accuracy_breakdown.get("evidence_coverage") or 0.0)
        mapping = float(result.accuracy_breakdown.get("requirement_mapping_coverage") or 0.0)
        if ev < 60.0:
            st.warning(
                f"Evidence Coverage is only {ev:.1f}%. Attach or reference "
                "policy documents, audit trails or artefacts in your "
                "answers to raise the Accuracy Score."
            )
        if mapping < 60.0:
            st.warning(
                f"Requirement Mapping Coverage is only {mapping:.1f}%. "
                "Re-run Agent 3 or edit the questionnaire so every question "
                "maps to at least one BRD requirement or obligation."
            )


def _readiness_gap_severity_from_score(score: float) -> str:
    """Convert a 0-100 score into the gap-severity vocabulary.

    Mirrors :func:`services.readiness_score.gap_severity` but reads from
    the score (not the gap) so the helper works when the caller only has
    the readiness value handy.
    """
    from services.readiness_score import gap_severity as _gs
    return _gs(max(0.0, 100.0 - float(score or 0.0)))


# ---------------------------------------------------------------------------
# Weighted impact panel (DORA demo profile)
# ---------------------------------------------------------------------------


def _impact_rating_css(rating: str) -> str:
    """Map an impact rating label to the shared CSS severity vocabulary.

    ``Very High Impact`` and ``High Impact`` land on ``crit`` / ``risk`` -
    the two most attention-grabbing colours in the palette so the tiles
    stand out on the dashboard.
    """
    r = str(rating or "").lower()
    if "very high" in r or "critical" in r:
        return "crit"
    if "high" in r:
        return "risk"
    if "medium" in r:
        return "watch"
    return "ready"


def _render_weighted_impact_panel(
    result: WeightedImpactResult,
    readiness_result: Optional[WeightedReadinessResult] = None,
) -> None:
    """Render the full weighted Impact section for the Dashboard page.

    Layout:

    1. Section header + rating pill.
    2. Four KPI cards (Overall Impact, Rating, Overall Priority, Coverage).
    3. Weighted Impact Factor Table (Factor, Weight, Score, Weighted, Rating).
    4. Top impacted business capabilities / processes / systems / controls /
       third parties - each as a compact list.
    5. Priority Areas table (High-Impact / Low-Readiness).
    6. Area x Function impact heatmap (top 25 rows).

    Impact and Readiness are calculated separately here - only the
    Priority column combines the two.
    """
    st.markdown(
        '<h4 class="rap-dash-hdr">Weighted Impact (DORA)</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Overall Impact Score is the weighted average of the seven DORA "
        "impact factors. Weights sum to 100 and are validated at startup. "
        "Impact answers *how much the regulation affects the organisation* - "
        "it is calculated independently of Readiness."
    )

    overall = float(result.overall_impact_score)
    imp_css = _impact_rating_css(result.impact_rating)
    priority = float(result.overall_priority_score)
    readiness_overall = (
        float(readiness_result.overall_readiness_score)
        if readiness_result is not None else 0.0
    )
    priority_css = _severity_class(priority) if priority < 40 else "risk" if priority < 60 else "crit"

    # --- KPI cards --------------------------------------------------------
    kpi_html = (
        '<div class="dash-kpis">'
        f'<div class="dash-kpi">'
        f'<div class="dash-kpi-label">Overall Impact Score</div>'
        f'<div class="dash-kpi-value">{overall:.1f} / 100</div>'
        f'<div class="dash-kpi-bar {imp_css}"><span style="width:{overall:.1f}%"></span></div>'
        f'</div>'
        f'<div class="dash-kpi">'
        f'<div class="dash-kpi-label">Impact Rating</div>'
        f'<div class="dash-kpi-value" style="font-size:1.05rem;">{html.escape(result.impact_rating)}</div>'
        f'<div class="dash-kpi-bar {imp_css}"><span style="width:{overall:.1f}%"></span></div>'
        f'</div>'
        f'<div class="dash-kpi" title="Priority = Impact * (100 - Readiness) / 100">'
        f'<div class="dash-kpi-label">Overall Priority</div>'
        f'<div class="dash-kpi-value">{priority:.1f}</div>'
        f'<div class="dash-kpi-bar {priority_css}"><span style="width:{priority:.1f}%"></span></div>'
        f'</div>'
        f'<div class="dash-kpi" title="Weighted readiness reported for context">'
        f'<div class="dash-kpi-label">Overall Readiness (context)</div>'
        f'<div class="dash-kpi-value">{readiness_overall:.1f}%</div>'
        f'<div class="dash-kpi-bar {_severity_class(readiness_overall)}"><span style="width:{readiness_overall:.1f}%"></span></div>'
        f'</div>'
        '</div>'
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

    # --- Weighted factor table -------------------------------------------
    st.markdown("**Weighted Impact Factors**")
    table_rows: List[Dict[str, Any]] = []
    for row in result.factor_details:
        table_rows.append({
            "Factor": row.factor,
            "Weight (%)": row.weight,
            "Factor Score": row.factor_score,
            "Weighted Score": row.weighted_score,
            "Rating": row.rating,
        })
    if table_rows:
        df = pd.DataFrame(table_rows)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "Weight (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Factor Score": st.column_config.ProgressColumn(
                    "Factor Score", min_value=0, max_value=100, format="%.1f",
                ),
                "Weighted Score": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    # Rationale per factor (short caption block so users can trace numbers).
    with st.expander("Factor rationale + signals", expanded=False):
        for row in result.factor_details:
            st.markdown(
                f"**{html.escape(row.factor)}** - "
                f"score **{row.factor_score:.1f}** "
                f"({html.escape(row.rating)}) - "
                f"weighted **{row.weighted_score:.2f}**"
            )
            st.caption(row.rationale)
            for sig in row.signals[:5]:
                st.write(f"- {sig}")

    # --- Top impacted lists ----------------------------------------------
    st.markdown("**Top Impacted Business Areas**")
    top_caps = result.top_impacted_business_capabilities
    if top_caps:
        cap_df = pd.DataFrame([
            {"Business Capability": row.get("area", ""), "Signal Hits": row.get("hit_count", 0)}
            for row in top_caps
        ])
        st.dataframe(cap_df, width="stretch", hide_index=True)
    else:
        st.caption("No obligation / requirement signals mapped to business areas yet.")

    _cols = st.columns(2)
    with _cols[0]:
        st.markdown("**Top Impacted Processes**")
        if result.top_impacted_processes:
            for line in result.top_impacted_processes:
                st.write(f"- {line}")
        else:
            st.caption("No process requirements above the priority threshold.")
        st.markdown("**Top Impacted Systems / Technology**")
        if result.top_impacted_systems:
            for line in result.top_impacted_systems:
                st.write(f"- {line}")
        else:
            st.caption("No technology-related requirements detected.")
    with _cols[1]:
        st.markdown("**Top Impacted Controls**")
        if result.top_impacted_controls:
            for line in result.top_impacted_controls:
                st.write(f"- {line}")
        else:
            st.caption("No control expectations captured in the obligations set.")
        st.markdown("**Top Impacted Third Parties / Vendors**")
        if result.top_impacted_third_parties:
            for line in result.top_impacted_third_parties:
                st.write(f"- {line}")
        else:
            st.caption("No third-party-themed obligations detected.")

    # --- Priority areas (high-impact / low-readiness) --------------------
    st.markdown("**High-Impact / Low-Readiness Priority Areas**")
    st.caption(
        "Priority = Impact x (100 - Readiness) / 100. Higher numbers "
        "signal an area where the regulation hits hard *and* the "
        "organisation is under-prepared."
    )
    if result.priority_areas:
        pa_rows = [
            {
                "Area": row.area,
                "Impact Score": row.impact_score,
                "Readiness Score": row.readiness_score,
                "Priority Score": row.priority_score,
                "Obligation Hits": row.signal_count,
            }
            for row in result.priority_areas[:10]
        ]
        pa_df = pd.DataFrame(pa_rows)
        st.dataframe(
            pa_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Impact Score": st.column_config.ProgressColumn(
                    "Impact Score", min_value=0, max_value=100, format="%.1f",
                ),
                "Readiness Score": st.column_config.ProgressColumn(
                    "Readiness Score", min_value=0, max_value=100, format="%.1f",
                ),
                "Priority Score": st.column_config.ProgressColumn(
                    "Priority Score", min_value=0, max_value=100, format="%.1f",
                ),
            },
        )
    else:
        st.caption("No area-level priority computed - no obligations mapped.")

    # --- Impact heatmap (Area x Function) --------------------------------
    if result.heatmap_rows:
        st.markdown("**Impact Heatmap (Area x Function)**")
        hm_rows = [
            {
                "Area": row.get("area", ""),
                "Function": row.get("function", ""),
                "Impact Score": row.get("impact_score", 0.0),
                "Readiness Score": row.get("readiness_score", 0.0),
                "Priority Score": row.get("priority_score", 0.0),
                "Requirement Hits": row.get("signal_count", 0),
            }
            for row in result.heatmap_rows[:25]
        ]
        hm_df = pd.DataFrame(hm_rows)
        st.dataframe(
            hm_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Impact Score": st.column_config.ProgressColumn(
                    "Impact Score", min_value=0, max_value=100, format="%.1f",
                ),
                "Readiness Score": st.column_config.ProgressColumn(
                    "Readiness Score", min_value=0, max_value=100, format="%.1f",
                ),
                "Priority Score": st.column_config.ProgressColumn(
                    "Priority Score", min_value=0, max_value=100, format="%.1f",
                ),
            },
        )


def _render_dashboard_hero(
    *,
    readiness_pct: float,
    confidence_pct: float,
    impact_pct: Optional[float] = None,
) -> None:
    """Render the two-tile hero strip for overall Impact and Readiness.

    Impact and Readiness are calculated **independently** now:

    - Readiness comes from the weighted readiness model (7 DORA areas).
    - Impact comes from the weighted impact model (7 DORA factors).

    If ``impact_pct`` is supplied (the weighted impact overall) we use
    it directly; otherwise we fall back to the legacy derivation
    ``100 - readiness`` so this function is safe to call from paths that
    have not yet computed a weighted impact result.
    """
    readiness = max(0.0, min(100.0, readiness_pct))
    if impact_pct is None:
        impact = max(0.0, min(100.0, 100.0 - readiness))
    else:
        impact = max(0.0, min(100.0, float(impact_pct)))
    read_label, read_css = _readiness_severity_from_score(readiness)
    imp_label, imp_css = _impact_severity_from_score(impact)

    # Both hero tiles now surface their severity label as a pill so the
    # user can read the rating (Critical / At Risk / Watch / Ready) at
    # a glance without having to interpret the raw percentage.
    html_out = (
        '<div class="dash-hero">'
        f'<div class="dash-hero-tile impact-tile {imp_css}">'
        '<div class="dash-hero-cap">Overall Impact Score</div>'
        f'<div class="dash-hero-value">{impact:.1f}%</div>'
        f'<div class="dash-hero-sub">Impact severity: '
        f'<span class="dash-pill {imp_css}">{imp_label}</span></div>'
        f'<div class="dash-hero-bar {imp_css}"><span style="width:{impact:.1f}%"></span></div>'
        '</div>'
        f'<div class="dash-hero-tile readiness-tile {read_css}">'
        '<div class="dash-hero-cap">Overall Readiness Score</div>'
        f'<div class="dash-hero-value">{readiness:.1f}%</div>'
        f'<div class="dash-hero-sub">Readiness rating: '
        f'<span class="dash-pill {read_css}">{read_label}</span></div>'
        f'<div class="dash-hero-bar {read_css}"><span style="width:{readiness:.1f}%"></span></div>'
        '</div>'
        '</div>'
    )
    st.markdown(html_out, unsafe_allow_html=True)


def _render_dashboard_readiness_cards(area_summary: Dict[str, Dict[str, Any]]) -> None:
    """Render the Readiness Overview By Area as coloured progress cards.

    Sorted by readiness ascending so the least-ready areas surface at the
    top - the same ranking a remediation lead would want.
    """
    if not area_summary:
        st.info("No area-level scores yet - answer more closed questions.")
        return
    rows: List[Tuple[str, float, str, int]] = []
    for name, summary in area_summary.items():
        try:
            comp = float(summary.get("compliance_score_pct") or summary.get("Compliance %") or 0.0)
        except (TypeError, ValueError):
            comp = 0.0
        status = str(summary.get("CXO status") or "").strip() or "—"
        try:
            qcount = int(summary.get("questions_scored") or summary.get("Questions scored") or 0)
        except (TypeError, ValueError):
            qcount = 0
        rows.append((str(name), comp, status, qcount))
    rows.sort(key=lambda r: r[1])

    cards: List[str] = ['<div class="dash-cards readiness-grid">']
    for name, comp, status, qcount in rows:
        css = _severity_label_from_status(status) or _severity_class(comp)
        cards.append(
            f'<div class="dash-card {css}">'
            f'<div class="dash-card-title">{html.escape(name)}</div>'
            f'<div class="dash-card-meta">'
            f'<span class="dash-pill {css}">{html.escape(status)}</span> '
            f'&nbsp;<b>{comp:.1f}%</b> readiness &nbsp;\u00b7&nbsp; '
            f'<b>{qcount}</b> Q scored'
            '</div>'
            f'<div class="dash-card-bar {css}"><span style="width:{max(0.0, min(100.0, comp)):.1f}%"></span></div>'
            '</div>'
        )
    cards.append('</div>')
    st.markdown("".join(cards), unsafe_allow_html=True)


def _render_dashboard_impact_cards(area_summary: Dict[str, Dict[str, Any]]) -> None:
    """Render an area-wise impact assessment view.

    Impact per area is sourced from the weighted-impact model
    (``st.session_state["weighted_impact"].priority_areas``) whenever
    available so every impact number on the dashboard comes from the
    same single source of truth. When an area is present in
    ``area_summary`` but has no weighted-impact entry (e.g. because no
    obligation was tagged to it), we fall back to the legacy
    ``100 - readiness`` derivation - this preserves the previous UI so
    nothing disappears when the impact model has not been computed yet.
    """
    if not area_summary:
        st.info("No area-level scores yet - answer more closed questions.")
        return

    weighted_impact: Optional[WeightedImpactResult] = st.session_state.get(
        "weighted_impact"
    )
    impact_by_area: Dict[str, float] = {}
    if weighted_impact is not None:
        for row in weighted_impact.priority_areas:
            impact_by_area[str(row.area)] = float(row.impact_score)

    rows: List[Tuple[str, float, float, str]] = []
    for name, summary in area_summary.items():
        try:
            comp = float(summary.get("compliance_score_pct") or summary.get("Compliance %") or 0.0)
        except (TypeError, ValueError):
            comp = 0.0
        # Prefer the weighted per-area impact; fall back to the legacy
        # `100 - readiness` derivation only when we have no signal.
        impact = impact_by_area.get(str(name))
        if impact is None:
            impact = max(0.0, min(100.0, 100.0 - comp))
        else:
            impact = max(0.0, min(100.0, float(impact)))
        status = str(summary.get("CXO status") or "").strip() or "—"
        rows.append((str(name), comp, impact, status))
    rows.sort(key=lambda r: -r[2])

    cards: List[str] = ['<div class="dash-cards impact-grid">']
    for name, readiness, impact, status in rows:
        label, css = _impact_severity_from_score(impact)
        cards.append(
            f'<div class="dash-card {css}">'
            f'<div class="dash-card-title">{html.escape(name)}</div>'
            f'<div class="dash-card-meta">'
            f'<span class="dash-pill {css}">{label}</span> '
            f'&nbsp;<b>{impact:.1f}%</b> impact'
            '</div>'
            f'<div class="dash-card-bar {css}"><span style="width:{impact:.1f}%"></span></div>'
            '</div>'
        )
    cards.append('</div>')
    st.markdown("".join(cards), unsafe_allow_html=True)


def _render_dashboard_kpis(
    *,
    readiness_pct: float,
    confidence_pct: float,
    answered: int,
    total: int,
    pairs: int,
    high_impact_area_count: int,
) -> None:
    """Render the KPI row shown just under the hero.

    The duplicate Readiness / Impact tiles have been removed - the hero
    strip already surfaces both percentages. The row now shows only the
    supporting KPIs (evaluation confidence, answered coverage,
    high-impact area count) with inline mini progress bars.
    """
    conf_class = _severity_class(confidence_pct)
    coverage_pct = round((answered / total) * 100.0, 1) if total else 0.0
    coverage_class = _severity_class(coverage_pct)

    # Native browser tooltip on the Evaluation Confidence tile so the
    # dashboard also documents the gap that composes the composite score.
    conf_tooltip_attr = html.escape(
        _confidence_gap_tooltip(
            st.session_state.get("confidence_assessment"),
            kind="evaluation",
        ),
        quote=True,
    )

    html_out = (
        '<div class="dash-kpis">'
        f'<div class="dash-kpi" title="{conf_tooltip_attr}">'
        f'<div class="dash-kpi-label">Evaluation Confidence '
        f'<span class="dash-help-hint" aria-hidden="true">ⓘ</span></div>'
        f'<div class="dash-kpi-value">{confidence_pct:.1f}%</div>'
        f'<div class="dash-kpi-bar {conf_class}"><span style="width:{confidence_pct:.1f}%"></span></div></div>'
        f'<div class="dash-kpi"><div class="dash-kpi-label">Answered / Applicable</div>'
        f'<div class="dash-kpi-value">{answered} / {total}</div>'
        f'<div class="dash-kpi-bar {coverage_class}"><span style="width:{coverage_pct:.1f}%"></span></div></div>'
        f'<div class="dash-kpi"><div class="dash-kpi-label">High-Impact Areas</div>'
        f'<div class="dash-kpi-value">{high_impact_area_count}</div>'
        f'<div class="dash-kpi-bar {"crit" if high_impact_area_count > 0 else "ready"}">'
        f'<span style="width:{min(100, high_impact_area_count * 20)}%"></span></div></div>'
        '</div>'
    )
    st.markdown(html_out, unsafe_allow_html=True)


def _classify_severity_distribution(
    values: Iterable[Optional[float]],
) -> Dict[str, int]:
    """Tally how many scores fall in each of the four canonical severity
    bands. Non-numeric / ``None`` scores are silently skipped so the
    counts always reflect real observations.
    """
    buckets = {"crit": 0, "risk": 0, "watch": 0, "ready": 0}
    for raw in values:
        if raw is None:
            continue
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        cls = _severity_class(score)
        if cls in buckets:
            buckets[cls] += 1
    return buckets


def _render_dashboard_legend(
    *,
    area_summary: Optional[Dict[str, Dict[str, Any]]] = None,
    function_summary: Optional[Dict[str, Dict[str, Any]]] = None,
    pair_scores: Optional[Dict[Any, float]] = None,
) -> None:
    """Live severity-distribution strip.

    Replaces the old static legend with four colourful cards - one per
    severity band - each showing the number of items currently in that
    band (areas + functions + area x function pairs) and its share of
    the total scored items. Updates automatically on every rerun because
    it is fed the freshly evaluated ``area_summary`` / ``function_summary``
    / ``pair_scores`` from the scoring engine.
    """
    all_scores: List[float] = []

    def _collect(summary: Optional[Dict[str, Dict[str, Any]]]) -> None:
        if not summary:
            return
        for row in summary.values():
            if not isinstance(row, dict):
                continue
            raw = row.get("compliance_pct")
            if raw is None:
                raw = row.get("score_pct")
            if raw is None:
                raw = row.get("readiness_pct")
            if raw is None:
                continue
            try:
                all_scores.append(float(raw))
            except (TypeError, ValueError):
                continue

    _collect(area_summary)
    _collect(function_summary)
    if pair_scores:
        for raw in pair_scores.values():
            try:
                all_scores.append(float(raw))
            except (TypeError, ValueError):
                continue

    buckets = _classify_severity_distribution(all_scores)
    total = sum(buckets.values()) or 1

    bands = [
        ("crit",  "Critical", "Readiness < 25%",       "Immediate action"),
        ("risk",  "At Risk",  "Readiness 25 - 50%",    "Elevated attention"),
        ("watch", "Watch",    "Readiness 50 - 75%",    "Monitor and refine"),
        ("ready", "Ready",    "Readiness \u2265 75%",  "Meeting expectations"),
    ]

    cards: List[str] = ['<div class="sev-strip">']
    for css, title, band, hint in bands:
        count = buckets[css]
        pct = (count / total) * 100.0 if total else 0.0
        cards.append(
            f'<div class="sev-card {css}" title="{html.escape(hint)}">'
            f'  <div class="sev-head">'
            f'    <div class="sev-title"><span class="sev-dot"></span>{title}</div>'
            f'    <div class="sev-count">{count}</div>'
            f'  </div>'
            f'  <div class="sev-range">{band}</div>'
            f'  <div class="sev-bar"><span style="width:{pct:.1f}%"></span></div>'
            f'  <div class="sev-share">{pct:.0f}% of {total} scored item(s)</div>'
            f'</div>'
        )
    cards.append('</div>')

    st.markdown(
        '<div class="sev-caption">'
        '<span class="sev-caption-title">Severity distribution</span>'
        '<span class="sev-caption-hint">Counts across scored areas, functions and area \u00d7 function pairs</span>'
        '<span class="sev-caption-live">Live</span>'
        '</div>'
        + "".join(cards),
        unsafe_allow_html=True,
    )


def _render_dashboard_summary_cards(df: pd.DataFrame, label: str) -> None:
    """Render an aggregate table (area or function summary) as a grid of
    severity-coloured cards, each with a mini progress bar and the CXO
    action from the scoring engine.
    """
    if df is None or df.empty:
        st.info(f"No {label.lower()} scores match the current filter.")
        return

    cards_html: List[str] = ['<div class="dash-cards">']
    for _, row in df.iterrows():
        name = html.escape(str(row.get(label) or "—"))
        try:
            comp = float(row.get("Compliance %") or 0.0)
        except (TypeError, ValueError):
            comp = 0.0
        status_label = str(row.get("CXO status") or "").strip() or "—"
        css = _severity_label_from_status(status_label) or _severity_class(comp)
        questions_scored = row.get("Questions scored") or 0
        action = html.escape(str(row.get("Recommended executive action") or ""))
        cards_html.append(
            f'<div class="dash-card {css}">'
            f'<div class="dash-card-title">{name}</div>'
            f'<div class="dash-card-meta">'
            f'<span class="dash-pill {css}">{html.escape(status_label)}</span> '
            f'&nbsp;<b>{comp:.1f}%</b> compliance &nbsp;·&nbsp; '
            f'<b>{int(questions_scored)}</b> Q scored'
            f'</div>'
            f'<div class="dash-card-bar {css}"><span style="width:{max(0.0, min(100.0, comp)):.1f}%"></span></div>'
            f'<div class="dash-card-body">{action}</div>'
            f'</div>'
        )
    cards_html.append("</div>")
    st.markdown("".join(cards_html), unsafe_allow_html=True)


def _render_dashboard_top_gap_cards(top_gaps: Any) -> None:
    """Render ``scoring.top_gaps`` as a grid of severity-coloured cards.
    Falls back to a friendly caption when no requirement-level scores
    exist yet.
    """
    if not top_gaps:
        st.caption("No requirement scores yet — answer more closed questions.")
        return

    cards: List[str] = ['<div class="dash-cards">']
    for gap in top_gaps:
        rid = str(gap.get("requirement_id") or "—")
        try:
            comp = float(gap.get("compliance_pct") or 0.0)
        except (TypeError, ValueError):
            comp = 0.0
        css = _severity_class(comp)
        label = {
            "crit": "Critical",
            "risk": "At risk",
            "watch": "Watch",
            "ready": "Ready",
            "none": "—",
        }.get(css, "—")
        cards.append(
            f'<div class="dash-card {css}">'
            f'<div class="dash-card-title">{html.escape(rid)}</div>'
            f'<div class="dash-card-meta">'
            f'<span class="dash-pill {css}">{label}</span> '
            f'&nbsp;<b>{comp:.1f}%</b> compliance'
            f'</div>'
            f'<div class="dash-card-bar {css}"><span style="width:{max(0.0, min(100.0, comp)):.1f}%"></span></div>'
            f'</div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def _render_dashboard_area_recommendations(
    recs: List[Any], area_summary: Dict[str, Dict[str, Any]]
) -> None:
    """Render one recommendation card per impacted area, each expanded
    into 3-4 concrete action bullets.

    - Bullets are synthesised deterministically from the Agent 4 output
      for the area (title, rationale, suggested action, owner, horizon,
      branch-log evidence) plus a per-severity playbook fallback so we
      always have at least three bullets.
    - Severity of the area card matches the executive HIGH / MEDIUM /
      LOW ladder used on the impact cards above, keeping the whole page
      internally consistent.
    """
    if not area_summary:
        st.info("No area scores available yet.")
        return

    def _get(r: Any, key: str, default: Any = "") -> Any:
        return getattr(r, key, None) if hasattr(r, key) else r.get(key, default)

    grouped: Dict[str, List[Any]] = {}
    for r in recs or []:
        area = str(_get(r, "area") or "").strip() or "Unmapped"
        grouped.setdefault(area, []).append(r)

    sorted_areas = sorted(
        area_summary.keys(),
        key=lambda a: float(
            area_summary[a].get("compliance_score_pct")
            or area_summary[a].get("Compliance %")
            or 0.0
        ),
    )

    cards: List[str] = ['<div class="dash-rec-grid">']
    for area in sorted_areas:
        summary = area_summary.get(area) or {}
        try:
            readiness = float(
                summary.get("compliance_score_pct")
                or summary.get("Compliance %")
                or 0.0
            )
        except (TypeError, ValueError):
            readiness = 0.0
        impact = max(0.0, min(100.0, 100.0 - readiness))
        status = str(summary.get("CXO status") or "").strip() or "\u2014"
        label, css = _impact_severity_from_score(impact)
        area_recs = grouped.get(area, [])
        bullets = _build_area_recommendation_bullets(
            area=area,
            readiness=readiness,
            impact=impact,
            status=status,
            severity_label=label,
            area_recs=area_recs,
        )
        bullet_html = "".join(
            f'<li><b>{html.escape(b["title"])}.</b> {html.escape(b["body"])}</li>'
            for b in bullets
        )
        exec_action = html.escape(str(summary.get("Recommended executive action") or ""))
        exec_action_html = (
            f'<div class="dash-rec-exec">{exec_action}</div>' if exec_action else ""
        )
        cards.append(
            f'<div class="dash-rec-card {css}">'
            f'<div class="dash-rec-hdr">'
            f'<div class="dash-rec-title">{html.escape(area)}</div>'
            f'<div class="dash-rec-tags">'
            f'<span class="dash-pill {css}">{label}</span>'
            f'<span class="dash-rec-scores">Readiness <b>{readiness:.1f}%</b> \u00b7 Impact <b>{impact:.1f}%</b></span>'
            '</div>'
            '</div>'
            f'{exec_action_html}'
            f'<ul class="dash-rec-bullets">{bullet_html}</ul>'
            '</div>'
        )
    cards.append('</div>')
    st.markdown("".join(cards), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Area-specific recommendation playbook (per-area × per-severity content)
# ---------------------------------------------------------------------------
#
# Each area entry carries:
#   - ``meta``: forum, executive owner, one-line "why it matters"
#   - ``tiers``: a dict of severity -> {first_move, actions, evidence, success}
#     with concrete, domain-appropriate content for that combination.
#
# Severity keys are the canonical labels used across the app:
# "critical" (< 25% readiness), "at_risk" (25-50%), "watch" (50-75%),
# "ready" (>= 75%). The lookup is prefix-based so "Cyber Security" also
# resolves "Cyber Security & Resilience", and unknown areas fall back to
# a neutral executive playbook.
#
# Adding a new area = add a new entry with 4 tiers × 4 bullet fragments.
# Rewording an existing severity = edit exactly one string.

_SEVERITY_KEYS = ("critical", "at_risk", "watch", "ready")


def _canon_severity(label: str) -> str:
    """Normalise a severity label (``Critical`` / ``At Risk`` / ``Watch``
    / ``Ready``) to the tier key used by the playbook."""
    s = (label or "").strip().lower()
    if s == "critical":
        return "critical"
    if s in ("at risk", "at_risk"):
        return "at_risk"
    if s == "watch":
        return "watch"
    if s == "ready":
        return "ready"
    return "watch"


_AREA_PLAYBOOK: Dict[str, Dict[str, Any]] = {
    "Governance": {
        "meta": {
            "forum": "management body / board risk committee",
            "owner": "Chief Compliance Officer",
            "why": "Governance gaps expose the whole DORA programme to supervisory findings and undermine every downstream control.",
        },
        "tiers": {
            "critical": {
                "first_move": "Convene an extraordinary board risk session, table a DORA-specific charter and reset delegated authorities so ICT-risk decisions are traceable to a named accountable executive.",
                "actions": "Publish a board-approved DORA governance charter, refresh the three-lines-of-defence RACI, and stand up a fortnightly management-body update until controls stabilise.",
                "evidence": "Board pack extracts, revised terms of reference, RACI matrix and the last 6 sets of committee minutes evidencing effective challenge.",
                "success": "Documented board approval of the DORA governance model plus two consecutive committee cycles with recorded challenge on ICT-risk items.",
            },
            "at_risk": {
                "first_move": "Escalate to the next scheduled board risk committee with a red-flagged governance paper covering charter, mandate and accountability gaps.",
                "actions": "Close open governance actions from the last 6 months, refresh delegated authorities and publish an updated ICT-risk policy for board sign-off.",
                "evidence": "Committee minutes, action-log burndown, updated policy set and independent challenge notes from Risk / Audit.",
                "success": "All open governance actions closed within 60 days and one clean quarter of board minutes evidencing DORA-specific challenge.",
            },
            "watch": {
                "first_move": "Bring a themed governance update to the next scheduled committee to close residual policy and delegation gaps.",
                "actions": "Confirm annual attestation of the DORA governance framework, refresh KRIs surfaced to the board and validate escalation thresholds.",
                "evidence": "Annual attestation memo, KRI pack extract and delegation-of-authority matrix.",
                "success": "Committee attestation signed on schedule and no open governance findings older than 90 days.",
            },
            "ready": {
                "first_move": "Keep governance oversight on the standard committee agenda and preserve the current cadence of ICT-risk reporting to the board.",
                "actions": "Run the annual governance refresh, benchmark against peer disclosures and archive prior-cycle evidence.",
                "evidence": "Annual governance review memo, peer benchmark note and archived committee packs.",
                "success": "Zero governance-related supervisory observations at the next thematic review.",
            },
        },
    },
    "Risk Management": {
        "meta": {
            "forum": "enterprise risk committee",
            "owner": "Chief Risk Officer",
            "why": "Weak ICT-risk oversight lets residual risk drift beyond appetite and undermines every downstream control decision.",
        },
        "tiers": {
            "critical": {
                "first_move": "Freeze ICT-risk appetite decisions pending a full re-baselining of the ICT-risk register, and appoint a dedicated risk lead to shepherd the rebuild.",
                "actions": "Rebuild the ICT-risk register end-to-end, redefine tolerance thresholds and stress-test each Tier-1 scenario with a fresh challenge session.",
                "evidence": "Re-baselined ICT-risk register, revised tolerance schedule and Tier-1 scenario stress-test outputs with independent challenge notes.",
                "success": "New risk register approved by the ERC and residual risk mapped inside tolerance for at least the top 10 ICT scenarios.",
            },
            "at_risk": {
                "first_move": "Table a red-status ICT-risk update at the next ERC with a remediation plan for the top 5 out-of-tolerance risks.",
                "actions": "Refresh the top-25 ICT risks, revalidate tolerance breaches and drive owner sign-off on mitigation plans with dated milestones.",
                "evidence": "Updated top-25 heat map, breach log with mitigation plans and owner-signed remediation charters.",
                "success": "Top-5 out-of-tolerance risks brought back inside appetite within 90 days.",
            },
            "watch": {
                "first_move": "Refresh the ICT-risk challenge cadence at the next ERC and validate that KRIs still trigger the intended escalation.",
                "actions": "Re-test KRI trigger points, review appetite thresholds for drift and refresh scenario libraries with fresh threat intel.",
                "evidence": "KRI back-test results, threshold review memo and updated scenario library.",
                "success": "All KRIs proven to trigger inside their SLA and no scenario library entry older than 12 months.",
            },
            "ready": {
                "first_move": "Retain the current ICT-risk cadence at the ERC and roll the framework into the annual risk review.",
                "actions": "Annual back-test of ICT-risk tolerances, peer-benchmark KRIs and archive the year's risk artefacts.",
                "evidence": "Annual back-test report, benchmark memo and archived risk register versions.",
                "success": "Annual risk attestation signed with zero material findings.",
            },
        },
    },
    "Business Continuity": {
        "meta": {
            "forum": "operational resilience steering committee",
            "owner": "Business Continuity Manager",
            "why": "Resilience gaps translate directly into breach of DORA Article 11 impact tolerances and undermine severe-but-plausible scenario response.",
        },
        "tiers": {
            "critical": {
                "first_move": "Halt any new critical-service go-lives, run a rapid BIA on the top 5 critical services and treat as a live crisis until impact tolerances are demonstrably deliverable.",
                "actions": "Rebuild the BIA, redefine impact tolerances and execute a severe-but-plausible test for each Tier-1 service within 30 days.",
                "evidence": "Refreshed BIA, tolerance schedule, scenario-test playbooks and post-exercise reports with root-cause and remediation actions.",
                "success": "Every Tier-1 critical service demonstrably recoverable within its stated impact tolerance in a live test.",
            },
            "at_risk": {
                "first_move": "Escalate the resilience gaps to the operational resilience committee with a dated recovery plan for the top exposed services.",
                "actions": "Refresh the BIA for the top 20 critical services, close open scenario-test findings and re-run the two weakest scenarios.",
                "evidence": "Updated BIA extracts, scenario-test findings register and re-run reports.",
                "success": "All scenario-test findings closed within SLA and re-run scenarios pass tolerance.",
            },
            "watch": {
                "first_move": "Confirm annual tolerance attestation is on track and validate the exercise calendar for the coming year.",
                "actions": "Run the scheduled scenario cycle, refresh the third-party dependency map and stress-test the incident bridge.",
                "evidence": "Exercise calendar, third-party dependency map and bridge-test after-action review.",
                "success": "Scheduled scenarios executed on time with zero critical after-action items outstanding.",
            },
            "ready": {
                "first_move": "Preserve the current test cadence and roll resilience into the annual DORA attestation.",
                "actions": "Annual resilience review, external benchmark against peer tests and archive prior exercise evidence.",
                "evidence": "Annual review memo, benchmark note and archived scenario evidence.",
                "success": "Annual attestation signed with tolerances demonstrably met.",
            },
        },
    },
    "Incident": {
        "meta": {
            "forum": "ICT incident review board",
            "owner": "ICT Incident Response Lead",
            "why": "Incident classification and reporting gaps trigger DORA Article 19 notification failures and expose the firm to supervisory escalation.",
        },
        "tiers": {
            "critical": {
                "first_move": "Stand up a 24×7 incident bridge, rehearse the DORA classification workflow against the last 20 incidents and pre-stage regulator notifications.",
                "actions": "Rebuild the incident classification model, dry-run the DORA 4-hour / 72-hour timelines and align severity thresholds with the ERC.",
                "evidence": "Classification decision tree, walk-back of last 20 incidents, dry-run timelines and pre-staged regulator notification templates.",
                "success": "Every dry-run classification decision defensible and the 4/72 hour timeline demonstrably met on at least three rehearsed incidents.",
            },
            "at_risk": {
                "first_move": "Bring a themed incident-management update to the incident review board with a remediation plan for reporting and classification gaps.",
                "actions": "Close the open post-incident actions, refresh the notification templates and rehearse the DORA timeline with front-line teams.",
                "evidence": "Post-incident action log, refreshed notification templates and rehearsal after-action reviews.",
                "success": "All open post-incident actions closed and one full rehearsal completed within SLA.",
            },
            "watch": {
                "first_move": "Confirm the quarterly rehearsal cadence and validate that classification thresholds still align with recent incident trends.",
                "actions": "Run the quarterly rehearsal, refresh the incident taxonomy against the latest trend data and update the on-call schedule.",
                "evidence": "Rehearsal report, refreshed taxonomy and up-to-date on-call schedule.",
                "success": "Quarterly rehearsal signed off with zero critical findings.",
            },
            "ready": {
                "first_move": "Preserve the current incident-response cadence and roll capability into the annual attestation.",
                "actions": "Annual review of incident metrics, external benchmark of MTTR/MTTC and archive rehearsal evidence.",
                "evidence": "Annual metrics memo, benchmark note and archived rehearsal evidence.",
                "success": "Annual attestation signed with all DORA notification timelines demonstrably met.",
            },
        },
    },
    "Third": {
        "meta": {
            "forum": "third-party risk oversight forum",
            "owner": "Head of Vendor / Third-Party Risk",
            "why": "Third-party gaps become concentration risk under the DORA critical-provider regime and expose contracts to unenforceable obligations.",
        },
        "tiers": {
            "critical": {
                "first_move": "Freeze on-boarding of new ICT third parties, treat the critical-provider register as a live artefact and stand up a war room to remediate contracts and exit plans.",
                "actions": "Re-tier every ICT vendor against Chapter V criteria, renegotiate contract clauses on audit, sub-contracting and exit, and re-execute exit tests for Tier-1 providers.",
                "evidence": "Re-tiered critical-provider register, remediated contract clauses, updated exit playbooks and exit-test evidence for each Tier-1 provider.",
                "success": "Every Tier-1 provider covered by a compliant contract clause set and a demonstrably executable exit plan.",
            },
            "at_risk": {
                "first_move": "Bring a red-status third-party update to the oversight forum focused on Chapter V clause gaps and exit-plan freshness.",
                "actions": "Remediate contract clauses on the top-30 providers, refresh sub-contractor visibility and re-run exit walkthroughs for the weakest 5 providers.",
                "evidence": "Contract remediation tracker, sub-contractor register and exit-walkthrough after-action reviews.",
                "success": "Top-30 provider contracts remediated within 90 days and sub-contractor visibility current.",
            },
            "watch": {
                "first_move": "Confirm the annual third-party attestation timeline and validate concentration KRIs at the oversight forum.",
                "actions": "Refresh the concentration heat-map, re-run the annual exit test for one Tier-1 provider and update the sub-contractor register.",
                "evidence": "Concentration heat-map, annual exit-test evidence and refreshed sub-contractor register.",
                "success": "Annual attestation signed with concentration KRIs inside tolerance.",
            },
            "ready": {
                "first_move": "Retain current provider oversight cadence and roll third-party into the annual DORA attestation.",
                "actions": "Annual third-party review, benchmark critical-provider KPIs and archive prior-cycle exit-test evidence.",
                "evidence": "Annual review memo, benchmark note and archived exit-test evidence.",
                "success": "Annual attestation signed with zero material third-party findings.",
            },
        },
    },
    "Cyber": {
        "meta": {
            "forum": "cyber steering committee",
            "owner": "Chief Information Security Officer",
            "why": "Cyber gaps drive the residual likelihood of every operational-risk scenario and are the first line supervisors probe under DORA.",
        },
        "tiers": {
            "critical": {
                "first_move": "Convene a cyber war room, freeze non-essential change, and pre-scope a Threat-Led Penetration Test (TLPT) against Tier-1 services within 30 days.",
                "actions": "Halve open critical vulnerabilities, expand MITRE ATT&CK detection coverage over the crown-jewels estate and rehearse ransomware playbooks with the incident bridge.",
                "evidence": "12-month vulnerability trend, ATT&CK coverage matrix, TLPT scoping brief, ransomware playbook rehearsal report.",
                "success": "Open critical vulnerabilities halved in 30 days and TLPT scope agreed with independent testers.",
            },
            "at_risk": {
                "first_move": "Bring a red-status cyber update to the steering committee focused on detection coverage and TLPT readiness.",
                "actions": "Close the open red-team findings, extend detection coverage to gap areas and run a targeted purple-team exercise on the weakest control family.",
                "evidence": "Red-team findings tracker, detection coverage matrix and purple-team after-action review.",
                "success": "All red-team findings closed within SLA and detection coverage above the agreed target.",
            },
            "watch": {
                "first_move": "Confirm the annual TLPT calendar and validate that detective controls still fire against current-year threat scenarios.",
                "actions": "Run the annual purple-team cycle, refresh the ATT&CK coverage baseline and validate response SLAs on the SOC.",
                "evidence": "Annual purple-team report, refreshed ATT&CK baseline and SOC SLA report.",
                "success": "Annual purple-team signed off with zero unmitigated critical findings.",
            },
            "ready": {
                "first_move": "Retain the current cyber cadence and roll defensive posture into the annual DORA attestation.",
                "actions": "Annual cyber posture review, peer benchmark of ATT&CK coverage and archive prior-cycle test evidence.",
                "evidence": "Annual posture memo, benchmark note and archived TLPT / purple-team evidence.",
                "success": "Annual attestation signed with zero unmitigated critical cyber findings.",
            },
        },
    },
    "Technology": {
        "meta": {
            "forum": "technology risk & operations forum",
            "owner": "Chief Technology Officer",
            "why": "Unresolved technology gaps propagate to every business service that runs on the platform and degrade the reliability of every critical business flow.",
        },
        "tiers": {
            "critical": {
                "first_move": "Institute a change freeze on Tier-1 platforms, force a full asset-inventory reconciliation and stand up a daily production stability call.",
                "actions": "Reconcile the asset inventory, force a baseline configuration on Tier-1 platforms and close all open Sev-1/2 production incidents within 30 days.",
                "evidence": "Reconciled CMDB, baseline configuration report, patch compliance trend and post-incident review pack.",
                "success": "CMDB reconciliation at 100% for Tier-1 assets and open Sev-1/2 incident tail cleared.",
            },
            "at_risk": {
                "first_move": "Bring a red-status IT operations update to the technology forum focused on change failure rate and patch compliance.",
                "actions": "Reduce change-failure rate below the agreed threshold, close the vulnerability backlog on Tier-1 assets and refresh the DR runbook for the two weakest services.",
                "evidence": "Change-failure trend, vulnerability backlog burndown and refreshed DR runbook.",
                "success": "Change-failure rate below target for three consecutive cycles and Tier-1 vulnerability backlog cleared.",
            },
            "watch": {
                "first_move": "Confirm quarterly platform-health metrics are trending correctly and validate the DR test schedule.",
                "actions": "Run the scheduled DR test, refresh the configuration baseline and validate SRE golden signals on Tier-1 services.",
                "evidence": "DR-test after-action report, refreshed baseline and SRE golden-signals dashboard.",
                "success": "DR test executed within SLA and golden signals green for three consecutive cycles.",
            },
            "ready": {
                "first_move": "Maintain the current technology cadence and roll platform stability into the annual DORA attestation.",
                "actions": "Annual platform-health review, peer benchmark of MTTR and archive DR-test evidence.",
                "evidence": "Annual health memo, benchmark note and archived DR-test evidence.",
                "success": "Annual attestation signed with platform-stability KPIs at or above target.",
            },
        },
    },
    "Data": {
        "meta": {
            "forum": "data governance council",
            "owner": "Chief Data Officer",
            "why": "Data-governance gaps directly compromise the accuracy of regulatory reporting and surface first in supervisory data-quality reviews.",
        },
        "tiers": {
            "critical": {
                "first_move": "Freeze new reporting go-lives, launch a lineage rebuild for the top 10 regulatory reports and stand up a daily data-quality bridge.",
                "actions": "Rebuild end-to-end lineage for Tier-1 reports, remediate reference-data ownership gaps and re-baseline reconciliation controls.",
                "evidence": "Lineage diagrams, reference-data ownership matrix, reconciliation exception log and root-cause pack.",
                "success": "Tier-1 reports each have signed-off lineage plus reconciliation controls proven for two consecutive cycles.",
            },
            "at_risk": {
                "first_move": "Escalate a red-status data-quality update to the governance council focused on reconciliation exceptions and reference-data ownership.",
                "actions": "Close the top reconciliation exceptions, remediate reference-data owner assignments and re-run controls on the weakest reports.",
                "evidence": "Reconciliation exception burndown, reference-data ownership tracker and control re-run evidence.",
                "success": "Reconciliation exceptions cleared inside SLA and reference-data ownership at 100% for critical domains.",
            },
            "watch": {
                "first_move": "Confirm the annual data-quality attestation is on track and validate lineage completeness against the current reporting inventory.",
                "actions": "Refresh the lineage completeness dashboard, re-run reference-data monitoring and validate controls sampling.",
                "evidence": "Lineage completeness dashboard, monitoring reports and control-sample evidence.",
                "success": "Annual attestation signed and lineage completeness above the agreed threshold.",
            },
            "ready": {
                "first_move": "Preserve the current data-governance cadence and roll data quality into the annual DORA attestation.",
                "actions": "Annual data-quality review, benchmark reconciliation timings and archive control-test evidence.",
                "evidence": "Annual review memo, benchmark note and archived control-test evidence.",
                "success": "Annual attestation signed with zero material data findings.",
            },
        },
    },
    "Reporting": {
        "meta": {
            "forum": "regulatory reporting steering group",
            "owner": "Head of Regulatory Reporting",
            "why": "Reporting gaps surface first in supervisory data-quality reviews and drive restatement risk.",
        },
        "tiers": {
            "critical": {
                "first_move": "Freeze new report go-lives, initiate a restatement-risk review across the last 4 reporting cycles and appoint a dedicated remediation lead.",
                "actions": "Rebuild the reporting inventory, redesign sign-off gates and re-run reconciliations for the last two cycles.",
                "evidence": "Reporting inventory, sign-off gate design, reconciliation packs and restatement-risk assessment.",
                "success": "All Tier-1 reports produced with clean reconciliations for two consecutive cycles.",
            },
            "at_risk": {
                "first_move": "Bring a red-status reporting update to the steering group focused on sign-off timeliness and reconciliation quality.",
                "actions": "Close the top reconciliation exceptions, tighten sign-off gates and refresh reviewer training.",
                "evidence": "Exception burndown, sign-off gate memo and refreshed training record.",
                "success": "Sign-off gates clean for three consecutive cycles.",
            },
            "watch": {
                "first_move": "Confirm the annual reporting attestation is on track and validate reviewer coverage on Tier-1 reports.",
                "actions": "Refresh the reporting KRIs, re-run reconciliation sampling and validate reviewer allocation.",
                "evidence": "KRI pack, reconciliation sample report and reviewer roster.",
                "success": "Annual attestation signed with KRIs inside tolerance.",
            },
            "ready": {
                "first_move": "Maintain the current reporting cadence and roll into annual attestation.",
                "actions": "Annual reporting review, peer benchmark and archive control-test evidence.",
                "evidence": "Annual review memo, benchmark note, archived evidence.",
                "success": "Annual attestation signed with zero material reporting findings.",
            },
        },
    },
    "Audit": {
        "meta": {
            "forum": "audit committee",
            "owner": "Head of Internal Audit",
            "why": "Audit-coverage gaps prevent independent assurance over the DORA programme and limit challenge of control effectiveness.",
        },
        "tiers": {
            "critical": {
                "first_move": "Table an out-of-cycle audit-committee paper covering DORA-audit-coverage gaps and mobilise co-source support to close scope shortfalls.",
                "actions": "Rebuild the DORA audit universe, refresh the annual audit plan and launch targeted deep-dives on Tier-1 domains.",
                "evidence": "Refreshed audit universe, revised annual plan and Tier-1 deep-dive reports.",
                "success": "DORA audit coverage at 100% for Tier-1 domains and audit-committee approval of the refreshed plan.",
            },
            "at_risk": {
                "first_move": "Escalate an out-of-cycle status update to the audit committee focused on open finding tail and coverage gaps.",
                "actions": "Close the tail of open audit findings past their SLA, refresh coverage on the weakest domains and rehearse regulator-facing narrative.",
                "evidence": "Open-finding burndown, refreshed coverage plan and regulator-facing walk-through pack.",
                "success": "Open findings past SLA reduced to zero within 90 days.",
            },
            "watch": {
                "first_move": "Confirm the annual audit plan is on track and validate coverage on Tier-1 domains.",
                "actions": "Run the scheduled audit cycle, refresh coverage KRIs and validate committee-reporting quality.",
                "evidence": "Cycle report, KRI pack and committee-reporting quality review.",
                "success": "Scheduled audits delivered on time with clean committee reporting.",
            },
            "ready": {
                "first_move": "Maintain the current audit cadence and roll assurance evidence into the annual attestation.",
                "actions": "Annual assurance review, peer benchmark and archive prior-cycle audit evidence.",
                "evidence": "Annual review memo, benchmark note and archived audit evidence.",
                "success": "Annual attestation signed with zero unmitigated audit findings.",
            },
        },
    },
    "Operations": {
        "meta": {
            "forum": "operations risk forum",
            "owner": "Head of Operations",
            "why": "Operational-process gaps compound into settlement, reconciliation and client-impact risk that supervisors flag quickly.",
        },
        "tiers": {
            "critical": {
                "first_move": "Institute a daily ops-risk bridge, freeze non-critical process change and re-baseline end-to-end process maps for the top client-impacting flows.",
                "actions": "Rebuild the top 10 process maps, redesign key controls and re-run the two weakest end-to-end walkthroughs.",
                "evidence": "Refreshed process maps, key-control design memos and end-to-end walkthrough reports.",
                "success": "Top 10 client-impacting flows each covered by a signed process map plus a live key-control test.",
            },
            "at_risk": {
                "first_move": "Escalate a themed ops-risk update to the risk forum focused on reconciliation breaks and manual-workaround dependencies.",
                "actions": "Close the top reconciliation breaks, retire priority manual workarounds and refresh reviewer sign-off.",
                "evidence": "Break-log burndown, workaround retirement tracker and reviewer sign-off pack.",
                "success": "Top reconciliation break categories cleared and priority workarounds retired.",
            },
            "watch": {
                "first_move": "Confirm quarterly ops-risk metrics are trending correctly and validate control testing coverage.",
                "actions": "Run the scheduled control-testing cycle, refresh workaround inventory and validate reconciliation KRIs.",
                "evidence": "Control-testing report, workaround inventory and KRI pack.",
                "success": "Scheduled control tests delivered on time with clean sign-off.",
            },
            "ready": {
                "first_move": "Maintain the current operations cadence and roll into annual attestation.",
                "actions": "Annual ops-risk review, peer benchmark and archive prior-cycle evidence.",
                "evidence": "Annual review memo, benchmark note and archived evidence.",
                "success": "Annual attestation signed with zero material ops-risk findings.",
            },
        },
    },
    "Compliance": {
        "meta": {
            "forum": "compliance and financial-crime oversight forum",
            "owner": "Chief Compliance Officer",
            "why": "Compliance-monitoring gaps leave DORA obligations unmapped to executable controls and expose the firm to enforcement risk.",
        },
        "tiers": {
            "critical": {
                "first_move": "Publish a red-status DORA-compliance dashboard to the oversight forum and mobilise a dedicated compliance rebuild team.",
                "actions": "Rebuild the obligations register, remap DORA articles to executable controls and refresh compliance monitoring for the weakest domains.",
                "evidence": "Refreshed obligations register, article-to-control map and monitoring plan for the next 6 months.",
                "success": "Every DORA article mapped to an owner and a testable control by the next oversight-forum cycle.",
            },
            "at_risk": {
                "first_move": "Escalate a themed compliance status update focused on monitoring frequency and closed-finding evidence.",
                "actions": "Close the tail of open compliance findings, refresh monitoring cadence and rehearse regulator-facing narrative.",
                "evidence": "Finding-burndown log, refreshed monitoring plan and regulator-facing walk-through pack.",
                "success": "Open compliance findings past SLA reduced to zero within 90 days.",
            },
            "watch": {
                "first_move": "Confirm the annual compliance attestation is on track and validate monitoring coverage across DORA domains.",
                "actions": "Run scheduled compliance monitoring, refresh KRIs and validate reviewer allocation.",
                "evidence": "Monitoring reports, KRI pack and reviewer roster.",
                "success": "Annual attestation signed with monitoring KRIs inside tolerance.",
            },
            "ready": {
                "first_move": "Preserve the current compliance cadence and roll monitoring evidence into the annual attestation.",
                "actions": "Annual compliance review, peer benchmark and archive prior-cycle evidence.",
                "evidence": "Annual review memo, benchmark note and archived evidence.",
                "success": "Annual attestation signed with zero material compliance findings.",
            },
        },
    },
    "Legal": {
        "meta": {
            "forum": "compliance and legal committee",
            "owner": "General Counsel",
            "why": "Legal-clause gaps expose contracts to unenforceable DORA obligations and undermine third-party risk remediation.",
        },
        "tiers": {
            "critical": {
                "first_move": "Freeze new material contract signings, launch a contract-clause remediation programme and pre-brief the legal committee.",
                "actions": "Remediate DORA clauses across all live Tier-1 contracts, refresh template libraries and rehearse enforcement scenarios.",
                "evidence": "Contract remediation tracker, refreshed template library and enforcement-scenario memos.",
                "success": "All Tier-1 contracts remediated within 90 days.",
            },
            "at_risk": {
                "first_move": "Escalate a red-status legal update focused on template freshness and horizon-scanning gaps.",
                "actions": "Refresh contract templates, close open regulatory-change log items and re-train contract owners.",
                "evidence": "Template refresh memo, regulatory-change log and training record.",
                "success": "Regulatory-change log cleared and templates refreshed on schedule.",
            },
            "watch": {
                "first_move": "Confirm the annual legal-risk review is on track and validate horizon scanning coverage.",
                "actions": "Refresh the legal-risk register, re-run horizon scan and validate template usage.",
                "evidence": "Legal-risk register, horizon-scan memo and template-usage report.",
                "success": "Annual review signed with legal risks inside tolerance.",
            },
            "ready": {
                "first_move": "Maintain the current legal cadence and roll into annual attestation.",
                "actions": "Annual legal review, peer benchmark and archive evidence.",
                "evidence": "Annual review memo, benchmark note and archived evidence.",
                "success": "Annual attestation signed with zero material legal findings.",
            },
        },
    },
    "Programme": {
        "meta": {
            "forum": "DORA programme steering committee",
            "owner": "DORA Programme Manager",
            "why": "Programme-management gaps delay the DORA-readiness timeline and hide dependencies until they become critical-path issues.",
        },
        "tiers": {
            "critical": {
                "first_move": "Trigger a full programme reset: re-baseline the plan, refresh the RAID log and stand up a weekly steering committee until the critical path is stable.",
                "actions": "Re-plan the DORA delivery roadmap, secure funding for the remaining critical path and refresh dependency management on Tier-1 workstreams.",
                "evidence": "Re-baselined plan, refreshed RAID log, funding-approval memo and dependency map.",
                "success": "Programme steering committee approves the re-baselined plan and burn-down starts trending on target.",
            },
            "at_risk": {
                "first_move": "Escalate a red-status programme paper to steering, focused on slippage on the critical path.",
                "actions": "Close open programme risks past SLA, refresh critical-path forecast and rehearse the go-live cutover plan.",
                "evidence": "Risk-log burndown, updated critical-path forecast and rehearsal after-action review.",
                "success": "Critical-path slippage cleared within the next reporting period.",
            },
            "watch": {
                "first_move": "Confirm milestone burn-down is on track and validate dependency risk against upcoming go-lives.",
                "actions": "Refresh the milestone plan, re-run dependency checks and validate benefit tracking.",
                "evidence": "Milestone plan, dependency report and benefit tracker.",
                "success": "Milestones tracked on schedule with benefits realised on plan.",
            },
            "ready": {
                "first_move": "Maintain the current programme cadence and prepare the closure pack for the DORA programme office.",
                "actions": "Draft the programme closure pack, capture lessons learned and archive artefacts.",
                "evidence": "Closure pack, lessons-learned memo and archived artefact index.",
                "success": "Programme closes on plan with no open Sev-1/2 issues.",
            },
        },
    },
    "Human": {
        "meta": {
            "forum": "people risk & training committee",
            "owner": "Head of HR / Talent",
            "why": "People and training gaps undermine the human side of every ICT control and become the failure mode supervisors probe fastest.",
        },
        "tiers": {
            "critical": {
                "first_move": "Mandate an emergency DORA training refresh for Tier-1 roles and re-issue role descriptions with named accountabilities.",
                "actions": "Redesign the DORA training curriculum, refresh the competency matrix and re-issue role descriptions for all DORA-critical roles.",
                "evidence": "Refreshed curriculum, competency matrix, role descriptions and 90-day completion tracker.",
                "success": "90%+ completion of the refreshed curriculum across all DORA-critical roles inside 90 days.",
            },
            "at_risk": {
                "first_move": "Escalate a themed people-risk paper focused on training completion and succession coverage for DORA-critical roles.",
                "actions": "Close training completion tail, refresh succession plans and validate DORA-role compensation alignment.",
                "evidence": "Completion tracker, succession plan and compensation-review memo.",
                "success": "Training completion tail cleared and succession coverage documented for every DORA-critical role.",
            },
            "watch": {
                "first_move": "Confirm quarterly training refresh is on track and validate succession coverage.",
                "actions": "Run the scheduled training refresh, refresh succession plans and validate role clarity.",
                "evidence": "Training report, succession plan and role clarity memo.",
                "success": "Scheduled training completed with succession coverage green.",
            },
            "ready": {
                "first_move": "Maintain the current training cadence and roll into annual attestation.",
                "actions": "Annual people-risk review, benchmark completion rates and archive evidence.",
                "evidence": "Annual review memo, benchmark note and archived evidence.",
                "success": "Annual attestation signed with zero material people-risk findings.",
            },
        },
    },
    "Execution": {
        "meta": {
            "forum": "front-office / business risk forum",
            "owner": "Front Office / Business Owner",
            "why": "Execution-layer gaps translate directly into client-impacting incidents and become supervisory conduct concerns quickly.",
        },
        "tiers": {
            "critical": {
                "first_move": "Freeze new product launches, run rapid client-impact analyses on the top 5 business flows and stand up daily front-office / risk / operations calls.",
                "actions": "Rebuild business-flow maps for Tier-1 activities, refresh product-approval gates and re-execute client-impact scenarios.",
                "evidence": "Refreshed flow maps, product-approval gate memo and client-impact scenario reports.",
                "success": "Tier-1 flows each have signed maps and demonstrably tested client-impact scenarios.",
            },
            "at_risk": {
                "first_move": "Escalate a red-status execution-risk update to the business risk forum focused on control coverage and product-approval delays.",
                "actions": "Close open execution-risk findings, refresh product-approval evidence and re-test the two weakest flows.",
                "evidence": "Finding-burndown log, product-approval evidence and flow re-test after-action.",
                "success": "Execution-risk findings past SLA cleared within 60 days.",
            },
            "watch": {
                "first_move": "Confirm quarterly execution KRIs are trending correctly and validate product-approval effectiveness.",
                "actions": "Refresh execution KRIs, run scheduled control tests and validate product-approval effectiveness.",
                "evidence": "KRI pack, control-test reports and product-approval effectiveness memo.",
                "success": "Execution KRIs green for three consecutive cycles.",
            },
            "ready": {
                "first_move": "Maintain the current execution cadence and roll into annual attestation.",
                "actions": "Annual execution-risk review, peer benchmark and archive evidence.",
                "evidence": "Annual review memo, benchmark note and archived evidence.",
                "success": "Annual attestation signed with zero material execution-risk findings.",
            },
        },
    },
}


# Alias table so distinct area labels reuse the closest playbook entry
# without duplicating content. Keys are lower-case substrings matched
# against the incoming area name; the value is the canonical playbook key.
_AREA_ALIAS: Dict[str, str] = {
    "vendor": "Third",
    "supplier": "Third",
    "outsourc": "Third",
    "security": "Cyber",
    "resilience": "Business Continuity",
    "continuity": "Business Continuity",
    "settlement": "Operations",
    "middle office": "Operations",
    "back office": "Operations",
    "client": "Execution",
    "front office": "Execution",
    "assurance": "Audit",
    "internal audit": "Audit",
    "hr": "Human",
    "training": "Human",
    "people": "Human",
    "it ": "Technology",
    "it,": "Technology",
    "systems": "Technology",
}


def _lookup_area_playbook(area: str) -> Dict[str, Any]:
    """Return the area-specific playbook entry (with ``meta`` + ``tiers``)
    for ``area``. Matching is case-insensitive and prefix / alias-aware so
    aliases like "IT, Systems & Technology" resolve to the "Technology"
    playbook and "Third-Party Management" resolves to "Third". Falls back
    to a neutral executive-sponsor playbook when nothing matches."""
    key = (area or "").strip()
    fallback: Dict[str, Any] = {
        "meta": {
            "forum": "executive risk forum",
            "owner": "Executive sponsor",
            "why": f"{key or 'This area'} contributes to overall DORA readiness and needs a named accountable owner.",
        },
        "tiers": {
            "critical": {
                "first_move": f"Escalate {key or 'this area'} to the executive risk forum immediately and mobilise a dedicated remediation team.",
                "actions": f"Rebuild {key or 'the area'} controls end-to-end, refresh evidence and close open findings within 30 days.",
                "evidence": f"{key or 'Area'} policies, control test evidence, remediation tracker and post-incident reviews.",
                "success": f"Every open {key or 'area'} finding closed inside 30 days and readiness above 75%.",
            },
            "at_risk": {
                "first_move": f"Escalate {key or 'this area'} at the next executive risk forum with a red-status paper and dated remediation plan.",
                "actions": f"Close the top 5 {key or 'area'} findings, refresh evidence packs and rehearse the reporting narrative.",
                "evidence": f"{key or 'Area'} finding tracker, refreshed evidence pack and reporting narrative memo.",
                "success": f"Top {key or 'area'} findings closed inside 90 days.",
            },
            "watch": {
                "first_move": f"Confirm {key or 'this area'} on the scheduled forum agenda and validate residual gaps.",
                "actions": f"Refresh {key or 'area'} KRIs, close residual findings and validate reviewer coverage.",
                "evidence": f"{key or 'Area'} KRI pack, residual-finding tracker and reviewer roster.",
                "success": f"Residual {key or 'area'} gaps cleared before the next review cycle.",
            },
            "ready": {
                "first_move": f"Maintain the current {key or 'area'} cadence and roll evidence into the annual attestation.",
                "actions": f"Annual {key or 'area'} review, peer benchmark and archive evidence.",
                "evidence": f"Annual {key or 'area'} review memo, benchmark note and archived evidence.",
                "success": f"Annual attestation signed with zero material {key or 'area'} findings.",
            },
        },
    }
    if not key:
        return fallback

    lower = key.lower()

    # Explicit alias match first
    for alias, target in _AREA_ALIAS.items():
        if alias in lower and target in _AREA_PLAYBOOK:
            return _AREA_PLAYBOOK[target]

    # Playbook prefix match
    for prefix, entry in _AREA_PLAYBOOK.items():
        if prefix.lower() in lower:
            return entry

    return fallback


# Severity-tier metadata reused across all areas: cadence, horizon, target
# and forum-verb. The area-specific action content lives in _AREA_PLAYBOOK.
_SEVERITY_FRAME: Dict[str, Dict[str, Any]] = {
    "critical": {
        "cadence": "weekly status reviews",
        "cadence_unit": "week",
        "horizon": "Immediate (0-30 days)",
        "forum_verb": "Escalate {area} to the {forum} this governance cycle",
        "target": 75.0,
    },
    "at_risk": {
        "cadence": "bi-weekly status reviews",
        "cadence_unit": "fortnight",
        "horizon": "Short-term (30-90 days)",
        "forum_verb": "Bring {area} to the next {forum}",
        "target": 75.0,
    },
    "watch": {
        "cadence": "monthly readiness check-ins",
        "cadence_unit": "month",
        "horizon": "Medium-term (90-180 days)",
        "forum_verb": "Keep {area} on the {forum} agenda",
        "target": 90.0,
    },
    "ready": {
        "cadence": "quarterly steady-state reviews",
        "cadence_unit": "quarter",
        "horizon": "Steady-state (periodic)",
        "forum_verb": "Retain the {forum} slot for {area}",
        "target": 92.0,
    },
}


def _build_area_recommendation_bullets(
    *,
    area: str,
    readiness: float,
    impact: float,
    status: str,
    severity_label: str,
    area_recs: List[Any],
) -> List[Dict[str, str]]:
    """Compose 4 area-specific action bullets tuned to the area's live
    severity band.

    Content is drawn from the per-area × per-severity playbook so every
    bullet reads as a domain-appropriate consulting recommendation
    rather than a generic escalation template. When Agent 4 has attached
    a top recommendation for the area (owner, horizon, branch evidence,
    mapped requirement IDs) those live values override the playbook
    defaults so the card reflects the real assessment state.

    Bullet layout (same 4 titles for every card so the grid is scannable):

      1. **Escalate & govern** - severity-aware forum action + why it matters.
      2. **First moves** - area × severity specific tasks for the owner.
      3. **Evidence & controls** - branch-log evidence when available,
         otherwise the area × severity evidence focus.
      4. **Success criteria** - measurable target derived from the
         current readiness score and severity horizon.
    """
    def _get(r: Any, key: str, default: Any = "") -> Any:
        return getattr(r, key, None) if hasattr(r, key) else r.get(key, default)

    top = area_recs[0] if area_recs else None
    branch_evidence = str(_get(top, "branch_evidence") or "").strip() if top else ""
    mapped_ids = list(_get(top, "mapped_requirement_ids") or []) if top else []
    rec_owner = str(_get(top, "suggested_owner") or "").strip() if top else ""
    rec_horizon = str(_get(top, "horizon") or "").strip() if top else ""

    playbook = _lookup_area_playbook(area)
    meta = playbook.get("meta", {}) or {}
    tiers = playbook.get("tiers", {}) or {}
    forum = str(meta.get("forum") or "executive risk forum")
    playbook_owner = str(meta.get("owner") or "Executive sponsor")
    why_it_matters = str(meta.get("why") or "").rstrip(".") + "."

    sev_key = _canon_severity(severity_label)
    tier = tiers.get(sev_key) or tiers.get("watch") or {}
    frame = _SEVERITY_FRAME.get(sev_key) or _SEVERITY_FRAME["watch"]

    first_move = str(tier.get("first_move") or "").strip()
    actions = str(tier.get("actions") or "").strip()
    evidence = str(tier.get("evidence") or "").strip()
    success_line = str(tier.get("success") or "").strip()

    cadence = str(frame["cadence"])
    horizon_default = str(frame["horizon"])
    forum_action = frame["forum_verb"].format(area=area, forum=forum)
    target = float(frame["target"])

    owner = rec_owner or playbook_owner
    horizon = rec_horizon or horizon_default
    gap = max(0.0, target - readiness)

    bullets: List[Dict[str, str]] = []

    # 1) Escalate & govern - area-specific forum + why-it-matters + live scores
    bullets.append({
        "title": "Escalate & govern",
        "body": (
            f"{forum_action}. {why_it_matters} Sustain {cadence} until "
            f"{area} readiness clears {target:.0f}% "
            f"(currently {readiness:.1f}% readiness / {impact:.1f}% impact)."
        ),
    })

    # 2) First moves - concrete area × severity actions + accountable owner.
    # The calendar-window horizon (e.g. "Short-term (30-90 days)") that
    # used to be appended here is intentionally omitted — product feedback
    # was that the fixed timeline read as arbitrary next to the concrete
    # action text.
    first_moves_parts: List[str] = []
    if first_move:
        first_moves_parts.append(first_move)
    if actions:
        first_moves_parts.append(actions)
    if not first_moves_parts:
        first_moves_parts.append(
            f"Assign {owner} to close the top gaps in {area}."
        )
    first_moves_body = " ".join(first_moves_parts)
    if owner:
        first_moves_body = f"{first_moves_body} Owned by {owner}."
    bullets.append({"title": "First moves", "body": first_moves_body})

    # 3) Evidence & controls - branch-log when available + area × severity focus
    if branch_evidence:
        evidence_body = (
            f"{branch_evidence.rstrip('.')}. "
            f"Rebuild the {area} evidence pack: {evidence or 'policies, control tests and remediation trackers'}."
        )
    else:
        evidence_body = (
            f"Refresh the {area} evidence pack: "
            f"{evidence or 'policies, control tests and remediation trackers'}. "
            "Re-attach each artefact to the impacted requirements in the RTM."
        )
    if mapped_ids:
        shortlist = ", ".join(str(mid) for mid in mapped_ids[:4])
        evidence_body += f" Priority requirement IDs: {shortlist}."
    bullets.append({"title": "Evidence & controls", "body": evidence_body})

    # 4) Success criteria - measurable, per-severity target for THIS area
    if success_line:
        success_body = (
            f"Target: lift {area} readiness from {readiness:.1f}% to at "
            f"least {target:.0f}% within the next review cycle "
            f"(+{gap:.1f} pts). Definition of done: {success_line} "
            f"Track weekly on the Agent 4 KPI panel."
        )
    else:
        success_body = (
            f"Lift {area} readiness from {readiness:.1f}% to at least "
            f"{target:.0f}% within the next review cycle "
            f"(+{gap:.1f} pts). Track the delta each {frame['cadence_unit']} "
            "on the Agent 4 recommendation KPIs."
        )
    bullets.append({"title": "Success criteria", "body": success_body})

    return bullets[:4]


def _render_dashboard_recommendation_cards(recs: List[Any]) -> None:
    """Render Agent 4 recommendations as severity-coloured cards. Falls
    back to a friendly caption when the user has not yet clicked
    "Run Agent 4".
    """
    if not recs:
        st.caption("Click **Run Agent 4** to produce the action list.")
        return

    def _get(r: Any, key: str, default: Any = "") -> Any:
        return getattr(r, key, None) if hasattr(r, key) else r.get(key, default)

    cards: List[str] = ['<div class="dash-cards">']
    for r in recs:
        rid = str(_get(r, "recommendation_id") or "—")
        severity = str(_get(r, "severity") or "—").strip()
        title = str(_get(r, "title") or _get(r, "recommendation_title") or "—")
        try:
            comp = float(_get(r, "compliance_pct") or 0.0)
        except (TypeError, ValueError):
            comp = 0.0
        owner = str(_get(r, "suggested_owner") or "")
        # Horizon field (e.g. "Short-term (30-90 days)") intentionally
        # omitted from the rendered card — the calendar-window pill
        # cluttered the Dashboard for readers who care about severity
        # + owner + action, not an arbitrary N-day timeline.
        action = str(_get(r, "suggested_action") or "")
        css = _severity_label_from_status(severity) or _severity_class(comp)
        cards.append(
            f'<div class="dash-card {css}">'
            f'<div class="dash-card-title">{html.escape(title)}</div>'
            f'<div class="dash-card-meta">'
            f'<span class="dash-pill {css}">{html.escape(severity)}</span> '
            f'&nbsp;<b>{comp:.1f}%</b> compliance &nbsp;·&nbsp; '
            f'<span title="Recommendation ID">{html.escape(rid)}</span>'
            f'</div>'
            f'<div class="dash-card-meta">'
            f'<b>Owner:</b> {html.escape(owner) or "—"}'
            f'</div>'
            f'<div class="dash-card-body">{html.escape(action) or "—"}</div>'
            f'</div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def _render_dashboard_question_scoring_table(
    questionnaire: QuestionnairePackage, scoring: Any
) -> None:
    """Emit a scrollable, reference-style table with one row per scored
    closed question. Columns: Q#, Section (area), Function, Answer,
    Requirement, Readiness, Impact, Question. Answers come from the live
    assessment state; readiness values come from
    ``scoring.evaluation["requirement_scores"]`` where available.
    """
    pkg = questionnaire.package
    questions = list(pkg.get("questions") or [])
    closed = [q for q in questions if not q.get("is_free_text")]
    if not closed:
        st.caption("No closed questions in this questionnaire.")
        return

    state: AssessmentState = st.session_state["assessment_state"]
    responses = state.responses or {}
    req_scores: Dict[str, float] = scoring.evaluation.get("requirement_scores") or {}

    def _fmt_answer(qid: str) -> str:
        resp = responses.get(qid)
        if resp is None:
            return "—"
        if isinstance(resp, dict):
            picked = resp.get("selected") or resp.get("answer")
        else:
            picked = resp
        if isinstance(picked, (list, tuple, set)):
            return ", ".join(str(p) for p in picked) or "—"
        return str(picked) if picked not in (None, "") else "—"

    def _readiness_for(q: Dict[str, Any]) -> Optional[float]:
        mapped = q.get("mapped_requirement_ids") or []
        vals = [req_scores[rid] for rid in mapped if rid in req_scores]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 1)

    header = (
        "<thead><tr>"
        "<th>Q#</th><th>Section</th><th>Function</th>"
        "<th>Type</th><th>Answer</th><th>Readiness</th>"
        "<th>Impact</th><th>Question</th>"
        "</tr></thead>"
    )
    body_rows: List[str] = []
    for q in closed:
        qid = str(q.get("question_id") or "")
        area = str(q.get("area") or "")
        function = str(q.get("function") or "")
        qtype = str(q.get("question_type") or "")
        answer = _fmt_answer(qid)
        readiness = _readiness_for(q)
        readiness_display = f"{readiness:.1f}%" if readiness is not None else "—"
        impact_display = f"{100 - readiness:.1f}%" if readiness is not None else "—"
        css = _severity_class(readiness)
        pill_label = {
            "crit": "Critical", "risk": "At risk",
            "watch": "Watch", "ready": "Ready", "none": "—",
        }.get(css, "—")
        question_text = str(q.get("question") or "")
        body_rows.append(
            "<tr>"
            f'<td>{html.escape(qid)}</td>'
            f'<td>{html.escape(area)}</td>'
            f'<td>{html.escape(function)}</td>'
            f'<td>{html.escape(qtype)}</td>'
            f'<td>{html.escape(answer)}</td>'
            f'<td>{readiness_display}</td>'
            f'<td><span class="dash-pill {css}">{pill_label}</span> '
            f'&nbsp;{impact_display}</td>'
            f'<td>{html.escape(question_text)}</td>'
            "</tr>"
        )
    st.markdown(
        '<div class="dash-qtable-wrap">'
        '<table class="dash-qtable">'
        f'{header}'
        f'<tbody>{"".join(body_rows)}</tbody>'
        "</table></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 5 — Gap Identification & Human-in-the-Loop review queue
# ---------------------------------------------------------------------------

_GAP_SEVERITY_CSS = {
    "critical": "crit",
    "high":     "risk",
    "medium":   "watch",
    "low":      "ready",
}

_GAP_TAB_LABELS = (
    ("missing_evidence",        "Missing evidence"),
    ("missing_interpretations", "Missing interpretations"),
    ("missing_requirements",    "Missing requirements"),
    ("low_confidence",          "Low confidence"),
    ("human_review",            "Human review required"),
)


def _render_gap_kpi_row(report: GapReport) -> None:
    """Render the KPI row across the top of the Gap page."""
    counts = report.by_severity()
    cols = st.columns(5)
    labels = [
        ("Total gaps",         report.total(), "none"),
        ("Critical",           counts.get("critical", 0), "crit"),
        ("High",               counts.get("high", 0),     "risk"),
        ("Medium",             counts.get("medium", 0),   "watch"),
        ("Low",                counts.get("low", 0),      "ready"),
    ]
    for col, (label, value, css) in zip(cols, labels):
        with col:
            st.markdown(
                f'<div class="dash-card {css}">'
                f'<div class="dash-card-title">{html.escape(label)}</div>'
                f'<div class="dash-card-metric">{int(value)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_gap_list(items: List[GapItem], *, empty_message: str) -> None:
    if not items:
        st.success(empty_message)
        return
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_items = sorted(items, key=lambda it: severity_rank.get(it.severity, 4))
    for it in sorted_items:
        css = _GAP_SEVERITY_CSS.get(it.severity, "none")
        header = html.escape(it.subject or it.item_type)
        st.markdown(
            f'<div class="dash-card {css}">'
            f'<div class="dash-card-title">{header}</div>'
            f'<div class="dash-card-meta">'
            f'<span class="dash-pill {css}">{html.escape(it.severity.title())}</span>'
            f'</div>'
            f'<div class="dash-card-body">{html.escape(it.detail)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Details / remediation", expanded=False):
            if it.obligation_id:
                st.markdown(f"**Obligation ID:** `{it.obligation_id}`")
            if it.requirement_id:
                st.markdown(f"**Requirement ID:** `{it.requirement_id}`")
            if it.question_id:
                st.markdown(f"**Question ID:** `{it.question_id}`")
            if it.remediation:
                st.markdown(f"**Remediation:** {it.remediation}")
            if it.metadata:
                st.json(it.metadata)


def _init_review_queue_state() -> None:
    """Ensure the HITL review queue lives in session state."""
    if "gap_review_state" not in st.session_state:
        st.session_state["gap_review_state"] = {}


def _review_key(item: GapItem) -> str:
    """Stable identifier for a gap item across reruns."""
    return "|".join([
        item.item_type,
        item.subject,
        item.obligation_id,
        item.requirement_id,
        item.question_id,
    ])


def render_gap_page() -> None:
    """Page 5 — Gap Identification + HITL review queue.

    All five tabs are rendered even if empty so users can see at a glance
    which gap families are clean and which need attention. The queue
    lives in ``st.session_state["gap_review_state"]`` and persists
    across reruns of the same session.
    """
    st.subheader("5. Gap Identification")
    st.caption(
        "Missing evidence, missing interpretations, missing requirements, "
        "low-confidence findings, and items flagged for human review — "
        "computed live from Agent 1 / 2 / 3 output."
    )

    analysis = st.session_state.get("analysis")
    rtm_artifact = st.session_state.get("rtm_artifact")
    scoring = _refresh_scoring_snapshot()
    evaluation = scoring.evaluation if scoring else st.session_state.get("evaluation")

    if analysis is None:
        st.warning("Run Agent 1 (Setup or BRD/FRD page) before opening this page.")
        return

    report = build_gap_report(
        analysis=analysis,
        rtm_artifact=rtm_artifact,
        scoring_evaluation=evaluation,
    )

    _render_gap_kpi_row(report)
    st.divider()

    _init_review_queue_state()
    review_state = st.session_state["gap_review_state"]

    tabs = st.tabs([label for _key, label in _GAP_TAB_LABELS])
    for tab, (key, label) in zip(tabs, _GAP_TAB_LABELS):
        with tab:
            items: List[GapItem] = getattr(report, key)
            _render_gap_list(
                items,
                empty_message=f"No {label.lower()} findings — all clear.",
            )

            # Review-queue actions (mentor #4 HITL): every item on this
            # tab can be marked "resolved" or "escalated". The state is
            # persisted in session so the queue tab reflects the same
            # decisions.
            if items and key == "human_review":
                st.divider()
                st.markdown("#### Reviewer actions")
                for it in items:
                    rkey = _review_key(it)
                    current = review_state.get(rkey, {"status": "open"})
                    with st.expander(
                        f"[{it.severity.title()}] {it.subject}",
                        expanded=False,
                    ):
                        status_choice = st.radio(
                            "Status",
                            options=("open", "resolved", "escalated"),
                            index=("open", "resolved", "escalated").index(
                                current.get("status", "open")
                            ),
                            horizontal=True,
                            key=f"gap_status__{rkey}",
                        )
                        notes = st.text_area(
                            "Reviewer notes",
                            value=str(current.get("notes", "")),
                            key=f"gap_notes__{rkey}",
                            max_chars=2000,
                            height=100,
                        )
                        if st.button("Save", key=f"gap_save__{rkey}"):
                            review_state[rkey] = {
                                "status": status_choice,
                                "notes": notes,
                                "subject": it.subject,
                                "severity": it.severity,
                                "item_type": it.item_type,
                            }
                            st.success("Saved.")

    # Roll-up download of the full gap report as JSON.
    st.divider()
    st.markdown("#### Export gap report")
    export_payload = {
        "regulation": st.session_state.get("regulation"),
        "client_roles": list(getattr(analysis, "client_roles", []) or []),
        "totals": {
            "total_gaps": report.total(),
            "by_severity": report.by_severity(),
        },
        "missing_evidence":        [it.__dict__ for it in report.missing_evidence],
        "missing_interpretations": [it.__dict__ for it in report.missing_interpretations],
        "missing_requirements":    [it.__dict__ for it in report.missing_requirements],
        "low_confidence":          [it.__dict__ for it in report.low_confidence],
        "human_review":            [it.__dict__ for it in report.human_review],
        "review_queue":            dict(review_state),
    }
    st.download_button(
        "Download gap report (JSON)",
        data=json.dumps(export_payload, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name=(
            f"{st.session_state.get('regulation','regulation')}_gap_report.json"
        ),
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Page 5 — Export
# ---------------------------------------------------------------------------

def render_export_page() -> None:
    st.subheader("5. Export")
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
        st.markdown("**Questionnaire Package (JSON)**")
        st.download_button(
            "Download Questionnaire JSON",
            data=json.dumps(pkg, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{st.session_state['regulation']}_questionnaire_package.json",
            mime="application/json",
            width="stretch",
        )

        st.markdown("**Responses & Live Results (JSON)**")
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
            "Download Responses JSON",
            data=json.dumps(responses_payload, ensure_ascii=False, indent=2,
                            default=str).encode("utf-8"),
            file_name=f"{st.session_state['regulation']}_responses.json",
            mime="application/json",
            width="stretch",
        )

    with cols[1]:
        st.markdown("**Excel Report (Questionnaire + Responses + Scores)**")
        # Build the Excel bytes eagerly on page render so the user can
        # download it with a single click, matching the "Download BRD +
        # FRD (DOCX)" pattern on Page 2. ``write_excel_from_package`` is
        # a deterministic ~1s CPU operation (no LLM), so it's cheap
        # enough to run on every render — but we still cache the bytes
        # in session state keyed on the questionnaire id so switching
        # away from the export page and back doesn't rebuild it.
        qid_for_cache = st.session_state.get("questionnaire_id") or id(pkg)
        cache_key = f"_excel_bytes__{qid_for_cache}"
        excel_bytes: Optional[bytes] = st.session_state.get(cache_key)
        excel_filename = f"{st.session_state['regulation']}_Readiness_Report.xlsx"
        if excel_bytes is None:
            try:
                target = OUTPUT_DIR / timestamped_name(
                    f"{st.session_state['regulation']}_Readiness_Report", ".xlsx"
                )
                write_excel_from_package(str(target), pkg)
                with open(target, "rb") as fh:
                    excel_bytes = fh.read()
                st.session_state[cache_key] = excel_bytes
                excel_filename = target.name
                st.session_state[f"{cache_key}__name"] = excel_filename
            except Exception as exc:
                excel_bytes = None
                st.error(f"Excel export failed: {exc}")
        else:
            excel_filename = st.session_state.get(
                f"{cache_key}__name", excel_filename,
            )

        if excel_bytes:
            st.download_button(
                "Download Excel Report",
                data=excel_bytes,
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                width="stretch",
                help=(
                    "One-click download of the Excel workbook with the "
                    "full questionnaire, your answers, and the scored "
                    "results."
                ),
            )

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
    elif page == "2. Generate BRD / FRD":
        render_brd_page()
    elif page == "3. Questionnaire":
        render_questionnaire_page()
    elif page == "4. Dashboard":
        render_dashboard_page()
    elif page == "5. Export":
        render_export_page()
    else:
        st.warning(f"Unknown page: {page}")


main()
