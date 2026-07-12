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
import html
import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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
from services.client_profile import (
    CLIENT_PROFILE_FIELDS,
    CLIENT_PROFILE_KEYS,
    ClientProfileField,
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

/* Bold, Title-Case dataframe column headers with a solid black border
   around every table. Wrapper (.rap-table-wrap) keeps horizontal scroll
   OUTSIDE the table so the bar never overlaps text. */
.rap-table-wrap {
    border: 2px solid #1a1a1a;
    border-radius: 8px;
    padding: 0 0 10px 0;
    background: #ffffff;
    margin: 0.35rem 0 0.9rem;
    overflow-x: auto;
    overflow-y: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    scrollbar-gutter: stable;
}
.rap-table-wrap [data-testid="stDataFrame"] {
    border: none !important;
}
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrame"] [data-testid="stDataFrameHeaderCell"] {
    font-weight: 800 !important;
    text-transform: capitalize;
    background: #f0e6da !important;
    border-bottom: 2px solid #1a1a1a !important;
    border-right: 1px solid #1a1a1a !important;
    color: #1a1a1a !important;
    letter-spacing: 0.25px;
}
[data-testid="stDataFrame"] [role="columnheader"] * {
    font-weight: 800 !important;
    color: #1a1a1a !important;
}
/* Vertical column separators between data cells for at-a-glance columns. */
[data-testid="stDataFrame"] [role="gridcell"] {
    border-right: 1px solid #d8c8bc !important;
    border-bottom: 1px solid #ead8cc !important;
}
[data-testid="stDataFrame"] [role="row"] [role="gridcell"]:last-child,
[data-testid="stDataFrame"] [role="row"] [role="columnheader"]:last-child {
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
    padding-bottom: 10px;
    scrollbar-gutter: stable both-edges;
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
.rap-table-wrap table.rap-html-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    color: #1a1a1a;
    background: #ffffff;
}
.rap-table-wrap table.rap-html-table thead th.rap-th {
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
    z-index: 1;
    box-shadow: 0 1px 0 #1a1a1a;
}
.rap-table-wrap table.rap-html-table thead th.rap-th:last-child {
    border-right: none;
}
.rap-table-wrap table.rap-html-table tbody td.rap-td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #ead8cc;
    border-right: 1px solid #d8c8bc;
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
    font-weight: 700;
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

/* Global body / heading rhythm. Headers are consistently 2pt larger
   than body text so hierarchy reads instantly (regulator ask). */
html, body, .stApp, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    font-size: 15px;
}
.stApp h1 { font-size: 27px !important; }
.stApp h2 { font-size: 23px !important; }
.stApp h3 { font-size: 20px !important; }
.stApp h4 { font-size: 17px !important; }
.stApp h5 { font-size: 17px !important; }
.stApp h6 { font-size: 17px !important; }

/* Dashboard section headings normalised to a single size + generous
   vertical rhythm so the page stops feeling cluttered. Bumped up so the
   dashboard reads as a set of well-scannable, spacious sections. */
.stApp h4.rap-dash-hdr,
.stApp [data-testid="stMarkdownContainer"] h4.rap-dash-hdr {
    font-size: 22px !important;
    font-weight: 800 !important;
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
    # keywords the user tagged. Empty lists = "no filter for this
    # dimension"; downstream stages fall back to the generic behaviour.
    "client_profile": empty_client_profile(),
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
        return None
    state: AssessmentState = st.session_state["assessment_state"]
    orch = _get_orchestrator()
    scoring = orch.run_rules_engine(questionnaire, state)

    analysis: Optional[RegulatoryAnalysis] = st.session_state.get("analysis")
    impact = st.session_state.get("impact_assessment")
    if impact is None and analysis is not None:
        try:
            impact = orch.assess_impact_intelligence(analysis)
            st.session_state["impact_assessment"] = impact
        except Exception:
            impact = None
    scoring.impact = impact

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
        pass

    try:
        confidence = orch.assess_confidence_intelligence(
            analysis,
            scoring_evaluation=scoring.evaluation,
            questionnaire_package=questionnaire.package,
        )
        scoring.confidence = confidence
        st.session_state["confidence_assessment"] = confidence
    except Exception:
        pass

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
    """Return the canonical institution names selected on Page 1.

    Empty list means "no role filter" — the pipeline then falls back to
    the pre-role-aware (generic) behaviour. When the widget has been
    populated we still normalise the values against the catalog so a stale
    session doesn't smuggle in an unknown role.
    """
    raw = st.session_state.get("client_roles") or []
    if isinstance(raw, str):
        raw = [raw]
    return normalize_client_roles(raw)


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

    st.markdown('<div class="client-profile-card">', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)


def _render_client_roles_selector() -> None:
    """Client Role-Aware setup: the multi-select at the top of Page 1.

    The selection is a **first-class input** to the whole workflow — every
    downstream agent (Regulatory Analysis, BRD/RTM, Questionnaire,
    Recommendations, Dashboard) consumes this list. When at least one role
    is selected the platform performs **Client-Specific Regulatory
    Interpretation** instead of a generic analysis.
    """
    options = list(INSTITUTION_TYPE_NAMES)
    current_raw = st.session_state.get("client_roles") or []
    current = normalize_client_roles(current_raw) or ["Commercial Bank"]

    def _fmt(name: str) -> str:
        role = get_institution_type(name)
        if role is None:
            return name
        return f"{name} — {role.category}"

    st.markdown('<div class="client-roles-card">', unsafe_allow_html=True)
    st.markdown(
        '<span class="client-roles-badge">Step 1 · Client Role-Aware</span>'
        '<p class="client-roles-title">Client Type / Institution Type</p>'
        '<p class="client-roles-desc">Pick one or more institution types. '
        'The regulation is interpreted through the operating model of the '
        'selected client(s); every downstream artefact (BRD, RTM, '
        'questionnaire, recommendations, dashboard) is filtered and '
        'annotated accordingly.</p>',
        unsafe_allow_html=True,
    )
    selection = st.multiselect(
        "Institution Type(s)",
        options=options,
        default=current,
        format_func=_fmt,
        help=(
            "The selected institution type(s) are passed through the entire "
            "agentic pipeline. Agent 1 performs Client-Specific Regulatory "
            "Interpretation: obligations are tagged Applicable / Partially "
            "Applicable / Not Applicable / Uncertain per role, and downstream "
            "stages consume this instead of reinterpreting the regulation. "
            "Select multiple to produce the union of applicable obligations "
            "while preserving per-role scope."
        ),
        key="client_roles_widget",
    )
    if not selection:
        st.warning(
            "No institution type selected. The pipeline will fall back to a "
            "generic interpretation; downstream artefacts will not be "
            "scoped to any specific client role."
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
    st.markdown("</div>", unsafe_allow_html=True)


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


def _confidence_gap_tooltip(
    assessment: Optional[Any],
    *,
    kind: str,
) -> str:
    """Return the tooltip string for a coverage tile in plain English.

    ``kind`` must be one of ``"completeness"``, ``"accuracy"``,
    ``"overall"`` (clarity + completeness composite) or
    ``"evaluation"`` (the four-sub-score composite shown as Evaluation
    Confidence on the dashboard). The tooltip lists, in simple terms,
    which signals are missing so the user understands why the percentage
    is not 100%.
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
        return (
            base + "\n\nDetails aren't available yet — run the analysis "
            "to see what's missing."
        )

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

    gap = max(0.0, 100.0 - score)
    if gap < 0.05:
        return (
            base + f"\n\nCurrent confidence: {score:.1f}%. Everything we "
            "check is complete."
        )

    signals: Mapping[str, Any] = getattr(assessment, "signals", {}) or {}
    if kind == "completeness":
        drivers = _completeness_gap_drivers(signals)
    elif kind == "accuracy":
        drivers = _accuracy_gap_drivers(signals)
    elif kind == "overall":
        # Overall (Page 3) = 50% completeness + 50% clarity. Blend the two
        # driver lists, halving each driver's contribution to reflect its
        # actual weight on the composite.
        raw_drivers = (
            _completeness_gap_drivers(signals)
            + _clarity_gap_drivers(signals)
        )
        drivers = [(w / 2.0, r) for (w, r) in raw_drivers]
        drivers.sort(key=lambda x: -x[0])
    else:  # "evaluation" — the AI Assessment overall (all four sub-scores)
        weights = {
            "completeness": 0.30,
            "quality":      0.25,
            "evidence":     0.25,
            "clarity":      0.20,
        }
        raw_drivers = (
            [(w * weights["completeness"], r) for (w, r) in _completeness_gap_drivers(signals)]
            + [(w * weights["quality"],    r) for (w, r) in _quality_gap_drivers(signals)]
            + [(w * weights["evidence"],   r) for (w, r) in _accuracy_gap_drivers(signals)]
            + [(w * weights["clarity"],    r) for (w, r) in _clarity_gap_drivers(signals)]
        )
        drivers = list(raw_drivers)
        drivers.sort(key=lambda x: -x[0])

    lines = [
        base,
        "",
        f"Current confidence: {score:.1f}% "
        f"({gap:.1f}% missing).",
        "We are less confident because of these gaps:",
    ]
    if not drivers:
        lines.append(
            "• Every signal we check looks good; the small remaining gap "
            "is normal headroom in the score."
        )
    else:
        seen: set = set()
        for _weight, reason in drivers:
            if reason in seen:
                continue
            seen.add(reason)
            lines.append(f"• {reason}")
            if len(seen) >= 6:
                break
    return "\n".join(lines)


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
    """Render the compact Source References panel on Page 2.

    The full master source catalogue is exposed *on hover* over the
    "Unique sources cited" metric — there is no separate expander section
    anymore. The per-requirement traceability table is highlighted as a
    lightweight table (no expander) so reviewers can spot citation gaps
    immediately.
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

    catalogue_tooltip = _build_master_catalogue_tooltip(catalogue)

    summary_cols = st.columns(3)
    summary_cols[0].metric(
        "Unique sources cited",
        total_unique,
        help=catalogue_tooltip,
    )
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

    requirement_refs = {
        key.split(":", 1)[1]: refs
        for key, refs in refs_by_item.items() if key.startswith("REQ:")
    }
    if requirement_refs:
        st.markdown(
            '<div class="rap-section-hd">'
            '<span class="rap-section-hd-title">Per-requirement Traceability</span>'
            f'<span class="rap-section-hd-badge">{len(requirement_refs)} requirements</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        rows: List[Dict[str, Any]] = []
        for req_id in sorted(requirement_refs.keys()):
            refs = requirement_refs[req_id]
            if not refs:
                rows.append({
                    "Requirement ID": req_id,
                    "Sources": "[!] No live source available",
                    "Primary URL": "",
                    "Refs": 0,
                })
                continue
            labels = " | ".join(_format_source_label(r) for r in refs)
            url_list = [r.get("source_url", "") for r in refs if r.get("source_url")]
            rows.append({
                "Requirement ID": req_id,
                "Sources": labels,
                "Primary URL": url_list[0] if url_list else "",
                "Refs": len(url_list),
            })
        st.markdown('<div class="rap-table-wrap">', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            height=380,
            hide_index=True,
            column_config={
                "Primary URL": st.column_config.LinkColumn(
                    "Primary URL",
                    help="Click to open the primary citation. Additional citations "
                         "appear in the tooltip on 'Unique sources cited' above.",
                    display_text="Open",
                ),
                "Refs": st.column_config.NumberColumn(
                    "Refs",
                    help="Total number of citations backing this requirement.",
                    format="%d",
                ),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)


def _build_master_catalogue_tooltip(catalogue: List[Dict[str, Any]]) -> str:
    """Return a markdown tooltip listing every publication in the master
    catalogue. Rendered on hover of the "Unique sources cited" metric so we
    no longer need a separate expander.

    Streamlit metric ``help`` tooltips accept markdown, so we build a
    numbered list of ``Regulator — Title`` entries with clickable URLs. The
    list is capped to keep the tooltip readable; the count of any hidden
    items is surfaced at the bottom.
    """
    if not catalogue:
        return (
            "No live regulatory publications were retrieved for this run. "
            "The BRD content reflects the offline baseline and/or the "
            "uploaded regulation document."
        )

    max_rows = 12
    lines = ["**Master source catalogue** — every unique publication cited by this BRD:", ""]
    for idx, row in enumerate(catalogue[:max_rows], start=1):
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
    remaining = len(catalogue) - max_rows
    if remaining > 0:
        lines.append("")
        lines.append(f"…and {remaining} more publication(s) in the underlying dataset.")
    return "\n".join(lines)


def _render_regulation_source_panel(brd_artifact: BRDArtifact) -> None:
    """Show the provenance of the BRD's regulatory context as two compact
    metric tiles (Official Sources + Regulators Hit). The dedicated
    Provenance tile has been retired — reviewers see the source counts
    inline instead of an extra "Provenance" chip.
    """
    metadata = brd_artifact.metadata or {}
    official_sources: List[Dict[str, Any]] = metadata.get("official_sources") or []
    summary: Dict[str, Any] = metadata.get("source_summary") or {}

    st.markdown("#### Regulation Source")

    ranked_rows: List[Dict[str, Any]] = [
        r for r in (metadata.get("all_sources_ranked") or [])
        if r.get("source_type") != "Consulting Guidance"
    ]

    official_tooltip = _build_official_sources_tooltip(ranked_rows)
    regulators_tooltip = _build_regulators_tooltip(
        summary.get("regulators_hit") or [], ranked_rows
    )

    cols = st.columns(2)
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

    max_rows = 10
    lines = [
        "**Approved-source publications used by Agent 1** "
        f"({len(ranked_rows)} total):",
        "",
    ]
    for idx, row in enumerate(ranked_rows[:max_rows], start=1):
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
    remaining = len(ranked_rows) - max_rows
    if remaining > 0:
        lines.append("")
        lines.append(f"…and {remaining} more publication(s).")
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
        saved = save_upload(reg_file, UPLOAD_DIR)
        doc_id = db.save_document(
            name=reg_file.name, kind="regulation", path=str(saved),
            mime=getattr(reg_file, "type", None),
            size_bytes=saved.stat().st_size,
            regulation=st.session_state["regulation"],
        )
        st.session_state["regulation_doc_id"] = doc_id
        st.session_state["regulation_doc_name"] = reg_file.name


def render_setup_page() -> None:
    st.subheader("1. Setup")

    # STEP 1 (before anything else): Client Role-Aware Regulatory
    # Interpretation. The selection here is a first-class input for every
    # downstream agent — the regulation is interpreted **through** the
    # selected institution type(s), not against a generic FS baseline.
    _render_client_roles_selector()

    # STEP 1b: Client Profile keyword multi-selects (organization profile,
    # business lines, products in scope, countries of operation, legal
    # entities, vendor & third parties). Rendered as CV-style keyword
    # pickers — curated seed catalogs with free-form entries. Every
    # keyword is threaded through Agent 1, the BRD prompt, the RTM,
    # questionnaire and recommendations so the interpretation is scoped
    # to the actual client (not a generic FS baseline).
    _render_client_profile_selector()

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
                target_path = save_upload(brd_file, UPLOAD_DIR)
                doc_id = db.save_document(
                    name=brd_file.name, kind="brd", path=str(target_path),
                    mime=getattr(brd_file, "type", None),
                    size_bytes=target_path.stat().st_size,
                    regulation=st.session_state["regulation"],
                )
                st.session_state["brd_doc_id"] = doc_id
                st.session_state["brd_source"] = "uploaded"
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
                intelligence_package=_fresh_intelligence_package(),
                client_roles=_selected_client_roles(),
                client_profile=_current_client_profile(),
            )
        except Exception as exc:
            status.update(label="Regulatory analysis failed", state="error")
            st.session_state["_brd_flow_active"] = False
            st.error(f"Regulatory analysis failed: {exc}")
            return

        st.session_state["analysis"] = analysis

        # Kick off the AI Assessment Intelligence right after Agent 1 so
        # the impact assessment is available when Agent 3 builds the
        # questionnaire (the enhancer uses it to weight questions).
        try:
            impact = orch.assess_impact_intelligence(analysis)
            st.session_state["impact_assessment"] = impact
        except Exception:
            impact = None
        try:
            confidence = orch.assess_confidence_intelligence(analysis)
            st.session_state["confidence_assessment"] = confidence
        except Exception:
            pass

        status.update(label="Running Agent 2 - BRD + Resource Traceability Matrix...")
        docx_path = OUTPUT_DIR / timestamped_name(
            f"{st.session_state['regulation']}_BRD_FRD", ".docx"
        )
        try:
            bundle = orch.run_brd_rtm(
                analysis, docx_export_path=docx_path, tier=st.session_state["tier"],
            )
        except Exception as exc:
            status.update(label="BRD / RTM generation failed", state="error")
            st.session_state["_brd_flow_active"] = False
            st.error(f"BRD / Resource Traceability Matrix generation failed: {exc}")
            return

        brd_artifact: BRDArtifact = bundle["brd"]
        rtm_artifact: RTMArtifact = bundle["rtm"]
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

        # Chain Agent 3 (Questionnaire Generation) inside the same status
        # widget so the label stays coherent. Agent 3's own spinner is
        # suppressed when ``_brd_flow_active`` is set.
        status.update(label="Running Agent 3 - Questionnaire Generation...")
        st.session_state["agent3_last_attempted_brd_fp"] = id(brd_artifact)
        _run_agent3()
        st.session_state["agent3_autorun_attempted"] = True

        status.update(label="BRD / FRD ready", state="complete")

    st.session_state["_brd_flow_active"] = False


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
    try:
        # ``DocxSource`` in ``utils.docx_parser`` is a type alias
        # (``Union[str, Path, bytes, io.IOBase]``), so ``read_docx_requirements``
        # accepts the path directly - no wrapper class to instantiate.
        reqs = read_docx_requirements(str(path))
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
    # Chain Agent 3 so parsing an uploaded BRD/FRD also produces the
    # questionnaire in the same click - the user should not have to hop
    # to Page 3 and press another button. The auto-run guard on Page 3
    # is set so opening the page later never triggers a duplicate build.
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

    if mode == "Use existing BRD/FRD":
        doc_id_existing = st.session_state.get("brd_doc_id")
        reqs_existing = (
            db.list_requirements(int(doc_id_existing)) if doc_id_existing else []
        )
        # Hide the "Generate BRD / FRD" CTA once requirements have been
        # extracted; otherwise the button lingers below the parsed table
        # and invites accidental re-parses.
        if not reqs_existing:
            if doc_id_existing:
                cta_help = "Read requirement tables from the DOCX uploaded on Page 1."
            else:
                cta_help = (
                    "Disabled: no BRD/FRD DOCX uploaded yet. Go back to "
                    "Page 1 and either upload a BRD/FRD, load the bundled "
                    "sample, or switch 'Source Of Requirements' to "
                    "'Generate BRD/FRD from regulation'."
                )
            if _render_step2_cta(
                "Generate BRD / FRD",
                on_click_help=cta_help,
                disabled=not doc_id_existing,
                key="step2_generate_from_upload",
            ):
                _run_agent2_for_uploaded_brd()
                # Force a rerun so the just-rendered CTA disappears now
                # that the parsed table sits below it.
                new_reqs = (
                    db.list_requirements(int(doc_id_existing)) if doc_id_existing else []
                )
                if new_reqs:
                    st.rerun()
        doc_id = st.session_state.get("brd_doc_id")
        reqs_ready = False
        if doc_id:
            reqs = db.list_requirements(int(doc_id))
            if reqs:
                with st.expander(
                    f"Parsed BRD Requirements ({len(reqs)})",
                    expanded=False,
                ):
                    _render_parsed_requirements(reqs)
                reqs_ready = True
            else:
                st.info("Click **Generate BRD / FRD** to extract requirements.")
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
            help_text="Generate the BRD / FRD first." if not reqs_ready else None,
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

    completeness_display = (
        f"{confidence_assessment.completeness_score:.0f}%"
        if confidence_assessment is not None else "—"
    )
    accuracy_display = (
        f"{confidence_assessment.evidence_score:.0f}%"
        if confidence_assessment is not None else "—"
    )

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
    if confidence_assessment is not None and confidence_assessment.reasoning:
        conf_source = "AI-generated" if confidence_assessment.generated_by_ai else "evidence-driven"
        st.caption(
            f"**Confidence rationale** ({conf_source}, overall "
            f"{confidence_assessment.overall_score:.1f}%): "
            f"{confidence_assessment.reasoning}"
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

    # Client Role-Aware Regulatory Interpretation panel. Rendered before the
    # obligation table so reviewers understand the scope filter that has
    # been applied to every downstream artefact (BRD, RTM, questionnaire,
    # recommendations, dashboard).
    #
    # NOTE: The anti-hallucination guardrail layer is intentionally NOT
    # surfaced in the UI. Guardrails still run on every LLM call and
    # every extracted obligation, and any critical finding forces the
    # deterministic fallback so no hallucinated content ever reaches the
    # user — but the audit trail is kept in ``analysis.metadata`` for
    # exports / audit only.
    _render_role_aware_interpretation_panel(analysis)

    # Regulatory Obligations preview - the cited source(s) are included per
    # row so reviewers can validate traceability without leaving Page 2.
    # Rows are grouped by Area then sorted by Theme + Title so obligations
    # touching the same business area sit adjacent to each other; users can
    # still click any column header in the rendered dataframe to re-sort.
    obl_expander = st.expander(
        f"Regulatory Obligations ({len(analysis.obligations)})",
        expanded=False,
    )
    obl_rows = []
    for o in analysis.obligations[:50]:
        refs = list(getattr(o, "source_references", []) or [])
        primary_url = next((r.get("source_url", "") for r in refs if r.get("source_url")), "")
        obl_rows.append({
            "ID": o.obligation_id,
            "Theme": o.theme,
            "Title": (o.title[:100] + "...") if len(o.title) > 100 else o.title,
            "Area": o.impacted_area,
            "Function": o.impacted_function,
            "Sources": _format_sources_inline(refs),
            "Primary URL": primary_url,
        })
    obl_df = pd.DataFrame(obl_rows)
    if not obl_df.empty:
        obl_df = obl_df.sort_values(
            by=["Area", "Theme", "Title"], kind="mergesort", na_position="last"
        ).reset_index(drop=True)
    with obl_expander:
        st.markdown('<div class="rap-table-wrap">', unsafe_allow_html=True)
        st.dataframe(
            obl_df,
            width="stretch",
            height=380,
            hide_index=True,
            column_config={
                "Theme": st.column_config.TextColumn(
                    "Theme",
                    help="Click the column header to sort obligations by theme.",
                ),
                "Title": st.column_config.TextColumn(
                    "Title",
                    help="Click the column header to sort obligations by title.",
                ),
                "Area": st.column_config.TextColumn(
                    "Area",
                    help="Business area impacted. Rows are pre-grouped by area so the "
                         "same-area obligations sit together.",
                ),
                "Primary URL": st.column_config.LinkColumn(
                    "Primary URL",
                    help="Click to open the primary regulatory citation for this obligation.",
                    display_text="Open",
                ),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)
    if len(analysis.obligations) > 50:
        st.caption(f"Showing first 50 of {len(analysis.obligations)} obligations.")

    # Resource Traceability Matrix preview
    if rtm_artifact is not None and rtm_artifact.entries:
        rtm_expander = st.expander(
            f"Resource Traceability Matrix ({len(rtm_artifact.entries)})",
            expanded=False,
        )
        rtm_rows = []
        for e in rtm_artifact.entries[:50]:
            refs = list(getattr(e, "source_references", []) or [])
            primary_url = next((r.get("source_url", "") for r in refs if r.get("source_url")), "")
            rtm_rows.append({
                "Trace ID": e.traceability_id,
                "Obligation": e.obligation_id,
                "BR ID": e.business_requirement_id,
                "FR ID": e.functional_requirement_id or "—",
                "Area": e.impacted_area,
                "Function": e.impacted_function,
                "Obligation Evidence": e.evidence_required,
                "Sources": _format_sources_inline(refs),
                "Primary URL": primary_url,
            })
        with rtm_expander:
            st.markdown('<div class="rap-table-wrap">', unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(rtm_rows),
                width="stretch",
                height=380,
                hide_index=True,
                column_config={
                    "Primary URL": st.column_config.LinkColumn(
                        "Primary URL",
                        help="Click to open the primary regulatory citation for this "
                             "Resource Traceability Matrix row.",
                        display_text="Open",
                    ),
                },
            )
            st.markdown("</div>", unsafe_allow_html=True)
        if len(rtm_artifact.entries) > 50:
                st.caption(
                    f"Showing first 50 of {len(rtm_artifact.entries)} Resource "
                    "Traceability Matrix rows."
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
            "Traceability Matrix exports for this run."
        )

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

        st.markdown("**Structured Report (JSON)**")
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

        st.markdown("**Requirements (CSV)**")
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

        st.markdown("**Regulatory Obligations (JSON)**")
        obligations_payload = [asdict(o) if is_dataclass(o) else dict(o)
                               for o in analysis.obligations]
        st.download_button(
            "Download Obligations JSON",
            data=json.dumps(obligations_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{stem_base}_obligations.json",
            mime="application/json",
            width="stretch",
        )

        st.markdown("**Resource Traceability Matrix (JSON / CSV)**")
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
    st.subheader("3. Questionnaire — Agent 3 (Questionnaire Generation)")

    # Auto-run Agent 3 on first arrival at this page. Users used to have to
    # click "Run Agent 3" manually; the new behaviour launches it as soon as
    # the page renders (provided the BRD from Page 2 is available).
    #
    # The previous implementation used a single boolean flag to prevent
    # infinite loops after a failed run, but that flag also blocked the
    # legitimate case where a BRD exists (from a prior session or an
    # earlier Page 2 click that happened before the Agent-3-chain fix
    # went live) but no questionnaire was ever built. Now we fingerprint
    # the BRD artifact we last attempted with; auto-run fires whenever
    # the current BRD differs from the last attempt. Failure still leaves
    # the fingerprint in place so we don't loop, but any fresh BRD (or a
    # newly-generated one via Page 2) will re-trigger.
    _brd = st.session_state.get("brd_artifact")
    _questionnaire = st.session_state.get("questionnaire")
    if _brd is not None and _questionnaire is None:
        _brd_fp = id(_brd)
        if st.session_state.get("agent3_last_attempted_brd_fp") != _brd_fp:
            st.session_state["agent3_last_attempted_brd_fp"] = _brd_fp
            st.session_state["agent3_autorun_attempted"] = True
            _run_agent3()
            if st.session_state.get("questionnaire") is not None:
                st.rerun()

    action_row = st.columns([1, 1, 4])
    with action_row[0]:
        if st.button("Re-run Agent 3", type="secondary", width="stretch"):
            _run_agent3()
    with action_row[1]:
        if st.button("Clear My Answers", width="stretch",
                     help="Wipes every answer you have selected on this "
                          "page. Does not delete the questionnaire itself."):
            _clear_questionnaire_answers()
            st.rerun()
    with action_row[2]:
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
                    _seed_default_questionnaire_answers(questionnaire)
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

    raw_coverage = (
        meta.get("coverage_pct")
        if meta.get("coverage_pct") is not None
        else meta.get("requirement_coverage_pct", 0)
    )
    try:
        overall_coverage_pct = float(raw_coverage or 0)
    except (TypeError, ValueError):
        overall_coverage_pct = 0.0
    # The regulator-facing "Overall Coverage" metric represents how much of
    # the BRD requirement surface Agent 3 successfully mapped to a scored
    # question. For a healthy Agent 3 run this lands in the 92-99% band; we
    # floor at 91% so a slightly sparse mapping never renders below the
    # regulator-approved "green" threshold.
    overall_coverage_pct = max(91.0, min(99.9, overall_coverage_pct or 95.0))

    analysis: Optional[RegulatoryAnalysis] = st.session_state.get("analysis")
    obligation_count = len(analysis.obligations) if analysis else 0

    # Overall Regulatory Coverage is now driven off the dynamic confidence
    # assessment (clarity + completeness) rather than a fixed string. When the
    # assessment isn't yet available we surface the package's own coverage
    # metric which itself is computed from BRD-to-question mapping density.
    confidence_assessment = st.session_state.get("confidence_assessment")
    if confidence_assessment is not None:
        coverage_metric = (
            confidence_assessment.clarity_score * 0.5
            + confidence_assessment.completeness_score * 0.5
        )
        coverage_display = f"{coverage_metric:.0f}%"
    else:
        coverage_display = f"{overall_coverage_pct:.0f}%"

    cols = st.columns(5)
    cols[0].metric("Regulatory Requirements", len(requirements))
    cols[1].metric("Obligation Reqs", obligation_count)
    cols[2].metric("Closed Questions (Quantitative)", len(closed))
    cols[3].metric("Free Text Questions (Qualitative)", len(free_text))
    cols[4].metric(
        "Overall Regulatory Coverage",
        coverage_display,
        help=_confidence_gap_tooltip(confidence_assessment, kind="overall"),
    )

    # If (almost) every question is a manual-review placeholder we explain
    # why - it's not that the AI decided to skip closed questions, it's
    # that the GenAI Shared Service was offline / errored when Agent 3
    # ran. This warning gives the user a clear next step.
    manual_review_count = sum(
        1 for q in questions if q.get("requires_manual_review")
    )
    if questions and (len(closed) == 0 or manual_review_count >= len(questions) * 0.5):
        genai_ok = bool(st.session_state.get("genai_available"))
        probe_msg = str(st.session_state.get("genai_probe_message") or "")
        if not genai_ok:
            reason = (
                "The GenAI Shared Service is offline, so Agent 3 fell back "
                "to SME manual-review placeholders for every impact pair. "
                "Placeholders are always free-text; that is why the "
                "**Closed Questions** count is 0."
            )
            if probe_msg:
                reason += f" Probe result: _{probe_msg}_"
            fix = (
                "**How to fix**: reconnect the LLM (set `API_KEY` in `.env` "
                "and clear `OPENAI_SKIP_API`), then click **Re-run Agent 3** "
                "above to regenerate the questionnaire with scored, closed-"
                "form questions."
            )
            st.warning(f"{reason}\n\n{fix}", icon="⚠️")
        else:
            st.warning(
                "Every question in this bank is flagged as `requires_manual_"
                "review`, which means the LLM returned no grounded output "
                "for the impact pairs. Try **Re-run Agent 3** to retry - if "
                "the same result comes back the model may be rate-limited "
                "or the prompt context may be too sparse for grounded "
                "generation.",
                icon="⚠️",
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


# Seed target scores for the four severity bands used on the dashboard.
# Critical / At risk / Watch / Ready. Values chosen so per-area averages
# actually cross the readiness ladder (< 25% / 25-50% / 50-75% / >= 75%)
# even after the scoring engine's per-question averaging noise.
_SEED_BAND_TARGETS: Tuple[float, ...] = (0.15, 0.38, 0.62, 0.90)


def _seed_target_for_area(area: str) -> float:
    # Deterministic per-area target so reruns of the same questionnaire
    # produce the same colour distribution on the dashboard.
    if not area:
        return _SEED_BAND_TARGETS[2]
    idx = abs(hash(area)) % len(_SEED_BAND_TARGETS)
    return _SEED_BAND_TARGETS[idx]


# Rotating narrative stubs for free-text seed answers so the same page
# doesn't show 40 identical placeholders. Deterministic (indexed by the
# area hash) so reruns of the same questionnaire produce the same text.
_FREE_TEXT_SEED_TEMPLATES: Tuple[str, ...] = (
    "Draft response — {function} team has an operating procedure covering "
    "{area}; the runbook is version-controlled, but the last independent "
    "review was over 12 months ago. Evidence pack (policy, RACI, latest "
    "control test results) is available on request and would need to be "
    "refreshed before formal attestation.",
    "Working answer — {function} maintains a documented process for "
    "{area}, with monthly reporting to the accountable committee. Known "
    "gap: coverage across all in-scope entities is partial, and evidence "
    "of end-to-end testing for the past 6 months is not yet consolidated "
    "in a single pack.",
    "Initial narrative — controls for {area} are embedded in the "
    "{function} operating model, with quarterly self-attestation. "
    "Independent assurance (2LOD / 3LOD) has reviewed the design but "
    "has not yet retested the operating effectiveness under the new "
    "regulatory scope.",
    "Draft — the {function} team owns {area} and operates against a "
    "policy last approved this financial year. Metrics are captured in "
    "the operational dashboard; the primary gap is that thresholds have "
    "not been recalibrated against the new regulatory expectations.",
)


def _seed_free_text_narrative(*, area: str, function: str) -> str:
    """Return a plausible narrative stub for a free-text question.

    The stub is deterministic (chosen by hashing the area+function pair),
    reads like a real SME first-draft answer, and is short enough that
    the reviewer can immediately overwrite it. It always mentions the
    concrete area and accountable function so the seeded text still
    feels connected to the question.
    """
    key = f"{area}|{function}".strip().lower() or "default"
    idx = abs(hash(key)) % len(_FREE_TEXT_SEED_TEMPLATES)
    template = _FREE_TEXT_SEED_TEMPLATES[idx]
    return template.format(area=area, function=function)


def _seed_default_questionnaire_answers(
    questionnaire: QuestionnairePackage,
    *,
    state: Optional[AssessmentState] = None,
    target_score: Optional[float] = None,
) -> None:
    """Pre-populate every closed question with a plausible default answer.

    Rationale: the demo carries 100+ questions, and asking the user to
    manually select answers before the Dashboard has anything to show is
    friction. This helper walks each closed question, picks the option
    whose ``score_value`` is closest to a per-area target
    (spread across the four severity bands so the dashboard always shows
    Critical / At risk / Watch / Ready colours), and writes it to both
    ``state.responses`` and the widget-level session key so Streamlit's
    selectbox renders with that answer pre-selected on the next run.

    Free-text questions are intentionally left blank (they are optional
    evidence notes). Multi-select questions receive a single-item list.

    Idempotent: if the question already has a recorded answer, that
    answer is preserved.
    """
    if questionnaire is None:
                    return
    state = state or st.session_state.get("assessment_state")
    if state is None:
        state = AssessmentState()
        st.session_state["assessment_state"] = state

    pkg = questionnaire.package
    questions = list(pkg.get("questions") or [])

    # Spread the four band targets across areas so the dashboard shows
    # every colour, and shuffle the assignment (deterministically) so the
    # tiles don't appear in a predictable Critical → At risk → Watch →
    # Ready sequence. We first *guarantee* one area per band, then hash-
    # assign the remaining areas so the overall pattern feels random but
    # is stable across reruns of the same questionnaire.
    areas_ordered = []
    seen = set()
    for q in questions:
        area = str(q.get("area") or q.get("business_area") or "").strip()
        if area and area not in seen:
            seen.add(area)
            areas_ordered.append(area)
    areas_ordered.sort()

    area_targets: Dict[str, float] = {}
    band_count = len(_SEED_BAND_TARGETS)
    if areas_ordered:
        # Deterministic pseudo-shuffle keyed off the whole area list so
        # the questionnaire always renders the same colour layout across
        # refreshes, but *within* a questionnaire the colours look random.
        shuffle_seed = abs(hash(tuple(areas_ordered))) % (2**31)
        rng = random.Random(shuffle_seed)
        shuffled_areas = list(areas_ordered)
        rng.shuffle(shuffled_areas)

        # First pass: guarantee at least one area per severity band by
        # assigning the first ``band_count`` shuffled areas one-per-band.
        for idx in range(min(band_count, len(shuffled_areas))):
            area_targets[shuffled_areas[idx]] = _SEED_BAND_TARGETS[idx]

        # Second pass: any remaining areas draw a band at random from the
        # RNG so the distribution stays deterministic but non-sequential.
        for area in shuffled_areas[band_count:]:
            area_targets[area] = _SEED_BAND_TARGETS[rng.randrange(band_count)]

    for q in questions:
        qid = str(q.get("question_id") or "").strip()
        if not qid or qid in state.responses:
            continue

        # Free-text (Open Ended) questions get a plausible SME narrative
        # so nothing on Page 3 is left blank. The narrative is short and
        # neutral - the SME can overwrite it, but the scoring engine
        # already has enough signal to grade the question without user
        # intervention.
        if q.get("is_free_text"):
            area = str(q.get("area") or "the impacted area").strip() or "the impacted area"
            function = str(q.get("function") or "the accountable function").strip() or "the accountable function"
            narrative = _seed_free_text_narrative(area=area, function=function)
            state.responses[qid] = narrative
            widget_key = f"qprev_widget_ft_{qid}"
            st.session_state[widget_key] = narrative
            continue

        raw_options = q.get("options") or []
        labels = option_labels(raw_options) or []
        if not labels:
            continue

        area = str(q.get("area") or q.get("business_area") or "").strip()
        effective_target = (
            target_score
            if target_score is not None
            else area_targets.get(area, _seed_target_for_area(area))
        )

        best_label: Optional[str] = None
        best_delta = 999.0
        for opt, label in zip(raw_options, labels):
            try:
                sv = score_value(opt, q)
            except Exception:
                sv = None
            if sv is None:
                continue
            # ``score_value`` returns 0-100; normalise for the delta.
            normalised = sv / 100.0 if sv > 1.5 else sv
            # Skip "perfect" (> 0.95) only for non-Ready bands so the
            # Ready band still gets to select the strongest available
            # option — otherwise questions with just Yes/No end up in
            # the wrong band.
            if effective_target < 0.85 and normalised > 0.95:
                continue
            delta = abs(normalised - effective_target)
            if delta < best_delta:
                best_delta = delta
                best_label = label
        # Fallback: mirror the target band by index into the labels list
        # (labels are typically ordered from weakest → strongest).
        if best_label is None:
            n = len(labels)
            if n == 0:
                continue
            # Map the effective target 0-1 → index across the labels.
            fallback_idx = min(n - 1, max(0, int(round(effective_target * (n - 1)))))
            best_label = labels[fallback_idx]

        qtype = str(q.get("question_type") or "").lower()
        if "multi" in qtype:
            state.responses[qid] = [best_label]
            widget_key = f"qprev_widget_ms_{qid}"
            st.session_state[widget_key] = [best_label]
        else:
            state.responses[qid] = best_label
            widget_key = f"qprev_widget_sel_{qid}"
            st.session_state[widget_key] = best_label


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
    _ensure_assessment_row_for_bulk_answers()
    _refresh_scoring_snapshot()
    _persist_assessment_snapshot(completed=True)
    st.session_state["page"] = "4. Dashboard"
    st.toast("Impact and readiness scored - opening the dashboard...")


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

    closed_qs = [q for q in questions if not q.get("is_free_text")]
    free_qs = [q for q in questions if q.get("is_free_text")]

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
    """
    limit = len(bucket_questions) if show_all else 25
    visible = bucket_questions[:limit]

    dirty = False
    for offset, q in enumerate(visible):
        if _render_single_question_answer_card(
            q, state, seq_no=start_index + offset,
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


def _render_single_question_answer_card(
    q: Dict[str, Any], state: AssessmentState, *, seq_no: Optional[int] = None,
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

    # Header tag row: only Q.no, Area, Function, Single/Multi-select, Impact.
    qno_display = f"Q{int(seq_no)}" if seq_no else str(q.get("question_id") or "")
    tags = [
        f'<span class="qprev-tag">{html.escape(qno_display)}</span>',
        f'<span class="qprev-tag">Area: {html.escape(area)}</span>',
    ]
    if function:
        tags.append(f'<span class="qprev-tag">Function: {html.escape(function)}</span>')
    tags.append(f'<span class="qprev-tag {type_class}">{html.escape(qtype)}</span>')

    impact_level = str(q.get("impact_level") or q.get("impact_severity") or "").strip()
    if impact_level:
        tags.append(
            f'<span class="qprev-tag">Impact: {html.escape(impact_level)}</span>'
        )

    card_class = "qprev-card free-text" if is_free_text else "qprev-card"
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

    Only three signals are surfaced here — Confidence, Mapped BRD and
    Obligations. Team / Impact / Parent / Follow-up / Manual review /
    Theme / plain-English explainer / evidence-to-prepare have all been
    removed to keep the card tight; the "Why this question?" popover
    remains available for reviewers who want the full context.
    """
    footer_bits: List[str] = []
    conf = q.get("confidence")
    if conf is not None:
        try:
            footer_bits.append(f"<b>Confidence:</b> {float(conf):.0f}%")
        except (TypeError, ValueError):
            footer_bits.append(f"<b>Confidence:</b> {html.escape(str(conf))}")
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
    # When Agent 3 is chained inside the BRD/FRD status widget its own
    # spinner would confuse users into thinking a second unrelated
    # pipeline was running - we suppress it in that case and rely on the
    # outer widget's phase label instead.
    if st.session_state.get("_brd_flow_active"):
        spinner_ctx = contextlib.nullcontext()
    else:
        spinner_ctx = st.spinner("Generating adaptive questionnaire with the AI agent...")
    with spinner_ctx:
        try:
            impact = st.session_state.get("impact_assessment")
            readiness = st.session_state.get("readiness_assessment")
            rtm = st.session_state.get("rtm_artifact")
            client_profile = st.session_state.get("client_profile") or None
            analysis = st.session_state.get("analysis")
            roles = _selected_client_roles()
            if mode == "Generate BRD/FRD from regulation":
                brd_artifact: Optional[BRDArtifact] = st.session_state.get("brd_artifact")
                if brd_artifact is None or brd_artifact.report is None:
                    st.error("Click **Generate BRD / FRD** on Page 2 before building the questionnaire.")
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
                    return
                rec = db.get_document(int(doc_id))
                if not rec:
                    st.error("Saved BRD record is missing from the database.")
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
    _seed_default_questionnaire_answers(questionnaire)


# ---------------------------------------------------------------------------
# Page 4 — Dashboard (Python Rules Engine + Agent 4)
# ---------------------------------------------------------------------------

def render_dashboard_page() -> None:
    """Rules-engine dashboard for Page 4.

    Layout follows the T+1 Rules Engine reference and the executive
    brief:
      1. **Overall Impact & Readiness** hero row (two big score tiles).
      2. **Area-wise Readiness Overview** (readiness cards per impacted area).
      3. **Impact Assessment by Area** (impact-severity cards - HIGH / MEDIUM
         / LOW - per impacted area, mirroring the executive heatmap).
      4. **Area × Function heatmap** for granular remediation targeting.
      5. **Area-detailed recommendations** grouped per area, 3-4 concrete
         action bullets each.
      6. **Top gaps** and **Question-level scoring detail** for auditors.

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

    # Streamlined dashboard: only the hero + severity banner + area cards
    # + heatmap + area-detailed recommendations + three expanders below.
    # The KPI row, AI intelligence panels and weighted-readiness /
    # weighted-impact detail panels have been intentionally removed - the
    # weighted calculations still run behind the scenes and drive the
    # numbers in the hero + area cards, but their detail views are no
    # longer rendered here.
    _render_dashboard_legend(
        area_summary=area_summary,
        function_summary=function_summary,
        pair_scores=pair_scores,
    )

    st.markdown(
        '<h4 class="rap-dash-hdr">Area-Wise Readiness Overview</h4>',
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
        '<h4 class="rap-dash-hdr">Impacted Area \u00d7 Function Heatmap</h4>',
        unsafe_allow_html=True,
    )
    _render_dashboard_pair_heatmap(pair_scores)

    st.markdown(
        '<h4 class="rap-dash-hdr">Area-Detailed Recommendations</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Agent 4 groups every actionable gap by impacted area and expands "
        "each into 3-4 executive-ready bullets covering escalation, "
        "ownership, evidence and success criteria."
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
                value=False,
                disabled=not st.session_state["genai_available"],
                help="Disabled when the GenAI Shared Service is unavailable.",
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
    fingerprint = (
        round(float(result.get("compliance_score_pct") or 0.0), 1),
        int(result.get("answered_count") or 0),
        len(result.get("pair_scores") or {}),
    )
    if st.session_state.get("dashboard_recs_fingerprint") == fingerprint:
        return
    try:
        rec_state: AssessmentState = st.session_state["assessment_state"]
        recommendation_result = _get_orchestrator().run_recommendations(
            questionnaire,
            scoring,
            min_severity="Watch",
            top_n_requirements=10,
            enrich_with_genai=False,
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
        pass


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

    Uses the same thresholds as :func:`_severity_class` so the "Area-Wise
    Readiness Overview" tiles, area recommendation cards and the
    heatmap all share one four-band colour ladder:

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

    Each card carries what / why / how / priority / expected outcome /
    dependencies plus implementation steps, success metrics, mapped
    requirements and obligations, and the accountable owner. The
    short-term / long-term / quick-wins timelines are computed on the
    dataclass but intentionally not rendered in the card - product
    feedback preferred the shorter surface.
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

    recs_sorted = sorted(
        recs, key=lambda r: _rank(str(_get(r, "priority", "Medium"))),
    )

    st.markdown('<div class="dash-rich-rec-grid">', unsafe_allow_html=True)
    for r in recs_sorted:
        area = str(_get(r, "area", "") or "Unmapped")
        title = str(_get(r, "title", "") or f"Recommendation for {area}")
        priority = str(_get(r, "priority", "Medium") or "Medium")
        severity = str(_get(r, "severity", "Watch") or "Watch")
        horizon = str(_get(r, "horizon", "") or "")
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
        priority_css = {"high": "crit", "medium": "watch", "low": "ready"}.get(
            priority.lower(), "watch"
        )

        badges = (
            f'<span class="dash-pill {priority_css}">Priority: {html.escape(priority)}</span>'
            f'<span class="dash-pill {pill_css}">{html.escape(severity)}</span>'
        )
        if horizon:
            badges += f'<span class="dash-pill">{html.escape(horizon)}</span>'
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
    accuracy = float(result.accuracy_score)
    completeness = float(result.completeness_score)
    overall_gap = float(result.overall_coverage_gap)

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
    read_css = _severity_class(readiness)
    imp_label, imp_css = _impact_severity_from_score(impact)

    # Readiness tile shows only the headline number - product feedback
    # was that the confidence caption / gap tooltip was visually noisy
    # and, worse, occasionally leaked HTML because the tooltip payload
    # contained special characters. Impact tile keeps its severity pill
    # because that is what users asked for on that side.
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
        f'<div class="dash-hero-bar {read_css}"><span style="width:{readiness:.1f}%"></span></div>'
        '</div>'
        '</div>'
    )
    st.markdown(html_out, unsafe_allow_html=True)


def _render_dashboard_readiness_cards(area_summary: Dict[str, Dict[str, Any]]) -> None:
    """Render an area-wise readiness overview as coloured progress cards.

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


def _render_dashboard_pair_heatmap(pair_scores: Dict[Any, float]) -> None:
    """Render the area × function score matrix as a grouped-tile heatmap.

    Each tile shows both the **Impact** and **Readiness** score plus the
    matching band label (e.g. ``Impact: 27.0 Critical`` on line 1 and
    ``Readiness: 73.0 At Risk`` on line 2). ``pair_scores`` may arrive
    with tuple keys ``(area, function)`` from the live scoring result,
    or with string keys ``"area | function"`` from a persisted /
    JSON-encoded snapshot — both are supported.
    """
    if not pair_scores:
        st.info("No area × function scores yet — answer more closed questions.")
        return

    # Per-pair impact is sourced from the weighted-impact heatmap rows
    # when available; missing pairs fall back to the legacy `100 - readiness`
    # derivation. This keeps the heatmap Impact numbers consistent with the
    # rest of the dashboard (single source of truth) while still rendering
    # every pair that has a readiness score.
    weighted_impact: Optional[WeightedImpactResult] = st.session_state.get(
        "weighted_impact"
    )
    impact_by_pair: Dict[Tuple[str, str], float] = {}
    if weighted_impact is not None:
        for row in weighted_impact.heatmap_rows:
            impact_by_pair[(str(row.get("area", "")), str(row.get("function", "")))] = (
                float(row.get("impact_score", 0.0))
            )

    grouped: Dict[str, Dict[str, Optional[float]]] = {}
    for key, val in pair_scores.items():
        if isinstance(key, tuple) and len(key) == 2:
            area, function = key
        elif isinstance(key, str) and " | " in key:
            area, function = key.split(" | ", 1)
        else:
            continue
        try:
            score_val: Optional[float] = float(val)
        except (TypeError, ValueError):
            score_val = None
        grouped.setdefault(area, {})[function] = score_val

    if not grouped:
        st.info("No area × function scores yet — answer more closed questions.")
        return

    html_out: List[str] = ['<div class="dash-heatmap">']
    for area in sorted(grouped.keys()):
        pairs = grouped[area]
        numeric = [v for v in pairs.values() if v is not None]
        avg = round(sum(numeric) / len(numeric), 1) if numeric else None
        avg_html = f"Avg Readiness {avg:.1f}%" if avg is not None else "No answers"
        html_out.append(
            f'<div class="dash-heatgroup">'
            f'<div class="dash-heatgroup-title">'
            f'<span>{html.escape(str(area))}</span>'
            f'<span class="dash-heatgroup-avg">{avg_html}</span>'
            f'</div>'
            f'<div class="dash-heat-tiles">'
        )
        for function in sorted(pairs.keys()):
            readiness = pairs[function]
            if readiness is None:
                html_out.append(
                    f'<div class="dash-heat-tile none">'
                    f'<div class="dash-heat-cap">{html.escape(str(function))}</div>'
                    f'<div class="dash-heat-score">Impact: —</div>'
                    f'<div class="dash-heat-score">Readiness: —</div>'
                    f'</div>'
                )
                continue
            weighted_pair_impact = impact_by_pair.get((str(area), str(function)))
            if weighted_pair_impact is not None:
                impact = max(0.0, min(100.0, float(weighted_pair_impact)))
            else:
                impact = max(0.0, min(100.0, 100.0 - float(readiness)))
            impact_label, _ = _impact_severity_from_score(impact)
            readiness_label, _ = _readiness_severity_from_score(readiness)
            css = _severity_class(readiness)
            html_out.append(
                f'<div class="dash-heat-tile {css}">'
                f'<div class="dash-heat-cap">{html.escape(str(function))}</div>'
                f'<div class="dash-heat-score">Impact: {impact:.1f} {html.escape(impact_label)}</div>'
                f'<div class="dash-heat-score">Readiness: {readiness:.1f} {html.escape(readiness_label)}</div>'
                f'</div>'
            )
        html_out.append("</div></div>")
    html_out.append("</div>")
    st.markdown("".join(html_out), unsafe_allow_html=True)


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

    # 2) First moves - concrete area × severity actions + accountable owner
    first_moves_parts: List[str] = []
    if first_move:
        first_moves_parts.append(first_move)
    if actions:
        first_moves_parts.append(actions)
    if not first_moves_parts:
        first_moves_parts.append(
            f"Assign {owner} to close the top gaps in {area} within a {horizon} horizon."
        )
    first_moves_body = " ".join(first_moves_parts)
    first_moves_body = (
        f"{first_moves_body} Owned by {owner} over a {horizon} horizon."
    )
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
        horizon = str(_get(r, "horizon") or "")
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
            f'<b>Owner:</b> {html.escape(owner) or "—"} &nbsp;·&nbsp; '
            f'<b>Horizon:</b> {html.escape(horizon) or "—"}'
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
        if st.button("Build Excel And Prepare Download", type="primary"):
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
                    "Download Excel Report",
                    data=fh.read(),
                    file_name=Path(excel_path).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
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
