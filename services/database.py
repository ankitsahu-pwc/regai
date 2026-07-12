"""SQLite persistence layer for the regulatory readiness MVP.

Single-file SQLite database (``data/app.db`` by default). All schema creation
is idempotent, all writes commit immediately, all queries return plain dicts
(not :class:`sqlite3.Row`) so they can be JSON-serialised by Streamlit.

Tables
------
``documents``
    Every file the user uploads (regulation PDF/DOCX, BRD/FRD DOCX).
``requirements``
    One row per requirement extracted from a BRD/FRD, denormalised for the UI.
``questionnaires``
    One row per generated questionnaire package. The full package JSON is
    stored verbatim in ``package_json`` to avoid lossy shredding.
``assessments``
    One row per assessment session. Holds the serialised
    :class:`~services.scoring_engine.AssessmentState`, the latest evaluation
    snapshot, and the latest recommendations bundle.
``responses``
    One row per answered question for fine-grained reporting (the same data
    is also available in ``assessments.state_json``, which keeps the
    in-memory state lossless including dynamic follow-ups and skipped IDs).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_DB_PATH = Path("data") / "app.db"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-persistence guardrail wiring
# ---------------------------------------------------------------------------
#
# Mentor guidance #1: every write path SHOULD pass its payload through the
# guardrail stack BEFORE INSERT so hallucinated / meta-leaked content
# never lands in ``data/app.db``. Callers pass ``guardrail_context``
# (regulation / roles / source_corpus) so the citation and role validators
# can do their job; when absent we still run meta-leakage scrubbing and
# speculation detection.
#
# The behaviour is controlled by ``APP_PERSIST_GUARDRAIL`` env var:
#
# * ``off``    — no guardrail sweep (use for tests / seeding).
# * ``warn``   — run the sweep; on critical findings, LOG and continue.
# * ``strict`` — run the sweep; on critical findings, refuse the write.
#
# Default is ``warn`` so existing user data continues to load even if it
# contains legacy strings that would trip a modern validator.

def _persist_guardrail_mode() -> str:
    """Return the current pre-persistence guardrail mode.

    Mode is read from the ``APP_PERSIST_GUARDRAIL`` env var on every call
    so tests / operators can flip behaviour without a restart.
    """
    return (os.getenv("APP_PERSIST_GUARDRAIL") or "warn").strip().lower()


def _run_persist_guardrail(
    payload: Any,
    *,
    component: str,
    context: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run the pre-persistence guardrail and return its serialised report.

    Returns ``None`` when the mode is ``off``. Otherwise runs the sweep
    and either raises (``strict`` mode) or logs (``warn`` mode) on
    critical findings. The returned dict can be embedded inside JSON
    payloads (``package_json`` / ``evaluation_json``) for later review.
    """
    mode = _persist_guardrail_mode()
    if mode == "off":
        return None

    try:
        # Deferred import to avoid a circular import at module load time
        # (``services.guardrails`` imports from other services that may
        # transitively touch persistence during rare code paths).
        from .guardrails import (
            PersistenceGuardrailError,
            check_before_persist,
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception("Could not import guardrails; skipping persistence sweep.")
        return None

    ctx = dict(context or {})
    try:
        report = check_before_persist(
            payload,
            component=component,
            source_corpus=ctx.get("source_corpus"),
            regulation=ctx.get("regulation"),
            client_roles=ctx.get("client_roles"),
            strict=(mode == "strict"),
        )
    except PersistenceGuardrailError:
        # Re-raise so the caller can surface the error to the user.
        raise
    except Exception:  # pragma: no cover - defensive; never block writes on a guardrail bug
        logger.exception(
            "Pre-persistence guardrail crashed for component=%s; continuing without sweep.",
            component,
        )
        return None
    if not report.ok:
        logger.warning(
            "Pre-persistence guardrail flagged findings for component=%s: %s",
            component,
            report.summary(),
        )
    return report.to_dict()


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def _resolve_db_path(db_path: Optional[os.PathLike[str] | str] = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    env_override = os.getenv("APP_DB_PATH")
    if env_override:
        return Path(env_override)
    return DEFAULT_DB_PATH


def connect(db_path: Optional[os.PathLike[str] | str] = None) -> sqlite3.Connection:
    """Return a configured sqlite3 connection. Creates parent dirs on demand."""
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(db_path: Optional[os.PathLike[str] | str] = None):
    """Context manager that yields a connection and commits on exit."""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
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
    impacted_areas      TEXT,
    impacted_functions  TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_requirements_doc ON requirements(document_id);

CREATE TABLE IF NOT EXISTS questionnaires (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id             INTEGER,
    regulation              TEXT,
    name                    TEXT    NOT NULL,
    package_json            TEXT    NOT NULL,
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
    state_json                  TEXT,
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
"""


def init_db(db_path: Optional[os.PathLike[str] | str] = None) -> Path:
    """Create the schema if missing. Returns the resolved DB path."""
    path = _resolve_db_path(db_path)
    with session(path) as conn:
        conn.executescript(_SCHEMA)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def _rows_to_dicts(rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    id: int
    name: str
    kind: str
    path: str
    mime: Optional[str]
    size_bytes: Optional[int]
    regulation: Optional[str]
    uploaded_at: str


def save_document(
    *,
    name: str,
    kind: str,
    path: str,
    mime: Optional[str] = None,
    size_bytes: Optional[int] = None,
    regulation: Optional[str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> int:
    """Register an uploaded file in the ``documents`` table.

    The document row itself carries only metadata (name, path, size,
    regulation label) — no LLM-generated text — so we don't run the
    guardrail sweep here. Guardrails apply to the derived artefacts
    (requirements, questionnaires, assessments) generated *from* the
    document, which are written through the other ``save_*`` functions.
    """
    if kind not in {"regulation", "brd", "frd", "other"}:
        raise ValueError(f"Invalid document kind: {kind!r}")
    with session(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO documents (name, kind, path, mime, size_bytes, regulation, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (name, kind, str(path), mime, size_bytes, regulation, _now_iso()),
        )
        return int(cur.lastrowid)


def list_documents(
    kind: Optional[str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> List[Dict[str, Any]]:
    with session(db_path) as conn:
        if kind:
            rows = conn.execute(
                "SELECT * FROM documents WHERE kind = ? ORDER BY id DESC", (kind,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY id DESC"
            ).fetchall()
    return _rows_to_dicts(rows)


def get_document(document_id: int, db_path: Optional[os.PathLike[str] | str] = None) -> Optional[Dict[str, Any]]:
    with session(db_path) as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Requirements (denormalised view of a parsed BRD)
# ---------------------------------------------------------------------------

def save_requirements(
    *,
    document_id: int,
    requirements: Iterable[Mapping[str, Any]],
    db_path: Optional[os.PathLike[str] | str] = None,
    guardrail_context: Optional[Mapping[str, Any]] = None,
) -> int:
    """Replace requirement rows for a document. Returns rows inserted.

    Runs the pre-persistence guardrail against the incoming requirement
    list before deleting the old rows so a hallucinated / meta-leaked
    payload cannot destroy the existing set. Any critical finding in
    strict mode aborts the write.
    """
    requirements_list = list(requirements)
    _run_persist_guardrail(
        requirements_list, component="requirements", context=guardrail_context,
    )
    rows_written = 0
    with session(db_path) as conn:
        conn.execute("DELETE FROM requirements WHERE document_id = ?", (document_id,))
        for req in requirements_list:
            conn.execute(
                """
                INSERT INTO requirements (document_id, requirement_id, section, description, impacted_areas, impacted_functions)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    str(req.get("requirement_id", "")),
                    req.get("section"),
                    req.get("description"),
                    json.dumps(list(req.get("impacted_areas", []) or [])),
                    json.dumps(list(req.get("impacted_functions", []) or [])),
                ),
            )
            rows_written += 1
    return rows_written


def list_requirements(
    document_id: int,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> List[Dict[str, Any]]:
    with session(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM requirements WHERE document_id = ? ORDER BY id", (document_id,)
        ).fetchall()
    out = _rows_to_dicts(rows)
    for r in out:
        try:
            r["impacted_areas"] = json.loads(r.get("impacted_areas") or "[]")
        except json.JSONDecodeError:
            r["impacted_areas"] = []
        try:
            r["impacted_functions"] = json.loads(r.get("impacted_functions") or "[]")
        except json.JSONDecodeError:
            r["impacted_functions"] = []
    return out


# ---------------------------------------------------------------------------
# Questionnaires
# ---------------------------------------------------------------------------

def save_questionnaire(
    *,
    name: str,
    package: Mapping[str, Any],
    document_id: Optional[int] = None,
    regulation: Optional[str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
    guardrail_context: Optional[Mapping[str, Any]] = None,
) -> int:
    """Persist a questionnaire package after running the pre-persistence guardrail.

    ``guardrail_context`` accepts ``source_corpus`` (concatenated regulation
    text), ``regulation`` (label — defaults to the argument above), and
    ``client_roles`` (list of institution types). The guardrail report is
    embedded inside ``package_json`` under the ``_persistence_guardrail``
    key so the review-queue page can surface it.
    """
    meta = dict(package.get("metadata", {})) if isinstance(package.get("metadata"), Mapping) else {}
    questions = list(package.get("questions") or [])
    requirements = list(package.get("requirements") or [])

    ctx = dict(guardrail_context or {})
    ctx.setdefault("regulation", regulation or meta.get("regulation"))
    guardrail_dict = _run_persist_guardrail(
        package, component="questionnaire", context=ctx,
    )
    package_to_write = dict(package)
    if guardrail_dict is not None:
        package_to_write["_persistence_guardrail"] = guardrail_dict

    with session(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO questionnaires
                (document_id, regulation, name, package_json, question_count, requirement_count, overall_confidence_pct, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                regulation or meta.get("regulation"),
                name,
                json.dumps(package_to_write, ensure_ascii=False),
                len(questions),
                len(requirements),
                float(meta.get("overall_confidence_pct") or 0.0),
                _now_iso(),
            ),
        )
        return int(cur.lastrowid)


def list_questionnaires(db_path: Optional[os.PathLike[str] | str] = None) -> List[Dict[str, Any]]:
    with session(db_path) as conn:
        rows = conn.execute(
            "SELECT id, document_id, regulation, name, question_count, requirement_count, "
            "overall_confidence_pct, created_at FROM questionnaires ORDER BY id DESC"
        ).fetchall()
    return _rows_to_dicts(rows)


def get_questionnaire(
    questionnaire_id: int,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Optional[Dict[str, Any]]:
    with session(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM questionnaires WHERE id = ?", (questionnaire_id,)
        ).fetchone()
    rec = _row_to_dict(row)
    if rec and rec.get("package_json"):
        try:
            rec["package"] = json.loads(rec["package_json"])
        except json.JSONDecodeError:
            rec["package"] = None
    return rec


# ---------------------------------------------------------------------------
# Assessments + responses
# ---------------------------------------------------------------------------

def create_assessment(
    *,
    questionnaire_id: int,
    name: str,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> int:
    now = _now_iso()
    with session(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO assessments
                (questionnaire_id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (questionnaire_id, name, now, now),
        )
        return int(cur.lastrowid)


def update_assessment_snapshot(
    *,
    assessment_id: int,
    state_json: Optional[str] = None,
    evaluation: Optional[Mapping[str, Any]] = None,
    recommendations: Optional[Iterable[Mapping[str, Any]]] = None,
    completed: bool = False,
    db_path: Optional[os.PathLike[str] | str] = None,
    guardrail_context: Optional[Mapping[str, Any]] = None,
) -> None:
    """Update the snapshot columns on an assessment row.

    Runs the pre-persistence guardrail against ``evaluation`` and
    ``recommendations`` (whichever are supplied) before writing. Any
    resulting report is embedded alongside the JSON so the review queue
    can render it later.
    """
    recommendations_list = list(recommendations) if recommendations is not None else None

    guardrail_dicts: List[Dict[str, Any]] = []
    if evaluation is not None:
        report_eval = _run_persist_guardrail(
            _jsonable(evaluation),
            component="assessment_evaluation",
            context=guardrail_context,
        )
        if report_eval:
            guardrail_dicts.append(report_eval)
    if recommendations_list is not None:
        report_recs = _run_persist_guardrail(
            [dict(r) for r in recommendations_list],
            component="assessment_recommendations",
            context=guardrail_context,
        )
        if report_recs:
            guardrail_dicts.append(report_recs)

    evaluation_to_write = evaluation
    if evaluation is not None and guardrail_dicts:
        try:
            evaluation_to_write = dict(evaluation)
            evaluation_to_write["_persistence_guardrail"] = guardrail_dicts
        except Exception:  # pragma: no cover
            evaluation_to_write = evaluation

    eval_json = (
        json.dumps(_jsonable(evaluation_to_write), ensure_ascii=False)
        if evaluation_to_write is not None else None
    )
    recs_json = (
        json.dumps([dict(r) for r in recommendations_list], ensure_ascii=False)
        if recommendations_list is not None
        else None
    )
    compliance = None
    confidence = None
    answered = None
    if evaluation:
        compliance = evaluation.get("compliance_score_pct")
        confidence = evaluation.get("evaluation_confidence_pct")
        answered = evaluation.get("answered_count")
    now = _now_iso()
    fields: List[str] = ["updated_at = ?"]
    values: List[Any] = [now]
    if state_json is not None:
        fields.append("state_json = ?")
        values.append(state_json)
    if eval_json is not None:
        fields.append("evaluation_json = ?")
        values.append(eval_json)
        fields.append("compliance_score_pct = ?")
        values.append(compliance)
        fields.append("evaluation_confidence_pct = ?")
        values.append(confidence)
        fields.append("answered_count = ?")
        values.append(answered)
    if recs_json is not None:
        fields.append("recommendations_json = ?")
        values.append(recs_json)
    if completed:
        fields.append("completed_at = ?")
        values.append(now)
    values.append(assessment_id)
    with session(db_path) as conn:
        conn.execute(
            f"UPDATE assessments SET {', '.join(fields)} WHERE id = ?",
            tuple(values),
        )


def upsert_responses(
    *,
    assessment_id: int,
    responses: Mapping[str, Any],
    db_path: Optional[os.PathLike[str] | str] = None,
) -> int:
    """Replace the response set for an assessment with the supplied mapping."""
    written = 0
    now = _now_iso()
    with session(db_path) as conn:
        conn.execute("DELETE FROM responses WHERE assessment_id = ?", (assessment_id,))
        for qid, answer in responses.items():
            if qid.endswith("__display_sequence") or qid.endswith("__comments"):
                continue
            conn.execute(
                """
                INSERT INTO responses (assessment_id, question_id, answer_json, answered_at)
                VALUES (?, ?, ?, ?)
                """,
                (assessment_id, qid, json.dumps(answer, ensure_ascii=False), now),
            )
            written += 1
    return written


def list_assessments(
    questionnaire_id: Optional[int] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> List[Dict[str, Any]]:
    with session(db_path) as conn:
        if questionnaire_id is None:
            rows = conn.execute(
                "SELECT id, questionnaire_id, name, created_at, updated_at, completed_at, "
                "compliance_score_pct, evaluation_confidence_pct, answered_count "
                "FROM assessments ORDER BY id DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, questionnaire_id, name, created_at, updated_at, completed_at, "
                "compliance_score_pct, evaluation_confidence_pct, answered_count "
                "FROM assessments WHERE questionnaire_id = ? ORDER BY id DESC",
                (questionnaire_id,),
            ).fetchall()
    return _rows_to_dicts(rows)


def get_assessment(
    assessment_id: int,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Optional[Dict[str, Any]]:
    with session(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM assessments WHERE id = ?", (assessment_id,)
        ).fetchone()
    rec = _row_to_dict(row)
    if rec:
        for k in ("evaluation_json", "recommendations_json"):
            raw = rec.get(k)
            if raw:
                try:
                    rec[k.replace("_json", "")] = json.loads(raw)
                except json.JSONDecodeError:
                    rec[k.replace("_json", "")] = None
    return rec


def get_responses(
    assessment_id: int,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    with session(db_path) as conn:
        rows = conn.execute(
            "SELECT question_id, answer_json FROM responses WHERE assessment_id = ?",
            (assessment_id,),
        ).fetchall()
    out: Dict[str, Any] = {}
    for r in rows:
        try:
            out[r["question_id"]] = json.loads(r["answer_json"]) if r["answer_json"] else None
        except json.JSONDecodeError:
            out[r["question_id"]] = r["answer_json"]
    return out


# ---------------------------------------------------------------------------
# JSON-safety helper (tuple keys, sets, etc.)
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Convert non-JSON-native structures (sets, tuple keys) into JSON-safe ones."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key = " | ".join(str(p) for p in k) if isinstance(k, tuple) else str(k)
            out[key] = _jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_jsonable(x) for x in obj)
    return obj


__all__ = [
    "DEFAULT_DB_PATH",
    "DocumentRecord",
    "connect",
    "create_assessment",
    "get_assessment",
    "get_document",
    "get_questionnaire",
    "get_responses",
    "init_db",
    "list_assessments",
    "list_documents",
    "list_questionnaires",
    "list_requirements",
    "save_document",
    "save_questionnaire",
    "save_requirements",
    "session",
    "update_assessment_snapshot",
    "upsert_responses",
]
