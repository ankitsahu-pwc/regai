"""Persistence facade.

The SQLite persistence implementation lives in :mod:`services.database`. This
module is a thin re-export so the new pipeline can import a name that matches
the proposed architecture (``services/persistence.py``) without forcing a
rename of the underlying schema/migrations file.

Both module names are valid import paths and behave identically:

    from services import persistence as db
    # is equivalent to
    from services import database as db
"""

from __future__ import annotations

from .database import (  # noqa: F401  (re-export)
    DEFAULT_DB_PATH,
    DocumentRecord,
    connect,
    create_assessment,
    get_assessment,
    get_document,
    get_questionnaire,
    get_responses,
    init_db,
    list_assessments,
    list_documents,
    list_questionnaires,
    list_requirements,
    save_document,
    save_questionnaire,
    save_requirements,
    session,
    update_assessment_snapshot,
    upsert_responses,
)

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
