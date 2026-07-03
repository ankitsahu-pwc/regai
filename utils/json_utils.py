"""JSON helpers for reading, writing, and lightly validating questionnaire packages.

The schema mirrors what ``generate_brd_questionnaire_streamlit_v11.py`` already
produces (see ``sample_data/dora_questionnaire_package_v10.json``). We do not
introduce a new schema — we just enforce that loaded JSON conforms to the one
the existing generator emits, so Phase 7's "load sample / load existing
questionnaire" paths are contract-safe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

PathLike = Union[str, Path]

REQUIRED_TOP_KEYS = ("metadata", "requirements", "impact_pairs", "questions")
REQUIRED_REQUIREMENT_KEYS = ("normalized_id", "category", "requirement", "detail")
REQUIRED_PAIR_KEYS = ("area", "function", "requirement_ids")
REQUIRED_QUESTION_KEYS = (
    "question_id",
    "area",
    "function",
    "question_type",
    "question",
    "options",
    "mapped_requirement_ids",
    "confidence",
)


def read_json(path: PathLike) -> Any:
    """Read any JSON file with UTF-8 encoding."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, data: Any, *, indent: int = 2) -> Path:
    """Write JSON with UTF-8 + indentation, returning the resolved path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return path


def read_package_json(path: PathLike) -> Dict[str, Any]:
    """Read and validate a questionnaire package JSON file.

    Raises ``ValueError`` if the file structure does not match the contract
    produced by the questionnaire generator.
    """
    data = read_json(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected top-level object, got {type(data).__name__}")
    issues = validate_package_schema(data)
    if issues:
        raise ValueError(
            f"{path}: questionnaire package JSON failed validation:\n  - "
            + "\n  - ".join(issues)
        )
    return dict(data)


def write_package_json(path: PathLike, data: Mapping[str, Any]) -> Path:
    """Validate then write a questionnaire package JSON file."""
    issues = validate_package_schema(data)
    if issues:
        raise ValueError(
            "Refusing to write invalid questionnaire package JSON:\n  - "
            + "\n  - ".join(issues)
        )
    return write_json(path, dict(data))


def validate_package_schema(data: Mapping[str, Any]) -> List[str]:
    """Return a list of human-readable schema issues. Empty list = valid."""
    issues: List[str] = []
    if not isinstance(data, Mapping):
        return [f"top-level must be an object, got {type(data).__name__}"]

    for key in REQUIRED_TOP_KEYS:
        if key not in data:
            issues.append(f"missing top-level key '{key}'")

    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        issues.append("'metadata' must be an object")

    requirements = data.get("requirements") or []
    if not isinstance(requirements, list):
        issues.append("'requirements' must be a list")
    else:
        for idx, req in enumerate(requirements[:5]):
            if not isinstance(req, Mapping):
                issues.append(f"requirements[{idx}] must be an object")
                continue
            for key in REQUIRED_REQUIREMENT_KEYS:
                if key not in req:
                    issues.append(f"requirements[{idx}] missing '{key}'")

    pairs = data.get("impact_pairs") or []
    if not isinstance(pairs, list):
        issues.append("'impact_pairs' must be a list")
    else:
        for idx, pair in enumerate(pairs[:5]):
            if not isinstance(pair, Mapping):
                issues.append(f"impact_pairs[{idx}] must be an object")
                continue
            for key in REQUIRED_PAIR_KEYS:
                if key not in pair:
                    issues.append(f"impact_pairs[{idx}] missing '{key}'")

    questions = data.get("questions") or []
    if not isinstance(questions, list):
        issues.append("'questions' must be a list")
    else:
        for idx, q in enumerate(questions[:5]):
            if not isinstance(q, Mapping):
                issues.append(f"questions[{idx}] must be an object")
                continue
            for key in REQUIRED_QUESTION_KEYS:
                if key not in q:
                    issues.append(f"questions[{idx}] missing '{key}'")

    return issues


__all__ = [
    "REQUIRED_PAIR_KEYS",
    "REQUIRED_QUESTION_KEYS",
    "REQUIRED_REQUIREMENT_KEYS",
    "REQUIRED_TOP_KEYS",
    "read_json",
    "read_package_json",
    "validate_package_schema",
    "write_json",
    "write_package_json",
]
