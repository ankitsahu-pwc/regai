"""Filesystem helpers for uploads, outputs, and Streamlit file handling.

Keeps file-IO concerns out of the Streamlit page code so app.py can stay
focused on UX wiring.
"""

from __future__ import annotations

import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

PathLike = Union[str, Path]

_INVALID_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._\-]+")


def ensure_dirs(*paths: PathLike) -> None:
    """Create every directory if it does not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def safe_filename(name: str, fallback: str = "upload") -> str:
    """Return a filesystem-safe filename derived from ``name``.

    Non-ASCII / unsafe characters are replaced with underscores. Empty names
    fall back to ``fallback``.
    """
    base = name.strip().replace("\\", "/").split("/")[-1]
    cleaned = _INVALID_FILENAME_CHARS.sub("_", base).strip("._-")
    return cleaned or fallback


def save_upload(uploaded_file, dest_dir: PathLike, *, prefix: Optional[str] = None) -> Path:
    """Persist a Streamlit ``UploadedFile`` (or anything with ``.name`` + ``.getbuffer``) to disk.

    A short uuid is added to the filename so re-uploading the same name does not
    overwrite the previous file.
    """
    if uploaded_file is None:
        raise ValueError("uploaded_file is None")

    ensure_dirs(dest_dir)
    original_name = safe_filename(getattr(uploaded_file, "name", "upload"))
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix
    unique = uuid.uuid4().hex[:8]
    parts = [p for p in (prefix, stem, unique) if p]
    final_name = "_".join(parts) + suffix
    target = Path(dest_dir) / final_name

    if hasattr(uploaded_file, "getbuffer"):
        data = uploaded_file.getbuffer()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        raise TypeError(
            f"Unsupported uploaded file object: {type(uploaded_file).__name__}"
        )

    with open(target, "wb") as f:
        f.write(bytes(data))
    return target


def read_bytes(path: PathLike) -> bytes:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_bytes()


def copy_into(src: PathLike, dest_dir: PathLike) -> Path:
    """Copy ``src`` into ``dest_dir``, returning the new path."""
    ensure_dirs(dest_dir)
    src = Path(src)
    target = Path(dest_dir) / src.name
    shutil.copy2(src, target)
    return target


def timestamped_name(stem: str, suffix: str) -> str:
    """Return ``stem_YYYYMMDDTHHMMSS.suffix`` — handy for export artefacts."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return f"{safe_filename(stem)}_{ts}{clean_suffix}"


def iter_files(directory: PathLike, patterns: Iterable[str]) -> Iterable[Path]:
    """Yield files matching any of the given glob patterns under ``directory``."""
    directory = Path(directory)
    if not directory.exists():
        return
    seen: set[Path] = set()
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


__all__ = [
    "PathLike",
    "copy_into",
    "ensure_dirs",
    "iter_files",
    "read_bytes",
    "safe_filename",
    "save_upload",
    "timestamped_name",
]
