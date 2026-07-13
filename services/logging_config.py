"""Centralised logging configuration for Reg AI RAP.

The Streamlit app, orchestrator, agents, and every service module all
funnel their logs through the standard :mod:`logging` package. This
module wires up two handlers with sensible defaults:

* **Rotating file handler** — writes to ``logs/regai.log`` with 5 MB
  rotation and 5 backups so we always have the last ~25 MB of history.
* **Console handler** — mirrors WARNING+ to stderr so operational
  problems still surface in the terminal that launched Streamlit.

The config is idempotent — calling :func:`setup_logging` more than once
(for example because Streamlit re-imports ``app.py`` on hot-reload)
will not duplicate handlers. Log level is configurable via the
``REGAI_LOG_LEVEL`` environment variable (default ``INFO``); the file
log always captures at least DEBUG so we never lose diagnostic detail
when something breaks.

Usage in any module::

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Something notable happened")
    logger.exception("Bad thing happened, stack trace attached")

The root call to :func:`setup_logging` lives at the top of ``app.py``.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_INITIALISED = False


def _resolve_log_dir() -> Path:
    """Return the directory where log files should live, creating it if needed."""
    env_path = os.environ.get("REGAI_LOG_DIR")
    if env_path:
        base = Path(env_path).expanduser().resolve()
    else:
        base = Path(__file__).resolve().parent.parent / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_level(default: str = "INFO") -> int:
    """Read ``REGAI_LOG_LEVEL`` (or fall back to ``default``) and return the numeric level."""
    raw = os.environ.get("REGAI_LOG_LEVEL", default).strip().upper()
    level = getattr(logging, raw, None)
    if not isinstance(level, int):
        level = logging.INFO
    return level


def setup_logging(
    *,
    log_dir: Optional[Path] = None,
    console_level: Optional[int] = None,
    file_level: int = logging.DEBUG,
) -> Path:
    """Configure the root logger with a rotating file + console handler.

    Safe to call multiple times — handlers are only attached once.

    Args:
        log_dir: override the log directory (defaults to ``<repo>/logs``).
        console_level: minimum level printed to stderr. Defaults to the
            level implied by ``REGAI_LOG_LEVEL`` or INFO.
        file_level: minimum level written to the file. Defaults to DEBUG
            so the file always has the most detail.

    Returns:
        The absolute path to the rotating log file that will receive
        every module's output.
    """
    global _INITIALISED

    resolved_log_dir = log_dir or _resolve_log_dir()
    log_file = resolved_log_dir / "regai.log"

    root = logging.getLogger()
    # The root logger must be at least as verbose as the most permissive
    # handler; otherwise DEBUG records get dropped before reaching the
    # file handler.
    root.setLevel(min(file_level, _resolve_level()))

    if _INITIALISED and getattr(root, "_regai_configured", False):
        return log_file

    for handler in list(root.handlers):
        if getattr(handler, "_regai_handler", False):
            root.removeHandler(handler)

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    file_handler._regai_handler = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setLevel(console_level if console_level is not None else _resolve_level())
    stream_handler.setFormatter(formatter)
    stream_handler._regai_handler = True  # type: ignore[attr-defined]
    root.addHandler(stream_handler)

    # Quieten a handful of extremely chatty third-party loggers that
    # would otherwise flood the log file on every LLM / search / DNS call.
    for noisy in (
        "urllib3", "httpx", "httpcore", "openai._base_client",
        "asyncio", "watchdog", "streamlit.runtime",
        # DDGS + hickory-resolver + reqwest/rustls (used by the regulator
        # search pipeline) emit hundreds of DEBUG lines per query for
        # DNS / TLS handshakes / cookies. We only care about their warnings.
        "ddgs", "ddgs.ddgs", "duckduckgo_search",
        "hickory_net", "hickory_resolver",
        "hickory_net.udp.udp_stream", "hickory_net.udp.udp_client_stream",
        "hickory_resolver.name_server", "hickory_resolver.lookup_ip",
        "reqwest", "reqwest.connect",
        "rustls", "rustls.webpki.anchors",
        "cookie_store", "cookie_store.cookie_store",
        "hyper_util", "hyper_util.client", "hyper_util.client.legacy",
        "hyper_util.client.legacy.connect", "hyper_util.client.legacy.pool",
        "hyper_util.client.legacy.connect.http",
        "h2", "h2.client", "h2.codec", "h2.frame",
        "h2.codec.framed_read", "h2.codec.framed_write",
        "h2.frame.settings", "h2.proto", "h2.proto.connection",
        "h2.proto.settings",
        "primp",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._regai_configured = True  # type: ignore[attr-defined]
    _INITIALISED = True

    logging.getLogger(__name__).info(
        "Reg AI RAP logging initialised. file=%s file_level=%s console_level=%s",
        log_file,
        logging.getLevelName(file_level),
        logging.getLevelName(
            console_level if console_level is not None else _resolve_level()
        ),
    )
    return log_file


def get_log_file_path() -> Path:
    """Return the current rotating log file path (initialises logging if needed)."""
    return setup_logging()


__all__ = ["setup_logging", "get_log_file_path"]
