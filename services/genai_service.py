"""PwC GenAI Shared Service client.

This module is a refactor of the HTTP / SSL / LLM-client infrastructure that
originally lived inside ``GenAISharedServiceBRDFRDv5.py``. The logic is
preserved verbatim where possible so the existing BRD/FRD generator (lifted in
Phase 4) can re-use these helpers unchanged.

Behavioural differences vs. the original script
-----------------------------------------------
1. ``get_llm_api_key()`` never blocks on ``input()``. If ``API_KEY`` is missing
   it raises ``GenAIConfigError``. Callers (Streamlit, CLI, tests) decide
   whether to fall back to offline mode.
2. ``load_dotenv()`` is called at import time so a workspace ``.env`` file is
   the single source of configuration truth.
3. All configuration is exposed via the ``GenAISettings`` dataclass so callers
   can introspect / override programmatically; the module-level constants stay
   for backwards compatibility with the lifted BRD code.
"""

from __future__ import annotations

import os
import ssl
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import httpx

try:
    import certifi
except ImportError:
    certifi = None

try:
    import truststore
except ImportError:
    truststore = None

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError
except ImportError:
    APIConnectionError = Exception
    APIStatusError = Exception
    APITimeoutError = Exception

try:
    # Newer openai SDKs raise this when the response is truncated by
    # max_tokens / model context window.
    from openai import LengthFinishReasonError as _LengthFinishReasonError
except ImportError:
    _LengthFinishReasonError = None


def _is_length_limit_error(exc: BaseException) -> bool:
    """True when ``exc`` looks like an LLM length / token-limit truncation.

    The OpenAI SDK raises :class:`openai.LengthFinishReasonError` for this
    specific case, but we also catch the more generic message form so we
    keep working against older SDKs or proxies that surface the error as a
    plain ``ValueError`` / ``Exception``.
    """
    if _LengthFinishReasonError is not None and isinstance(exc, _LengthFinishReasonError):
        return True
    if type(exc).__name__ == "LengthFinishReasonError":
        return True
    msg = str(exc).lower()
    return (
        "length limit" in msg
        or "finish_reason" in msg and "length" in msg
        or "max_tokens" in msg and "exceed" in msg
    )

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class GenAIConfigError(RuntimeError):
    """Raised when required GenAI Shared Service configuration is missing."""


class GenAIServiceUnavailable(RuntimeError):
    """Raised when the GenAI Shared Service endpoint is unreachable."""


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class GenAISettings:
    """Snapshot of GenAI Shared Service configuration resolved from the environment."""

    base_url: str
    model: str
    timeout_seconds: float
    max_tokens: int
    context_chars: int
    verify_ssl: bool
    skip_api: bool
    ssl_strategy: str
    explicit_ca_bundle: Optional[str]
    https_proxy: Optional[str]
    http_proxy: Optional[str]

    @classmethod
    def from_env(cls) -> "GenAISettings":
        return cls(
            base_url=os.getenv(
                "GENAI_SHARED_SERVICE_BASE",
                "https://genai-sharedservice-americas.pwcinternal.com",
            ).strip().rstrip("/"),
            model=os.getenv("GENAI_SHARED_SERVICE_MODEL", "azure.gpt-4o").strip(),
            timeout_seconds=_env_float("OPENAI_TIMEOUT_SECONDS", "180"),
            # 6000 is large enough for the heaviest BRD bundle
            # (functional + non-functional requirements). The historical
            # default of 2200 truncated that bundle and tripped the
            # LengthFinishReasonError fallback.
            max_tokens=_env_int("OPENAI_MAX_TOKENS", "6000"),
            context_chars=_env_int("GENAI_CONTEXT_CHARS", "6000"),
            verify_ssl=os.getenv("OPENAI_VERIFY_SSL", "true").lower() != "false",
            skip_api=_env_bool("OPENAI_SKIP_API", "false"),
            ssl_strategy=os.getenv("OPENAI_SSL_STRATEGY", "auto").strip().lower(),
            explicit_ca_bundle=os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE"),
            https_proxy=os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"),
            http_proxy=os.getenv("HTTP_PROXY") or os.getenv("http_proxy"),
        )


def get_settings() -> GenAISettings:
    """Return a freshly-resolved settings snapshot (re-reads the environment)."""
    return GenAISettings.from_env()


def get_llm_api_key() -> str:
    """Return the PwC GenAI Shared Service bearer token from ``API_KEY``.

    Unlike the original script this function never calls ``input()`` because
    Streamlit cannot block on stdin. If ``API_KEY`` is absent the caller
    decides whether to fall back to offline mode.
    """
    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        raise GenAIConfigError(
            "API_KEY is not set. Add it to your .env file or export it before "
            "running the application."
        )
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key


def build_ssl_verify_setting(settings: Optional[GenAISettings] = None) -> Any:
    """Return the SSL verification setting for httpx, mirroring the original script.

    Preferred corporate fix on Windows: ``pip install truststore``. When
    ``OPENAI_SSL_STRATEGY=auto`` and the package is available, Python uses the
    operating system certificate store where corporate roots are installed.
    """
    settings = settings or get_settings()

    if not settings.verify_ssl:
        return False

    if settings.explicit_ca_bundle:
        return settings.explicit_ca_bundle

    use_os_store = settings.ssl_strategy in {"auto", "windows_store", "system", "os"}

    if use_os_store and truststore is not None:
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    return certifi.where() if certifi else True


def build_http_client(settings: Optional[GenAISettings] = None) -> httpx.Client:
    """Build the corporate-friendly httpx client used for all GenAI calls."""
    settings = settings or get_settings()
    verify_value = build_ssl_verify_setting(settings)
    return httpx.Client(
        trust_env=True,
        verify=verify_value,
        timeout=httpx.Timeout(
            timeout=settings.timeout_seconds,
            connect=45.0,
            read=settings.timeout_seconds,
            write=60.0,
        ),
    )


def preflight_openai_connectivity(
    http_client: httpx.Client,
    settings: Optional[GenAISettings] = None,
) -> bool:
    """Lightweight chat-completions ping. Returns True if a 200 was observed."""
    settings = settings or get_settings()

    if settings.skip_api:
        return False

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
    if not api_key:
        return False

    token_param = os.getenv("OPENAI_TOKEN_PARAM_NAME", "max_completion_tokens").strip()
    payload: dict = {
        "model": settings.model,
        "messages": [{"role": "user", "content": "Return the word OK."}],
        token_param: 5,
    }
    if os.getenv("OPENAI_SEND_TEMPERATURE", "false").strip().lower() == "true":
        payload["temperature"] = 0

    try:
        response = http_client.post(
            f"{settings.base_url}/chat/completions",
            headers={
                "accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    except Exception:
        return False

    if response.status_code == 200:
        return True
    return False


def create_configured_llm(
    api_key: str,
    custom_http_client: httpx.Client,
    settings: Optional[GenAISettings] = None,
) -> ChatOpenAI:
    """Construct a LangChain ChatOpenAI bound to the PwC GenAI Shared Service."""
    settings = settings or get_settings()
    token_param = os.getenv("OPENAI_TOKEN_PARAM_NAME", "max_completion_tokens").strip()
    send_temperature = os.getenv("OPENAI_SEND_TEMPERATURE", "false").strip().lower() == "true"
    kwargs: dict = {
        "model": settings.model,
        "max_retries": 3,
        "timeout": settings.timeout_seconds,
        "api_key": api_key,
        "base_url": settings.base_url,
        "http_client": custom_http_client,
        "model_kwargs": {token_param: settings.max_tokens},
    }
    if send_temperature:
        kwargs["temperature"] = 0.10
    return ChatOpenAI(**kwargs)


_DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a principal regulatory compliance architect, DORA SME, and senior business analyst. "
    "Generate consulting-grade BRD/FRD content for a regulated financial services company.\n\n"
    "Important output rules:\n"
    "- Return valid structured output only.\n"
    "- Be detailed, practical, and implementation-oriented.\n"
    "- Keep the response bounded; do not over-generate.\n"
    "- Use formal business English.\n"
    "- Apply Tier-2 proportionality: detailed but not over-engineered.\n"
    "- Include diagnostic cockpit concepts where relevant: data ingestion, mapping, rules, "
    "exceptions, dashboards, evidence, controls, workflow, and traceability."
)


def generate_structured_component(
    llm: ChatOpenAI,
    schema_model: Any,
    component_name: str,
    component_instruction: str,
    context: str,
    system_instruction: str = _DEFAULT_SYSTEM_INSTRUCTION,
) -> Any:
    """Invoke the LLM once for a single Pydantic schema slice (token-optimised)."""
    structured_llm = llm.with_structured_output(schema_model)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            (
                "user",
                "Component to generate:\n{component_name}\n\n"
                "Component-specific instructions:\n{component_instruction}\n\n"
                "Regulatory and reference context:\n{context}\n",
            ),
        ]
    )
    chain = prompt | structured_llm
    return chain.invoke(
        {
            "component_name": component_name,
            "component_instruction": component_instruction,
            "context": context,
        }
    )


class GenAIClient:
    """Convenience facade so callers can hold one configured client.

    Usage::

        client = GenAIClient.try_create()
        if client is None:
            # Use offline fallback content
            ...
        else:
            output = client.generate(MyPydanticModel, "Section 1", "Instruction...", context)
    """

    def __init__(
        self,
        api_key: str,
        http_client: httpx.Client,
        settings: GenAISettings,
        llm: ChatOpenAI,
    ) -> None:
        self.api_key = api_key
        self.http_client = http_client
        self.settings = settings
        self.llm = llm

    @classmethod
    def try_create(cls) -> Optional["GenAIClient"]:
        """Return a ready client, or None if the service is unavailable or unconfigured.

        The caller can branch on ``None`` to use the offline fallback content.
        """
        settings = get_settings()
        if settings.skip_api:
            return None
        try:
            api_key = get_llm_api_key()
        except GenAIConfigError:
            return None

        http_client = build_http_client(settings)
        if not preflight_openai_connectivity(http_client, settings):
            http_client.close()
            return None

        llm = create_configured_llm(api_key, http_client, settings)
        return cls(api_key=api_key, http_client=http_client, settings=settings, llm=llm)

    def generate(
        self,
        schema_model: Any,
        component_name: str,
        component_instruction: str,
        context: str,
        system_instruction: str = _DEFAULT_SYSTEM_INSTRUCTION,
    ) -> Any:
        return generate_structured_component(
            self.llm,
            schema_model,
            component_name,
            component_instruction,
            context[: self.settings.context_chars],
            system_instruction=system_instruction,
        )

    def generate_with_length_retry(
        self,
        schema_model: Any,
        component_name: str,
        component_instruction: str,
        context: str,
        system_instruction: str = _DEFAULT_SYSTEM_INSTRUCTION,
        *,
        max_retry_tokens: int = 12000,
        on_retry: Optional[Any] = None,
    ) -> Any:
        """Like :meth:`generate`, but retries once with a higher ``max_tokens``
        if the model trips a length-limit error.

        Several BRD bundles (Functional + Non-Functional requirements, the
        Governance bundle, the Workshop Delivery Plan) regularly need more
        than the default ``OPENAI_MAX_TOKENS`` budget when the model decides
        to be verbose. Bumping the default helps; this retry mops up the
        remaining edge cases without making every call slow.

        ``on_retry`` is an optional ``callable(message: str)`` for status
        logging (compatible with the Streamlit ``st.status.write`` writer).
        """
        try:
            return self.generate(
                schema_model, component_name, component_instruction,
                context, system_instruction,
            )
        except Exception as exc:
            if not _is_length_limit_error(exc):
                raise
            if on_retry is not None:
                try:
                    on_retry(
                        f"{component_name}: hit length limit; retrying once "
                        f"with max_tokens={max_retry_tokens}."
                    )
                except Exception:
                    pass
            original_max = getattr(self.llm, "max_tokens", None)
            try:
                self.llm.max_tokens = max_retry_tokens
                return self.generate(
                    schema_model, component_name, component_instruction,
                    context, system_instruction,
                )
            finally:
                if original_max is not None:
                    self.llm.max_tokens = original_max

    def close(self) -> None:
        try:
            self.http_client.close()
        except Exception:
            pass


__all__ = [
    "APIConnectionError",
    "APIStatusError",
    "APITimeoutError",
    "GenAIClient",
    "GenAIConfigError",
    "GenAIServiceUnavailable",
    "GenAISettings",
    "build_http_client",
    "build_ssl_verify_setting",
    "create_configured_llm",
    "generate_structured_component",
    "get_llm_api_key",
    "get_settings",
    "preflight_openai_connectivity",
]
