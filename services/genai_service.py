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

import logging
import os
import ssl
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

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
from langchain_openai import AzureChatOpenAI, ChatOpenAI

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


def _detect_provider(base_url: str, explicit: str = "") -> str:
    """Return ``"azure"`` or ``"openai"`` based on env override + base URL.

    Rules (in order):

    * If ``GENAI_PROVIDER`` is set to ``azure`` or ``openai`` we honour it.
    * If the base URL contains ``.azure.com`` (Azure OpenAI Service or
      Azure AI Foundry) we treat it as Azure.
    * Otherwise we assume an OpenAI-compatible endpoint (this preserves
      the original PwC Shared Service behaviour by default).
    """
    override = (explicit or "").strip().lower()
    if override in {"azure", "openai"}:
        return override
    host = (base_url or "").lower()
    if ".azure.com" in host or ".openai.azure.com" in host:
        return "azure"
    return "openai"


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
    # New: Azure-only fields. ``provider`` is auto-detected from ``base_url``
    # (or forced via the ``GENAI_PROVIDER`` env var). ``api_version`` and
    # ``azure_deployment`` are only used when ``provider == "azure"``.
    provider: str = "openai"
    api_version: str = "2024-02-15-preview"
    azure_deployment: str = ""
    # Reasoning-model tuning. GPT-5 and the o-series are "reasoning" models
    # that spend hidden ``reasoning_tokens`` thinking before writing output.
    # The default effort is "high", which routinely burns the entire
    # ``max_completion_tokens`` budget on internal reasoning and produces
    # zero output text — the ``LengthFinishReasonError`` we see in production.
    # ``reasoning_effort`` (values: minimal / low / medium / high) is the
    # OpenAI-standard knob to cap this thinking budget. Empty string keeps
    # the SDK default (== "high") for backwards compatibility.
    reasoning_effort: str = ""

    @classmethod
    def from_env(cls) -> "GenAISettings":
        base_url = os.getenv(
            "GENAI_SHARED_SERVICE_BASE",
            "https://genai-sharedservice-americas.pwcinternal.com",
        ).strip().rstrip("/")
        model = os.getenv("GENAI_SHARED_SERVICE_MODEL", "azure.gpt-4o").strip()
        provider = _detect_provider(base_url, os.getenv("GENAI_PROVIDER", ""))
        # For Azure the "model" env var is actually the deployment name. We
        # allow ``AZURE_DEPLOYMENT`` as an explicit override, and strip the
        # historical ``azure.`` prefix from the model string when present so
        # existing .env files keep working.
        raw_deployment = os.getenv("AZURE_DEPLOYMENT", "").strip() or model
        if raw_deployment.startswith("azure."):
            raw_deployment = raw_deployment[len("azure."):]

        # Auto-pick a sensible reasoning_effort when the deployment looks
        # like a reasoning model (GPT-5, o-series). The user can always
        # override via GENAI_REASONING_EFFORT. We default to "minimal" for
        # GPT-5/o-* because our BRD prompts are already highly structured
        # and the default "high" effort routinely burns 6000 tokens on
        # internal reasoning alone, producing no output.
        explicit_effort = os.getenv("GENAI_REASONING_EFFORT", "").strip().lower()
        if explicit_effort:
            reasoning_effort = explicit_effort
        elif any(
            marker in raw_deployment.lower()
            for marker in ("gpt-5", "gpt5", "o1", "o3", "o4")
        ):
            reasoning_effort = "minimal"
        else:
            reasoning_effort = ""

        return cls(
            base_url=base_url,
            model=model,
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
            provider=provider,
            api_version=os.getenv("GENAI_API_VERSION", "2024-02-15-preview").strip(),
            azure_deployment=raw_deployment,
            reasoning_effort=reasoning_effort,
        )

    @property
    def is_azure(self) -> bool:
        """True when the configured endpoint is Azure OpenAI / Azure AI Foundry."""
        return self.provider == "azure"


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
    """Lightweight chat-completions ping. Returns True if a 200 was observed.

    Speaks two dialects transparently:

    * **OpenAI-native** (default) — ``POST {base_url}/chat/completions`` with
      ``Authorization: Bearer <key>``. This is what the PwC GenAI Shared
      Service proxy expects.
    * **Azure OpenAI / Azure AI Foundry** — when ``settings.is_azure`` is
      True the request goes to
      ``POST {base_url}/openai/deployments/{deployment}/chat/completions
      ?api-version={version}`` with the ``api-key`` header. Azure returns
      404 on the OpenAI-native URL, which is why the current app is
      falling back to offline mode.
    """
    settings = settings or get_settings()

    if settings.skip_api:
        return False

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
    if not api_key:
        return False

    token_param = os.getenv("OPENAI_TOKEN_PARAM_NAME", "max_completion_tokens").strip()
    # Preflight is a 5-token ping. We need a much larger budget for GPT-5
    # even for a tiny prompt, because reasoning tokens are counted against
    # the same cap. Give reasoning models a generous cushion so the
    # preflight never fails with LengthFinishReasonError.
    preflight_tokens = 4096 if settings.reasoning_effort else 5
    payload: dict = {
        "messages": [{"role": "user", "content": "Return the word OK."}],
        token_param: preflight_tokens,
    }
    if settings.reasoning_effort:
        payload["reasoning_effort"] = settings.reasoning_effort
    if os.getenv("OPENAI_SEND_TEMPERATURE", "false").strip().lower() == "true":
        payload["temperature"] = 0

    if settings.is_azure:
        deployment = settings.azure_deployment or settings.model
        url = (
            f"{settings.base_url}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={settings.api_version}"
        )
        headers = {
            "accept": "application/json",
            "api-key": api_key,
            "Content-Type": "application/json",
        }
        # Azure ignores "model" in the body (deployment is in the URL),
        # but including it does no harm and helps a few Foundry proxies.
        payload["model"] = deployment
    else:
        url = f"{settings.base_url}/chat/completions"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload["model"] = settings.model

    try:
        response = http_client.post(url, headers=headers, json=payload)
    except Exception:
        logger.warning(
            "GenAI preflight request raised. provider=%s url=%s (will run offline).",
            settings.provider, url,
            exc_info=True,
        )
        return False

    if response.status_code == 200:
        logger.info(
            "GenAI preflight OK. provider=%s base_url=%s model=%s",
            settings.provider, settings.base_url,
            settings.azure_deployment if settings.is_azure else settings.model,
        )
        return True
    logger.warning(
        "GenAI preflight non-200. provider=%s status=%s url=%s body=%s",
        settings.provider, response.status_code, url,
        (response.text or "")[:400],
    )
    return False


def create_configured_llm(
    api_key: str,
    custom_http_client: httpx.Client,
    settings: Optional[GenAISettings] = None,
) -> ChatOpenAI:
    """Construct a LangChain LLM bound to the configured GenAI backend.

    Returns a :class:`ChatOpenAI` for OpenAI-native / PwC Shared Service
    endpoints, or an :class:`AzureChatOpenAI` when ``settings.is_azure``
    is True. Both classes expose the same LangChain runnable / structured
    output API, so every caller downstream continues to work unchanged.
    """
    settings = settings or get_settings()
    token_param = os.getenv("OPENAI_TOKEN_PARAM_NAME", "max_completion_tokens").strip()
    send_temperature = os.getenv("OPENAI_SEND_TEMPERATURE", "false").strip().lower() == "true"

    # Build the shared model_kwargs bag. Reasoning-model tuning
    # (reasoning_effort) goes here because LangChain doesn't yet expose it
    # as a top-level constructor parameter — anything in ``model_kwargs``
    # is forwarded verbatim to the underlying OpenAI SDK.
    shared_model_kwargs: dict = {token_param: settings.max_tokens}
    if settings.reasoning_effort:
        shared_model_kwargs["reasoning_effort"] = settings.reasoning_effort

    if settings.is_azure:
        deployment = settings.azure_deployment or settings.model
        kwargs: dict = {
            "azure_endpoint": settings.base_url,
            "azure_deployment": deployment,
            "api_version": settings.api_version,
            "api_key": api_key,
            "http_client": custom_http_client,
            "max_retries": 3,
            "timeout": settings.timeout_seconds,
            "model_kwargs": shared_model_kwargs,
        }
        if send_temperature:
            kwargs["temperature"] = 0.10
        logger.info(
            "Creating AzureChatOpenAI. endpoint=%s deployment=%s api_version=%s reasoning_effort=%s max_tokens=%d",
            settings.base_url, deployment, settings.api_version,
            settings.reasoning_effort or "(default)", settings.max_tokens,
        )
        return AzureChatOpenAI(**kwargs)

    kwargs = {
        "model": settings.model,
        "max_retries": 3,
        "timeout": settings.timeout_seconds,
        "api_key": api_key,
        "base_url": settings.base_url,
        "http_client": custom_http_client,
        "model_kwargs": shared_model_kwargs,
    }
    if send_temperature:
        kwargs["temperature"] = 0.10
    logger.info(
        "Creating ChatOpenAI. base_url=%s model=%s reasoning_effort=%s max_tokens=%d",
        settings.base_url, settings.model,
        settings.reasoning_effort or "(default)", settings.max_tokens,
    )
    return ChatOpenAI(**kwargs)


_DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a principal regulatory compliance architect and senior business analyst "
    "authoring consulting-grade BRD / FRD / RTM artefacts for a regulated financial "
    "services company. Work strictly within the current regulation in scope; do not "
    "invent references to other frameworks.\n\n"
    "Frameworks you MUST follow when structuring output:\n"
    "- IIBA BABOK v3 (Business Analysis Body of Knowledge) for BRD structure: "
    "business need, scope, in-scope / out-of-scope, stakeholders, assumptions, "
    "constraints, dependencies, risks, business requirements, acceptance criteria, "
    "success measures, and traceability.\n"
    "- IREB CPRE (Certified Professional for Requirements Engineering) for FRD "
    "requirement quality: each requirement must be atomic, unambiguous, testable, "
    "prioritised (MoSCoW), traced to a source, and free of implementation bias.\n"
    "- SDLC discipline for RTM entries: every requirement traces to (a) a "
    "regulatory clause, (b) a business capability / function, (c) an assessment "
    "question, (d) a recommendation, and (e) supporting evidence.\n\n"
    "Every BRD MUST include, as first-class sections, the following BABOK "
    "artefacts (do not omit or merge them):\n"
    "  1. Prerequisites / preconditions\n"
    "  2. In-scope and out-of-scope\n"
    "  3. Assumptions\n"
    "  4. Dependencies\n"
    "  5. Risks (with impact and likelihood)\n"
    "  6. Controls (preventive / detective / corrective / governance)\n"
    "  7. Recommendations\n"
    "  8. Success criteria / acceptance criteria\n\n"
    "Output rules:\n"
    "- Return valid structured output only (matching the requested schema).\n"
    "- Be detailed, practical, and implementation-oriented; never speculative.\n"
    "- Keep the response bounded; do not over-generate.\n"
    "- Use formal business English.\n"
    "- Apply proportionality (Tier-1 / Tier-2 / Tier-3) as directed: detailed "
    "but not over-engineered.\n"
    "- Include diagnostic cockpit concepts where relevant: data ingestion, "
    "mapping, rules, exceptions, dashboards, evidence, controls, workflow, and "
    "traceability."
)


def _harden(text: str) -> str:
    """Prepend the shared anti-hallucination directive to a prompt string.

    Deferred / local import of :mod:`services.guardrails` so we avoid a
    circular import at module load time (guardrails imports the default
    system instruction for its own ``safe_generate`` wrapper).
    """
    try:
        from .guardrails import harden_instruction
    except Exception:  # pragma: no cover - defensive
        return text
    return harden_instruction(text or "")


def generate_structured_component(
    llm: ChatOpenAI,
    schema_model: Any,
    component_name: str,
    component_instruction: str,
    context: str,
    system_instruction: str = _DEFAULT_SYSTEM_INSTRUCTION,
) -> Any:
    """Invoke the LLM once for a single Pydantic schema slice (token-optimised).

    Every call passes through the shared anti-hallucination guardrails in
    :mod:`services.guardrails`. Both the system prompt AND the component
    instruction are hardened before dispatch — so every existing caller
    in the codebase is automatically protected without any signature
    change. Callers that need the *stronger* post-hoc validation
    (citation checking, regulation-scope validation, role-scope
    validation) should use :func:`services.guardrails.safe_generate`
    instead.
    """
    structured_llm = llm.with_structured_output(schema_model)

    system_prompt = _harden(system_instruction)
    hardened_component = _harden(component_instruction)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
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
            "component_instruction": hardened_component,
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
        # Guards the max_tokens mutation inside generate_with_length_retry.
        # httpx.Client and the underlying openai client are thread-safe for
        # concurrent .invoke() calls, so we do NOT hold this lock for
        # normal generations — only for the retry path that temporarily
        # rewrites the shared llm.max_tokens attribute. In practice
        # retries are very rare (<5% of calls once reasoning_effort=minimal
        # is set) so this lock costs virtually nothing under parallel load.
        self._retry_lock = threading.Lock()
        # Concurrency semaphore: caps how many LLM calls run in flight
        # simultaneously so we don't burst past Azure's per-minute
        # tokens-per-minute (TPM) budget and trigger 429 rate limits.
        # A 429 forces the openai SDK into 60–120s exponential backoff,
        # which is far more expensive than briefly queueing here.
        #
        # The default of 5 is calibrated for a GPT-5 deployment with the
        # standard 40k TPM quota when each call consumes ~6–8k tokens
        # (5 * 8k = 40k tokens/min sustained). It's overridable via
        # ``GENAI_MAX_CONCURRENCY`` in the environment when a customer
        # runs on a higher-tier deployment.
        try:
            _max_conc = int(os.environ.get("GENAI_MAX_CONCURRENCY", "5"))
        except (TypeError, ValueError):
            _max_conc = 5
        _max_conc = max(1, _max_conc)
        self._call_semaphore = threading.BoundedSemaphore(_max_conc)
        self._max_concurrency = _max_conc

    @classmethod
    def try_create(cls) -> Optional["GenAIClient"]:
        """Return a ready client, or None if the service is unavailable or unconfigured.

        The caller can branch on ``None`` to use the offline fallback content.
        """
        settings = get_settings()
        if settings.skip_api:
            logger.info("GenAI client construction skipped (skip_api=true, running offline).")
            return None
        try:
            api_key = get_llm_api_key()
        except GenAIConfigError:
            logger.warning("GenAI API key missing — running in offline mode.")
            return None

        http_client = build_http_client(settings)
        if not preflight_openai_connectivity(http_client, settings):
            logger.warning("GenAI preflight failed — falling back to offline mode.")
            http_client.close()
            return None

        llm = create_configured_llm(api_key, http_client, settings)
        instance = cls(api_key=api_key, http_client=http_client, settings=settings, llm=llm)
        logger.info(
            "GenAI client ready. provider=%s base_url=%s model=%s max_concurrency=%d",
            settings.provider, settings.base_url,
            settings.azure_deployment if settings.is_azure else settings.model,
            instance._max_concurrency,
        )
        return instance

    def generate(
        self,
        schema_model: Any,
        component_name: str,
        component_instruction: str,
        context: str,
        system_instruction: str = _DEFAULT_SYSTEM_INSTRUCTION,
        *,
        regulation: Optional[str] = None,
        client_roles: Optional[Any] = None,
    ) -> Any:
        """Invoke the LLM with anti-hallucination guardrails applied.

        The optional ``regulation`` and ``client_roles`` arguments let the
        caller specialise the shared guardrail directive with the current
        scope so the LLM knows exactly which regulation to stay inside and
        which institution types to consider. Callers can also skip these
        arguments — the generic (unspecialised) directive still applies.
        """
        component_hardened = component_instruction
        if regulation or client_roles:
            try:
                from .guardrails import harden_instruction
                component_hardened = harden_instruction(
                    component_instruction or "",
                    regulation=regulation,
                    client_roles=list(client_roles) if client_roles else None,
                )
            except Exception:  # pragma: no cover
                logger.exception("Guardrail hardening failed; using raw instruction.")
                component_hardened = component_instruction
        logger.debug(
            "LLM call. component=%s regulation=%s roles=%s context_chars=%d",
            component_name, regulation, list(client_roles or []) or None,
            len(context or ""),
        )
        # Concurrency gate: acquire before the LLM invoke and release
        # after. Parallel callers block briefly here rather than storming
        # Azure and getting 429'd — a 429 costs 60–120s of SDK backoff,
        # while local queuing on the semaphore costs milliseconds.
        acquired = self._call_semaphore.acquire(timeout=300)
        if not acquired:  # pragma: no cover — extreme starvation
            raise RuntimeError(
                "GenAI concurrency semaphore timed out (300s) — the LLM "
                "backend appears wedged. component=" + str(component_name)
            )
        try:
            # Manual polite retry for HTTP 429 rate-limit responses. The
            # openai SDK does its own retry, but its backoff can stretch
            # to ~60s per attempt. When we're already past the burst we
            # want to retry quickly. We attempt up to 3 times with short
            # jittered sleeps; any other error propagates unchanged.
            import time as _time
            import random as _random
            attempts = 0
            max_attempts = 3
            while True:
                try:
                    return generate_structured_component(
                        self.llm,
                        schema_model,
                        component_name,
                        component_hardened,
                        context[: self.settings.context_chars],
                        system_instruction=system_instruction,
                    )
                except Exception as exc:
                    lower = str(exc).lower()
                    is_429 = (
                        "429" in lower
                        or "rate limit" in lower
                        or "rate_limit_exceeded" in lower
                        or "too_many_requests" in lower
                        or type(exc).__name__ == "RateLimitError"
                    )
                    if is_429 and attempts < max_attempts - 1:
                        attempts += 1
                        wait_s = min(30.0, 2.0 * (2 ** (attempts - 1))) + _random.uniform(0, 1.0)
                        logger.warning(
                            "LLM 429 rate limit on '%s' (attempt %d/%d). "
                            "Sleeping %.1fs before retry.",
                            component_name, attempts, max_attempts, wait_s,
                        )
                        _time.sleep(wait_s)
                        continue
                    logger.exception(
                        "LLM call FAILED. component=%s regulation=%s roles=%s",
                        component_name, regulation, list(client_roles or []) or None,
                    )
                    raise
        finally:
            self._call_semaphore.release()

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
        regulation: Optional[str] = None,
        client_roles: Optional[Any] = None,
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
                regulation=regulation, client_roles=client_roles,
            )
        except Exception as exc:
            if not _is_length_limit_error(exc):
                raise
            logger.warning(
                "LLM length-limit hit for component=%s; retrying with max_tokens=%d",
                component_name, max_retry_tokens,
            )
            if on_retry is not None:
                try:
                    on_retry(
                        f"{component_name}: hit length limit; retrying once "
                        f"with max_tokens={max_retry_tokens}."
                    )
                except Exception:
                    logger.debug("on_retry callback raised; ignoring.", exc_info=True)
            # Thread-safe retry: the shared self.llm.max_tokens mutation is
            # guarded by ``self._retry_lock`` so parallel BRD / questionnaire
            # workers don't race here. Because retries are rare (<5% of
            # calls under reasoning_effort=minimal), serialising them
            # doesn't materially hurt throughput.
            with self._retry_lock:
                original_max = getattr(self.llm, "max_tokens", None)
                try:
                    self.llm.max_tokens = max_retry_tokens
                    return self.generate(
                        schema_model, component_name, component_instruction,
                        context, system_instruction,
                        regulation=regulation, client_roles=client_roles,
                    )
                finally:
                    if original_max is not None:
                        self.llm.max_tokens = original_max

    def close(self) -> None:
        try:
            self.http_client.close()
            logger.debug("GenAI HTTP client closed.")
        except Exception:
            logger.debug("GenAI HTTP client close raised; ignoring.", exc_info=True)


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
