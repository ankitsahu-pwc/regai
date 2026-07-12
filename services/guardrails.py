"""Central AI-guardrails layer for the RegAI pipeline.

The whole platform relies on the PwC GenAI Shared Service (see
``services/genai_service.py``) for structured content generation. LLMs
are prone to *hallucination*: fabricating citations, inventing
applicability, using generic language that does not match the source
regulation, or leaking meta-content ("As an AI language model…"). This
module is the single source of truth for **anti-hallucination
guardrails** applied to every LLM call in the codebase.

Guardrail categories (each implemented below):

1. **Instruction hardening** — a shared anti-hallucination directive is
   prepended to every system prompt so the model is told, before any
   task-specific instructions, to (a) cite only from provided context,
   (b) mark uncertainty explicitly instead of inventing content, (c)
   avoid AI meta-language, and (d) stay inside the scope of the selected
   regulation and client role(s).

2. **Citation validation** — every ``Article N``, ``RTS N``,
   ``Chapter N``, ``Section N`` and ``paragraph N`` reference produced
   by the LLM is cross-checked against the source corpus. Any citation
   that does not appear in the corpus is flagged (and, at the caller's
   discretion, stripped from the output).

3. **Regulation-scope validation** — when the pipeline is analysing
   regulation ``X`` we must not emit content that talks about regulation
   ``Y``. ``RegulationScopeValidator`` detects mismatched regulation
   names and either downgrades confidence or triggers a retry.

4. **Client-role scope validation** — when the user selected e.g.
   ``["Commercial Bank"]`` we must not emit content that claims the
   regulation applies to ``Insurance Company`` — that would silently
   invent scope. ``RoleScopeValidator`` flags any role mentioned in the
   output that is not in the selected list.

5. **Meta-leakage detection** — regex patterns that catch AI meta
   language ("As an AI language model", "I cannot", "I'm sorry, but…",
   "As of my knowledge cutoff", "OpenAI"). These are stripped from the
   output before it is returned to the caller.

6. **Safe generation wrapper** — :func:`safe_generate` combines all of
   the above and provides retry-on-validation logic so callers can drop
   it in as a replacement for ``client.generate(...)`` without changing
   their signatures.

The module is **pure** — no I/O, no session state, no Streamlit imports
— so it is safe to import from any layer (services, agents, UI). Every
validator carries a :class:`GuardrailReport` describing what fired, why,
and what remediation was applied; callers can persist the report so the
UI can render a "Guardrails audit trail" panel.

The design principles:

* **Never crash the caller.** A guardrail is *additive*: if the model
  produced acceptable output the guardrails must be transparent. If the
  model produced unacceptable output the guardrails must degrade
  gracefully to the deterministic fallback, not raise.
* **Never invent content.** Guardrails are *subtractive* — they strip
  suspicious text, mark it uncertain, or force retry. They never patch
  hallucinated content with new invented content.
* **Fully auditable.** Every attenuation must be recorded on the
  :class:`GuardrailReport` so reviewers can trace why a field is empty
  or was rewritten.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anti-hallucination directive (prepended to every LLM system prompt)
# ---------------------------------------------------------------------------
#
# This is the single-source shared prompt fragment used by
# :func:`harden_instruction`. Keep it deliberately short and machine-
# unambiguous. Longer directives tend to be selectively followed.

ANTI_HALLUCINATION_DIRECTIVE = (
    "\n\n--- ANTI-HALLUCINATION GUARDRAILS (MANDATORY) ---\n"
    "You MUST follow these rules for every field you emit:\n"
    "1. Cite only from the regulatory context supplied below. If a fact "
    "is not in the context, either omit the field or mark it "
    "'Not stated in the source' — never invent citations, article "
    "numbers, RTS numbers, chapter numbers, dates or figures.\n"
    "2. If you are uncertain about applicability, scope or a numeric "
    "value, say so explicitly (e.g. 'uncertain — SME review required') "
    "rather than producing a confident-sounding guess.\n"
    "3. Do NOT reference any regulation other than the one named in the "
    "task description. If the context includes text from other "
    "regulations, treat that as background only; your output must be "
    "scoped to the named regulation.\n"
    "4. Do NOT claim applicability to institution types that are not in "
    "the selected client-role list (when a client-role list is "
    "provided). Silently expanding scope is a hallucination.\n"
    "5. Do NOT use AI meta-language ('as an AI language model', "
    "'I cannot', 'as of my knowledge cutoff', 'OpenAI', 'ChatGPT'). "
    "Speak as a regulatory practitioner.\n"
    "6. Do NOT fabricate names of people, organisations, systems, "
    "regulators, dates or URLs. If a name is not in the context, use a "
    "role-based placeholder (e.g. 'the responsible business owner').\n"
    "7. Prefer conservative, defensible wording. Every requirement, "
    "control expectation, obligation, impact, recommendation, question "
    "and citation you emit must be traceable to the context.\n"
    "--- END ANTI-HALLUCINATION GUARDRAILS ---"
)


# Patterns for AI meta-leakage — split by category. "Phrase" patterns
# strip only the exact phrase (and one trailing punctuation) so the rest
# of the sentence — often containing legitimate content like an Article
# citation — is preserved. "Sentence" patterns strip whole apology / refusal
# sentences up to the next sentence terminator, because in practice those
# sentences carry no downstream value ("I cannot answer that.").
_META_LEAKAGE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    # -- Phrase-level scrubs (strip the tag + optional trailing comma).
    re.compile(r"\bas an?\s+ai\s+language\s+model\b\s*[,.:;\-]?\s*", re.I),
    re.compile(r"\bas an?\s+ai\s+(?:assistant|model|chatbot)\b\s*[,.:;\-]?\s*", re.I),
    re.compile(r"\bas\s+of\s+my\s+(?:knowledge\s+cutoff|last\s+update|training\s+data)\b\s*[,.:;\-]?\s*", re.I),
    re.compile(r"\bopenai\b", re.I),
    re.compile(r"\bchatgpt\b", re.I),
    re.compile(r"\banthropic\b", re.I),
    re.compile(r"\bclaude(?:\s+by\s+anthropic)?\b", re.I),
    re.compile(r"\bgpt[- ]?\d(?:\.\d+)?\b", re.I),
    re.compile(r"\blanguage\s+model\b", re.I),
    # Guardrail metaphor from the prompt itself leaking into output:
    re.compile(r"\banti[- ]hallucination\s+guardrails?\b", re.I),
    re.compile(r"\bmandatory\s+rules?\b(?=\s*[:.])", re.I),
    # Placeholder text that some prompts leave behind:
    re.compile(r"\[\s*insert[^\]]{1,40}\]", re.I),
    re.compile(r"\{\{\s*[a-z_ ]{1,30}\s*\}\}", re.I),
    # -- Sentence-level scrubs (whole apology / refusal sentences).
    # These are safe to eat wholesale because they never contain useful
    # regulatory content. Match the phrase + everything up to the next
    # sentence terminator (or end of string).
    re.compile(r"\bi\s+(?:cannot|can[' ]?t)\s+(?:provide|answer|assist|help)\b[^.!?]*[.!?]?", re.I),
    re.compile(r"\bi[' ]?m\s+sorry(?:,|\s+but)?\b[^.!?]*[.!?]?", re.I),
    re.compile(r"\bi\s+don[' ]?t\s+have\s+(?:access|the\s+ability)\b[^.!?]*[.!?]?", re.I),
)


# Speculation / hedging patterns. When an LLM cannot ground a claim in
# the source it tends to fall back on soft language like "generally",
# "typically", "many organisations", "research shows". These phrases are
# highly correlated with hallucinated content in regulatory work — we
# don't strip them (they occasionally have legitimate uses) but we
# *flag* them so the containing sentence is treated as low-confidence
# and, if enough of them stack up in a single field, we force the
# deterministic fallback for that field.
_SPECULATION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:generally|typically|usually|often|in\s+most\s+cases|by\s+and\s+large)\b", re.I),
    re.compile(r"\b(?:many|some|several|numerous|most)\s+(?:organi[sz]ations?|firms?|institutions?|banks?|companies|regulators?|experts?)\b", re.I),
    re.compile(r"\baccording\s+to\s+(?:industry\s+(?:best\s+)?practices?|market\s+practice|leading\s+practice)\b", re.I),
    re.compile(r"\b(?:it\s+is\s+(?:well[- ]known|widely\s+accepted|generally\s+agreed))\b", re.I),
    re.compile(r"\b(?:research|studies|surveys?|analysts?)\s+(?:show|indicate|suggest|find|report)\b", re.I),
    re.compile(r"\b(?:based\s+on\s+my\s+(?:analysis|knowledge|understanding|training))\b", re.I),
    re.compile(r"\b(?:it\s+is\s+(?:likely|probable|possible|reasonable\s+to\s+assume))\b", re.I),
    re.compile(r"\b(?:on\s+average|approximately|roughly|around)\s+\d+\s*(?:%|percent|days?|hours?|months?|years?)\b", re.I),
)


# Fabricated-URL patterns. LLMs love to invent authoritative-looking
# URLs. We flag any URL and let the citation validator check whether the
# domain / path actually appears in the corpus. Placeholder URLs are
# always critical.
_URL_PATTERN = re.compile(
    r"https?://[^\s\)\]<>\"']+", re.I,
)
_PLACEHOLDER_URL_DOMAINS: Tuple[str, ...] = (
    "example.com", "example.org", "example.net",
    "placeholder.com", "yourdomain.com",
    "tbd.com", "todo.com",
    "somewebsite.com", "regulator.com",
)


# Numeric-fabrication patterns. Any specific quantity the model emits
# (dates, day counts, percentages, currency amounts, thresholds) must
# appear in the corpus. We deliberately DON'T match tiny integers like
# "3" (too many false positives on section numbers) — the pattern
# targets domain-shaped numerics.
_NUMERIC_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("percentage", re.compile(r"\b\d{1,3}(?:\.\d+)?\s*(?:%|percent)\b", re.I)),
    ("duration",   re.compile(r"\b\d{1,4}\s*(?:calendar\s+days?|business\s+days?|working\s+days?|days?|hours?|weeks?|months?|years?)\b", re.I)),
    ("currency",   re.compile(r"(?:€|£|\$|USD|EUR|GBP)\s*\d[\d,.]*\s*(?:million|billion|thousand|mn|bn|m|k)?", re.I)),
    ("year",       re.compile(r"\b(?:19|20)\d{2}\b")),
    ("date_iso",   re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("threshold",  re.compile(r"\b(?:at\s+least|no\s+more\s+than|up\s+to|minimum\s+of|maximum\s+of)\s+\d[\d,.]*\b", re.I)),
)


# Regulation-name detector — anything that looks like a well-known FS
# regulation but is not the one we're currently analysing is treated as
# a scope leak. Extend as new regulations come into scope.
_REGULATION_NAME_TOKENS: Tuple[str, ...] = (
    "DORA", "MiFID II", "MiFID", "MiFIR",
    "CRR", "CRD", "CRD IV", "CRD V", "CRD VI",
    "SFDR", "AIFMD", "UCITS", "EMIR", "CSDR",
    "PSD2", "PSD3", "EMD", "EMD2",
    "AML5", "AMLD5", "AMLD6", "MLR",
    "GDPR", "NIS2", "NIS",
    "Solvency II", "IDD", "PRIIPs",
    "MiCA", "MAR",
    "Basel III", "Basel IV",
    "SOX", "Sarbanes-Oxley",
    "FATCA", "CRS",
    "IFR", "IFD",
    "BRRD",
    "SREP",
    "IRRBB",
    "FRTB",
)


# ---------------------------------------------------------------------------
# GuardrailReport — the audit trail
# ---------------------------------------------------------------------------


@dataclass
class GuardrailFinding:
    """Single guardrail firing event.

    Every attenuation / rejection creates one finding. Findings are
    aggregated on the enclosing :class:`GuardrailReport` so the UI can
    render a full audit trail of what the guardrails did.
    """

    category: str
    severity: str  # "info" | "warning" | "critical"
    field_path: str
    message: str
    matched_snippet: str = ""
    remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "field_path": self.field_path,
            "message": self.message,
            "matched_snippet": self.matched_snippet,
            "remediation": self.remediation,
        }


@dataclass
class GuardrailReport:
    """Aggregated audit trail for one LLM call.

    Attach to whatever payload the LLM produces so downstream code (the
    orchestrator, exports, the Streamlit UI) can display the trail. When
    ``ok`` is ``False`` the caller is expected to fall back to the
    deterministic baseline rather than surface the (rejected) LLM output.
    """

    component: str
    ok: bool = True
    used_llm: bool = False
    used_fallback: bool = False
    retry_count: int = 0
    findings: List[GuardrailFinding] = field(default_factory=list)
    citations_verified: int = 0
    citations_flagged: int = 0
    meta_leaks_scrubbed: int = 0
    off_scope_regulations: List[str] = field(default_factory=list)
    off_scope_roles: List[str] = field(default_factory=list)

    def add(self, finding: GuardrailFinding) -> None:
        self.findings.append(finding)
        if finding.severity == "critical":
            self.ok = False

    def summary(self) -> str:
        parts: List[str] = []
        if self.used_llm:
            parts.append("LLM used")
        else:
            parts.append("deterministic")
        if self.retry_count:
            parts.append(f"{self.retry_count} retry")
        if self.citations_verified:
            parts.append(f"{self.citations_verified} citations verified")
        if self.citations_flagged:
            parts.append(f"{self.citations_flagged} citations flagged")
        if self.meta_leaks_scrubbed:
            parts.append(f"{self.meta_leaks_scrubbed} meta leaks scrubbed")
        if self.off_scope_regulations:
            parts.append(f"{len(self.off_scope_regulations)} off-scope regulation(s)")
        if self.off_scope_roles:
            parts.append(f"{len(self.off_scope_roles)} off-scope role(s)")
        if self.used_fallback:
            parts.append("fallback used")
        return ", ".join(parts) or "no findings"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "ok": self.ok,
            "used_llm": self.used_llm,
            "used_fallback": self.used_fallback,
            "retry_count": self.retry_count,
            "findings": [f.to_dict() for f in self.findings],
            "citations_verified": self.citations_verified,
            "citations_flagged": self.citations_flagged,
            "meta_leaks_scrubbed": self.meta_leaks_scrubbed,
            "off_scope_regulations": list(self.off_scope_regulations),
            "off_scope_roles": list(self.off_scope_roles),
            "summary": self.summary(),
        }


# ---------------------------------------------------------------------------
# Instruction hardening
# ---------------------------------------------------------------------------


def harden_instruction(
    instruction: str,
    *,
    regulation: Optional[str] = None,
    client_roles: Optional[Sequence[str]] = None,
    source_present: bool = True,
) -> str:
    """Prepend the mandatory anti-hallucination directive to an LLM prompt.

    Parameters
    ----------
    instruction
        The caller's original system / component instruction.
    regulation
        Optional regulation label (e.g. ``"DORA"``). When provided the
        directive explicitly names the regulation the LLM must stay
        inside.
    client_roles
        Optional list of selected client-role names. When provided the
        directive tells the LLM which institution types the output must
        be scoped to.
    source_present
        Set to ``False`` when the caller has no regulation corpus to
        pass in (e.g. UI-side ad-hoc rewrites). In that case the
        directive is relaxed to "cite only what you can defend" instead
        of the stricter "cite only from the context".
    """
    if not instruction:
        instruction = ""

    # Idempotency: if the caller passed a string that already contains
    # the directive header we do not prepend again. This lets the same
    # helper be called on both the system prompt AND the component
    # instruction without doubling the token budget.
    if "ANTI-HALLUCINATION GUARDRAILS" in instruction:
        return instruction

    header = ANTI_HALLUCINATION_DIRECTIVE
    if not source_present:
        header = header.replace(
            "Cite only from the regulatory context supplied below.",
            "Cite only claims you can defend. No supplied regulatory context "
            "is available for this call.",
        )
    scope_lines: List[str] = []
    if regulation:
        scope_lines.append(
            f"Regulation in scope: {regulation}. Do not answer as if a "
            f"different regulation applies."
        )
    if client_roles:
        role_list = ", ".join(str(r) for r in client_roles if r)
        if role_list:
            scope_lines.append(
                f"Selected institution type(s): {role_list}. Do not "
                f"claim applicability to any other institution type."
            )
    scope_block = ("\n" + "\n".join(scope_lines)) if scope_lines else ""

    return f"{header}{scope_block}\n\n{instruction}"


# ---------------------------------------------------------------------------
# Citation extraction + validation
# ---------------------------------------------------------------------------


# Citation patterns keyed by their kind. Each pattern captures the
# *raw* citation surface form (e.g. ``"Article 5(2)"``) plus a
# named ``value`` group with just the identifying number so we can
# generate canonical variants when validating against the corpus.
_CITATION_PATTERN_SPECS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("Article",  re.compile(r"\bArt(?:icle|\.)\s*(?P<value>\d+[a-z]?(?:\(\d+\))?(?:\(\d+\))?)", re.I)),
    ("RTS",      re.compile(r"\bRTS\s+(?:on\s+)?(?P<value>\d+[/\-]?\d*)", re.I)),
    ("ITS",      re.compile(r"\bITS\s+(?:on\s+)?(?P<value>\d+[/\-]?\d*)", re.I)),
    ("Regulation", re.compile(r"\bRegulation\s+\(EU\)\s+(?P<value>\d{4}/\d+)", re.I)),
    ("Directive", re.compile(r"\bDirective\s+(?P<value>\d{4}/\d+/[A-Z]+)", re.I)),
    ("Chapter",  re.compile(r"\bChapter\s+(?P<value>[IVX]+|\d+)\b", re.I)),
    ("Section",  re.compile(r"\bSection\s+(?P<value>\d+(?:\.\d+)*)", re.I)),
    ("Paragraph", re.compile(r"\bpar(?:agraph|a)\.?\s*(?P<value>\d+)", re.I)),
    ("Recital",  re.compile(r"\bRecital\s+(?P<value>\d+)", re.I)),
    ("Annex",    re.compile(r"\bAnnex\s+(?P<value>[IVX]+|\d+|[A-Z])", re.I)),
)


@dataclass(frozen=True)
class Citation:
    """One citation match — raw surface form plus its normalised parts."""

    kind: str          # "Article", "RTS", "Chapter", …
    value: str         # "5(2)", "27/2020", "III"
    raw: str           # the exact substring that matched in the text
    span: Tuple[int, int]  # (start, end) offset in the source text


def extract_citations_detailed(text: str) -> List[Citation]:
    """Return a list of :class:`Citation` matches from ``text``."""
    if not text:
        return []
    out: List[Citation] = []
    for kind, pattern in _CITATION_PATTERN_SPECS:
        for m in pattern.finditer(text):
            value = m.group("value")
            if not value:
                continue
            out.append(Citation(
                kind=kind,
                value=value.strip(),
                raw=m.group(0),
                span=(m.start(), m.end()),
            ))
    return out


def extract_citations(text: str) -> List[Tuple[str, str]]:
    """Return a list of ``(citation_kind, citation_value)`` from ``text``.

    Kept as a light wrapper around :func:`extract_citations_detailed` so
    existing callers (and unit tests) don't need to work with the
    dataclass. Example::

        >>> extract_citations("See Article 5(2) and RTS 27/2020")
        [('Article', '5(2)'), ('RTS', '27/2020')]
    """
    return [(c.kind, c.value) for c in extract_citations_detailed(text)]


def _corpus_contains_citation(corpus_lower: str, kind: str, value: str) -> bool:
    """Return True when the citation appears somewhere in the corpus.

    Uses a small set of canonical surface variants so a citation typed
    as ``"Art. 5(2)"`` still validates against a corpus that contains
    ``"Article 5(2)"``.
    """
    if not corpus_lower or not value:
        return False
    kind_lower = kind.lower()
    value_lower = value.lower()

    # Canonical variants per citation kind.
    variants: List[str] = []
    if kind_lower == "article":
        variants += [
            f"article {value_lower}",
            f"art. {value_lower}",
            f"art {value_lower}",
        ]
    elif kind_lower in {"rts", "its"}:
        variants += [
            f"{kind_lower} {value_lower}",
            f"{kind_lower.upper()} {value_lower}",
            f"{kind_lower} on {value_lower}",
        ]
    elif kind_lower == "regulation":
        variants += [
            f"regulation (eu) {value_lower}",
        ]
    elif kind_lower == "directive":
        variants += [
            f"directive {value_lower}",
        ]
    else:  # chapter, section, paragraph, recital, annex, …
        variants += [
            f"{kind_lower} {value_lower}",
        ]

    return any(v in corpus_lower for v in variants)


class CitationValidator:
    """Validate citations produced by the LLM against a source corpus.

    Usage::

        validator = CitationValidator(source_corpus, regulation="DORA")
        report = GuardrailReport(component="assess_impact")
        validated_text = validator.attenuate_text(
            text, field_path="rationale", report=report,
        )

    The validator does not raise. When a citation is missing from the
    corpus it is *replaced* with a neutral marker (``"[citation not "
    "verified against source]"``) and a warning finding is added to the
    report. When ``strict=True`` a critical finding is added so
    :meth:`GuardrailReport.ok` flips to ``False``.
    """

    UNVERIFIED_MARKER = "[citation not verified against source]"

    def __init__(
        self,
        source_corpus: str,
        *,
        regulation: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        self.corpus_lower = (source_corpus or "").lower()
        self.regulation = (regulation or "").strip()
        self.strict = strict

    def has_corpus(self) -> bool:
        return len(self.corpus_lower) > 0

    def attenuate_text(
        self,
        text: str,
        *,
        field_path: str,
        report: GuardrailReport,
    ) -> str:
        """Return ``text`` with unverifiable citations replaced.

        When the corpus is empty this is a no-op — we cannot claim a
        citation is invented if we have nothing to check against. The
        replacement uses the exact raw surface form from the source
        text so we don't accidentally leave a partial match behind.
        """
        if not text or not self.has_corpus():
            return text

        # Sort matches right-to-left so each replacement's offsets stay
        # valid even after we mutate the string. Track which raw forms
        # were already replaced so we don't double-flag identical repeats
        # (still count them, though — the LLM emitted them twice).
        citations = extract_citations_detailed(text)
        citations.sort(key=lambda c: c.span[0], reverse=True)
        # Track (kind, value) so we don't file two findings for the same
        # citation appearing twice; the replacement itself must still be
        # performed on every occurrence.
        finding_key_seen: set = set()

        out = text
        for cit in citations:
            if _corpus_contains_citation(self.corpus_lower, cit.kind, cit.value):
                report.citations_verified += 1
                continue
            report.citations_flagged += 1
            key = (cit.kind.lower(), cit.value.lower())
            if key not in finding_key_seen:
                finding_key_seen.add(key)
                report.add(GuardrailFinding(
                    category="citation",
                    severity="critical" if self.strict else "warning",
                    field_path=field_path,
                    message=(
                        f"'{cit.raw}' was cited by the LLM but does not "
                        f"appear in the source corpus for regulation "
                        f"{self.regulation or 'in scope'}."
                    ),
                    matched_snippet=cit.raw,
                    remediation=(
                        "Replaced with an 'unverified citation' marker so "
                        "the downstream artefact does not silently "
                        "propagate a fabricated reference."
                    ),
                ))
            start, end = cit.span
            out = out[:start] + self.UNVERIFIED_MARKER + out[end:]
        return out


# ---------------------------------------------------------------------------
# Regulation-scope validation
# ---------------------------------------------------------------------------


class RegulationScopeValidator:
    """Flag outputs that reference off-scope regulations.

    The pipeline runs one regulation at a time. If the LLM's output
    mentions another well-known FS regulation as *the applicable
    regulation* (rather than as background), that is a scope leak.
    We take a conservative approach: any distinct regulation name that
    appears in the output but is not the one currently being analysed
    is flagged. Callers can still keep the text — the finding surfaces
    on the audit trail so reviewers know to double-check.
    """

    def __init__(self, regulation: str) -> None:
        self.regulation = (regulation or "").strip()
        self.regulation_lower = self.regulation.lower()

    def validate(
        self,
        text: str,
        *,
        field_path: str,
        report: GuardrailReport,
    ) -> None:
        if not text or not self.regulation:
            return
        text_lower = text.lower()
        seen: set = set()
        for token in _REGULATION_NAME_TOKENS:
            token_lower = token.lower()
            if token_lower == self.regulation_lower:
                continue
            # Skip token if the current regulation contains it (e.g.
            # analysing "MiFID II" should not flag mentions of "MiFID").
            if token_lower in self.regulation_lower:
                continue
            # Case-sensitive word-boundary match; the well-known tokens
            # are capitalised in canonical form so we require the same.
            if re.search(r"\b" + re.escape(token) + r"\b", text):
                if token in seen:
                    continue
                seen.add(token)
                report.off_scope_regulations.append(token)
                report.add(GuardrailFinding(
                    category="regulation_scope",
                    severity="warning",
                    field_path=field_path,
                    message=(
                        f"Output mentions '{token}' but the analysis is "
                        f"scoped to '{self.regulation}'. Reviewer should "
                        f"confirm the reference is background context "
                        f"(not a scope leak)."
                    ),
                    matched_snippet=token,
                    remediation=(
                        "Left in place — reviewer to adjudicate. Consider "
                        "re-running with a tighter prompt if reviewer "
                        "flags this as an actual scope leak."
                    ),
                ))


# ---------------------------------------------------------------------------
# Client-role scope validation
# ---------------------------------------------------------------------------


class RoleScopeValidator:
    """Flag outputs that claim applicability to unselected institution types.

    Uses the canonical catalog from :mod:`services.client_roles` at
    lookup time. Because the catalog has ~35 entries we look for
    surface-form mentions and only escalate a hit to a finding when the
    mention is *not* in the selected list.
    """

    def __init__(self, selected_roles: Optional[Sequence[str]]) -> None:
        self.selected = [str(r) for r in (selected_roles or []) if r]
        self.selected_lower = {r.lower() for r in self.selected}

    def validate(
        self,
        text: str,
        *,
        field_path: str,
        report: GuardrailReport,
    ) -> None:
        if not text or not self.selected:
            return
        # Late import to avoid a circular dep at module load time.
        try:
            from .client_roles import INSTITUTION_TYPE_NAMES
        except Exception:  # pragma: no cover - defensive
            return
        text_lower = text.lower()
        seen: set = set()
        for role_name in INSTITUTION_TYPE_NAMES:
            role_lower = role_name.lower()
            if role_lower in self.selected_lower:
                continue
            if role_lower in seen:
                continue
            # Require the full canonical name (case-insensitive, with
            # word boundaries) — avoids false-positives on generic
            # nouns like "bank" that appear inside longer names.
            if re.search(
                r"\b" + re.escape(role_name) + r"\b", text, flags=re.I,
            ):
                seen.add(role_lower)
                report.off_scope_roles.append(role_name)
                report.add(GuardrailFinding(
                    category="role_scope",
                    severity="warning",
                    field_path=field_path,
                    message=(
                        f"Output mentions institution type '{role_name}' "
                        f"but that role is not in the selected list "
                        f"({', '.join(self.selected)}). Silently expanding "
                        f"scope is a hallucination — reviewer should "
                        f"confirm."
                    ),
                    matched_snippet=role_name,
                    remediation=(
                        "Left in place — reviewer to adjudicate. Consider "
                        "adding this role on Page 1 if the reference is "
                        "intended."
                    ),
                ))


# ---------------------------------------------------------------------------
# Speculation / hedging validator
# ---------------------------------------------------------------------------


class SpeculationValidator:
    """Detect hedging / speculation phrases the LLM emits when un-grounded.

    Speculation words are common in general-purpose LLM output but rare
    in high-quality regulatory writing. We *do not strip* them — the
    signal is aggregated on the report and consumed by
    :func:`safe_generate` to decide whether to force the deterministic
    fallback (too many hedges = probably ungrounded).
    """

    def __init__(self, *, threshold: int = 4) -> None:
        # ``threshold``: allowed hedges in a single string before a
        # critical finding is raised for that string.
        self.threshold = threshold

    def validate(
        self,
        text: str,
        *,
        field_path: str,
        report: GuardrailReport,
    ) -> int:
        if not text:
            return 0
        hits: List[str] = []
        for pattern in _SPECULATION_PATTERNS:
            for m in pattern.finditer(text):
                hits.append(m.group(0))
        if not hits:
            return 0
        severity = "critical" if len(hits) >= self.threshold else "warning"
        report.add(GuardrailFinding(
            category="speculation",
            severity=severity,
            field_path=field_path,
            message=(
                f"Detected {len(hits)} hedging / speculation phrase(s) "
                f"— unsupported soft claims are a strong hallucination "
                f"signal. Phrases: {', '.join(sorted(set(hits))[:6])}."
            ),
            matched_snippet=", ".join(sorted(set(hits))[:6]),
            remediation=(
                "Speculation phrases are counted, not stripped. The "
                "aggregate count is used to decide whether to fall back "
                "to the deterministic baseline for this field."
                if severity == "warning"
                else "Threshold exceeded — deterministic fallback will "
                "be preferred for this field."
            ),
        ))
        return len(hits)


# ---------------------------------------------------------------------------
# Fabricated-URL validator
# ---------------------------------------------------------------------------


class UrlValidator:
    """Flag URLs that don't appear in the source corpus.

    We rely on the corpus containing the URL exactly (or the URL's
    domain). Placeholder domains (``example.com``, ``placeholder.com``)
    are always critical because they are pure hallucination markers.
    """

    def __init__(self, source_corpus: str) -> None:
        self.corpus_lower = (source_corpus or "").lower()

    def validate(
        self,
        text: str,
        *,
        field_path: str,
        report: GuardrailReport,
    ) -> None:
        if not text:
            return
        for m in _URL_PATTERN.finditer(text):
            url = m.group(0).rstrip(".,;:")
            url_lower = url.lower()
            # Placeholder domain — always critical, no exceptions.
            if any(dom in url_lower for dom in _PLACEHOLDER_URL_DOMAINS):
                report.add(GuardrailFinding(
                    category="fabricated_url",
                    severity="critical",
                    field_path=field_path,
                    message=(
                        f"URL '{url}' uses a placeholder domain — this "
                        f"is a hallucinated reference."
                    ),
                    matched_snippet=url,
                    remediation="Force deterministic fallback for this field.",
                ))
                continue
            if not self.corpus_lower:
                # No corpus to validate against — flag as unverifiable
                # (warning only, so single URLs don't nuke the payload).
                report.add(GuardrailFinding(
                    category="unverifiable_url",
                    severity="warning",
                    field_path=field_path,
                    message=(
                        f"URL '{url}' could not be validated (no source "
                        f"corpus available)."
                    ),
                    matched_snippet=url,
                ))
                continue
            # Verify by looking for the URL OR its host in the corpus.
            host = re.sub(r"^https?://", "", url_lower).split("/", 1)[0]
            if url_lower in self.corpus_lower or (host and host in self.corpus_lower):
                continue
            report.add(GuardrailFinding(
                category="fabricated_url",
                severity="warning",
                field_path=field_path,
                message=(
                    f"URL '{url}' does not appear in the source corpus."
                ),
                matched_snippet=url,
                remediation=(
                    "Warning only — reviewer to verify. Aggregate URL "
                    "fabrications may trigger fallback."
                ),
            ))


# ---------------------------------------------------------------------------
# Numeric fabrication validator
# ---------------------------------------------------------------------------


class NumericValidator:
    """Flag domain-shaped numerics that don't appear in the source corpus.

    Regulations are precise about numbers (72 hours, 15 business days,
    €5 million). When the LLM emits a specific quantity that isn't in
    the corpus we treat it as fabricated. To avoid drowning the report
    in false positives we normalise both sides (strip commas, lower-case,
    collapse whitespace).
    """

    def __init__(self, source_corpus: str) -> None:
        self.corpus_norm = self._normalise(source_corpus or "")

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower().replace(",", ""))

    def validate(
        self,
        text: str,
        *,
        field_path: str,
        report: GuardrailReport,
    ) -> None:
        if not text or not self.corpus_norm:
            return
        text_norm = self._normalise(text)
        flagged: Dict[str, List[str]] = {}
        for kind, pattern in _NUMERIC_PATTERNS:
            for m in pattern.finditer(text):
                raw = m.group(0)
                norm = self._normalise(raw)
                if norm in self.corpus_norm:
                    continue
                # For years, be lenient — many years appear in both
                # sides without semantic meaning. Only flag as warning.
                # For percentages / durations / currency / thresholds:
                # treat as critical because they encode obligations.
                flagged.setdefault(kind, []).append(raw)
        for kind, values in flagged.items():
            severity = "warning" if kind == "year" else "critical"
            unique = sorted({v.strip() for v in values})
            report.add(GuardrailFinding(
                category=f"fabricated_{kind}",
                severity=severity,
                field_path=field_path,
                message=(
                    f"{kind.capitalize()} value(s) not found in source: "
                    f"{', '.join(unique[:6])}. Numeric obligations MUST "
                    f"be grounded — this is a strong hallucination signal."
                ),
                matched_snippet=", ".join(unique[:6]),
                remediation=(
                    "Critical — deterministic fallback will be preferred."
                    if severity == "critical"
                    else "Warning only — reviewer to verify."
                ),
            ))


# ---------------------------------------------------------------------------
# Meta-leakage scrubbing
# ---------------------------------------------------------------------------


def scrub_meta_leakage(
    text: str,
    *,
    field_path: str,
    report: GuardrailReport,
) -> str:
    """Return ``text`` with any AI meta-language removed.

    Any match against :data:`_META_LEAKAGE_PATTERNS` is stripped and a
    warning finding is added to ``report``. When the entire text was
    meta-language the function returns an empty string — the caller is
    expected to fall back to the deterministic baseline for that field.
    """
    if not text:
        return text
    out = text
    hits = 0
    for pattern in _META_LEAKAGE_PATTERNS:
        new, count = pattern.subn(" ", out)
        if count:
            hits += count
            out = new
    if hits:
        report.meta_leaks_scrubbed += hits
        report.add(GuardrailFinding(
            category="meta_leakage",
            severity="warning",
            field_path=field_path,
            message=(
                f"Scrubbed {hits} AI meta-language pattern(s) from the "
                f"LLM output."
            ),
            remediation="Stripped in place — safe to keep the remaining text.",
        ))
        # Collapse whitespace produced by the substitutions.
        out = re.sub(r"\s{2,}", " ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Text-attenuation convenience
# ---------------------------------------------------------------------------


def apply_text_guardrails(
    text: str,
    *,
    field_path: str,
    report: GuardrailReport,
    citation_validator: Optional[CitationValidator] = None,
    regulation_validator: Optional[RegulationScopeValidator] = None,
    role_validator: Optional[RoleScopeValidator] = None,
    speculation_validator: Optional[SpeculationValidator] = None,
    url_validator: Optional[UrlValidator] = None,
    numeric_validator: Optional[NumericValidator] = None,
) -> str:
    """Run every text-level guardrail in the correct order.

    Order matters:

    1. Scrub meta-leakage first (removes phrases we would otherwise
       cross-validate).
    2. Validate citations (attenuates in place).
    3. Validate regulation scope (annotation-only).
    4. Validate role scope (annotation-only).
    5. Validate URLs (annotation + critical for placeholders).
    6. Validate numeric grounding (annotation-only).
    7. Detect speculation / hedging density (annotation-only).
    """
    if not text:
        return text
    out = scrub_meta_leakage(text, field_path=field_path, report=report)
    if citation_validator is not None:
        out = citation_validator.attenuate_text(
            out, field_path=field_path, report=report,
        )
    if regulation_validator is not None:
        regulation_validator.validate(out, field_path=field_path, report=report)
    if role_validator is not None:
        role_validator.validate(out, field_path=field_path, report=report)
    if url_validator is not None:
        url_validator.validate(out, field_path=field_path, report=report)
    if numeric_validator is not None:
        numeric_validator.validate(out, field_path=field_path, report=report)
    if speculation_validator is not None:
        speculation_validator.validate(out, field_path=field_path, report=report)
    return out


def guard_string_fields(
    payload: Any,
    *,
    field_names: Sequence[str],
    report: GuardrailReport,
    citation_validator: Optional[CitationValidator] = None,
    regulation_validator: Optional[RegulationScopeValidator] = None,
    role_validator: Optional[RoleScopeValidator] = None,
    speculation_validator: Optional[SpeculationValidator] = None,
    url_validator: Optional[UrlValidator] = None,
    numeric_validator: Optional[NumericValidator] = None,
    field_prefix: str = "",
) -> None:
    """Guard a set of string attributes on a Pydantic/dataclass payload.

    Mutates ``payload`` in place — every field named in ``field_names``
    that carries a string is passed through :func:`apply_text_guardrails`.
    Non-string fields (lists, ints, dicts) are ignored. Lists of strings
    are traversed and each element is guarded individually.
    """
    if payload is None:
        return

    def _run(value: str, path: str) -> str:
        return apply_text_guardrails(
            value,
            field_path=path,
            report=report,
            citation_validator=citation_validator,
            regulation_validator=regulation_validator,
            role_validator=role_validator,
            speculation_validator=speculation_validator,
            url_validator=url_validator,
            numeric_validator=numeric_validator,
        )

    for name in field_names:
        try:
            value = getattr(payload, name)
        except AttributeError:
            continue
        if value is None:
            continue
        if isinstance(value, str):
            new_value = _run(value, field_prefix + name)
            if new_value != value:
                try:
                    setattr(payload, name, new_value)
                except Exception:  # pragma: no cover - frozen models
                    pass
        elif isinstance(value, list):
            new_list: List[Any] = []
            for idx, entry in enumerate(value):
                if isinstance(entry, str):
                    new_list.append(_run(entry, f"{field_prefix}{name}[{idx}]"))
                else:
                    new_list.append(entry)
            try:
                setattr(payload, name, new_list)
            except Exception:  # pragma: no cover
                pass


# ---------------------------------------------------------------------------
# Persistence guardrail — run BEFORE writing to SQLite
# ---------------------------------------------------------------------------


class PersistenceGuardrailError(RuntimeError):
    """Raised when a payload fails the pre-persistence guardrail sweep.

    Semantic contract:

    * ``report`` is a fully populated :class:`GuardrailReport` describing
      every finding, including at least one critical finding that could
      not be auto-remediated.
    * Callers should log the report, refuse the write, and surface the
      problem to the user via the audit / review queue.
    """

    def __init__(self, report: "GuardrailReport", scrubbed_paths: List[str]):
        self.report = report
        self.scrubbed_paths = scrubbed_paths
        critical = [f for f in report.findings if f.severity == "critical"]
        message = (
            f"Persistence guardrail rejected write for '{report.component}': "
            f"{len(critical)} critical finding(s) after scrubbing {len(scrubbed_paths)} field(s)."
        )
        super().__init__(message)


def _walk_string_leaves(
    payload: Any,
    *,
    prefix: str = "",
    _visited: Optional[set] = None,
) -> List[Tuple[str, str]]:
    """Return ``[(field_path, string_value), ...]`` for every string leaf.

    Recurses into dicts, lists, tuples, dataclass instances, and Pydantic
    models. Cyclic references are broken by tracking ``id(obj)``.
    """
    _visited = _visited if _visited is not None else set()
    if payload is None:
        return []
    marker = id(payload)
    if marker in _visited:
        return []
    _visited.add(marker)

    out: List[Tuple[str, str]] = []
    if isinstance(payload, str):
        out.append((prefix or "value", payload))
    elif isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_walk_string_leaves(value, prefix=path, _visited=_visited))
    elif isinstance(payload, (list, tuple)):
        for idx, value in enumerate(payload):
            path = f"{prefix}[{idx}]"
            out.extend(_walk_string_leaves(value, prefix=path, _visited=_visited))
    elif hasattr(payload, "model_dump"):
        # Pydantic v2 model — recurse into its dict.
        try:
            out.extend(_walk_string_leaves(payload.model_dump(), prefix=prefix, _visited=_visited))
        except Exception:  # pragma: no cover - defensive
            pass
    elif hasattr(payload, "__dataclass_fields__"):
        for name in payload.__dataclass_fields__:
            try:
                value = getattr(payload, name)
            except AttributeError:
                continue
            path = f"{prefix}.{name}" if prefix else name
            out.extend(_walk_string_leaves(value, prefix=path, _visited=_visited))
    return out


def run_persistence_guardrail(
    payload: Any,
    *,
    component: str,
    source_corpus: Optional[str] = None,
    regulation: Optional[str] = None,
    client_roles: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> Tuple[GuardrailReport, Dict[str, str]]:
    """Scan a payload for meta-leakage / bad citations / placeholder URLs.

    Runs the same text-guardrail stack that ``safe_generate`` runs, but
    against an *already-generated* payload — the last line of defence
    before we commit to SQLite.

    The function is *non-mutating* on the input payload (call sites are
    diverse — dicts, dataclasses, Pydantic models — and mutating them
    all safely is error-prone). Instead it returns:

    * a :class:`GuardrailReport` describing every finding, and
    * a ``{field_path -> scrubbed_value}`` map that the caller can splice
      back into the payload before writing.

    When ``strict`` is ``True`` and the report contains any critical
    finding, callers are expected to :class:`raise PersistenceGuardrailError`
    (see :func:`check_before_persist` for the standard wrapper).
    """
    report = GuardrailReport(component=component)

    # Build only the validators we can — the citation/regulation/role/URL/
    # numeric validators need a corpus to compare against. When the caller
    # cannot provide a corpus, we still run meta-leakage scrubbing and
    # speculation detection.
    citation_validator = None
    regulation_validator = None
    role_validator = None
    url_validator = None
    numeric_validator = None
    if source_corpus:
        citation_validator = CitationValidator(
            source_corpus, regulation=regulation or "", strict=strict,
        )
        regulation_validator = RegulationScopeValidator(regulation or "")
        url_validator = UrlValidator(source_corpus)
        numeric_validator = NumericValidator(source_corpus)
    if client_roles:
        role_validator = RoleScopeValidator(client_roles)
    speculation_validator = SpeculationValidator(threshold=6)

    scrubbed_by_path: Dict[str, str] = {}
    for field_path, text in _walk_string_leaves(payload):
        # Skip empty strings and short tokens (IDs, dates, enums).
        if not text or len(text) < 3:
            continue
        cleaned = apply_text_guardrails(
            text,
            field_path=field_path,
            report=report,
            citation_validator=citation_validator,
            regulation_validator=regulation_validator,
            role_validator=role_validator,
            speculation_validator=speculation_validator,
            url_validator=url_validator,
            numeric_validator=numeric_validator,
        )
        if cleaned != text:
            scrubbed_by_path[field_path] = cleaned

    return report, scrubbed_by_path


def check_before_persist(
    payload: Any,
    *,
    component: str,
    source_corpus: Optional[str] = None,
    regulation: Optional[str] = None,
    client_roles: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> GuardrailReport:
    """Run :func:`run_persistence_guardrail` and raise on critical findings.

    Standard call site::

        report = check_before_persist(
            package, component="questionnaire",
            source_corpus=corpus, regulation="DORA", client_roles=roles,
        )
        # ... proceed with the DB write; embed report.to_dict() alongside.

    In ``strict=False`` mode the function never raises and simply returns
    the report — useful for warn-only mode during rollout.
    """
    report, scrubbed = run_persistence_guardrail(
        payload,
        component=component,
        source_corpus=source_corpus,
        regulation=regulation,
        client_roles=client_roles,
        strict=strict,
    )
    critical = [f for f in report.findings if f.severity == "critical"]
    if strict and critical:
        raise PersistenceGuardrailError(report, list(scrubbed.keys()))
    return report


# ---------------------------------------------------------------------------
# safe_generate — the main wrapper
# ---------------------------------------------------------------------------


@dataclass
class ValidatorBundle:
    """The full set of text-level validators for one LLM call."""
    citation: CitationValidator
    regulation: RegulationScopeValidator
    role: RoleScopeValidator
    speculation: SpeculationValidator
    url: UrlValidator
    numeric: NumericValidator


def build_validators(
    *,
    source_corpus: str,
    regulation: Optional[str],
    client_roles: Optional[Sequence[str]],
    strict_citations: bool = True,
    speculation_threshold: int = 4,
) -> ValidatorBundle:
    """Convenience: build every text validator from one config.

    ``strict_citations`` defaults to ``True`` — unverifiable citations are
    treated as critical findings so :func:`safe_generate` forces the
    deterministic fallback instead of surfacing attenuated content.
    """
    return ValidatorBundle(
        citation=CitationValidator(
            source_corpus,
            regulation=regulation or "",
            strict=strict_citations,
        ),
        regulation=RegulationScopeValidator(regulation or ""),
        role=RoleScopeValidator(client_roles),
        speculation=SpeculationValidator(threshold=speculation_threshold),
        url=UrlValidator(source_corpus),
        numeric=NumericValidator(source_corpus),
    )


def safe_generate(
    client: Any,
    schema_model: Any,
    component_name: str,
    instruction: str,
    context: str,
    *,
    regulation: Optional[str] = None,
    client_roles: Optional[Sequence[str]] = None,
    source_corpus: Optional[str] = None,
    system_instruction: Optional[str] = None,
    text_fields: Sequence[str] = (),
    on_retry: Optional[Callable[[str], None]] = None,
    strict_citations: bool = True,
    max_retries: int = 1,
    prefer_generate_with_length_retry: bool = False,
    min_citation_ratio: float = 0.5,
) -> Tuple[Optional[Any], GuardrailReport]:
    """Run ``client.generate`` behind the anti-hallucination guardrails.

    This is the **strict** entry point. Behaviour:

    * Every prompt is hardened with the anti-hallucination directive plus
      the caller's regulation + client-role scope block.
    * Every text field named in ``text_fields`` is passed through the
      full text-guardrail stack (meta-leakage scrub, citation validator,
      regulation-scope validator, role-scope validator, URL validator,
      numeric validator, speculation detector).
    * When any *critical* finding is raised the payload is REJECTED —
      ``(None, report)`` is returned so the caller falls back to its
      deterministic baseline. This is the "no hallucination reaches the
      user" contract.
    * When ``min_citation_ratio`` is > 0 and the LLM emitted citations
      that could not be verified against the corpus at a higher rate
      than ``1 - min_citation_ratio``, the payload is also rejected.

    Parameters
    ----------
    strict_citations
        Defaults to ``True`` — every unverifiable citation is a critical
        finding.
    min_citation_ratio
        Defaults to ``0.5`` — at least 50% of the citations the LLM
        emits must appear in the source corpus. Set to ``0`` to disable
        the ratio check.

    Returns
    -------
    payload, report
        ``payload`` is the Pydantic model (or ``None`` if generation
        failed / was rejected by the guardrails). ``report`` is always
        returned so callers can persist it on the audit trail.
    """
    report = GuardrailReport(component=component_name)

    if client is None:
        report.used_fallback = True
        report.add(GuardrailFinding(
            category="fallback",
            severity="info",
            field_path=component_name,
            message="GenAI client is None — deterministic fallback used.",
        ))
        return None, report

    hardened_instruction = harden_instruction(
        instruction or "",
        regulation=regulation,
        client_roles=client_roles,
        source_present=bool(source_corpus or context),
    )
    system = system_instruction
    if system is None:
        try:
            from .genai_service import _DEFAULT_SYSTEM_INSTRUCTION
            system = _DEFAULT_SYSTEM_INSTRUCTION
        except Exception:  # pragma: no cover
            system = ""
    hardened_system = harden_instruction(
        system or "",
        regulation=regulation,
        client_roles=client_roles,
        source_present=bool(source_corpus or context),
    )

    validators = build_validators(
        source_corpus=source_corpus or context or "",
        regulation=regulation,
        client_roles=client_roles,
        strict_citations=strict_citations,
    )

    last_error: Optional[Exception] = None
    payload: Optional[Any] = None
    for attempt in range(max_retries + 1):
        try:
            if prefer_generate_with_length_retry and hasattr(
                client, "generate_with_length_retry",
            ):
                payload = client.generate_with_length_retry(
                    schema_model,
                    component_name,
                    hardened_instruction,
                    context,
                    system_instruction=hardened_system,
                    on_retry=on_retry,
                )
            else:
                payload = client.generate(
                    schema_model,
                    component_name,
                    hardened_instruction,
                    context,
                    system_instruction=hardened_system,
                )
            report.used_llm = True
            break
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            report.retry_count = attempt + 1
            report.add(GuardrailFinding(
                category="generation_error",
                severity="warning" if attempt < max_retries else "critical",
                field_path=component_name,
                message=f"LLM invocation failed on attempt {attempt + 1}: {exc}",
                remediation=(
                    "Retrying with the same hardened prompt."
                    if attempt < max_retries
                    else "Falling back to deterministic content."
                ),
            ))
            payload = None

    if payload is None:
        report.used_fallback = True
        if last_error is not None:
            logger.warning(
                "guardrails.safe_generate(%s) fell back after error: %s",
                component_name, last_error,
            )
        return None, report

    # Post-hoc validation of the specified text fields.
    if text_fields:
        guard_string_fields(
            payload,
            field_names=list(text_fields),
            report=report,
            citation_validator=validators.citation,
            regulation_validator=validators.regulation,
            role_validator=validators.role,
            speculation_validator=validators.speculation,
            url_validator=validators.url,
            numeric_validator=validators.numeric,
            field_prefix=f"{component_name}.",
        )

    # Citation-ratio enforcement: if the LLM cited N things and fewer
    # than min_citation_ratio * N were verifiable, the output is
    # ungrounded — reject it.
    if min_citation_ratio > 0.0 and validators.citation.has_corpus():
        total_cites = report.citations_verified + report.citations_flagged
        if total_cites > 0:
            ratio = report.citations_verified / total_cites
            if ratio < min_citation_ratio:
                report.add(GuardrailFinding(
                    category="citation_ratio",
                    severity="critical",
                    field_path=component_name,
                    message=(
                        f"Only {report.citations_verified}/{total_cites} "
                        f"citations ({ratio:.0%}) were verifiable against "
                        f"the source corpus (threshold: "
                        f"{min_citation_ratio:.0%}). The output is likely "
                        f"ungrounded — deterministic fallback preferred."
                    ),
                    remediation="Payload rejected; caller falls back.",
                ))

    # STRICT ENFORCEMENT: if any critical finding was raised, reject the
    # payload so the caller uses the deterministic fallback. This is
    # what makes hallucinated content unable to reach the user.
    if not report.ok:
        report.used_fallback = True
        logger.warning(
            "guardrails.safe_generate(%s) rejected payload due to critical "
            "findings: %s",
            component_name,
            [f.category for f in report.findings if f.severity == "critical"],
        )
        return None, report

    return payload, report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


__all__ = [
    "ANTI_HALLUCINATION_DIRECTIVE",
    "CitationValidator",
    "GuardrailFinding",
    "GuardrailReport",
    "NumericValidator",
    "RegulationScopeValidator",
    "RoleScopeValidator",
    "SpeculationValidator",
    "UrlValidator",
    "ValidatorBundle",
    "apply_text_guardrails",
    "build_validators",
    "extract_citations",
    "guard_string_fields",
    "harden_instruction",
    "safe_generate",
    "scrub_meta_leakage",
]
