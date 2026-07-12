"""2-of-3 voting + LLM-as-judge for critical AI outputs.

Mentor guidance #5 asked for a **voting mechanism** on critical outputs:
combine a deterministic baseline (vote 1), the primary LLM call (vote 2),
and — when the first two disagree — an LLM-as-judge adjudicator (vote 3).
The winner is whichever candidate carries a 2-of-3 majority; ties fall
back to the deterministic baseline (safest default).

The module is intentionally light. It does not know anything about the
concrete payload shape — callers pass Pydantic models, dataclasses,
plain dicts, or strings, and the shared helpers below normalise to a
comparable form.

Usage
-----

```
from services.llm_judge import voted_safe_generate

winner, vote_report = voted_safe_generate(
    client=genai,
    schema_model=_ConfidencePayload,
    component_name="Regulatory Confidence Assessment",
    instruction=instruction,
    context=context,
    deterministic_fallback_fn=lambda: baseline,
    regulation=regulation,
    client_roles=client_roles,
    source_corpus=source_corpus,
    text_fields=("reasoning",),
)
```

``vote_report.winner`` is one of ``"llm" / "deterministic" / "tie"``.
``vote_report.agreement_score`` is the structural-similarity metric
between the two LLM candidates. ``vote_report.judge_verdict`` (if run)
carries the adjudicator's raw output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class JudgeVerdict:
    """Adjudicator verdict from an LLM-as-judge call.

    Populated by :func:`llm_judge_vote`. The winner is one of
    ``"A" | "B" | "TIE" | "REVIEW"``. ``REVIEW`` means the judge could
    not decide — the caller should surface both candidates to a human.
    """

    winner: str
    confidence: float = 0.0
    rationale: str = ""
    disagreement_notes: List[str] = field(default_factory=list)


@dataclass
class VoteReport:
    """Audit trail for one voted generation call.

    Attach to whatever payload the caller returns so the UI / review
    queue can trace *how* the winning value was selected.
    """

    component: str
    winner: str = "deterministic"  # "llm" | "deterministic" | "tie"
    agreement_score: float = 0.0
    n_llm_candidates: int = 0
    llm_ok: bool = True
    judge_verdict: Optional[JudgeVerdict] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "winner": self.winner,
            "agreement_score": round(self.agreement_score, 3),
            "n_llm_candidates": self.n_llm_candidates,
            "llm_ok": self.llm_ok,
            "judge_verdict": {
                "winner": self.judge_verdict.winner,
                "confidence": round(self.judge_verdict.confidence, 3),
                "rationale": self.judge_verdict.rationale,
                "disagreement_notes": list(self.judge_verdict.disagreement_notes),
            } if self.judge_verdict is not None else None,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Structural agreement — cheap, no LLM
# ---------------------------------------------------------------------------


def _as_comparable(payload: Any) -> Any:
    """Coerce a payload to a comparable primitive tree (dict / list / str)."""
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload.strip().lower()
    if isinstance(payload, (int, float, bool)):
        return payload
    if isinstance(payload, (list, tuple)):
        return [_as_comparable(x) for x in payload]
    if isinstance(payload, dict):
        return {k: _as_comparable(v) for k, v in sorted(payload.items())}
    if hasattr(payload, "model_dump"):
        try:
            return _as_comparable(payload.model_dump())
        except Exception:  # pragma: no cover - defensive
            return None
    if hasattr(payload, "__dataclass_fields__"):
        return _as_comparable({
            name: getattr(payload, name, None)
            for name in payload.__dataclass_fields__
        })
    # Fallback: string coercion for anything else (rare — enums etc.).
    return str(payload)


def structural_agreement(a: Any, b: Any) -> float:
    """Return a 0..1 score describing how similar two payloads are.

    Uses a recursive equality walk on the normalised structure:
    * primitives contribute 1 (equal) or 0 (unequal);
    * containers contribute the ratio of matching children (Jaccard-style
      for dicts, positional for lists);
    * missing / mismatched shapes short-circuit to 0.

    The score is intentionally coarse — it exists to answer "is the LLM
    self-consistent across two temperature-perturbed samples?" rather
    than "are these two payloads semantically identical". A high score
    (>= 0.85) means the LLM output is deterministic enough that we don't
    need to run the judge.
    """
    return _score(_as_comparable(a), _as_comparable(b))


def _score(a: Any, b: Any) -> float:
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    if type(a) != type(b):
        return 0.0
    if isinstance(a, (str, int, float, bool)):
        return 1.0 if a == b else 0.0
    if isinstance(a, list):
        if not a and not b:
            return 1.0
        if len(a) != len(b):
            # Partial credit for overlap.
            pairs = min(len(a), len(b))
            scores = [_score(a[i], b[i]) for i in range(pairs)]
            return sum(scores) / max(len(a), len(b))
        scores = [_score(x, y) for x, y in zip(a, b)]
        return sum(scores) / len(scores) if scores else 1.0
    if isinstance(a, dict):
        keys = set(a) | set(b)
        if not keys:
            return 1.0
        scores = [_score(a.get(k), b.get(k)) for k in keys]
        return sum(scores) / len(scores)
    return 0.0


# ---------------------------------------------------------------------------
# LLM-as-judge — only called when the deterministic vote disagrees with the LLM
# ---------------------------------------------------------------------------


_JUDGE_INSTRUCTION = (
    "You are an independent regulatory-compliance quality reviewer.\n"
    "You will see:\n"
    "  * ``candidate_a``: the deterministic baseline.\n"
    "  * ``candidate_b``: the LLM-generated candidate.\n"
    "  * ``source_context``: the regulation text and evidence available.\n\n"
    "Decide which candidate is more faithful to the source and free of "
    "hallucination.\n\n"
    "Respond with a single JSON object of the shape:\n"
    "  {\n"
    "    \"winner\": \"A\" | \"B\" | \"TIE\" | \"REVIEW\",\n"
    "    \"confidence\": 0.0..1.0,\n"
    "    \"rationale\": \"one short paragraph\",\n"
    "    \"disagreement_notes\": [\"bullet 1\", \"bullet 2\", ...]\n"
    "  }\n\n"
    "Rules:\n"
    "* Prefer candidates that quote the source directly and cite an "
    "article / clause reference.\n"
    "* Penalise speculation, hedging language, or claims not supported "
    "by the source context.\n"
    "* Return ``TIE`` when both candidates are equally faithful.\n"
    "* Return ``REVIEW`` when neither candidate is trustworthy — this "
    "flags the output for human adjudication."
)


def llm_judge_vote(
    client: Any,
    *,
    candidate_a: Any,
    candidate_b: Any,
    source_context: str,
    component: str,
    regulation: Optional[str] = None,
) -> JudgeVerdict:
    """Ask an LLM to adjudicate between two candidates.

    Uses the same guardrail-protected ``safe_generate`` path as any
    other LLM call so a fabricated verdict still gets meta-leakage and
    citation scrubbing.

    On any failure (client unavailable, malformed JSON) the function
    returns a ``JudgeVerdict(winner="REVIEW")`` — the caller should
    treat that as an escalation, not a silent default.
    """
    try:
        from pydantic import BaseModel, Field
    except ImportError:  # pragma: no cover
        return JudgeVerdict(winner="REVIEW", rationale="pydantic unavailable")

    class _JudgeSchema(BaseModel):
        winner: str = Field(description="A | B | TIE | REVIEW")
        confidence: float = Field(default=0.5, ge=0.0, le=1.0)
        rationale: str = Field(default="")
        disagreement_notes: List[str] = Field(default_factory=list)

    try:
        from .guardrails import safe_generate
    except Exception:  # pragma: no cover
        return JudgeVerdict(winner="REVIEW", rationale="guardrails unavailable")

    context_payload = {
        "candidate_a": _as_comparable(candidate_a),
        "candidate_b": _as_comparable(candidate_b),
        "source_context": source_context[:8000],  # cap for token safety
        "component_under_review": component,
    }
    payload, _report = safe_generate(
        client,
        _JudgeSchema,
        f"LLM-as-Judge · {component}",
        _JUDGE_INSTRUCTION,
        json.dumps(context_payload, default=str),
        regulation=regulation,
        source_corpus=source_context,
        text_fields=("rationale",),
        strict_citations=False,
        max_retries=0,
    )
    if payload is None:
        return JudgeVerdict(
            winner="REVIEW",
            rationale="Judge call failed (guardrail veto or LLM error).",
        )
    return JudgeVerdict(
        winner=str(payload.winner or "REVIEW").upper().strip(),
        confidence=float(payload.confidence or 0.0),
        rationale=str(payload.rationale or ""),
        disagreement_notes=list(payload.disagreement_notes or []),
    )


# ---------------------------------------------------------------------------
# The 2-of-3 vote
# ---------------------------------------------------------------------------


#: When two LLM samples agree above this threshold, we skip the judge call
#: and take the LLM as the winner. Below this threshold we escalate to the
#: judge (or the deterministic fallback if the judge is unavailable).
AGREEMENT_THRESHOLD = 0.85


def voted_generate(
    *,
    component: str,
    deterministic_fn: Callable[[], Any],
    llm_fn: Callable[[], Optional[Any]],
    second_llm_fn: Optional[Callable[[], Optional[Any]]] = None,
    judge_client: Any = None,
    source_corpus: Optional[str] = None,
    regulation: Optional[str] = None,
    llm_ok_flag: bool = True,
) -> Tuple[Any, VoteReport]:
    """Run a 2-of-3 vote and return the winning payload + audit report.

    Voting protocol
    ~~~~~~~~~~~~~~~
    1. Compute the deterministic baseline (``vote 1``). This always
       succeeds and is the safest fallback.
    2. Run the primary LLM call (``vote 2``) via ``llm_fn``. Any
       exception, empty return, or guardrail veto (``llm_ok_flag=False``)
       counts as an abstention and we return the deterministic baseline
       with ``winner="deterministic"``.
    3. Optionally run a second LLM call (``vote 3``) with a different
       temperature via ``second_llm_fn``. When the two LLM samples agree
       above :data:`AGREEMENT_THRESHOLD`, the LLM wins (2-of-3 majority:
       LLM + LLM against deterministic).
    4. When the two LLM samples disagree — or when ``second_llm_fn`` is
       ``None`` — we invoke :func:`llm_judge_vote` with the deterministic
       baseline vs. the LLM candidate. The judge breaks the tie.

    The report always describes the decision path so a reviewer can
    trace why one candidate won.
    """
    baseline = deterministic_fn()
    report = VoteReport(component=component)

    # Vote 2 — primary LLM.
    try:
        llm_a = llm_fn()
    except Exception:
        logger.exception("Primary LLM call raised in voted_generate for %s.", component)
        llm_a = None
        llm_ok_flag = False

    if llm_a is None or not llm_ok_flag:
        report.winner = "deterministic"
        report.llm_ok = bool(llm_ok_flag and llm_a is not None)
        report.n_llm_candidates = 0
        report.reason = "LLM abstained (empty output or guardrail veto)."
        return baseline, report

    report.n_llm_candidates = 1

    # Vote 3 — second LLM sample (optional; costs another API call).
    llm_b = None
    if second_llm_fn is not None:
        try:
            llm_b = second_llm_fn()
        except Exception:
            logger.exception("Secondary LLM call raised in voted_generate for %s.", component)
            llm_b = None
    if llm_b is not None:
        report.n_llm_candidates = 2

    # Self-agreement check between the two LLM samples.
    if llm_b is not None:
        agreement = structural_agreement(llm_a, llm_b)
        report.agreement_score = agreement
        if agreement >= AGREEMENT_THRESHOLD:
            report.winner = "llm"
            report.reason = (
                f"Both LLM samples agree ({agreement:.2f} >= "
                f"{AGREEMENT_THRESHOLD}); LLM wins 2-of-3."
            )
            return llm_a, report

    # Judge round — only when we have a judge client.
    if judge_client is not None:
        verdict = llm_judge_vote(
            judge_client,
            candidate_a=baseline,
            candidate_b=llm_a,
            source_context=source_corpus or "",
            component=component,
            regulation=regulation,
        )
        report.judge_verdict = verdict
        w = verdict.winner
        if w == "B":
            report.winner = "llm"
            report.reason = f"Judge picked LLM ({verdict.confidence:.2f})."
            return llm_a, report
        if w == "A":
            report.winner = "deterministic"
            report.reason = f"Judge picked deterministic ({verdict.confidence:.2f})."
            return baseline, report
        if w == "TIE":
            report.winner = "tie"
            report.reason = "Judge tied — defaulting to deterministic."
            return baseline, report
        # REVIEW / unknown -> escalate to human via deterministic fallback.
        report.winner = "deterministic"
        report.reason = "Judge escalated to REVIEW; using deterministic baseline."
        return baseline, report

    # No judge available and LLM samples not confidently in agreement.
    report.winner = "deterministic"
    report.reason = (
        "Judge unavailable and LLM samples disagreed; "
        "defaulting to deterministic baseline."
    )
    return baseline, report


__all__ = [
    "AGREEMENT_THRESHOLD",
    "JudgeVerdict",
    "VoteReport",
    "structural_agreement",
    "llm_judge_vote",
    "voted_generate",
]
