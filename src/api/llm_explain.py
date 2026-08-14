"""
LLM-generated, plain-English explanations of churn predictions.

Supports two providers, tried in this order:
  1. Groq (GROQ_API_KEY) - genuinely free tier, no credit card, recommended
  2. Anthropic (ANTHROPIC_API_KEY) - paid, used if configured instead/as well

Kept isolated from the core prediction path: if neither key is set, or the
API call fails for any reason, callers get a clear "not available" message
instead of a broken request. The churn model itself never depends on this
module - predictions work identically with or without an LLM configured.
"""
from __future__ import annotations

import os

_GROQ_MODEL = os.environ.get("GROQ_EXPLAIN_MODEL", "llama-3.1-8b-instant")
_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_EXPLAIN_MODEL", "claude-haiku-4-5-20251001")

_SYSTEM_PROMPT = (
    "You are a churn-analysis assistant embedded in a telecom customer churn "
    "prediction tool. Given a customer's account details and a model's churn "
    "probability, explain in 2-3 short sentences, in plain English, why the "
    "model likely scored this customer the way it did. Reference specific "
    "fields from the input (e.g. contract type, tenure, support tickets, "
    "charges). Be concise and concrete - no preamble, no disclaimers, no "
    "restating the probability number itself. Write for a non-technical "
    "business user."
)


def _active_provider() -> str | None:
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def is_configured() -> bool:
    return _active_provider() is not None


def _build_user_prompt(customer: dict, churn_probability: float, risk_tier: str) -> str:
    customer_summary = "\n".join(f"- {k}: {v}" for k, v in customer.items() if k != "customer_id")
    return (
        f"Customer churn probability: {churn_probability * 100:.1f}% (risk tier: {risk_tier})\n\n"
        f"Customer details:\n{customer_summary}"
    )


def _call_groq(user_prompt: str) -> str:
    from groq import Groq

    client = Groq()  # reads GROQ_API_KEY from env
    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        max_tokens=200,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _call_anthropic(user_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=_ANTHROPIC_MODEL,
        max_tokens=200,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()


def explain_prediction(customer: dict, churn_probability: float, risk_tier: str) -> str:
    """Returns a short natural-language explanation, or a friendly fallback
    message if no LLM is configured or the call fails."""
    provider = _active_provider()
    if provider is None:
        return (
            "AI explanations aren't enabled on this deployment (no GROQ_API_KEY or "
            "ANTHROPIC_API_KEY configured). The prediction above is still the real "
            "model output."
        )

    user_prompt = _build_user_prompt(customer, churn_probability, risk_tier)
    try:
        if provider == "groq":
            text = _call_groq(user_prompt)
        else:
            text = _call_anthropic(user_prompt)
        return text or "The model did not return an explanation for this prediction."
    except Exception as e:  # noqa: BLE001
        return f"Couldn't generate an AI explanation right now ({type(e).__name__}: {e})."
