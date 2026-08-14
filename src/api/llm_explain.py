"""
LLM-generated, plain-English explanations of churn predictions using the
Anthropic API.

Kept isolated from the core prediction path: if ANTHROPIC_API_KEY isn't set,
or the API call fails for any reason, callers get a clear "not available"
message instead of a broken request. The churn model itself never depends on
this module - predictions work identically with or without an LLM configured.
"""
from __future__ import annotations

import os

_MODEL = os.environ.get("ANTHROPIC_EXPLAIN_MODEL", "claude-haiku-4-5-20251001")

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


def is_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def explain_prediction(customer: dict, churn_probability: float, risk_tier: str) -> str:
    """Returns a short natural-language explanation, or a friendly fallback
    message if the LLM isn't configured or the call fails."""
    if not is_configured():
        return (
            "AI explanations aren't enabled on this deployment (no ANTHROPIC_API_KEY "
            "configured). The prediction above is still the real model output."
        )

    try:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        customer_summary = "\n".join(f"- {k}: {v}" for k, v in customer.items() if k != "customer_id")
        user_prompt = (
            f"Customer churn probability: {churn_probability * 100:.1f}% (risk tier: {risk_tier})\n\n"
            f"Customer details:\n{customer_summary}"
        )
        response = client.messages.create(
            model=_MODEL,
            max_tokens=200,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return text.strip() or "The model did not return an explanation for this prediction."
    except Exception as e:  # noqa: BLE001
        return f"Couldn't generate an AI explanation right now ({type(e).__name__}: {e})."
