"""Provider selection: returns the configured LLM reviewer (gemini | groq).

Both reviewers expose the same `call(prompt) -> str` interface, so callers
(DiffReviewer) and the findings validation downstream are provider-agnostic.
"""
from app.config import settings


def get_reviewer():
    """Builds the reviewer selected by LLM_PROVIDER.

    Raises ValueError with a clear message for an unknown provider so a
    typo fails loudly instead of silently falling back to the default.
    """
    provider = (settings.LLM_PROVIDER or "gemini").strip().lower()
    if provider == "gemini":
        from app.ai.reviewer import GeminiReviewer
        return GeminiReviewer()
    if provider == "groq":
        from app.ai.groq import GroqReviewer
        return GroqReviewer()
    raise ValueError(
        f"Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}' "
        "(expected 'gemini' or 'groq')"
    )
