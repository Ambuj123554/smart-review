"""Provider selection: returns the configured LLM reviewer (gemini | groq).

Both reviewers expose the same `call(prompt) -> str` interface and an
`unconfigured_reason() -> str | None` check, so callers (DiffReviewer,
/Review in app.main) and the findings validation downstream are
provider-agnostic.
"""
from app.config import settings


def active_provider() -> str:
    """The configured provider name, normalized ('gemini' | 'groq').

    Read at call time (not import time) so a settings override made after
    app.config was imported always applies.
    """
    return (settings.LLM_PROVIDER or "gemini").strip().lower()


def active_model() -> str:
    """The model id of the active provider. Never includes API keys."""
    return settings.GROQ_MODEL if active_provider() == "groq" else settings.GEMINI_MODEL


def get_reviewer():
    """Builds the reviewer selected by LLM_PROVIDER.

    This is the ONLY place a reviewer is constructed in the app: the
    webhook pipeline (DiffReviewer) and /review (app.main) both call it.

    Raises ValueError with a clear message for an unknown provider so a
    typo fails loudly instead of silently falling back to the default.
    """
    provider = active_provider()
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
