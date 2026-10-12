"""Groq provider: OpenAI-compatible chat completions for prompt -> text.

Mirrors the GeminiReviewer interface (`call(prompt) -> str`) so the rest of
the pipeline (prompt building, findings schema, validation) is unchanged.
Uses the project's existing HTTP client (requests).
"""
import logging
import time
from typing import Optional

import requests

from app.config import settings

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2.0


class GroqError(RuntimeError):
    """Raised when a Groq request fails permanently (config or API error)."""


class GroqReviewer:
    """Calls Groq's chat completions endpoint with JSON output mode."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        # Read settings at call time so tests can override them.
        self.api_key = settings.GROQ_API_KEY if api_key is None else api_key
        self.model = settings.GROQ_MODEL if model is None else model
        self.session = session or requests.Session()

    def unconfigured_reason(self) -> Optional[str]:
        """Reason this provider cannot be called, or None when it is ready.

        Shared with /review (app.main) so a missing key becomes a clear
        status instead of a pointless network call.
        """
        if not self.api_key or self.api_key.strip() in (
            "", "your_groq_api_key_here", "gsk_your_key_here",
        ):
            return "GROQ_API_KEY is not configured; set it or use LLM_PROVIDER=gemini."
        return None

    def call(self, prompt: str) -> str:
        """Sends the prompt and returns the assistant message content."""
        reason = self.unconfigured_reason()
        if reason:
            logger.error(reason)
            raise GroqError(reason)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            # JSON output mode: guarantees well-formed JSON text; the
            # existing prompt still specifies the array schema and the
            # existing validator still enforces it.
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = self.session.post(
                    GROQ_CHAT_URL, json=payload, headers=headers, timeout=60
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Groq network error (attempt %d/%d): %s",
                    attempt, MAX_ATTEMPTS, exc,
                )
                if attempt < MAX_ATTEMPTS:
                    time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
                continue

            if resp.status_code == 429:
                last_error = GroqError("Groq rate limited (429)")
                retry_after = resp.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if retry_after and retry_after.replace(".", "", 1).isdigit()
                    else BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                )
                logger.warning(
                    "Groq rate limited (attempt %d/%d); sleeping %.1fs: %s",
                    attempt, MAX_ATTEMPTS, wait, resp.text[:200],
                )
                if attempt < MAX_ATTEMPTS:
                    time.sleep(wait)
                continue

            if resp.status_code in (401, 403):
                msg = (
                    f"Groq rejected the API key (HTTP {resp.status_code}); "
                    "check GROQ_API_KEY."
                )
                logger.error(msg)
                raise GroqError(msg)

            if resp.status_code == 404:
                msg = (
                    f"Groq model id '{self.model}' not found (HTTP 404); "
                    "check GROQ_MODEL."
                )
                logger.error(msg)
                raise GroqError(msg)

            if resp.status_code >= 400:
                msg = f"Groq API error {resp.status_code}: {resp.text[:200]}"
                logger.error(msg)
                raise GroqError(msg)

            try:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                msg = f"Unexpected Groq response shape: {exc}"
                logger.error(msg)
                raise GroqError(msg) from exc

            if not isinstance(content, str) or not content.strip():
                msg = "Groq returned empty message content."
                logger.error(msg)
                raise GroqError(msg)
            return content

        raise GroqError(
            f"Groq request failed after {MAX_ATTEMPTS} attempts: {last_error}"
        )
