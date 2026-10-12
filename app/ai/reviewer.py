import json
import logging
import re
from typing import List, Optional
from app.config import settings
from app.schemas.review import Finding

logger = logging.getLogger(__name__)


def build_review_prompt(code: str, rule_findings: List[Finding]) -> str:
    """Constructs the full-file contextual review prompt.

    Provider-agnostic: the same text is sent by whichever reviewer
    app.ai.providers.get_reviewer() selects (Gemini or Groq).
    """
    rule_summary_str = json.dumps([f.model_dump() for f in rule_findings], indent=2) if rule_findings else "None"

    return f"""You are a Senior Python Code Reviewer AI.
Analyze the following Python source code for contextual issues, logical bugs, edge cases, missing error handling, off-by-one errors, resource leaks, or poor naming choices.

Source Code:
```python
{code}
```

Deterministic AST Findings already detected:
```json
{rule_summary_str}
```

Instructions:
1. Do NOT duplicate or re-report AST findings that are already listed above, unless you are adding deeper contextual insight.
2. Focus on contextual logic issues that AST static rules cannot easily detect.
3. Return ONLY a valid JSON array of objects. Do not include markdown formatting around JSON if possible, or use standard ```json block.

Each JSON object in the array MUST strictly follow this structure:
{{
  "rule_id": "AI-LOGIC-001",
  "category": "Logic",
  "severity": "HIGH",
  "line": 5,
  "message": "Short summary of issue",
  "explanation": "Clear explanation of why this matters",
  "suggestion": "How to fix it"
}}

Valid Severities: CRITICAL, HIGH, MEDIUM, LOW.
Valid Categories: Security, Logic, Performance, Code Quality, Error Handling, Bug Risk.
"""


def parse_findings_response(text: Optional[str]) -> Optional[List[Finding]]:
    """Parses an LLM reply into validated findings.

    Returns None when the output is unusable (empty, non-JSON, or not a
    JSON array of findings) so callers can report a clear failure instead
    of silently dropping findings; a valid, empty review returns [].
    """
    if not text or not str(text).strip():
        return None

    # Extract JSON block if surrounded by markdown fence
    json_text = str(text).strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_text)
    if match:
        json_text = match.group(1).strip()

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    if isinstance(data, dict):
        # JSON output mode (Groq) can wrap the array in an object such as
        # {"findings": [...]}; unwrap the first list of objects.
        data = next((v for v in data.values() if isinstance(v, list)), None)

    if not isinstance(data, list):
        return None

    ai_findings: List[Finding] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        try:
            finding = Finding(
                rule_id=str(item.get("rule_id", f"AI-CONTEXT-{idx+1}")),
                category=str(item.get("category", "Logic")),
                severity=str(item.get("severity", "MEDIUM")).upper(),
                line=int(item.get("line", 1)),
                message=str(item.get("message", "AI Contextual finding")),
                explanation=str(item.get("explanation", "")),
                suggestion=str(item.get("suggestion", "")),
                source="AI Analysis"
            )
            ai_findings.append(finding)
        except Exception as e:
            logger.warning(f"Skipping malformed AI finding item: {e}")

    return ai_findings


class GeminiReviewer:
    """Contextual AI Code Reviewer using Google Gemini API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.GEMINI_API_KEY

    def unconfigured_reason(self) -> Optional[str]:
        """Reason this provider cannot be called, or None when it is ready.

        Mirrors GroqReviewer.unconfigured_reason() so callers (e.g.
        /review) can surface a clear status without a pointless call.
        """
        if not self.api_key or self.api_key.strip() in ("", "your_gemini_api_key_here"):
            return "GEMINI_API_KEY is not configured"
        return None

    def call(self, prompt: str) -> str:
        """Provider-neutral entry point (same interface as GroqReviewer.call)."""
        return self._call_gemini_api(prompt)

    def analyze(self, code: str, rule_findings: List[Finding]) -> List[Finding]:
        """Sends code and AST findings to Gemini for contextual analysis."""
        reason = self.unconfigured_reason()
        if reason:
            logger.info("%s. Skipping AI contextual review.", reason)
            return []

        prompt = self._build_prompt(code, rule_findings)

        try:
            raw_response_text = self._call_gemini_api(prompt)
            return self._parse_and_validate_response(raw_response_text)
        except Exception as err:
            logger.warning(f"Gemini AI review failed gracefully: {str(err)}")
            return []

    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API using available google-genai or google.generativeai SDK."""
        # Try google-genai SDK first
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except ImportError:
            pass

        # Fallback to google-generativeai SDK
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            model = genai_legacy.GenerativeModel(settings.GEMINI_MODEL)
            response = model.generate_content(prompt)
            return response.text
        except ImportError:
            raise RuntimeError("Neither 'google-genai' nor 'google-generativeai' package is installed.")

    def _build_prompt(self, code: str, rule_findings: List[Finding]) -> str:
        """Constructs prompt for Gemini AI model."""
        return build_review_prompt(code, rule_findings)

    def _parse_and_validate_response(self, text: str) -> List[Finding]:
        """Safely parses JSON response from Gemini output.

        Legacy analyze() path: invalid output logs a warning and degrades
        to no findings; parse_findings_response() returns None for that case.
        """
        if not text or not str(text).strip():
            return []

        findings = parse_findings_response(text)
        if findings is None:
            logger.warning("Failed to decode JSON from Gemini AI response.")
            return []
        return findings
