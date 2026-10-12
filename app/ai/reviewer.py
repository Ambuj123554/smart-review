import json
import logging
import re
from typing import List
from app.config import settings
from app.schemas.review import Finding

logger = logging.getLogger(__name__)


class GeminiReviewer:
    """Contextual AI Code Reviewer using Google Gemini API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.GEMINI_API_KEY

    def analyze(self, code: str, rule_findings: List[Finding]) -> List[Finding]:
        """Sends code and AST findings to Gemini for contextual analysis."""
        if not self.api_key or self.api_key.strip() in ("", "your_gemini_api_key_here"):
            logger.info("GEMINI_API_KEY not configured. Skipping AI contextual review.")
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

    def _parse_and_validate_response(self, text: str) -> List[Finding]:
        """Safely parses JSON response from Gemini output."""
        if not text:
            return []

        # Extract JSON block if surrounded by markdown fence
        json_text = text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_text)
        if match:
            json_text = match.group(1).strip()

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON from Gemini AI response.")
            return []

        if not isinstance(data, list):
            return []

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
