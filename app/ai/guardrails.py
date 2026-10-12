"""LLM security guardrails: neutralize markdown/code fences in untrusted text.

Prompt injection via diff content is a real risk: an attacker can craft a line
like "```json ... ```" inside a Python file to break out of the code fence and
inject instructions. Replacing backticks plus explicit re-fencing mitigates
this by making the fence count deterministic and closing any open one.
"""

# Replace backticks to prevent closing/building code fences from untrusted text.
_BACKTICK_MAP = {"`": "'"}


def fence_code(text: str) -> str:
    """Wraps untrusted text in a code fence after replacing backticks.

    This ensures the fence is well-formed regardless of what the untrusted
    text contains and that the model cannot close it early or open new ones.
    """
    neutralized = text
    for old, new in _BACKTICK_MAP.items():
        neutralized = neutralized.replace(old, new)
    return "\n".join(["```", neutralized, "```"])
