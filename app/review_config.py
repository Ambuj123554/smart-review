"""Optional per-repository review config: `.reviewer.yml` at the PR head.

The file is fetched through the GitHub contents API at the PR's head SHA
and may be absent, malformed, or hostile. The contract for ALL of those
cases is the same: log a warning, fall back to defaults, and let the
review proceed — a config problem must never fail a review.

The file content is UNTRUSTED (it lives in the reviewed repo and can be
edited by any PR author):

- it is parsed with yaml.safe_load only (never an unsafe loader),
- every value is validated against a strict schema; ANY problem rejects
  the whole file (unknown keys included) instead of half-applying it,
- custom_instructions is stripped of control characters and capped at
  MAX_CUSTOM_INSTRUCTIONS characters before it can reach a prompt, and
  the prompt layer fences it as data (see app/ai/diff_reviewer.py),
- nothing from settings or the environment is ever combined with it, so
  it cannot be used to elicit or log secrets.
"""
import fnmatch
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# File name looked up in the repo root at the PR's head SHA.
CONFIG_FILE_NAME = ".reviewer.yml"

# Supported keys — anything else rejects the file (typo protection).
SUPPORTED_KEYS = frozenset(
    {"ignore_paths", "min_severity", "disable_rules", "custom_instructions"}
)

VALID_SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# custom_instructions is a SHORT hint, hard-capped before it can reach a prompt.
MAX_CUSTOM_INSTRUCTIONS = 500
# List-valued keys: bounded so a hostile file cannot blow up memory/prompt cost.
MAX_LIST_ENTRIES = 100
MAX_LIST_ENTRY_CHARS = 200

# Rule ids look like SEC-001 / AST-002 / AI-DIFF-001 / TEST-001.
_RULE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")

# C0/C1 control characters except newline and tab (kept for readable text).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ReviewConfigError(ValueError):
    """Raised when .reviewer.yml content fails validation.

    Callers turn this into "defaults + warning"; the review never fails.
    """


def sanitize_custom_instructions(text: Optional[str]) -> str:
    """Strips control characters and caps at MAX_CUSTOM_INSTRUCTIONS.

    Defense in depth: applied when parsing the file AND again where the
    text enters an LLM prompt, so no caller can bypass the cap.
    """
    if not text or not isinstance(text, str):
        return ""
    cleaned = _CONTROL_CHARS_RE.sub("", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(cleaned) > MAX_CUSTOM_INSTRUCTIONS:
        logger.warning(
            "custom_instructions exceeds %d characters; capping it (the "
            "truncated remainder is not used).",
            MAX_CUSTOM_INSTRUCTIONS,
        )
        cleaned = cleaned[:MAX_CUSTOM_INSTRUCTIONS].rstrip()
    return cleaned


@dataclass(frozen=True)
class ReviewConfig:
    """Validated .reviewer.yml values (all optional)."""

    ignore_paths: Tuple[str, ...] = field(default_factory=tuple)
    min_severity: Optional[str] = None  # None -> the MIN_SEVERITY setting applies
    disable_rules: frozenset = field(default_factory=frozenset)
    custom_instructions: str = ""
    applied: bool = False  # True only when a file was fetched AND validated

    @staticmethod
    def default() -> "ReviewConfig":
        """Plain defaults: analyse everything, setting-based severity floor."""
        return ReviewConfig()

    def ignores_path(self, path: Optional[str]) -> bool:
        """True when `path` (repo-relative, forward slashes) matches ignore_paths.

        fnmatch-style globs on the full path: `tests/**` skips everything
        under tests/, `*.md` skips markdown anywhere (fnmatch's `*` spans
        separators), and `tests/**` also matches the bare `tests` entry.
        """
        if not path or not self.ignore_paths:
            return False
        normalized = str(path).replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        for pattern in self.ignore_paths:
            if fnmatch.fnmatchcase(normalized, pattern):
                return True
            if pattern.endswith("/**"):
                directory = pattern[:-2]  # "tests/" -> also skip "tests" itself
                if normalized == pattern[:-3] or normalized.startswith(directory):
                    return True
        return False


def _validate_string_list(data: dict, key: str) -> Tuple[str, ...]:
    """Validates a list-of-strings key: present values must be well-formed.

    Raises ReviewConfigError on null, a non-list, non-string members,
    empty entries, or absurd sizes.
    """
    value = data.get(key, [])
    if value is None:
        raise ReviewConfigError(f"'{key}' must be a list of strings, got null")
    if not isinstance(value, list):
        raise ReviewConfigError(
            f"'{key}' must be a list of strings, got {type(value).__name__}"
        )
    if len(value) > MAX_LIST_ENTRIES:
        raise ReviewConfigError(
            f"'{key}' has {len(value)} entries (max {MAX_LIST_ENTRIES})"
        )
    entries: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ReviewConfigError(
                f"'{key}' entries must be strings, got {type(item).__name__}"
            )
        entry = item.strip()
        if not entry:
            raise ReviewConfigError(f"'{key}' contains an empty entry")
        if len(entry) > MAX_LIST_ENTRY_CHARS:
            raise ReviewConfigError(
                f"'{key}' entry exceeds {MAX_LIST_ENTRY_CHARS} characters"
            )
        entries.append(entry)
    return tuple(entries)


def parse_review_config(text: str) -> ReviewConfig:
    """Parses and validates .reviewer.yml content (UNTRUSTED).

    Returns a ReviewConfig with applied=True. Raises ReviewConfigError on
    ANY problem — malformed YAML, non-mapping content, unknown keys, wrong
    types, bad severities, bad rule ids — so callers can fall back to
    defaults wholesale rather than half-applying a hostile file.
    custom_instructions is capped, not rejected, when too long.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as err:
        # Truncated: YAML errors echo the source line, which is repo content.
        raise ReviewConfigError(f"not valid YAML: {str(err)[:200]}") from None

    if data is None:
        raise ReviewConfigError("file is empty")
    if not isinstance(data, dict):
        raise ReviewConfigError(
            f"expected a mapping of keys, got {type(data).__name__}"
        )

    unknown = sorted(str(k) for k in data if k not in SUPPORTED_KEYS)
    if unknown:
        raise ReviewConfigError(
            "unknown key(s): "
            + ", ".join(unknown)
            + " (supported: "
            + ", ".join(sorted(SUPPORTED_KEYS))
            + ")"
        )

    ignore_paths = _validate_string_list(data, "ignore_paths")
    disable_rules = frozenset(_validate_string_list(data, "disable_rules"))
    for rule_id in sorted(disable_rules):
        if not _RULE_ID_RE.match(rule_id):
            raise ReviewConfigError(
                f"'disable_rules' entry {rule_id!r} is not a valid rule id"
            )

    min_severity: Optional[str] = None
    if "min_severity" in data:
        raw = data["min_severity"]
        if not isinstance(raw, str) or not raw.strip():
            raise ReviewConfigError(
                "'min_severity' must be one of: " + ", ".join(VALID_SEVERITIES)
            )
        candidate = raw.strip().upper()
        if candidate not in VALID_SEVERITIES:
            raise ReviewConfigError(
                f"'min_severity' value {raw.strip()!r} is not one of: "
                + ", ".join(VALID_SEVERITIES)
            )
        min_severity = candidate

    custom_instructions = ""
    if "custom_instructions" in data:
        raw = data["custom_instructions"]
        if not isinstance(raw, str):
            raise ReviewConfigError("'custom_instructions' must be a string")
        custom_instructions = sanitize_custom_instructions(raw)

    return ReviewConfig(
        ignore_paths=ignore_paths,
        min_severity=min_severity,
        disable_rules=disable_rules,
        custom_instructions=custom_instructions,
        applied=True,
    )


def fetch_review_config(
    client, repo_full_name: str, head_sha: str
) -> ReviewConfig:
    """Fetches CONFIG_FILE_NAME at head_sha via the GitHub contents API.

    Missing, unfetchable, or invalid content NEVER fails the review: each
    case logs a warning and returns defaults. Returns a config with
    applied=True only when a file was fetched AND validated.
    """
    try:
        text = client.get_file_content(
            repo_full_name, CONFIG_FILE_NAME, head_sha)
    except Exception as err:  # noqa: BLE001 - a config problem must not fail a review
        logger.warning(
            "Could not fetch %s for %s (%s); using default review settings.",
            CONFIG_FILE_NAME, repo_full_name, str(err)[:200],
        )
        return ReviewConfig.default()

    if text is None or not str(text).strip():
        logger.warning(
            "%s not found for %s at %s; using default review settings.",
            CONFIG_FILE_NAME, repo_full_name, str(head_sha)[:12],
        )
        return ReviewConfig.default()

    try:
        config = parse_review_config(text)
    except ReviewConfigError as err:
        logger.warning(
            "Ignoring invalid %s for %s: %s; using default review settings.",
            CONFIG_FILE_NAME, repo_full_name, err,
        )
        return ReviewConfig.default()

    logger.info(
        "%s applied for %s (ignore_paths=%d, min_severity=%s, "
        "disable_rules=%d, custom_instructions=%s).",
        CONFIG_FILE_NAME, repo_full_name, len(config.ignore_paths),
        config.min_severity or "setting", len(config.disable_rules),
        "yes" if config.custom_instructions else "no",
    )
    return config
