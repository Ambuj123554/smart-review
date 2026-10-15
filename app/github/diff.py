"""Parsing of GitHub unified diffs (`patch` field from GET /pulls/{n}/files)."""
import re
from dataclasses import dataclass, field
from typing import List, Optional

# Matches "^@@ -l,c +l,c @@ context" (the context tail is optional).
HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

# Binary patch markers we treat as "no analysable content".
BINARY_MARKERS = ("Binary files ", "GIT binary patch")

# Caveat: a deleted file's diff technically has "+/-" lines too (nothing added
# past line 0 of the new side); the caller filters deleted + no-patch files so
# we never see those here. Renamed files keep a valid patch with "+" lines.


@dataclass
class AddedLine:
    """A single added line in a diff, positioned on the new side."""
    line: int          # 1-indexed line number on the new ("+") side
    content: str       # the added text without the leading "+"

    def __repr__(self) -> str:
        return f"AddedLine({self.line}, {self.content!r})"


@dataclass
class FilePatch:
    """A parsed patch for one file, grouped by hunk for review batching."""
    path: str                       # new path after rename, if renamed
    old_path: Optional[str] = None  # present when the file was renamed
    added_lines: List[AddedLine] = field(default_factory=list)
    truncated: bool = False         # True when the line cap cut this file short

    @property
    def is_renamed(self) -> bool:
        return self.old_path is not None and self.old_path != self.path


def parse_patch(
    path: str,
    patch: str,
    old_path: Optional[str] = None,
    max_lines: int = 0,
) -> FilePatch:
    """Parses the unified diff of one file into its added lines.

    `max_lines` caps how many added lines are kept for this file
    (0 = unlimited). Later hunks are dropped once the cap is hit and
    `truncated` is set so callers can still react.
    """
    result = FilePatch(path=path, old_path=old_path)
    new_line = 0
    cap_hit = False

    for raw in patch.splitlines():
        if cap_hit:
            # Still scanning? No — once capped we stop tracking: later hunks
            # would exceed the cap. Skip rest but keep old_path/path intact.
            break

        header = HUNK_HEADER_RE.match(raw)
        if header:
            new_line = int(header.group(1))
            continue

        if raw.startswith("+"):
            # Can't be a file header; hunks already set new_line > 0.
            # `+++ b/...` lines are handled by the caller, not here.
            if new_line > 0 and not raw.startswith("+++"):
                if max_lines > 0 and len(result.added_lines) >= max_lines:
                    cap_hit = True
                    result.truncated = True
                    break
                result.added_lines.append(
                    AddedLine(line=new_line, content=raw[1:])
                )
            new_line += 1
        elif raw.startswith("-"):
            # Deleted line: does not advance the new-side counter.
            continue
        elif raw.startswith(" "):
            new_line += 1
        elif raw.startswith("\\"):
            # "\ No newline at end of file" – does not advance counters.
            continue

        # Anything else (e.g. bare "\n" inside) – ignore.

    return result


def parse_files_payload(files: List[dict], max_files: int = 0, max_lines: int = 0,
                        max_lines_per_file: int = 0) -> List[FilePatch]:
    """Parses the JSON array from GET /repos/{o}/{r}/pulls/{n}/files.

    Skips:
      - binary files (`filename` from response means `patch` is absent,
        or the patch text contains binary markers),
      - deleted files (`status == "removed"`; nothing to review),
      - files with no patch (e.g. mode-only changes).

    Caps the total number of files (max_files, 0 = no cap), the total
    number of added lines across files (max_lines, 0 = no cap), and the
    added lines kept per file (max_lines_per_file, 0 = no cap). Files cut
    by a line cap stay in the result up to the cap with `truncated` set.
    """
    parsed: List[FilePatch] = []
    total_lines_used = 0

    for f in files:
        if max_files > 0 and len(parsed) >= max_files:
            break

        path = f.get("filename", "")
        if not path:
            continue

        status = f.get("status", "")
        if status == "removed":
            continue  # deleted file: nothing on the new side to analyse

        patch_text = f.get("patch")
        if not patch_text or any(
            marker in patch_text[:200] for marker in BINARY_MARKERS
        ):
            continue  # binary or no patch (e.g. mode-only change)

        old_path = None
        previous = f.get("previous_filename")
        if previous and previous != path:
            old_path = previous   # GitHub sets this for renamed files

        remaining = 0 if max_lines <= 0 else max_lines - total_lines_used
        if max_lines > 0 and remaining <= 0:
            break  # global line cap already reached

        # Distinguish "uncapped" (remaining == 0 with max_lines == 0) from
        # "capped to exactly this many" (remaining > 0 under a global cap):
        # pass a sentinel so truncated still reflects a real truncation.

        # Per-file cap (0 = unlimited); never exceed the remaining global
        # budget when one is active (remaining == 0 means "no global cap").
        per_file = remaining
        if max_lines_per_file > 0:
            per_file = (max_lines_per_file if remaining == 0
                        else min(max_lines_per_file, remaining))

        fp = parse_patch(path, patch_text, old_path=old_path, max_lines=per_file)

        # Keep the file even when truncated: its lines are valid up to the cap.
        if fp.added_lines:
            parsed.append(fp)

        if fp.truncated:
            # Per-file parse hit its allowance; the global cap is now exhausted.
            if max_lines > 0:
                total_lines_used = max_lines
                break  # cap reached, no point looking at more files
            total_lines_used += len(fp.added_lines)
        else:
            total_lines_used += len(fp.added_lines)

    return parsed
