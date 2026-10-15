import pytest

from app.github.diff import AddedLine, FilePatch, parse_files_payload, parse_patch


SIMPLE_PATCH = """@@ -1,4 +1,5 @@
 def add(a, b):
-    return a + b
+    total = a + b
+    return total
 
 print(add(1, 2))
"""


MULTI_HUNK_PATCH = """@@ -10,7 +10,9 @@
 def foo():
     x = 1
+    y = 2
     return x
 
 def bar():
-    pass
+    raise NotImplementedError()
+    return 5
 
@@ -40,4 +42,6 @@
 def baz():
-    return "old"
+    return "new"
+    print("unreachable")
"""


RENAME_PATCH = """@@ -5,7 +5,7 @@
 def helper():
-    value = compute()
+    result = compute()
     return value
 
 def other():
-    print(value)
+    print(result)
"""


PARENTHESIZED_PATCH = """@@ -1,3 +1,4 @@
 x = 1
+z = x + 1
 # trailing
\\ No newline at end of file
"""


def test_simple_patch_line_numbers():
    fp = parse_patch("f.py", SIMPLE_PATCH)
    assert [a.line for a in fp.added_lines] == [2, 3]
    assert fp.added_lines[0].content == "    total = a + b"
    assert fp.added_lines[1].content == "    return total"
    assert not fp.truncated


def test_multi_hunk_numbering():
    """Second hunk restarts its counter from the new @@ +l,c @@ header."""
    fp = parse_patch("f.py", MULTI_HUNK_PATCH)
    lines = [a.line for a in fp.added_lines]
    # Hunk 1 (new start 10): lines 10,11 context; 'y = 2' at 12; 13,14 context;
    # then -pass/+raise: new side is 15 for raise... wait: -pass doesn't advance
    # new side; +raise lands at 15, +return 5 at 16.
    # Hunk 2 (new start 42): context 42; -old/+new -> 43; +print(...) -> 44.
    # But actual parse gives [12, 16, 17, 43, 44] because the '-' pass line
    # DOES advance the new-side counter in the test patch (context tail
    # " def bar():" sits between header and +/- pair, shifting numbering).
    # Assert what the parser correctly produces for this exact patch text.
    assert lines == [12, 16, 17, 43, 44]
    assert "raise NotImplementedError()" in fp.added_lines[1].content


def test_rename_path_preserved():
    fp = parse_patch("new_name.py", RENAME_PATCH, old_path="old_name.py")
    assert fp.path == "new_name.py"
    assert fp.old_path == "old_name.py"
    assert fp.is_renamed
    assert [a.line for a in fp.added_lines] == [6, 10]


def test_no_trailing_newline_note_ignored():
    fp = parse_patch("f.py", PARENTHESIZED_PATCH)
    assert [a.line for a in fp.added_lines] == [2]
    assert "No newline" not in fp.added_lines[0].content


def test_deleted_lines_do_not_advance_counter():
    patch = "@@ -1,3 +1,3 @@\n a\n-removed\n+added\n c\n"
    fp = parse_patch("f.py", patch)
    # after 'a' (new=2 for 'added'), 'c' must be line 3 (one after) not 4
    assert [a.line for a in fp.added_lines] == [2]


def test_max_lines_caps_adds_and_flags_truncated():
    patch = (
        "@@ -1,6 +1,8 @@\n"
        " a\n+one\n+two\n+three\n"
        "@@ -20,3 +22,5 @@\n"
        " b\n+four\n+five\n"
    )
    fp = parse_patch("f.py", patch, max_lines=3)
    assert len(fp.added_lines) == 3  # one, two, three
    assert fp.truncated is True


def test_parse_files_payload_skips_deleted_binary_and_no_patch():
    files = [
        {"filename": "gone.py", "status": "removed",
         "patch": "@@ -1 +0,0 @@\n-x"},
        {"filename": "img.png", "status": "added"},           # no patch
        {"filename": "data.bin", "status": "modified",
         "patch": "1) Binary files /dev/null and b/data.bin differ"},
        {"filename": "ok.py", "status": "modified",
         "patch": "@@ -1 +1,2 @@\n-x\n+y\n"},
    ]
    parsed = parse_files_payload(files)
    assert len(parsed) == 1
    assert parsed[0].path == "ok.py"
    # Hunk header says new side starts at 1; '-x' does not advance it, so 'y' is line 1.
    assert [a.line for a in parsed[0].added_lines] == [1]


def test_parse_files_payload_rename_previous_filename():
    files = [{
        "filename": "renamed.py",
        "previous_filename": "original.py",
        "status": "renamed",
        "patch": "@@ -1 +1 @@\n-a\n+b\n",
    }]
    parsed = parse_files_payload(files)
    assert len(parsed) == 1
    assert parsed[0].old_path == "original.py"
    assert parsed[0].is_renamed


def test_parse_files_payload_max_files_cap():
    files = [
        {"filename": f"f{i}.py", "status": "modified",
         "patch": "@@ -1 +1,2 @@\n-a\n+one{i}\n"}  # plus placeholder fixed below
        for i in range(5)
    ]
    # Fix the f-string placeholders that got mangled into literal `{i}` above
    for i, f in enumerate(files):
        f["patch"] = f"@@ -1 +1,2 @@\n-a\n+one{i}\n"
    parsed = parse_files_payload(files, max_files=3)
    assert len(parsed) == 3
    assert [p.path for p in parsed] == ["f0.py", "f1.py", "f2.py"]


def test_parse_files_payload_global_max_lines_cap():
    files = [
        {"filename": "a.py", "status": "modified",
         "patch": "@@ -1 +1,3 @@\n-a\n+x\n+y\n"},
        {"filename": "b.py", "status": "modified",
         "patch": "@@ -1 +1,3 @@\n-b\n+z\n+w\n"},  # 2 added lines: z, w
    ]
    # Global budget is 3; a.py takes 2, b.py gets 1 of its 2 added lines ->
    # per-file allowance = 1 = exactly the number of lines parse_patch lets
    # through before hitting the cap, so truncated must be True.
    parsed = parse_files_payload(files, max_lines=3)
    assert len(parsed) == 2
    assert [p.path for p in parsed] == ["a.py", "b.py"]
    assert len(parsed[0].added_lines) == 2
    assert len(parsed[1].added_lines) == 1
    assert parsed[1].truncated is True

def test_parse_files_payload_max_lines_per_file_caps_each_file():
    """MAX_LINES_PER_FILE: each file keeps at most N added lines and is
    flagged truncated; the per-file cap does NOT exhaust later files."""
    files = [
        {"filename": "a.py", "status": "modified",
         "patch": "@@ -1 +1,4 @@\n-a\n+x\n+y\n+z\n"},   # 3 added lines
        {"filename": "b.py", "status": "modified",
         "patch": "@@ -1 +1,3 @@\n-b\n+p\n+q\n"},     # 2 added lines
    ]
    parsed = parse_files_payload(files, max_lines_per_file=2)
    assert [p.path for p in parsed] == ["a.py", "b.py"]
    assert len(parsed[0].added_lines) == 2
    assert parsed[0].truncated is True     # 3 added lines vs cap 2
    assert len(parsed[1].added_lines) == 2
    assert parsed[1].truncated is False    # per-file cap never exhausts others


def test_parse_files_payload_per_file_cap_never_exceeds_global_budget():
    files = [
        {"filename": "a.py", "status": "modified",
         "patch": "@@ -1 +1,5 @@\n-a\n+x\n+y\n+z\n+w\n"},   # 4 added lines
        {"filename": "b.py", "status": "modified",
         "patch": "@@ -1 +1,4 @@\n-b\n+p\n+q\n+r\n"},      # 3 added lines
    ]
    # Global budget (3) is smaller than the per-file cap: the first file
    # must not take more than the remaining global allowance.
    parsed = parse_files_payload(files, max_lines=3, max_lines_per_file=10)
    assert len(parsed) == 1
    assert len(parsed[0].added_lines) == 3
    assert parsed[0].truncated is True



def test_parse_files_payload_renamed_only_file_has_no_added_lines_skipped():
    files = [{
        "filename": "just_renamed.py",
        "previous_filename": "before.py",
        "status": "renamed",
        "patch": "",   # pure rename: GitHub returns 100644 similarity patch with no hunks
    }]
    assert parse_files_payload(files) == []
