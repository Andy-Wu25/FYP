from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from code_similarity_tool.checking.check_utils import collect_staged_query_elements
from code_similarity_tool.core.runtime import parse_staged_new_ranges


class StagedHunksTest(unittest.TestCase):
    def test_parse_staged_new_ranges_handles_common_hunks(self) -> None:
        diff_text = """diff --git a/foo.py b/foo.py
index 1111111..2222222 100644
--- a/foo.py
+++ b/foo.py
@@ -1,2 +1,3 @@
+x = 1
@@ -20 +25 @@
+y = 2
@@ -40,4 +44,0 @@
-removed
"""
        ranges = parse_staged_new_ranges(diff_text)
        self.assertEqual(ranges, [(1, 3), (25, 25), (44, 44)])

    def test_parse_staged_new_ranges_merges_adjacent_ranges(self) -> None:
        diff_text = """@@ -1 +1 @@
+a
@@ -4 +2,2 @@
+b
"""
        ranges = parse_staged_new_ranges(diff_text)
        self.assertEqual(ranges, [(1, 3)])

    def test_collect_staged_query_elements_filters_to_changed_function_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            rel_path = "sample.py"
            abs_path = repo_root / rel_path
            abs_path.write_text("placeholder", encoding="utf-8")

            matcher = Mock()
            matcher.allows.return_value = True

            elements = [
                {
                    "id": "a",
                    "name": "first",
                    "kind": "function",
                    "start_line": 1,
                    "end_line": 4,
                    "hash": "h1",
                    "text": "def first(): pass",
                },
                {
                    "id": "b",
                    "name": "second",
                    "kind": "function",
                    "start_line": 10,
                    "end_line": 20,
                    "hash": "h2",
                    "text": "def second(): pass",
                },
            ]

            with patch("code_similarity_tool.checking.check_utils.load_ignore_file", return_value=matcher), patch(
                "code_similarity_tool.checking.check_utils.read_index_blob", return_value=b"dummy"
            ), patch("code_similarity_tool.checking.check_utils.staged_hunk_line_ranges", return_value=[(12, 12)]), patch(
                "code_similarity_tool.checking.check_utils.extract_code_elements", return_value=elements
            ):
                out = collect_staged_query_elements(repo_root, [rel_path])

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], rel_path)
        self.assertEqual(out[0][1]["name"], "second")

    def test_collect_staged_query_elements_skips_files_without_staged_hunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            rel_path = "sample.py"
            abs_path = repo_root / rel_path
            abs_path.write_text("placeholder", encoding="utf-8")

            matcher = Mock()
            matcher.allows.return_value = True

            with patch("code_similarity_tool.checking.check_utils.load_ignore_file", return_value=matcher), patch(
                "code_similarity_tool.checking.check_utils.read_index_blob", return_value=b"dummy"
            ), patch("code_similarity_tool.checking.check_utils.staged_hunk_line_ranges", return_value=[]), patch(
                "code_similarity_tool.checking.check_utils.extract_code_elements",
                return_value=[
                    {
                        "id": "a",
                        "name": "first",
                        "kind": "function",
                        "start_line": 1,
                        "end_line": 4,
                        "hash": "h1",
                        "text": "def first(): pass",
                    }
                ],
            ):
                out = collect_staged_query_elements(repo_root, [rel_path])

        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
