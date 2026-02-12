from __future__ import annotations

import unittest

from code_similarity_tool.check_self import _include_self_hit


class CheckSelfFilterTest(unittest.TestCase):
    def test_excludes_exact_same_file_and_content(self) -> None:
        meta = {
            "file_path": "src/a.py",
            "content_hash": "abc123",
        }
        self.assertFalse(_include_self_hit(meta, "src/a.py", "abc123"))

    def test_includes_same_file_with_different_content(self) -> None:
        meta = {
            "file_path": "src/a.py",
            "content_hash": "oldhash",
        }
        self.assertTrue(_include_self_hit(meta, "src/a.py", "newhash"))

    def test_includes_different_file_with_same_content(self) -> None:
        meta = {
            "file_path": "src/b.py",
            "content_hash": "abc123",
        }
        self.assertTrue(_include_self_hit(meta, "src/a.py", "abc123"))


if __name__ == "__main__":
    unittest.main()
