from __future__ import annotations

import unittest

from code_similarity_tool.public_links import (
    build_github_blob_url,
    build_github_commit_url,
    build_public_commit_permalink,
    build_public_match_permalink,
)


class PublicLinksTest(unittest.TestCase):
    def test_build_github_blob_url(self) -> None:
        url = build_github_blob_url(
            "https://github.com/acme/demo",
            "abc123",
            "src/main file.py",
        )
        self.assertEqual(url, "https://github.com/acme/demo/blob/abc123/src/main%20file.py")

    def test_build_github_commit_url(self) -> None:
        url = build_github_commit_url("https://github.com/acme/demo", "abc123")
        self.assertEqual(url, "https://github.com/acme/demo/commit/abc123")

    def test_match_permalink_uses_stored_file_url_and_lines(self) -> None:
        meta = {
            "source_file_url": "https://github.com/acme/demo/blob/abc123/src/a.py",
            "start_line": 10,
            "end_line": 20,
        }
        self.assertEqual(
            build_public_match_permalink(meta),
            "https://github.com/acme/demo/blob/abc123/src/a.py#L10-L20",
        )

    def test_match_permalink_can_be_reconstructed_from_metadata(self) -> None:
        meta = {
            "source_url": "https://github.com/acme/demo",
            "source_commit": "abc123",
            "file_path": "src/a.py",
            "start_line": 3,
            "end_line": 3,
        }
        self.assertEqual(
            build_public_match_permalink(meta),
            "https://github.com/acme/demo/blob/abc123/src/a.py#L3",
        )

    def test_commit_permalink_from_metadata(self) -> None:
        meta = {
            "source_url": "https://github.com/acme/demo",
            "source_commit": "abc123",
        }
        self.assertEqual(
            build_public_commit_permalink(meta),
            "https://github.com/acme/demo/commit/abc123",
        )


if __name__ == "__main__":
    unittest.main()
