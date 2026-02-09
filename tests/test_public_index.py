from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from code_similarity_tool.public_index import ensure_gnu_license, parse_github_url


class PublicIndexTest(unittest.TestCase):
    def test_parse_github_url_accepts_canonical(self) -> None:
        owner, repo, canonical = parse_github_url("https://github.com/openai/example-repo")
        self.assertEqual(owner, "openai")
        self.assertEqual(repo, "example-repo")
        self.assertEqual(canonical, "https://github.com/openai/example-repo")

    def test_parse_github_url_accepts_dot_git(self) -> None:
        owner, repo, canonical = parse_github_url("https://github.com/user/demo.git")
        self.assertEqual(owner, "user")
        self.assertEqual(repo, "demo")
        self.assertEqual(canonical, "https://github.com/user/demo")

    def test_parse_github_url_rejects_non_github(self) -> None:
        with self.assertRaises(ValueError):
            parse_github_url("https://gitlab.com/user/demo")

    def test_ensure_gnu_license_accepts_gpl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "LICENSE").write_text(
                "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n",
                encoding="utf-8",
            )
            self.assertEqual(ensure_gnu_license(root), "GPL-3.0")

    def test_ensure_gnu_license_rejects_non_gnu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "LICENSE").write_text("MIT License", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                ensure_gnu_license(root)


if __name__ == "__main__":
    unittest.main()
