from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from code_similarity_tool.public_index import ensure_gnu_license, index_public_github_repo, parse_github_url


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

    def test_debug_element_index_prints_target_and_exits_without_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            file_path = repo_root / "a.py"
            file_path.write_text("def f():\n    return 1\n", encoding="utf-8")

            def _clone(_url: str, _sha: str, repo_dir: Path) -> None:
                repo_dir.mkdir(parents=True, exist_ok=True)
                (repo_dir / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")

            output = io.StringIO()
            with patch("code_similarity_tool.public_index._configure_logging", return_value=Mock()), patch(
                "code_similarity_tool.public_index.load_runtime_context"
            ), patch("code_similarity_tool.public_index.resolve_remote_commit", return_value="abc123"), patch(
                "code_similarity_tool.public_index.clone_repo_at_commit", side_effect=_clone
            ), patch(
                "code_similarity_tool.public_index.ensure_gnu_license", return_value="GPL-3.0"
            ), patch(
                "code_similarity_tool.public_index.iter_public_repo_source_files",
                side_effect=lambda repo_dir: [repo_dir / "a.py"],
            ), patch(
                "code_similarity_tool.public_index.CodeVectorStore"
            ) as store_cls, patch(
                "code_similarity_tool.public_index.EmbeddingClient"
            ) as embedder_cls, redirect_stdout(output):
                total = index_public_github_repo(
                    "https://github.com/example/demo",
                    debug_element_index=1,
                )

            self.assertEqual(total, 0)
            self.assertIn("#1 file=a.py", output.getvalue())
            store_cls.assert_not_called()
            embedder_cls.assert_not_called()

    def test_debug_element_index_out_of_range_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            file_path = repo_root / "a.py"
            file_path.write_text("def f():\n    return 1\n", encoding="utf-8")

            def _clone(_url: str, _sha: str, repo_dir: Path) -> None:
                repo_dir.mkdir(parents=True, exist_ok=True)
                (repo_dir / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")

            with patch("code_similarity_tool.public_index._configure_logging", return_value=Mock()), patch(
                "code_similarity_tool.public_index.load_runtime_context"
            ), patch("code_similarity_tool.public_index.resolve_remote_commit", return_value="abc123"), patch(
                "code_similarity_tool.public_index.clone_repo_at_commit", side_effect=_clone
            ), patch(
                "code_similarity_tool.public_index.ensure_gnu_license", return_value="GPL-3.0"
            ), patch(
                "code_similarity_tool.public_index.iter_public_repo_source_files",
                side_effect=lambda repo_dir: [repo_dir / "a.py"],
            ):
                with self.assertRaises(RuntimeError):
                    index_public_github_repo(
                        "https://github.com/example/demo",
                        debug_element_index=2,
                    )


if __name__ == "__main__":
    unittest.main()
