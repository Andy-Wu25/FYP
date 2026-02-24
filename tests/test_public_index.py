from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from code_similarity_tool.public_index import (
    _get_file_last_commit,
    detect_public_license,
    ensure_gnu_license,
    index_public_github_repo,
    parse_github_url,
)


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

    def test_detect_public_license_accepts_gpl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "LICENSE").write_text(
                "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n",
                encoding="utf-8",
            )
            self.assertEqual(detect_public_license(root), "GPL-3.0")

    def test_detect_public_license_accepts_mit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "LICENSE").write_text(
                "MIT License\n\nPermission is hereby granted, free of charge...",
                encoding="utf-8",
            )
            self.assertEqual(detect_public_license(root), "MIT")

    def test_detect_public_license_returns_unknown_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(detect_public_license(root), "UNKNOWN")

    def test_ensure_gnu_license_wrapper_remains_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "LICENSE").write_text("MIT License", encoding="utf-8")
            self.assertEqual(ensure_gnu_license(root), "MIT")

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
                "code_similarity_tool.public_index.detect_public_license", return_value="GPL-3.0"
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
                "code_similarity_tool.public_index.detect_public_license", return_value="GPL-3.0"
            ), patch(
                "code_similarity_tool.public_index.iter_public_repo_source_files",
                side_effect=lambda repo_dir: [repo_dir / "a.py"],
            ):
                with self.assertRaises(RuntimeError):
                    index_public_github_repo(
                        "https://github.com/example/demo",
                        debug_element_index=2,
                    )


class GetFileLastCommitTest(unittest.TestCase):
    """Unit tests for _get_file_last_commit — no real git repo needed."""

    def test_returns_trimmed_sha_from_git_log(self) -> None:
        mock_result = Mock()
        mock_result.stdout = "881ddba1234567890abcdef1234567890abcdef12\n"
        with patch("code_similarity_tool.public_index._run_git", return_value=mock_result):
            sha = _get_file_last_commit(Path("/repo"), "tests/config/psql_config.py", "fallback")
        self.assertEqual(sha, "881ddba1234567890abcdef1234567890abcdef12")

    def test_falls_back_when_git_log_returns_empty(self) -> None:
        mock_result = Mock()
        mock_result.stdout = ""
        with patch("code_similarity_tool.public_index._run_git", return_value=mock_result):
            sha = _get_file_last_commit(Path("/repo"), "src/main.py", "head_fallback")
        self.assertEqual(sha, "head_fallback")

    def test_falls_back_when_git_raises_called_process_error(self) -> None:
        with patch(
            "code_similarity_tool.public_index._run_git",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            sha = _get_file_last_commit(Path("/repo"), "src/main.py", "head_fallback")
        self.assertEqual(sha, "head_fallback")

    def test_falls_back_when_git_raises_os_error(self) -> None:
        with patch(
            "code_similarity_tool.public_index._run_git",
            side_effect=OSError("git not found"),
        ):
            sha = _get_file_last_commit(Path("/repo"), "src/main.py", "head_fallback")
        self.assertEqual(sha, "head_fallback")

    def test_passes_correct_args_and_cwd_to_run_git(self) -> None:
        mock_result = Mock()
        mock_result.stdout = "deadbeef\n"
        repo_dir = Path("/some/repo")
        with patch("code_similarity_tool.public_index._run_git", return_value=mock_result) as mock_run:
            _get_file_last_commit(repo_dir, "tests/config/psql_config.py", "fallback")
        mock_run.assert_called_once_with(
            ["log", "-1", "--format=%H", "--", "tests/config/psql_config.py"],
            cwd=repo_dir,
        )

    def test_strips_whitespace_from_sha(self) -> None:
        mock_result = Mock()
        mock_result.stdout = "  abc123  \n"
        with patch("code_similarity_tool.public_index._run_git", return_value=mock_result):
            sha = _get_file_last_commit(Path("/repo"), "file.py", "fallback")
        self.assertEqual(sha, "abc123")


def _sample_element() -> dict:
    return {
        "id": "placeholder",
        "name": "f",
        "kind": "function",
        "start_line": 1,
        "end_line": 2,
        "text": "def f():\n    return 1",
        "hash": "h1",
    }


class PublicIndexFileCommitTest(unittest.TestCase):
    """Integration tests: source_file_url uses the per-file last-modifying commit."""

    def _base_patches(self, store: Mock, embedder: Mock, file_commit_sha: str):
        """Return a stack of patches common to integration tests."""
        def _clone(_url: str, _sha: str, repo_dir: Path) -> None:
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")

        return [
            patch("code_similarity_tool.public_index._configure_logging", return_value=Mock()),
            patch("code_similarity_tool.public_index.load_runtime_context"),
            patch("code_similarity_tool.public_index.resolve_remote_commit", return_value="head_sha"),
            patch("code_similarity_tool.public_index.clone_repo_at_commit", side_effect=_clone),
            patch("code_similarity_tool.public_index.detect_public_license", return_value="GPL-3.0"),
            patch(
                "code_similarity_tool.public_index.iter_public_repo_source_files",
                side_effect=lambda repo_dir: [repo_dir / "a.py"],
            ),
            patch(
                "code_similarity_tool.public_index.extract_code_elements",
                return_value=[_sample_element()],
            ),
            patch("code_similarity_tool.public_index.CodeVectorStore", return_value=store),
            patch("code_similarity_tool.public_index.EmbeddingClient", return_value=embedder),
            patch(
                "code_similarity_tool.public_index._get_file_last_commit",
                return_value=file_commit_sha,
            ),
        ]

    def test_source_file_url_always_uses_head_commit(self) -> None:
        """permalink must pin to the indexed snapshot (HEAD), not the per-file commit."""
        store = Mock()
        store.delete_public_source_stale_entries.return_value = 0
        embedder = Mock()
        embedder.embed_documents.return_value = [[0.1, 0.2]]

        patches = self._base_patches(store, embedder, file_commit_sha="file_specific_sha")
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
            index_public_github_repo("https://github.com/example/demo")

        base_meta = store.upsert_public_code_elements.call_args.kwargs["base_metadata"]
        self.assertIn("/blob/head_sha/", base_meta["source_file_url"])
        self.assertNotIn("file_specific_sha", base_meta["source_file_url"])

    def test_source_file_commit_stored_as_per_file_sha(self) -> None:
        """source_file_commit must be the per-file last-modifying SHA, not HEAD."""
        store = Mock()
        store.delete_public_source_stale_entries.return_value = 0
        embedder = Mock()
        embedder.embed_documents.return_value = [[0.1, 0.2]]

        patches = self._base_patches(store, embedder, file_commit_sha="file_specific_sha")
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
            index_public_github_repo("https://github.com/example/demo")

        base_meta = store.upsert_public_code_elements.call_args.kwargs["base_metadata"]
        self.assertEqual(base_meta["source_file_commit"], "file_specific_sha")

    def test_source_commit_remains_head_sha(self) -> None:
        store = Mock()
        store.delete_public_source_stale_entries.return_value = 0
        embedder = Mock()
        embedder.embed_documents.return_value = [[0.1, 0.2]]

        patches = self._base_patches(store, embedder, file_commit_sha="file_specific_sha")
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
            index_public_github_repo("https://github.com/example/demo")

        base_meta = store.upsert_public_code_elements.call_args.kwargs["base_metadata"]
        self.assertEqual(base_meta["source_commit"], "head_sha")

    def test_source_file_url_contains_correct_file_path(self) -> None:
        store = Mock()
        store.delete_public_source_stale_entries.return_value = 0
        embedder = Mock()
        embedder.embed_documents.return_value = [[0.1, 0.2]]

        patches = self._base_patches(store, embedder, file_commit_sha="file_sha_abc")
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
            index_public_github_repo("https://github.com/example/demo")

        base_meta = store.upsert_public_code_elements.call_args.kwargs["base_metadata"]
        self.assertTrue(base_meta["source_file_url"].endswith("/a.py"))

    def test_source_file_commit_equals_fallback_when_git_unavailable(self) -> None:
        """When _get_file_last_commit falls back, source_file_commit equals HEAD."""
        store = Mock()
        store.delete_public_source_stale_entries.return_value = 0
        embedder = Mock()
        embedder.embed_documents.return_value = [[0.1, 0.2]]

        patches = self._base_patches(store, embedder, file_commit_sha="head_sha")
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
            index_public_github_repo("https://github.com/example/demo")

        base_meta = store.upsert_public_code_elements.call_args.kwargs["base_metadata"]
        self.assertEqual(base_meta["source_file_commit"], "head_sha")

    def test_get_file_last_commit_called_with_repo_dir_and_rel_path(self) -> None:
        """_get_file_last_commit must be called with the cloned repo_dir, not CWD."""
        store = Mock()
        store.delete_public_source_stale_entries.return_value = 0
        embedder = Mock()
        embedder.embed_documents.return_value = [[0.1, 0.2]]

        captured_calls = []

        def _clone(_url: str, _sha: str, repo_dir: Path) -> None:
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")

        def _fake_file_commit(repo_dir: Path, rel_path: str, fallback: str) -> str:
            captured_calls.append((repo_dir, rel_path, fallback))
            return "captured_sha"

        with patch("code_similarity_tool.public_index._configure_logging", return_value=Mock()), \
             patch("code_similarity_tool.public_index.load_runtime_context"), \
             patch("code_similarity_tool.public_index.resolve_remote_commit", return_value="head_sha"), \
             patch("code_similarity_tool.public_index.clone_repo_at_commit", side_effect=_clone), \
             patch("code_similarity_tool.public_index.detect_public_license", return_value="GPL-3.0"), \
             patch(
                 "code_similarity_tool.public_index.iter_public_repo_source_files",
                 side_effect=lambda repo_dir: [repo_dir / "a.py"],
             ), \
             patch(
                 "code_similarity_tool.public_index.extract_code_elements",
                 return_value=[_sample_element()],
             ), \
             patch("code_similarity_tool.public_index.CodeVectorStore", return_value=store), \
             patch("code_similarity_tool.public_index.EmbeddingClient", return_value=embedder), \
             patch(
                 "code_similarity_tool.public_index._get_file_last_commit",
                 side_effect=_fake_file_commit,
             ):
            index_public_github_repo("https://github.com/example/demo")

        self.assertEqual(len(captured_calls), 1)
        _repo_dir, rel_path, fallback = captured_calls[0]
        self.assertEqual(rel_path, "a.py")
        self.assertEqual(fallback, "head_sha")


if __name__ == "__main__":
    unittest.main()
