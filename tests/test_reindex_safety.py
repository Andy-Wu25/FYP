from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from code_similarity_tool import public_index
from code_similarity_tool.code_parser import make_element_id
from code_similarity_tool.index import sync_current_repo
from code_similarity_tool.public_index import index_public_github_repo
from code_similarity_tool.runtime import RuntimeContext


def _runtime_ctx(repo_root: Path) -> RuntimeContext:
    return RuntimeContext(
        repo_root=repo_root,
        repo_name=repo_root.name,
        repo_id="repo-1",
        org_id="org-1",
        db_path=repo_root / "db",
        private_collection_name="private_collection",
        public_collection_name="public_collection",
        metric="cosine",
    )


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


class ReindexSafetyTest(unittest.TestCase):
    def test_private_sync_does_not_delete_when_embedding_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            file_path = repo_root / "a.py"
            file_path.write_text("def f():\n    return 1\n", encoding="utf-8")

            ctx = _runtime_ctx(repo_root)
            store = Mock()
            matcher = Mock()
            matcher.allows.return_value = True
            embedder = Mock()
            embedder.embed_documents.side_effect = RuntimeError("embedding unavailable")

            with patch("code_similarity_tool.index._configure_logging", return_value=Mock()), patch(
                "code_similarity_tool.index.load_runtime_context", return_value=ctx
            ), patch("code_similarity_tool.index.load_ignore_file", return_value=matcher), patch(
                "code_similarity_tool.index.iter_repo_source_files", return_value=[file_path]
            ), patch(
                "code_similarity_tool.index.extract_code_elements", return_value=[_sample_element()]
            ), patch(
                "code_similarity_tool.index.CodeVectorStore", return_value=store
            ), patch(
                "code_similarity_tool.index.EmbeddingClient", return_value=embedder
            ):
                with self.assertRaises(RuntimeError):
                    sync_current_repo(action_name="index")

            store.delete_private_repo_entries.assert_not_called()
            store.delete_private_repo_stale_entries.assert_not_called()
            store.upsert_private_code_elements.assert_not_called()

    def test_private_sync_upserts_then_prunes_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            file_path = repo_root / "a.py"
            file_path.write_text("def f():\n    return 1\n", encoding="utf-8")

            ctx = _runtime_ctx(repo_root)
            store = Mock()
            store.delete_private_repo_stale_entries.return_value = 1
            matcher = Mock()
            matcher.allows.return_value = True
            embedder = Mock()
            embedder.embed_documents.return_value = [[0.1, 0.2]]

            with patch("code_similarity_tool.index._configure_logging", return_value=Mock()), patch(
                "code_similarity_tool.index.load_runtime_context", return_value=ctx
            ), patch("code_similarity_tool.index.load_ignore_file", return_value=matcher), patch(
                "code_similarity_tool.index.iter_repo_source_files", return_value=[file_path]
            ), patch(
                "code_similarity_tool.index.extract_code_elements", return_value=[_sample_element()]
            ), patch(
                "code_similarity_tool.index.CodeVectorStore", return_value=store
            ), patch(
                "code_similarity_tool.index.EmbeddingClient", return_value=embedder
            ):
                total = sync_current_repo(action_name="index")

            expected_id = make_element_id("org-1", "repo-1", "a.py", "h1")
            self.assertEqual(total, 1)
            store.delete_private_repo_entries.assert_not_called()
            store.upsert_private_code_elements.assert_called_once()
            store.delete_private_repo_stale_entries.assert_called_once_with(
                "org-1",
                "repo-1",
                keep_ids=[expected_id],
            )

            method_names = [call[0] for call in store.method_calls]
            self.assertEqual(
                method_names,
                [
                    "upsert_private_code_elements",
                    "delete_private_repo_stale_entries",
                ],
            )

    def test_private_sync_no_elements_still_clears_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            file_path = repo_root / "a.py"
            file_path.write_text("# no funcs\n", encoding="utf-8")

            ctx = _runtime_ctx(repo_root)
            store = Mock()
            matcher = Mock()
            matcher.allows.return_value = True

            with patch("code_similarity_tool.index._configure_logging", return_value=Mock()), patch(
                "code_similarity_tool.index.load_runtime_context", return_value=ctx
            ), patch("code_similarity_tool.index.load_ignore_file", return_value=matcher), patch(
                "code_similarity_tool.index.iter_repo_source_files", return_value=[file_path]
            ), patch(
                "code_similarity_tool.index.extract_code_elements", return_value=[]
            ), patch(
                "code_similarity_tool.index.CodeVectorStore", return_value=store
            ), patch("code_similarity_tool.index.EmbeddingClient") as embedder_cls:
                total = sync_current_repo(action_name="index")

            self.assertEqual(total, 0)
            store.delete_private_repo_entries.assert_called_once_with("org-1", "repo-1")
            store.delete_private_repo_stale_entries.assert_not_called()
            store.upsert_private_code_elements.assert_not_called()
            embedder_cls.assert_not_called()

    def test_private_sync_indexes_unsupported_language_as_file_element(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            file_path = repo_root / "a.foolang"
            file_path.write_text("hello unsupported language", encoding="utf-8")

            ctx = _runtime_ctx(repo_root)
            store = Mock()
            matcher = Mock()
            matcher.allows.return_value = True
            embedder = Mock()
            embedder.embed_documents.return_value = [[0.1, 0.2]]

            with patch("code_similarity_tool.index._configure_logging", return_value=Mock()), patch(
                "code_similarity_tool.index.load_runtime_context", return_value=ctx
            ), patch("code_similarity_tool.index.load_ignore_file", return_value=matcher), patch(
                "code_similarity_tool.index.iter_repo_source_files", return_value=[file_path]
            ), patch(
                "code_similarity_tool.index.CodeVectorStore", return_value=store
            ), patch(
                "code_similarity_tool.index.EmbeddingClient", return_value=embedder
            ):
                total = sync_current_repo(action_name="index")

            self.assertEqual(total, 1)
            args = store.upsert_private_code_elements.call_args.args
            elements = args[0]
            self.assertEqual(elements[0]["kind"], "file")

    def test_public_index_does_not_delete_when_embedding_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ctx = _runtime_ctx(repo_root)
            store = Mock()
            embedder = Mock()
            embedder.embed_documents.side_effect = RuntimeError("embedding unavailable")

            def _clone(_url: str, _sha: str, repo_dir: Path) -> None:
                repo_dir.mkdir(parents=True, exist_ok=True)
                (repo_dir / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")

            with patch("code_similarity_tool.public_index._configure_logging", return_value=Mock()), patch(
                "code_similarity_tool.public_index.load_runtime_context", return_value=ctx
            ), patch("code_similarity_tool.public_index.resolve_remote_commit", return_value="abc123"), patch(
                "code_similarity_tool.public_index.clone_repo_at_commit", side_effect=_clone
            ), patch(
                "code_similarity_tool.public_index.detect_public_license", return_value="GPL-3.0"
            ), patch(
                "code_similarity_tool.public_index.iter_public_repo_source_files",
                side_effect=lambda repo_dir: [repo_dir / "a.py"],
            ), patch(
                "code_similarity_tool.public_index.extract_code_elements", return_value=[_sample_element()]
            ), patch(
                "code_similarity_tool.public_index.CodeVectorStore", return_value=store
            ), patch(
                "code_similarity_tool.public_index.EmbeddingClient", return_value=embedder
            ):
                with self.assertRaises(RuntimeError):
                    index_public_github_repo("https://github.com/example/demo")

            store.delete_public_source_entries.assert_not_called()
            store.delete_public_source_stale_entries.assert_not_called()
            store.upsert_public_code_elements.assert_not_called()

    def test_public_index_indexes_unsupported_language_as_file_element(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ctx = _runtime_ctx(repo_root)
            store = Mock()
            store.delete_public_source_stale_entries.return_value = 0
            embedder = Mock()
            embedder.embed_documents.return_value = [[0.1, 0.2]]

            def _clone(_url: str, _sha: str, repo_dir: Path) -> None:
                repo_dir.mkdir(parents=True, exist_ok=True)
                (repo_dir / "a.foolang").write_text("hello unsupported language", encoding="utf-8")

            with patch("code_similarity_tool.public_index._configure_logging", return_value=Mock()), patch(
                "code_similarity_tool.public_index.load_runtime_context", return_value=ctx
            ), patch("code_similarity_tool.public_index.resolve_remote_commit", return_value="abc123"), patch(
                "code_similarity_tool.public_index.clone_repo_at_commit", side_effect=_clone
            ), patch(
                "code_similarity_tool.public_index.detect_public_license", return_value="GPL-3.0"
            ), patch(
                "code_similarity_tool.public_index.iter_public_repo_source_files",
                side_effect=lambda repo_dir: [repo_dir / "a.foolang"],
            ), patch(
                "code_similarity_tool.public_index.CodeVectorStore", return_value=store
            ), patch(
                "code_similarity_tool.public_index.EmbeddingClient", return_value=embedder
            ):
                total = index_public_github_repo("https://github.com/example/demo")

            self.assertEqual(total, 1)
            args = store.upsert_public_code_elements.call_args.args
            elements = args[0]
            self.assertEqual(elements[0]["kind"], "file")

    def test_public_index_upserts_then_prunes_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ctx = _runtime_ctx(repo_root)
            store = Mock()
            store.delete_public_source_stale_entries.return_value = 2
            embedder = Mock()
            embedder.embed_documents.return_value = [[0.1, 0.2]]

            def _clone(_url: str, _sha: str, repo_dir: Path) -> None:
                repo_dir.mkdir(parents=True, exist_ok=True)
                (repo_dir / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")

            with patch("code_similarity_tool.public_index._configure_logging", return_value=Mock()), patch(
                "code_similarity_tool.public_index.load_runtime_context", return_value=ctx
            ), patch("code_similarity_tool.public_index.resolve_remote_commit", return_value="abc123"), patch(
                "code_similarity_tool.public_index.clone_repo_at_commit", side_effect=_clone
            ), patch(
                "code_similarity_tool.public_index.detect_public_license", return_value="GPL-3.0"
            ), patch(
                "code_similarity_tool.public_index.iter_public_repo_source_files",
                side_effect=lambda repo_dir: [repo_dir / "a.py"],
            ), patch(
                "code_similarity_tool.public_index.extract_code_elements", return_value=[_sample_element()]
            ), patch(
                "code_similarity_tool.public_index.CodeVectorStore", return_value=store
            ), patch(
                "code_similarity_tool.public_index.EmbeddingClient", return_value=embedder
            ):
                total = index_public_github_repo("https://github.com/example/demo")

            public_source_id = public_index._public_source_id("https://github.com/example/demo", "abc123")
            expected_id = public_index._public_element_id(public_source_id, "a.py", "h1")
            self.assertEqual(total, 1)
            store.delete_public_source_entries.assert_not_called()
            store.upsert_public_code_elements.assert_called_once()
            store.delete_public_source_stale_entries.assert_called_once_with(
                public_source_id,
                keep_ids=[expected_id],
            )

            method_names = [call[0] for call in store.method_calls]
            self.assertEqual(
                method_names,
                [
                    "upsert_public_code_elements",
                    "delete_public_source_stale_entries",
                ],
            )

    def test_public_index_no_elements_still_clears_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ctx = _runtime_ctx(repo_root)
            store = Mock()

            def _clone(_url: str, _sha: str, repo_dir: Path) -> None:
                repo_dir.mkdir(parents=True, exist_ok=True)

            with patch("code_similarity_tool.public_index._configure_logging", return_value=Mock()), patch(
                "code_similarity_tool.public_index.load_runtime_context", return_value=ctx
            ), patch("code_similarity_tool.public_index.resolve_remote_commit", return_value="abc123"), patch(
                "code_similarity_tool.public_index.clone_repo_at_commit", side_effect=_clone
            ), patch(
                "code_similarity_tool.public_index.detect_public_license", return_value="GPL-3.0"
            ), patch(
                "code_similarity_tool.public_index.iter_public_repo_source_files", return_value=[]
            ), patch(
                "code_similarity_tool.public_index.CodeVectorStore", return_value=store
            ), patch("code_similarity_tool.public_index.EmbeddingClient") as embedder_cls:
                total = index_public_github_repo("https://github.com/example/demo")

            public_source_id = public_index._public_source_id("https://github.com/example/demo", "abc123")
            self.assertEqual(total, 0)
            store.delete_public_source_entries.assert_called_once_with(public_source_id)
            store.delete_public_source_stale_entries.assert_not_called()
            store.upsert_public_code_elements.assert_not_called()
            embedder_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
