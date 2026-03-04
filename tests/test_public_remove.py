from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from code_similarity_tool.clients import CodeVectorStore
from code_similarity_tool.public_remove import remove_public_github_repo


class PublicRemoveStoreTest(unittest.TestCase):
    def test_delete_public_repo_entries_uses_source_url_filter(self) -> None:
        store = CodeVectorStore.__new__(CodeVectorStore)
        store.public_collection = Mock()
        store.public_collection.get.return_value = {"ids": ["id1", "id2"]}

        removed = store.delete_public_repo_entries("https://github.com/acme/demo")

        self.assertEqual(removed, 2)
        store.public_collection.get.assert_called_once_with(
            limit=5000,
            offset=0,
            where={"source_url": "https://github.com/acme/demo"},
        )
        store.public_collection.delete.assert_called_once_with(ids=["id1", "id2"])


class PublicRemoveCommandTest(unittest.TestCase):
    def test_remove_public_github_repo_canonicalizes_url_and_deletes_all_versions(self) -> None:
        ctx = SimpleNamespace(
            db_path="/tmp/code-sim",
            private_collection_name="private",
            public_collection_name="public",
            metric="cosine",
        )
        store = Mock()
        store.delete_public_repo_entries.return_value = 7

        with patch("code_similarity_tool.public_remove._configure_logging", return_value=Mock()) as log_cfg, patch(
            "code_similarity_tool.public_remove.load_runtime_context",
            return_value=ctx,
        ), patch(
            "code_similarity_tool.public_remove.CodeVectorStore",
            return_value=store,
        ) as store_cls:
            removed = remove_public_github_repo("https://github.com/acme/demo.git")

        self.assertEqual(removed, 7)
        log_cfg.assert_called_once()
        store_cls.assert_called_once_with(
            path="/tmp/code-sim",
            private_collection_name="private",
            public_collection_name="public",
            metric="cosine",
        )
        store.delete_public_repo_entries.assert_called_once_with("https://github.com/acme/demo")


if __name__ == "__main__":
    unittest.main()
