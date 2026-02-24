from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from code_similarity_tool.check_public import main as check_public_main
from code_similarity_tool.check_public import _normalize_license_filters
from code_similarity_tool.check_public import _suggest_license_keywords
from code_similarity_tool.clients import CodeVectorStore


class PublicLicenseFilterNormalizationTest(unittest.TestCase):
    def test_normalizes_repeatable_and_comma_separated_values(self) -> None:
        self.assertEqual(
            _normalize_license_filters(["mit", " apache-2.0, gpl-3.0 ", "MIT"]),
            ["MIT", "APACHE-2.0", "GPL-3.0"],
        )

    def test_suggests_similar_license_for_typo(self) -> None:
        suggestions = _suggest_license_keywords("mitt")
        self.assertIn("MIT", suggestions)


class PublicCollectionQueryFilterTest(unittest.TestCase):
    def test_query_public_without_license_filter(self) -> None:
        store = CodeVectorStore.__new__(CodeVectorStore)
        store.public_collection = Mock()
        store.public_collection.query.return_value = {"ids": [[]]}

        store.query_public_by_embedding([0.1, 0.2], n_results=7)

        store.public_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=7,
        )

    def test_query_public_with_license_filter(self) -> None:
        store = CodeVectorStore.__new__(CodeVectorStore)
        store.public_collection = Mock()
        store.public_collection.query.return_value = {"ids": [[]]}

        store.query_public_by_embedding([0.1, 0.2], n_results=7, licenses=["mit", " GPL-3.0 "])

        store.public_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=7,
            where={"license": {"$in": ["GPL-3.0", "MIT"]}},
        )


class CheckPublicCliLicenseFilterTest(unittest.TestCase):
    def test_license_works_with_scope_and_max_distance(self) -> None:
        ctx = SimpleNamespace(
            db_path=Path("/tmp/code-sim-db"),
            private_collection_name="private",
            public_collection_name="public",
            metric="cosine",
        )
        query_elements = [
            (
                "src/a.py",
                {
                    "name": "foo",
                    "kind": "function",
                    "start_line": 1,
                    "end_line": 2,
                    "text": "def foo():\n    return 1\n",
                    "hash": "h1",
                },
            )
        ]

        embedder = Mock()
        embedder.embed_documents.return_value = [[0.3, 0.4]]

        store = Mock()
        store.query_public_by_embedding.return_value = {
            "ids": [["id1"]],
            "distances": [[0.2]],
            "metadatas": [[
                {
                    "repo_name": "public/repo",
                    "file_path": "src/other.py",
                    "function_name": "bar",
                    "license": "MIT",
                    "source_commit": "abc123",
                    "source_url": "https://github.com/public/repo",
                    "source_file_commit": "abc123",
                }
            ]],
        }

        with patch.object(
            sys,
            "argv",
            [
                "code-sim-check-public",
                "--license",
                "mit",
                "--scope",
                "files",
                "src/a.py",
                "--max-distance",
                "0.5",
            ],
        ), patch("code_similarity_tool.check_public.configure_logging", return_value=Mock()), patch(
            "code_similarity_tool.check_public.query_elements_from_args",
            return_value=(ctx, ["src/a.py"], query_elements),
        ), patch(
            "code_similarity_tool.check_public.EmbeddingClient",
            return_value=embedder,
        ), patch(
            "code_similarity_tool.check_public.CodeVectorStore",
            return_value=store,
        ):
            with redirect_stdout(io.StringIO()):
                check_public_main()

        store.query_public_by_embedding.assert_called_once_with(
            [0.3, 0.4],
            n_results=20,
            licenses=["MIT"],
        )

    def test_prints_license_suggestion_on_no_hits_for_unknown_keyword(self) -> None:
        ctx = SimpleNamespace(
            db_path=Path("/tmp/code-sim-db"),
            private_collection_name="private",
            public_collection_name="public",
            metric="cosine",
        )
        query_elements = [
            (
                "src/a.py",
                {
                    "name": "foo",
                    "kind": "function",
                    "start_line": 1,
                    "end_line": 2,
                    "text": "def foo():\n    return 1\n",
                    "hash": "h1",
                },
            )
        ]

        embedder = Mock()
        embedder.embed_documents.return_value = [[0.3, 0.4]]

        store = Mock()
        store.query_public_by_embedding.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            [
                "code-sim-check-public",
                "--license",
                "mitt",
                "--scope",
                "files",
                "src/a.py",
                "--max-distance",
                "0.5",
            ],
        ), patch("code_similarity_tool.check_public.configure_logging", return_value=Mock()), patch(
            "code_similarity_tool.check_public.query_elements_from_args",
            return_value=(ctx, ["src/a.py"], query_elements),
        ), patch(
            "code_similarity_tool.check_public.EmbeddingClient",
            return_value=embedder,
        ), patch(
            "code_similarity_tool.check_public.CodeVectorStore",
            return_value=store,
        ), redirect_stdout(output):
            check_public_main()

        rendered = output.getvalue()
        self.assertIn("License filter 'MITT' not recognized.", rendered)
        self.assertIn("Did you mean: MIT?", rendered)

if __name__ == "__main__":
    unittest.main()
