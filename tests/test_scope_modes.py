from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from code_similarity_tool.checking.check_utils import add_scope_argument, query_elements_from_args, validate_scope_args
from code_similarity_tool.core.runtime import RuntimeContext


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


class ScopeModesTest(unittest.TestCase):
    def test_files_scope_requires_paths(self) -> None:
        parser = argparse.ArgumentParser(prog="scope-test")
        parser.add_argument("paths", nargs="*")
        add_scope_argument(parser)
        args = parser.parse_args(["--scope", "files"])
        with self.assertRaises(SystemExit):
            validate_scope_args(parser, args)

    def test_repo_scope_rejects_paths(self) -> None:
        parser = argparse.ArgumentParser(prog="scope-test")
        parser.add_argument("paths", nargs="*")
        add_scope_argument(parser)
        args = parser.parse_args(["--scope", "repo", "a.py"])
        with self.assertRaises(SystemExit):
            validate_scope_args(parser, args)

    def test_staged_scope_intersects_requested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ctx = _runtime_ctx(repo_root)
            with patch("code_similarity_tool.checking.check_utils.load_runtime_context", return_value=ctx), patch(
                "code_similarity_tool.checking.check_utils.staged_added_modified_renamed", return_value=["a.py", "b.py"]
            ), patch("code_similarity_tool.checking.check_utils.resolve_optional_paths", return_value=["b.py", "c.py"]), patch(
                "code_similarity_tool.checking.check_utils.collect_staged_query_elements", return_value=[]
            ):
                _, rel_paths, _ = query_elements_from_args(["ignored"], scope="staged")

        self.assertEqual(rel_paths, ["b.py"])

    def test_files_scope_uses_resolved_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ctx = _runtime_ctx(repo_root)
            with patch("code_similarity_tool.checking.check_utils.load_runtime_context", return_value=ctx), patch(
                "code_similarity_tool.checking.check_utils.resolve_optional_paths", return_value=["b.py", "a.py", "a.py"]
            ), patch("code_similarity_tool.checking.check_utils.collect_full_file_query_elements", return_value=[]):
                _, rel_paths, _ = query_elements_from_args(["ignored"], scope="files")

        self.assertEqual(rel_paths, ["a.py", "b.py"])

    def test_repo_scope_uses_repository_file_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            (repo_root / "sub").mkdir(parents=True, exist_ok=True)
            (repo_root / "sub" / "b.ts").write_text("export const x = 1;\n", encoding="utf-8")

            ctx = _runtime_ctx(repo_root)
            with patch("code_similarity_tool.checking.check_utils.load_runtime_context", return_value=ctx), patch(
                "code_similarity_tool.checking.check_utils.load_ignore_file", return_value=Mock()
            ), patch(
                "code_similarity_tool.checking.check_utils.iter_repo_source_files",
                return_value=[repo_root / "a.py", repo_root / "sub" / "b.ts"],
            ), patch("code_similarity_tool.checking.check_utils.collect_full_file_query_elements", return_value=[]):
                _, rel_paths, _ = query_elements_from_args([], scope="repo")

        self.assertEqual(rel_paths, ["a.py", "sub/b.ts"])


if __name__ == "__main__":
    unittest.main()
