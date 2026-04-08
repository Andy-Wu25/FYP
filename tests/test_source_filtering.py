from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from code_similarity_tool.core.runtime import iter_repo_source_files
from code_similarity_tool.core.source_filtering import is_noise_source_path


class _AllowAllMatcher:
    def allows(self, _path: Path, is_dir: bool = False) -> bool:
        _ = is_dir
        return True


class SourceFilteringTest(unittest.TestCase):
    def test_noise_paths_are_excluded(self) -> None:
        self.assertTrue(is_noise_source_path(Path("README.md")))
        self.assertTrue(is_noise_source_path(Path("pyproject.toml")))
        self.assertTrue(is_noise_source_path(Path("docs/guide.md")))
        self.assertTrue(is_noise_source_path(Path("package-lock.json")))
        self.assertTrue(is_noise_source_path(Path("dist/app.min.js")))
        self.assertTrue(is_noise_source_path(Path(".env.local")))
        self.assertTrue(is_noise_source_path(Path("assets/logo.png")))

    def test_unknown_code_like_path_not_excluded(self) -> None:
        self.assertFalse(is_noise_source_path(Path("src/main.foolang")))
        self.assertFalse(is_noise_source_path(Path("src/engine.abc")))

    def test_iter_repo_source_files_skips_noise_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src").mkdir(parents=True, exist_ok=True)
            (repo_root / "docs").mkdir(parents=True, exist_ok=True)
            (repo_root / "dist").mkdir(parents=True, exist_ok=True)
            (repo_root / "src" / "main.foolang").write_text("hello", encoding="utf-8")
            (repo_root / "docs" / "guide.md").write_text("text", encoding="utf-8")
            (repo_root / "dist" / "bundle.min.js").write_text("text", encoding="utf-8")
            (repo_root / "README.md").write_text("text", encoding="utf-8")

            files = iter_repo_source_files(repo_root, _AllowAllMatcher())
            rels = [str(p.relative_to(repo_root)) for p in files]

        self.assertEqual(rels, ["src/main.foolang"])


if __name__ == "__main__":
    unittest.main()
