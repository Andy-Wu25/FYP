from __future__ import annotations

import unittest

from code_similarity_tool.cli import _hook_block


class CliTest(unittest.TestCase):
    def test_pre_push_hook_runs_update(self) -> None:
        block = _hook_block("pre-push")
        self.assertIn("code-sim-update", block)

    def test_pre_commit_hook_runs_check(self) -> None:
        block = _hook_block("pre-commit")
        self.assertIn("code-sim-check", block)


if __name__ == "__main__":
    unittest.main()
