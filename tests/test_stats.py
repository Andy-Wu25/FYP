from __future__ import annotations

import unittest
from collections import Counter

from code_similarity_tool import stats


class StatsRenderingTest(unittest.TestCase):
    def test_render_index_block_shows_all_repository_rows(self) -> None:
        repos = Counter({f"owner/repo{i}": i for i in range(1, 12)})
        rendered = stats._render_index_block(
            {
                "total": sum(repos.values()),
                "repos": repos,
                "files": 11,
                "kinds": Counter({"function": 66}),
                "langs": Counter({"Python": 66}),
                "licenses": Counter(),
                "repo_commits": {},
            }
        )

        repo_header_index = rendered.index(stats._sub_header("Repositories"))
        repo_rows = rendered[repo_header_index + 1 : repo_header_index + 1 + len(repos)]

        self.assertEqual(len(repo_rows), len(repos))
        for name in repos:
            self.assertTrue(any(name in row for row in repo_rows), name)


if __name__ == "__main__":
    unittest.main()
