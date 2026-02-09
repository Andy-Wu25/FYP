from __future__ import annotations

import unittest

from code_similarity_tool.code_similarity import _extract_hits


class ExtractHitsTest(unittest.TestCase):
    def test_skips_exact_self_match(self) -> None:
        results = {
            "ids": [["self", "other"]],
            "distances": [[0.0, 0.2]],
            "metadatas": [[
                {
                    "repo_id": "repo-1",
                    "file_path": "a.py",
                    "content_hash": "h1",
                    "repo_name": "repo-a",
                    "function_name": "foo",
                    "start_line": 1,
                    "end_line": 2,
                },
                {
                    "repo_id": "repo-2",
                    "file_path": "b.py",
                    "content_hash": "h2",
                    "repo_name": "repo-b",
                    "function_name": "bar",
                    "start_line": 3,
                    "end_line": 4,
                },
            ]],
        }

        hits = _extract_hits(
            results,
            top_k=5,
            max_distance=None,
            query_rel="a.py",
            query_hash="h1",
            query_repo_id="repo-1",
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][1]["repo_id"], "repo-2")

    def test_respects_max_distance(self) -> None:
        results = {
            "ids": [["x", "y"]],
            "distances": [[0.1, 0.9]],
            "metadatas": [[
                {"repo_id": "r2", "file_path": "x.py", "content_hash": "x"},
                {"repo_id": "r3", "file_path": "y.py", "content_hash": "y"},
            ]],
        }

        hits = _extract_hits(
            results,
            top_k=5,
            max_distance=0.5,
            query_rel="q.py",
            query_hash="q",
            query_repo_id="r1",
        )

        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0][0], 0.1)


if __name__ == "__main__":
    unittest.main()
