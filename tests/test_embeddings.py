from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

import requests

from code_similarity_tool.infra.embeddings import EmbeddingClient


def _make_client(
    input_prefix: str = "",
) -> EmbeddingClient:
    """Construct an EmbeddingClient without hitting the network."""
    env = {
        "CODE_SIM_EMBEDDINGS_BACKEND": "vllm",
        "VLLM_VERIFY_MODELS": "0",
        "VLLM_INPUT_PREFIX": input_prefix,
    }
    with patch.dict(os.environ, env, clear=False):
        return EmbeddingClient()


class EmbeddingBatchingTest(unittest.TestCase):
    def test_short_text_passes_through_unchanged(self) -> None:
        client = _make_client()
        short = "x" * 50
        with patch.object(client, "_embed_batch", return_value=[[0.1, 0.2]]) as mock_batch:
            client.embed_documents([short])
        mock_batch.assert_called_once_with([short])

    def test_empty_texts_returns_empty_without_calling_batch(self) -> None:
        client = _make_client()
        with patch.object(client, "_embed_batch") as mock_batch:
            result = client.embed_documents([])
        self.assertEqual(result, [])
        mock_batch.assert_not_called()


class EmbeddingInputPrefixTest(unittest.TestCase):
    def test_input_prefix_env_var_is_stored(self) -> None:
        client = _make_client(input_prefix="\n")
        self.assertEqual(client.input_prefix, "\n")

    def test_empty_prefix_stored_as_empty(self) -> None:
        client = _make_client()
        self.assertEqual(client.input_prefix, "")

    def test_prefix_prepended_to_each_text(self) -> None:
        client = _make_client(input_prefix="\n")
        texts = ["import os\n", "def f():\n    pass\n"]
        with patch.object(client, "_embed_batch", return_value=[[0.1], [0.2]]) as mock_batch:
            client.embed_documents(texts)
        sent = mock_batch.call_args.args[0]
        self.assertEqual(sent[0], "\nimport os\n")
        self.assertEqual(sent[1], "\ndef f():\n    pass\n")

    def test_no_prefix_leaves_texts_untouched(self) -> None:
        client = _make_client(input_prefix="")
        text = "import os\n"
        with patch.object(client, "_embed_batch", return_value=[[0.1]]) as mock_batch:
            client.embed_documents([text])
        sent = mock_batch.call_args.args[0]
        self.assertEqual(sent[0], text)

    def test_prefix_applied_to_all_texts_in_batch(self) -> None:
        client = _make_client(input_prefix="passage: ")
        texts = ["alpha", "beta", "gamma"]
        with patch.object(client, "_embed_batch", return_value=[[0.1], [0.2], [0.3]]) as mock_batch:
            client.embed_documents(texts)
        sent = mock_batch.call_args.args[0]
        self.assertEqual(sent, ["passage: alpha", "passage: beta", "passage: gamma"])

    def test_return_count_unchanged_with_prefix(self) -> None:
        client = _make_client(input_prefix="\n")
        texts = ["import os\n", "import sys\n", "x = 1\n"]
        with patch.object(client, "_embed_batch", return_value=[[0.1], [0.2], [0.3]]):
            result = client.embed_documents(texts)
        self.assertEqual(len(result), 3)


class EmbeddingErrorReportingTest(unittest.TestCase):
    def test_runtime_error_includes_underlying_http_error_details(self) -> None:
        client = _make_client()
        response = Mock()
        response.status_code = 500
        response.reason = "Internal Server Error"
        response.text = "backend exploded"
        error = requests.HTTPError("500 Server Error")
        error.response = response

        with patch.object(client, "_embed_batch", side_effect=error):
            with self.assertRaises(RuntimeError) as cm:
                client.embed_documents(["def f():\n    return 1\n"], labels=["src/a.py"])

        message = str(cm.exception)
        self.assertIn("Embedding batch 1-1 of 1 failed.", message)
        self.assertIn("labels=[src/a.py]", message)
        self.assertIn("Cause: HTTPError: 500 Server Error", message)
        self.assertIn("status=500 Internal Server Error", message)
        self.assertIn("body=backend exploded", message)


if __name__ == "__main__":
    unittest.main()
