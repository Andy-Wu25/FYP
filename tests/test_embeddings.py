from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

import requests

from code_similarity_tool.embeddings import EmbeddingClient


def _make_client(
    max_chars: int = 0,
    input_prefix: str = "",
    long_text_mode: str = "chunk",
    chunk_overlap: int = 512,
) -> EmbeddingClient:
    """Construct an EmbeddingClient without hitting the network."""
    env = {
        "CODE_SIM_EMBEDDINGS_BACKEND": "vllm",
        "VLLM_VERIFY_MODELS": "0",
        "VLLM_MAX_CHARS": str(max_chars),
        "VLLM_LONG_TEXT_MODE": long_text_mode,
        "VLLM_CHUNK_OVERLAP": str(chunk_overlap),
        "VLLM_INPUT_PREFIX": input_prefix,
    }
    with patch.dict(os.environ, env, clear=False):
        return EmbeddingClient()


class EmbeddingLongTextHandlingTest(unittest.TestCase):
    def test_max_chars_env_var_is_stored(self) -> None:
        client = _make_client(max_chars=12345)
        self.assertEqual(client.max_chars, 12345)

    def test_max_chars_zero_stored_as_zero(self) -> None:
        client = _make_client(max_chars=0)
        self.assertEqual(client.max_chars, 0)

    def test_chunk_mode_is_default(self) -> None:
        client = _make_client()
        self.assertEqual(client.long_text_mode, "chunk")

    def test_short_text_passes_through_unchanged(self) -> None:
        client = _make_client(max_chars=100)
        short = "x" * 50
        with patch.object(client, "_embed_batch", return_value=[[0.1, 0.2]]) as mock_batch:
            client.embed_documents([short])
        mock_batch.assert_called_once_with([short])

    def test_long_text_is_split_into_chunks_by_default(self) -> None:
        client = _make_client(max_chars=10, chunk_overlap=0)
        long_text = "abcdefghijklmnopqrstuvwxyz"
        with patch.object(client, "_embed_batch", return_value=[[0.1], [0.2], [0.3]]) as mock_batch:
            client.embed_documents([long_text])
        sent = mock_batch.call_args.args[0]
        self.assertEqual(sent, ["abcdefghij", "klmnopqrst", "uvwxyz"])

    def test_chunked_embeddings_are_averaged_back_to_one_vector(self) -> None:
        client = _make_client(max_chars=10, chunk_overlap=0)
        long_text = "abcdefghijklmnopqrstuvwxyz"
        with patch.object(client, "_embed_batch", return_value=[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]):
            result = client.embed_documents([long_text])
        self.assertEqual(result, [[2.0, 5.0]])

    def test_truncate_mode_keeps_previous_behavior(self) -> None:
        client = _make_client(max_chars=10, long_text_mode="truncate", chunk_overlap=0)
        long_text = "a" * 100
        with patch.object(client, "_embed_batch", return_value=[[0.1, 0.2]]) as mock_batch:
            client.embed_documents([long_text])
        sent_text = mock_batch.call_args.args[0][0]
        self.assertEqual(len(sent_text), 10)
        self.assertEqual(sent_text, "a" * 10)

    def test_max_chars_zero_disables_truncation(self) -> None:
        client = _make_client(max_chars=0)
        long_text = "a" * 100_000
        with patch.object(client, "_embed_batch", return_value=[[0.1, 0.2]]) as mock_batch:
            client.embed_documents([long_text])
        sent_text = mock_batch.call_args.args[0][0]
        self.assertEqual(len(sent_text), 100_000)

    def test_chunking_logs_warning_with_provided_label(self) -> None:
        client = _make_client(max_chars=10, chunk_overlap=0)
        with patch.object(client, "_embed_batch", return_value=[[0.1, 0.2]]):
            with self.assertLogs("code_similarity_tool.embeddings", level="WARNING") as cm:
                client.embed_documents(["a" * 15], labels=["src/configure_db.py"])
        self.assertTrue(any("src/configure_db.py" in line for line in cm.output))
        self.assertTrue(any("splitting into" in line for line in cm.output))

    def test_truncation_logs_warning_with_index_when_in_truncate_mode(self) -> None:
        client = _make_client(max_chars=10, long_text_mode="truncate", chunk_overlap=0)
        with patch.object(client, "_embed_batch", return_value=[[0.1, 0.2]]):
            with self.assertLogs("code_similarity_tool.embeddings", level="WARNING") as cm:
                client.embed_documents(["a" * 100])
        self.assertTrue(any("index 0" in line for line in cm.output))
        self.assertTrue(any("truncating to 10 chars" in line for line in cm.output))

    def test_only_long_texts_are_split_in_mixed_batch(self) -> None:
        client = _make_client(max_chars=20, chunk_overlap=0)
        short = "x" * 10
        long = "y" * 45
        with patch.object(client, "_embed_batch", return_value=[[0.1], [0.2], [0.3], [0.4]]) as mock_batch:
            client.embed_documents([short, long])
        sent = mock_batch.call_args.args[0]
        self.assertEqual(sent[0], short)
        self.assertEqual(sent[1:], ["y" * 20, "y" * 20, "y" * 5])

    def test_no_warning_logged_for_short_text(self) -> None:
        client = _make_client(max_chars=100)
        short = "x" * 50
        with patch.object(client, "_embed_batch", return_value=[[0.1, 0.2]]):
            try:
                with self.assertLogs("code_similarity_tool.embeddings", level="WARNING"):
                    client.embed_documents([short])
                self.fail("No truncation warning should have been emitted for short text")
            except AssertionError as exc:
                if "no logs" not in str(exc).lower() and "0 log message" not in str(exc).lower():
                    raise

    def test_chunking_uses_correct_label_by_position(self) -> None:
        client = _make_client(max_chars=10, chunk_overlap=0)
        texts = ["short", "a" * 100, "also short"]
        labels = ["first.py", "second.py", "third.py"]
        with patch.object(client, "_embed_batch", return_value=[[0.1]] * 12):
            with self.assertLogs("code_similarity_tool.embeddings", level="WARNING") as cm:
                client.embed_documents(texts, labels=labels)
        self.assertTrue(any("second.py" in line for line in cm.output))
        self.assertFalse(any("first.py" in line for line in cm.output))
        self.assertFalse(any("third.py" in line for line in cm.output))

    def test_return_count_matches_input_count_after_chunking(self) -> None:
        client = _make_client(max_chars=10, chunk_overlap=0)
        texts = ["a" * 100, "b" * 200, "c" * 300]
        fake_vectors = [[0.1]] * 60
        with patch.object(client, "_embed_batch", return_value=fake_vectors):
            result = client.embed_documents(texts)
        self.assertEqual(len(result), 3)

    def test_empty_texts_returns_empty_without_truncation(self) -> None:
        client = _make_client(max_chars=10)
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

    def test_prefix_applied_after_chunking(self) -> None:
        client = _make_client(max_chars=10, input_prefix="\n", chunk_overlap=0)
        long_text = "a" * 100
        with patch.object(client, "_embed_batch", return_value=[[0.1]] * 10) as mock_batch:
            client.embed_documents([long_text])
        sent = mock_batch.call_args.args[0][0]
        self.assertEqual(sent, "\n" + "a" * 10)

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
