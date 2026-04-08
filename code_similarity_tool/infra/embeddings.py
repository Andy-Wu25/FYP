from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import re

import requests


log = logging.getLogger(__name__)

_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?(\d+) input tokens",
    re.IGNORECASE | re.DOTALL,
)


def _format_exception_details(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]

    response = getattr(exc, "response", None)
    if response is None:
        return " | ".join(parts)

    status_code = getattr(response, "status_code", None)
    reason = getattr(response, "reason", None)
    if status_code is not None:
        status = f"status={status_code}"
        if reason:
            status += f" {reason}"
        parts.append(status)

    try:
        body = str(getattr(response, "text", "") or "").strip()
    except Exception:  # noqa: BLE001
        body = ""

    if body:
        compact = " ".join(body.split())
        if len(compact) > 300:
            compact = compact[:297] + "..."
        parts.append(f"body={compact}")

    return " | ".join(parts)


class EmbeddingClient:
    """vLLM embeddings client (OpenAI-compatible)."""

    def __init__(
        self,
        *,
        truncate_tokens: int | None = None,
    ):
        backend = os.getenv("CODE_SIM_EMBEDDINGS_BACKEND", "vllm").strip().lower()
        if backend != "vllm":
            raise ValueError(
                f"Unsupported embeddings backend '{backend}'. "
                "Set CODE_SIM_EMBEDDINGS_BACKEND=vllm."
            )

        self.base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        self.api_key = os.getenv("VLLM_API_KEY", "").strip()
        self.model = os.getenv("VLLM_MODEL", "Octen/Octen-Embedding-8B").strip()
        self.timeout_s = float(os.getenv("VLLM_TIMEOUT_S", "120"))
        self.verify_models = os.getenv("VLLM_VERIFY_MODELS", "1").strip().lower() not in {
            "0",
            "false",
        }
        self.batch_size = max(1, int(os.getenv("VLLM_BATCH_SIZE", "64")))
        self.chunk_overlap = max(0, int(os.getenv("VLLM_CHUNK_OVERLAP", "512")))
        self.input_prefix = os.getenv("VLLM_INPUT_PREFIX", "\n")
        self.truncate_tokens = max(0, truncate_tokens if truncate_tokens is not None else int(os.getenv("VLLM_TRUNCATE_TOKENS", "0")))

        log.info(
            "Embeddings backend=vllm base_url=%s model=%s batch=%d truncate=%s timeout=%ds auth=%s",
            self.base_url,
            self.model,
            self.batch_size,
            f"{self.truncate_tokens} tokens" if self.truncate_tokens > 0 else "off (auto-chunk on overflow)",
            int(self.timeout_s),
            "set" if self.api_key else "none",
        )
        if self.input_prefix:
            log.info("Using input prefix %r for all embedding inputs.", self.input_prefix)

        if self.verify_models:
            self._log_available_models()

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _log_available_models(self) -> None:
        url = f"{self.base_url}/v1/models"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=5.0)
            resp.raise_for_status()
            payload = resp.json()
            model_ids = [m.get("id") for m in payload.get("data", []) if isinstance(m, dict)]
            if model_ids:
                log.info("vLLM server reports models: %s", ", ".join(model_ids))
                if self.model not in model_ids:
                    log.warning("Requested model '%s' not present in /v1/models.", self.model)
            else:
                log.warning("vLLM /v1/models returned no model ids.")
        except Exception as exc:  # noqa: BLE001
            log.warning("Unable to query %s (%s)", url, exc)

    @staticmethod
    def _parse_context_length_error(resp: requests.Response) -> tuple[int, int] | None:
        """Parse a 400 response for context-length info.

        Returns (max_tokens, input_tokens) or None if not a context-length error.
        """
        try:
            body = resp.json()
            msg = body.get("error", {}).get("message", "")
        except Exception:
            msg = getattr(resp, "text", "") or ""
        m = _CONTEXT_LENGTH_RE.search(msg)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/v1/embeddings"
        payload: Dict[str, Any] = {"model": self.model, "input": texts}
        if self.truncate_tokens > 0:
            payload["truncate_prompt_tokens"] = self.truncate_tokens

        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        if resp.status_code == 401:
            raise RuntimeError(
                "vLLM returned 401 Unauthorized. Set VLLM_API_KEY to match server --api-key."
            )
        if resp.status_code == 400:
            ctx_info = self._parse_context_length_error(resp)
            if ctx_info is not None:
                max_tokens = ctx_info[0]
                log.warning(
                    "Batch hit context length limit (%d tokens > %d limit); "
                    "retrying %d text(s) individually with auto-truncation.",
                    ctx_info[1], max_tokens, len(texts),
                )
                return [self._embed_single_chunked(t, max_tokens) for t in texts]
        resp.raise_for_status()

        data = resp.json()
        rows = data.get("data", [])
        if not isinstance(rows, list):
            raise RuntimeError("Unexpected embeddings response: 'data' is not a list.")

        rows = sorted(rows, key=lambda row: row.get("index", 0))
        embeddings = [row.get("embedding") for row in rows if isinstance(row, dict) and "embedding" in row]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Unexpected embeddings response shape: got {len(embeddings)} vectors for {len(texts)} inputs."
            )
        return embeddings

    def _embed_single_chunked(self, text: str, max_tokens: int) -> List[float]:
        """Embed a single text, auto-chunking if it exceeds the model's context limit."""
        url = f"{self.base_url}/v1/embeddings"
        resp = requests.post(
            url, headers=self._headers(),
            json={"model": self.model, "input": [text]},
            timeout=self.timeout_s,
        )
        if resp.status_code == 400:
            ctx_info = self._parse_context_length_error(resp)
            if ctx_info is not None:
                # Estimate safe char limit from token ratio
                safe_chars = max(1, int(len(text) * max_tokens / ctx_info[1] * 0.85))
                # Chunk with overlap, same logic as _split_long_text
                step = max(1, safe_chars - self.chunk_overlap)
                chunks: List[str] = []
                start = 0
                while start < len(text):
                    end = min(len(text), start + safe_chars)
                    chunks.append(text[start:end])
                    if end >= len(text):
                        break
                    start += step
                log.warning(
                    "Auto-chunking oversized input (%d chars, ~%d tokens) "
                    "into %d chunks of ~%d chars with overlap %d "
                    "(model limit: %d tokens).",
                    len(text), ctx_info[1], len(chunks), safe_chars,
                    self.chunk_overlap, max_tokens,
                )
                chunk_vectors: List[List[float]] = []
                for chunk in chunks:
                    vec = self._embed_one_with_retry(url, chunk, max_tokens)
                    chunk_vectors.append(vec)
                return self._average_vectors(chunk_vectors)
        resp.raise_for_status()
        return self._parse_single_embedding(resp)

    def _embed_one_with_retry(self, url: str, text: str, max_tokens: int) -> List[float]:
        """Embed a single text, progressively truncating if it still exceeds the limit."""
        for attempt in range(3):
            resp = requests.post(
                url, headers=self._headers(),
                json={"model": self.model, "input": [text]},
                timeout=self.timeout_s,
            )
            if resp.status_code != 400:
                resp.raise_for_status()
                return self._parse_single_embedding(resp)
            ctx_info = self._parse_context_length_error(resp)
            if ctx_info is None:
                resp.raise_for_status()  # not a context-length error — raise
            # Shrink proportionally to fit
            ratio = max_tokens / ctx_info[1] * 0.80
            new_len = max(1, int(len(text) * ratio))
            log.warning(
                "Chunk still too long (%d chars, ~%d tokens); "
                "truncating to %d chars (attempt %d).",
                len(text), ctx_info[1], new_len, attempt + 1,
            )
            text = text[:new_len]
        # Final attempt after truncation
        resp = requests.post(
            url, headers=self._headers(),
            json={"model": self.model, "input": [text]},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return self._parse_single_embedding(resp)

    @staticmethod
    def _parse_single_embedding(resp: requests.Response) -> List[float]:
        data = resp.json()
        rows = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
        embeddings = [r["embedding"] for r in rows if isinstance(r, dict) and "embedding" in r]
        if len(embeddings) != 1:
            raise RuntimeError(f"Expected 1 embedding, got {len(embeddings)}")
        return embeddings[0]

    @staticmethod
    def _average_vectors(vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []
        if len(vectors) == 1:
            return vectors[0]

        width = len(vectors[0])
        if any(len(vec) != width for vec in vectors):
            raise RuntimeError("Cannot average embeddings with mismatched vector sizes.")

        totals = [0.0] * width
        for vec in vectors:
            for idx, value in enumerate(vec):
                totals[idx] += float(value)
        count = float(len(vectors))
        return [value / count for value in totals]

    def embed_documents(self, texts: List[str], *, labels: Optional[List[str]] = None) -> List[List[float]]:
        if not texts:
            return []
        if labels is not None and len(labels) != len(texts):
            raise ValueError("labels length must match texts length.")

        if self.input_prefix:
            texts = [self.input_prefix + t for t in texts]

        if labels is None:
            labels = [f"index {i}" for i in range(len(texts))]

        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            label_batch = labels[i : i + len(batch)]
            files_in_batch = dict.fromkeys(
                lb.split("file=", 1)[1].split(" ", 1)[0]
                for lb in label_batch
                if "file=" in lb
            )
            if files_in_batch:
                names = [f.rsplit("/", 1)[-1] for f in files_in_batch]
                file_hint = " files=[" + ", ".join(names) + "]"
            else:
                file_hint = ""
            log.info("Embedding batch %d-%d of %d%s", i + 1, i + len(batch), len(texts), file_hint)
            try:
                all_vectors.extend(self._embed_batch(batch))
            except Exception as exc:  # noqa: BLE001
                preview = "; ".join(label_batch[:3])
                if len(label_batch) > 3:
                    preview += "; ..."
                cause = _format_exception_details(exc)
                raise RuntimeError(
                    f"Embedding batch {i + 1}-{i + len(batch)} of {len(texts)} failed. "
                    f"labels=[{preview}] Cause: {cause}"
                ) from exc
        return all_vectors


def add_embedding_args(parser: argparse.ArgumentParser) -> None:
    """Add --truncate flag to an argument parser."""
    group = parser.add_argument_group("embedding options")
    group.add_argument(
        "--truncate", type=int, nargs="?", const=32768, default=None, metavar="TOKENS",
        help="Truncate inputs exceeding the model's context window server-side. "
        "Pass without a value for 32768 tokens, or specify a custom limit. "
        "By default (without this flag), oversized inputs are auto-chunked and averaged.",
    )


def _config_path(db_path: Path, collection_type: str) -> Path:
    return db_path / f"_embedding_config_{collection_type}.json"


def _config_dict(client: EmbeddingClient) -> Dict[str, Any]:
    return {
        "truncate_tokens": client.truncate_tokens,
        "model": client.model,
    }


def save_embedding_config(db_path: Path, client: EmbeddingClient, collection_type: str) -> None:
    """Save the embedding config used at index time for a collection.

    The first index sets the canonical config.  Subsequent indexes that
    use a different config will warn but NOT overwrite — the canonical
    config stays so that eval queries match the majority of vectors.
    To change the canonical config, delete the collection and re-index.
    """
    path = _config_path(db_path, collection_type)
    new_cfg = _config_dict(client)

    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = None

        if existing and existing != new_cfg:
            log.warning(
                "Embedding config for %s collection differs from canonical: "
                "canonical=%s current=%s. "
                "Vectors in this collection are now mixed — distances may be "
                "unreliable. Consider re-indexing the entire collection with "
                "consistent settings.",
                collection_type,
                existing,
                new_cfg,
            )
            return  # keep the canonical config

    path.write_text(json.dumps(new_cfg, indent=2), encoding="utf-8")


def load_embedding_config(db_path: Path, collection_type: str) -> Optional[Dict[str, Any]]:
    """Load the embedding config saved at index time for a collection.

    Returns None if no config file exists.
    """
    path = _config_path(db_path, collection_type)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
