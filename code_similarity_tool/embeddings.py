from __future__ import annotations

import logging
import os
from typing import List, Optional

import requests


log = logging.getLogger(__name__)


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

    def __init__(self):
        backend = os.getenv("CODE_SIM_EMBEDDINGS_BACKEND", "vllm").strip().lower()
        if backend != "vllm":
            raise ValueError(
                f"Unsupported embeddings backend '{backend}'. "
                "Set CODE_SIM_EMBEDDINGS_BACKEND=vllm."
            )

        self.base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        self.api_key = os.getenv("VLLM_API_KEY", "").strip()
        self.model = os.getenv("VLLM_MODEL", "Octen/Octen-Embedding-8B").strip()
        self.timeout_s = float(os.getenv("VLLM_TIMEOUT_S", "60"))
        self.verify_models = os.getenv("VLLM_VERIFY_MODELS", "1").strip().lower() not in {
            "0",
            "false",
        }
        self.batch_size = max(1, int(os.getenv("VLLM_BATCH_SIZE", "64")))
        self.max_chars = max(0, int(os.getenv("VLLM_MAX_CHARS", "16000")))
        self.long_text_mode = os.getenv("VLLM_LONG_TEXT_MODE", "chunk").strip().lower() or "chunk"
        if self.long_text_mode not in {"chunk", "truncate"}:
            raise ValueError(
                f"Unsupported long text mode '{self.long_text_mode}'. "
                "Set VLLM_LONG_TEXT_MODE=chunk or truncate."
            )
        raw_chunk_overlap = max(0, int(os.getenv("VLLM_CHUNK_OVERLAP", "512")))
        if self.max_chars > 0:
            self.chunk_overlap = min(raw_chunk_overlap, self.max_chars // 8)
        else:
            self.chunk_overlap = raw_chunk_overlap
        self.input_prefix = os.getenv("VLLM_INPUT_PREFIX", "")

        log.info(
            "Embeddings backend=vllm base_url=%s model=%s batch=%d max_chars=%s long_text=%s overlap=%d auth=%s",
            self.base_url,
            self.model,
            self.batch_size,
            str(self.max_chars) if self.max_chars > 0 else "disabled",
            self.long_text_mode,
            self.chunk_overlap,
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

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.model, "input": texts}

        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        if resp.status_code == 401:
            raise RuntimeError(
                "vLLM returned 401 Unauthorized. Set VLLM_API_KEY to match server --api-key."
            )
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

    def _split_long_text(self, text: str) -> List[str]:
        if self.max_chars <= 0 or len(text) <= self.max_chars:
            return [text]
        if self.long_text_mode == "truncate":
            return [text[: self.max_chars]]

        step = max(1, self.max_chars - self.chunk_overlap)
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.max_chars)
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += step
        return chunks

    def embed_documents(self, texts: List[str], *, labels: Optional[List[str]] = None) -> List[List[float]]:
        if not texts:
            return []
        if labels is not None and len(labels) != len(texts):
            raise ValueError("labels length must match texts length.")

        expanded_texts: List[str] = []
        expanded_labels: Optional[List[str]] = [] if labels is not None else None
        chunk_counts: List[int] = []

        for i, text in enumerate(texts):
            label = labels[i] if labels is not None else f"index {i}"
            chunks = self._split_long_text(text)

            if len(chunks) > 1:
                log.warning(
                    "Input too long (%d chars); splitting into %d chunks of up to %d chars with overlap %d (%s).",
                    len(text),
                    len(chunks),
                    self.max_chars,
                    self.chunk_overlap,
                    label,
                )
            elif self.max_chars > 0 and len(text) > self.max_chars:
                log.warning(
                    "Input too long (%d chars); truncating to %d chars (%s).",
                    len(text),
                    self.max_chars,
                    label,
                )

            chunk_counts.append(len(chunks))
            for chunk_index, chunk in enumerate(chunks):
                expanded_texts.append(chunk)
                if expanded_labels is not None:
                    if len(chunks) == 1:
                        expanded_labels.append(label)
                    else:
                        expanded_labels.append(f"{label} [chunk {chunk_index + 1}/{len(chunks)}]")

        texts = expanded_texts

        if self.input_prefix:
            texts = [self.input_prefix + t for t in texts]

        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            log.info("Embedding batch %d-%d of %d", i + 1, i + len(batch), len(texts))
            try:
                all_vectors.extend(self._embed_batch(batch))
            except Exception as exc:  # noqa: BLE001
                details = ""
                if expanded_labels is not None:
                    label_batch = expanded_labels[i : i + len(batch)]
                    preview = "; ".join(label_batch[:3])
                    if len(label_batch) > 3:
                        preview += "; ..."
                    details = f" labels=[{preview}]"
                cause = _format_exception_details(exc)
                raise RuntimeError(
                    f"Embedding batch {i + 1}-{i + len(batch)} of {len(texts)} failed.{details} "
                    f"Cause: {cause}"
                ) from exc
        if all(count == 1 for count in chunk_counts):
            return all_vectors

        merged_vectors: List[List[float]] = []
        offset = 0
        for count in chunk_counts:
            merged_vectors.append(self._average_vectors(all_vectors[offset : offset + count]))
            offset += count
        return merged_vectors
