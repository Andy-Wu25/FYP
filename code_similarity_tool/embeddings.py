from __future__ import annotations

import logging
import os
from typing import List, Optional

import requests


log = logging.getLogger(__name__)


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
        self.input_prefix = os.getenv("VLLM_INPUT_PREFIX", "")

        log.info(
            "Embeddings backend=vllm base_url=%s model=%s batch=%d max_chars=%s auth=%s",
            self.base_url,
            self.model,
            self.batch_size,
            str(self.max_chars) if self.max_chars > 0 else "disabled",
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

    def embed_documents(self, texts: List[str], *, labels: Optional[List[str]] = None) -> List[List[float]]:
        if not texts:
            return []
        if labels is not None and len(labels) != len(texts):
            raise ValueError("labels length must match texts length.")

        if self.max_chars > 0:
            capped: List[str] = []
            for i, text in enumerate(texts):
                if len(text) > self.max_chars:
                    label = labels[i] if labels is not None else f"index {i}"
                    log.warning(
                        "Input too long (%d chars); truncating to %d chars (%s).",
                        len(text),
                        self.max_chars,
                        label,
                    )
                    capped.append(text[: self.max_chars])
                else:
                    capped.append(text)
            texts = capped

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
                if labels is not None:
                    label_batch = labels[i : i + len(batch)]
                    preview = "; ".join(label_batch[:3])
                    if len(label_batch) > 3:
                        preview += "; ..."
                    details = f" labels=[{preview}]"
                raise RuntimeError(
                    f"Embedding batch {i + 1}-{i + len(batch)} of {len(texts)} failed.{details}"
                ) from exc
        return all_vectors
