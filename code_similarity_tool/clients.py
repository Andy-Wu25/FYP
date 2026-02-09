from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings


class CodeVectorStore:
    """ChromaDB wrapper for org-wide private data and central public corpora."""

    def __init__(
        self,
        path: str,
        private_collection_name: str,
        public_collection_name: str,
        metric: str = "cosine",
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.private_collection = self.client.get_or_create_collection(
            name=private_collection_name,
            metadata={"hnsw:space": metric},
        )
        self.public_collection = self.client.get_or_create_collection(
            name=public_collection_name,
            metadata={"hnsw:space": metric},
        )

    @staticmethod
    def _where_private_repo(org_id: str, repo_id: str) -> Dict[str, Any]:
        return {
            "$and": [
                {"org_id": org_id},
                {"repo_id": repo_id},
            ]
        }

    @staticmethod
    def _where_public_source(public_source_id: str) -> Dict[str, Any]:
        return {"public_source_id": public_source_id}

    @staticmethod
    def _delete_ids_in_chunks(collection, ids: List[str], chunk_size: int = 500) -> int:
        if not ids:
            return 0
        for i in range(0, len(ids), chunk_size):
            collection.delete(ids=ids[i : i + chunk_size])
        return len(ids)

    def delete_private_repo_entries(self, org_id: str, repo_id: str) -> int:
        existing = self.private_collection.get(where=self._where_private_repo(org_id, repo_id))
        ids = existing.get("ids") or []
        return self._delete_ids_in_chunks(self.private_collection, ids)

    def delete_public_source_entries(self, public_source_id: str) -> int:
        existing = self.public_collection.get(where=self._where_public_source(public_source_id))
        ids = existing.get("ids") or []
        return self._delete_ids_in_chunks(self.public_collection, ids)

    def _upsert_code_elements(
        self,
        collection,
        elements: List[Dict[str, Any]],
        embeddings: List[List[float]],
        *,
        base_metadata: Dict[str, Any],
    ) -> None:
        if not elements:
            return

        ids = [el["id"] for el in elements]
        metadatas = [
            {
                **base_metadata,
                "function_name": el["name"],
                "kind": el["kind"],
                "start_line": el["start_line"],
                "end_line": el["end_line"],
                "content_hash": el["hash"],
            }
            for el in elements
        ]
        documents = [el["text"] for el in elements]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def upsert_private_code_elements(
        self,
        elements: List[Dict[str, Any]],
        embeddings: List[List[float]],
        *,
        base_metadata: Dict[str, Any],
    ) -> None:
        self._upsert_code_elements(
            self.private_collection,
            elements,
            embeddings,
            base_metadata=base_metadata,
        )

    def upsert_public_code_elements(
        self,
        elements: List[Dict[str, Any]],
        embeddings: List[List[float]],
        *,
        base_metadata: Dict[str, Any],
    ) -> None:
        self._upsert_code_elements(
            self.public_collection,
            elements,
            embeddings,
            base_metadata=base_metadata,
        )

    def query_private_by_embedding(self, embedding: List[float], *, org_id: str, n_results: int = 8) -> Dict[str, Any]:
        return self.private_collection.query(
            query_embeddings=[embedding],
            where={"org_id": org_id},
            n_results=n_results,
        )

    def query_public_by_embedding(self, embedding: List[float], *, n_results: int = 8) -> Dict[str, Any]:
        return self.public_collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )
