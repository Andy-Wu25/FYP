from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings


class CodeVectorStore:
    """ChromaDB wrapper for org-wide, multi-repo code similarity search."""

    def __init__(self, path: str, collection_name: str, metric: str = "cosine"):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": metric},
        )

    @staticmethod
    def _where_repo(org_id: str, repo_id: str) -> Dict:
        return {"$and": [{"org_id": org_id}, {"repo_id": repo_id}]}

    def delete_repo_entries(self, org_id: str, repo_id: str) -> int:
        where = self._where_repo(org_id, repo_id)
        existing = self.collection.get(where=where)
        ids = existing.get("ids") or []
        if not ids:
            return 0

        chunk_size = 500
        for i in range(0, len(ids), chunk_size):
            self.collection.delete(ids=ids[i : i + chunk_size])
        return len(ids)

    def upsert_code_elements(
        self,
        elements: List[Dict],
        embeddings: List[List[float]],
        *,
        org_id: str,
        repo_id: str,
        repo_name: str,
        file_path: str,
    ) -> None:
        if not elements:
            return

        ids = [el["id"] for el in elements]
        metadatas = [
            {
                "org_id": org_id,
                "repo_id": repo_id,
                "repo_name": repo_name,
                "file_path": file_path,
                "function_name": el["name"],
                "kind": el["kind"],
                "start_line": el["start_line"],
                "end_line": el["end_line"],
                "content_hash": el["hash"],
            }
            for el in elements
        ]
        documents = [el["text"] for el in elements]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query_by_embedding(self, embedding: List[float], *, org_id: str, n_results: int = 8) -> Dict:
        return self.collection.query(
            query_embeddings=[embedding],
            where={"org_id": org_id},
            n_results=n_results,
        )
