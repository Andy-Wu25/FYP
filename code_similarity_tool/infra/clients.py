from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
    def _where_public_repo(source_url: str) -> Dict[str, Any]:
        return {"source_url": source_url}

    @staticmethod
    def _get_all_ids(collection, where: Optional[Dict[str, Any]] = None, page_size: int = 5_000) -> List[str]:
        offset = 0
        out: List[str] = []
        while True:
            kw: Dict[str, Any] = {
                "limit": page_size,
                "offset": offset,
            }
            if where:
                kw["where"] = where
            result = collection.get(**kw)
            ids = result.get("ids") or []
            out.extend(ids)
            if len(ids) < page_size:
                break
            offset += page_size
        return out

    @staticmethod
    def _delete_ids_in_chunks(collection, ids: List[str], chunk_size: int = 500) -> int:
        if not ids:
            return 0
        for i in range(0, len(ids), chunk_size):
            collection.delete(ids=ids[i : i + chunk_size])
        return len(ids)

    @staticmethod
    def _compute_stale_ids(existing_ids: List[str], keep_ids: Set[str]) -> List[str]:
        if not existing_ids:
            return []
        if not keep_ids:
            return existing_ids
        return [id_ for id_ in existing_ids if id_ not in keep_ids]

    def delete_private_repo_entries(self, org_id: str, repo_id: str) -> int:
        ids = self._get_all_ids(self.private_collection, where=self._where_private_repo(org_id, repo_id))
        return self._delete_ids_in_chunks(self.private_collection, ids)

    def delete_public_source_entries(self, public_source_id: str) -> int:
        ids = self._get_all_ids(self.public_collection, where=self._where_public_source(public_source_id))
        return self._delete_ids_in_chunks(self.public_collection, ids)

    def delete_public_repo_entries(self, source_url: str) -> int:
        ids = self._get_all_ids(self.public_collection, where=self._where_public_repo(source_url))
        return self._delete_ids_in_chunks(self.public_collection, ids)

    def delete_private_repo_stale_entries(self, org_id: str, repo_id: str, keep_ids: List[str]) -> int:
        ids = self._get_all_ids(self.private_collection, where=self._where_private_repo(org_id, repo_id))
        stale_ids = self._compute_stale_ids(ids, set(keep_ids))
        return self._delete_ids_in_chunks(self.private_collection, stale_ids)

    def delete_public_source_stale_entries(self, public_source_id: str, keep_ids: List[str]) -> int:
        ids = self._get_all_ids(self.public_collection, where=self._where_public_source(public_source_id))
        stale_ids = self._compute_stale_ids(ids, set(keep_ids))
        return self._delete_ids_in_chunks(self.public_collection, stale_ids)

    def _upsert_code_elements(
        self,
        collection,
        elements: List[Dict[str, Any]],
        embeddings: List[List[float]],
        *,
        base_metadata: Dict[str, Any],
        store_code: bool = True,
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
                "code_stored": "true" if store_code else "false",
            }
            for el in elements
        ]

        upsert_kwargs: Dict[str, Any] = dict(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        if store_code:
            upsert_kwargs["documents"] = [el["text"] for el in elements]

        collection.upsert(**upsert_kwargs)

    def upsert_private_code_elements(
        self,
        elements: List[Dict[str, Any]],
        embeddings: List[List[float]],
        *,
        base_metadata: Dict[str, Any],
        store_code: bool = True,
    ) -> None:
        self._upsert_code_elements(
            self.private_collection,
            elements,
            embeddings,
            base_metadata=base_metadata,
            store_code=store_code,
        )

    def upsert_public_code_elements(
        self,
        elements: List[Dict[str, Any]],
        embeddings: List[List[float]],
        *,
        base_metadata: Dict[str, Any],
        store_code: bool = True,
    ) -> None:
        self._upsert_code_elements(
            self.public_collection,
            elements,
            embeddings,
            base_metadata=base_metadata,
            store_code=store_code,
        )

    def query_private_by_embedding(self, embedding: List[float], *, org_id: str, n_results: int = 8) -> Dict[str, Any]:
        return self.private_collection.query(
            query_embeddings=[embedding],
            where={"org_id": org_id},
            n_results=n_results,
        )

    def query_private_repo_by_embedding(
        self, embedding: List[float], *, org_id: str, repo_id: str, n_results: int = 8
    ) -> Dict[str, Any]:
        return self.private_collection.query(
            query_embeddings=[embedding],
            where=self._where_private_repo(org_id, repo_id),
            n_results=n_results,
        )

    def query_public_by_embedding(
        self,
        embedding: List[float],
        *,
        n_results: int = 8,
        licenses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": n_results,
        }

        normalized = sorted({lic.strip().upper() for lic in (licenses or []) if lic and lic.strip()})
        if normalized:
            query_kwargs["where"] = {"license": {"$in": normalized}}

        return self.public_collection.query(**query_kwargs)
