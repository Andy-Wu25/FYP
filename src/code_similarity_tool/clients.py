import os
from pathlib import Path
from typing import List, Dict, Optional
import chromadb

class CodeVectorStore:
    """ChromaDB wrapper (persistent, cosine metric, single collection)."""
    def __init__(self, path: str, collection_name: str, metric: str = "cosine"):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        # ensure metric (space) is set on the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": metric}
        )
        print(f"[INFO] ChromaDB ready. path={path}, collection={collection_name}, metric={metric}")

    def reset_collection(self):
        name = self.collection.name
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        # Recreate with same metadata
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata=self.collection.metadata
        )
        print(f"[INFO] Reset collection: {name}")

    # ---- Bulk add for indexer (embeddings already computed) ----
    def add_many(self, elements: List[Dict], base_repo: str):
        """Add many elements; assumes you already computed embeddings separately.
           Use this from indexer by calling embedder first, then upsert_code_elements.
           Here, we just provide a convenience if you embed elsewhere.
        """
        # This helper is optional; indexer can just call upsert_code_elements directly.

    # ---- Standard upserts / deletes / queries ----
    def upsert_code_elements(self, elements: List[Dict], embeddings: List[List[float]], file_path: str):
        ids = [el['id'] for el in elements]
        metadatas = [{
            "file_path": file_path,                 # store path relative to repo root
            "function_name": el['name'],
            "kind": el['kind'],
            "start_line": el['start_line'],
            "end_line": el['end_line'],
            "content_hash": el['hash'],
        } for el in elements]
        documents = [el["text"] for el in elements]

        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def delete_by_ids(self, ids: List[str]):
        if ids:
            self.collection.delete(ids=ids)

    def delete_by_file_path(self, file_path_str: str) -> int:
        existing = self.collection.get(where={"file_path": file_path_str})
        if existing and existing.get('ids'):
            self.collection.delete(ids=existing['ids'])
            return len(existing['ids'])
        return 0

    def query_by_embedding(self, embedding: List[float], n_results: int = 6) -> Dict:
        return self.collection.query(query_embeddings=[embedding], n_results=n_results)
