#!/usr/bin/env python3
import sys
import logging
import voyageai
import chromadb
from pathlib import Path
from typing import List, Dict, Optional
from code_parser import CodeElement 

log = logging.getLogger(__name__)

class EmbeddingClient:
    """Wraps all interactions with the Voyage AI embedding API."""
    def __init__(self):
        try:
            self.client = voyageai.Client()
            log.info("Voyage AI client initialized.")
        except Exception as e:
            log.error(f"Voyage AI client init failed (set VOYAGE_API_KEY): {e}")
            sys.exit(1)

    def embed_documents(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Embeds a list of code snippets."""
        if not texts:
            return []
        try:
            result = self.client.embed(texts, model="voyage-code-2", input_type="document")
            return result.embeddings
        except Exception as e:
            log.error(f"Voyage embedding failed: {e}")
            return None

class CodeVectorStore:
    """Wraps all interactions with the ChromaDB vector store."""
    def __init__(self, path: str = "vector_db", collection_name: str = "project_code"):
        self.collection_name = collection_name
        try:
            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            log.info(f"ChromaDB client initialized. Collection: {self.collection_name}")
        except Exception as e:
            log.error(f"ChromaDB client init failed: {e}")
            sys.exit(1)
    
    def reset_collection(self):
        """Deletes and recreates the collection."""
        try:
            log.info(f"Attempting to delete existing collection: '{self.collection_name}'...")
            self.client.delete_collection(name=self.collection_name)
            log.info("Previous collection deleted.")
        except chromadb.errors.NotFoundError:
            log.info("No existing collection found. Starting fresh.")
        except Exception as e:
            log.error(f"Failed to delete collection: {e}")
            sys.exit(1)
        
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        log.info(f"Created new collection: '{self.collection_name}'")

    def batch_add_elements(self, elements: List[CodeElement], embeddings: List[List[float]]):
        """Performs a bulk 'add' for initial indexing."""
        ids = [el['id'] for el in elements]
        documents = [el['text'] for el in elements]
        metadatas = [{
            "file_path": el['file_path'], "function_name": el['name'], "kind": el['kind'],
            "start_line": el['start_line'], "end_line": el['end_line'], "content_hash": el['hash']
        } for el in elements]

        # Use 'add' for bulk import (faster, but fails on duplicates)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def upsert_code_elements(self, elements: List[CodeElement], embeddings: List[List[float]]):
        """Adds or updates code elements (for the hook)."""
        ids = [el['id'] for el in elements]
        metadatas = [{
            "file_path": el['file_path'],
            "function_name": el['name'],
            "kind": el['kind'],
            "start_line": el['start_line'],
            "end_line": el['end_line'],
            "content_hash": el['hash']
        } for el in elements]
        documents = [el["text"] for el in elements]
        
        # Use 'upsert' for incremental updates
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def delete_by_ids(self, ids: List[str]):
        """Deletes a list of elements by their unique IDs."""
        if ids:
            self.collection.delete(ids=ids)

    def query_by_embedding(self, embedding: List[float], n_results: int = 6) -> Dict:
        """Finds similar items to a given embedding."""
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )