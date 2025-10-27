#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict

# Import your new shared modules
from .clients import CodeVectorStore, EmbeddingClient
from .code_parser import extract_code_elements, CodeElement

# -------- Setup Logging --------
# Use logging instead of print for better control
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

class ProjectIndexer:
    """
    Scans an entire project directory, deletes the old database,
    and builds a new one from scratch.
    """
    def __init__(self, root_dir: str, vector_store: CodeVectorStore, embed_client: EmbeddingClient):
        self.root_dir = root_dir
        self.vector_store = vector_store
        self.embed_client = embed_client

    def index_project(self):
        """Main method to run the indexing process."""
        log.info("--- Starting Initial Project Indexing ---")
        
        # 1. Reset the database
        self.vector_store.reset_collection()

        # 2. Scan all files and extract code elements
        all_elements = self._scan_project_files()
        if not all_elements:
            log.info("No code elements found to index.")
            return

        log.info(f"\nFound a total of {len(all_elements)} functions/methods to index.")
        log.info("Getting embeddings from Voyage AI (this may take a moment)...")

        # 3. Get embeddings in a single batch
        payloads = [it["text"] for it in all_elements]
        embeddings = self.embed_client.embed_documents(payloads)
        
        if embeddings is None:
            log.error("Failed to get embeddings. Aborting index.")
            return

        log.info("Embeddings received. Adding to vector database...")
        
        # 4. Add to database in a single batch
        self.vector_store.batch_add_elements(all_elements, embeddings)

        log.info("\n✅ Initial indexing complete!")
        log.info("The pre-commit hook will now handle incremental updates.")

    def _scan_project_files(self) -> List[CodeElement]:
        """Walks the directory and parses all valid files."""
        all_elements: List[CodeElement] = []
        for subdir, _, files in os.walk(self.root_dir):
            # Skip hidden folders and venv
            if any(part.startswith('.') for part in Path(subdir).parts) or 'venv' in subdir:
                continue

            for file in files:
                file_path = Path(os.path.join(subdir, file))
                if file_path.suffix not in ['.py', '.java']:
                    continue

                log.info(f"  -> Scanning: {file_path}")
                try:
                    content = file_path.read_bytes()
                    elements = extract_code_elements(file_path, content)
                    if elements:
                        all_elements.extend(elements)
                except Exception as e:
                    log.warning(f"    [warn] Could not process file: {e}")
        
        return all_elements

def main():
    """Main execution function."""
    try:
        # 1. Initialize services
        embed_client = EmbeddingClient()
        vector_store = CodeVectorStore(path=DB_PATH, collection_name=COLLECTION_NAME)
        
        # 2. Initialize and run the indexer
        # You can make root_dir an argument if you want, e.g., sys.argv[1]
        indexer = ProjectIndexer(root_dir=".", vector_store=vector_store, embed_client=embed_client)
        indexer.index_project()
        
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Define these constants here or get them from env/args
    DB_PATH = "vector_db"
    COLLECTION_NAME = "project_code"
    main()