#!/usr/bin/env python3
from __future__ import annotations

import sys
import hashlib
import subprocess
import requests
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, TypedDict, Set

from tree_sitter_language_pack import get_language, get_parser

from .clients import CodeVectorStore
from .ignore import load_ignore_file

# -------- Logging --------
LOG_LEVEL = os.getenv("CODE_SIM_LOG_LEVEL", "INFO").upper()
LOG_FORCE = os.getenv("CODE_SIM_LOG_FORCE", "0").lower() in {"1", "true", "yes"}
try:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format='[%(levelname)s] %(message)s',
        force=LOG_FORCE,
    )
except TypeError:
    # Python < 3.8 doesn't support force=
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format='[%(levelname)s] %(message)s',
    )
log = logging.getLogger(__name__)

# -------- Paths / constants --------
THIS_FILE = Path(__file__).resolve()
TOOL_DIR  = THIS_FILE.parent

# Repo root is the directory where the user runs the CLI (project root)
REPO_ROOT = Path.cwd().resolve()

DB_PATH   = REPO_ROOT / ".git" / ".code-sim-db"
COLLECTION_NAME = "project_code"
METRIC          = "cosine"

SUPPORTED_SUFFIXES = {".py", ".java"}


# -------- Types --------
class CodeElement(TypedDict):
    id: str
    name: str
    kind: str
    start_line: int
    end_line: int
    text: str
    hash: str


class QueryElement(TypedDict):
    name: str
    file_path: str


# -------- Helpers --------
def make_instance_id(rel_path: str, content_hash: str) -> str:
    """
    Stable per-file instance id: same body in different files => different id.
    """
    return hashlib.sha256(f"{rel_path}:{content_hash}".encode("utf-8")).hexdigest()


def detect_lang(path: Path) -> Optional[str]:
    s = path.suffix.lower()
    if s == ".py":
        return "python"
    if s == ".java":
        return "java"
    return None


def _slice(buf: bytes, node) -> str:
    return buf[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def get_git_file_content(commit_hash: str, file_path: str) -> Optional[bytes]:
    """
    Read content from git. If commit_hash == '', read the INDEX (staged).
    Returns None if the blob doesn't exist.
    """
    try:
        spec = f"{commit_hash}:{file_path}" if commit_hash else f":{file_path}"
        res = subprocess.run(['git', 'show', spec], capture_output=True, check=True, text=False)
        return res.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def extract_code_elements(file_path: Path, buf: Optional[bytes]) -> List[CodeElement]:
    """
    Extract functions/methods using tree-sitter.
    Returns [] if buf is None or language unsupported.
    """
    if not buf:
        return []
    lang = detect_lang(file_path)
    if not lang:
        return []

    language = get_language(lang)
    parser   = get_parser(lang)
    tree     = parser.parse(buf)
    root     = tree.root_node

    if lang == "python":
        query_str = r"(function_definition) @decl"
        kind_map  = {"function_definition": "function"}
    else:  # java
        query_str = r"""
          (method_declaration) @decl
          (constructor_declaration) @decl
        """
        kind_map  = {"method_declaration": "method", "constructor_declaration": "constructor"}

    query = language.query(query_str)
    items: List[CodeElement] = []

    for _, caps in query.matches(root):
        decl_nodes = caps.get("decl")
        if not decl_nodes:
            continue
        d = decl_nodes[0]
        name_node = d.child_by_field_name("name")
        name = _slice(buf, name_node) if name_node else "<no-name>"
        text = _slice(buf, d)

        content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        items.append({
            "id": content_sha,  # will be overwritten with per-file instance id later
            "name": name,
            "kind": kind_map.get(d.type, d.type),
            "start_line": d.start_point[0] + 1,
            "end_line": d.end_point[0] + 1,
            "text": text,
            "hash": content_sha,
        })
    return items


def get_staged_added_modified() -> List[Path]:
    """Return repo-absolute Paths for files staged as added/modified (diff-filter=AM)."""
    try:
        out = subprocess.check_output(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=AM', '-z'],
            cwd=str(REPO_ROOT)
        )
        parts = out.split(b'\x00')
        files = [p.decode('utf-8') for p in parts if p]
        return [(REPO_ROOT / f).resolve() for f in files]
    except Exception:
        return []


def get_staged_deletions() -> List[Path]:
    """Return repo-absolute Paths for files staged as deleted (diff-filter=D)."""
    try:
        out = subprocess.check_output(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=D', '-z'],
            cwd=str(REPO_ROOT)
        )
        parts = out.split(b'\x00')
        files = [p.decode('utf-8') for p in parts if p]
        return [(REPO_ROOT / f).resolve() for f in files]
    except Exception:
        return []


# -------- Embedding --------
class VoyageEmbedder:
    """VoyageAI embedding backend."""
    def __init__(self, api_key: Optional[str], model: str = "voyage-code-2", input_type: str = "document"):
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is not set, cannot use VoyageAI backend")
        try:
            import voyageai  # type: ignore
        except ImportError as e:
            raise RuntimeError("voyageai is not installed in this environment") from e
        self._model = model
        self._input_type = input_type
        self._client = voyageai.Client(api_key=api_key)
        log.info("Embeddings backend=voyageai model=%s", self._model)

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        log.debug("VoyageAI embed request: n_texts=%d", len(texts))
        result = self._client.embed(texts, model=self._model, input_type=self._input_type)
        embeddings = result.embeddings
        if embeddings and isinstance(embeddings[0], list):
            log.debug("VoyageAI embed response: n=%d dim=%d", len(embeddings), len(embeddings[0]))
        return embeddings


class VLLMEmbedder:
    """OpenAI-compatible embeddings backend served by vLLM."""
    def __init__(self, base_url: str, api_key: Optional[str], model: str, timeout_s: float = 60.0):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        auth = "set" if api_key else "not-set"
        log.info("Embeddings backend=vllm base_url=%s model=%s auth=%s", self._base_url, self._model, auth)
        self._maybe_log_server_models()

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _maybe_log_server_models(self) -> None:
        verify = os.getenv("CODE_SIM_VLLM_VERIFY_MODELS", "1").lower() not in {"0", "false", "no"}
        if not verify:
            log.debug("Skipping vLLM /v1/models verification (CODE_SIM_VLLM_VERIFY_MODELS=0)")
            return
        models_url = f"{self._base_url}/v1/models"
        try:
            timeout = float(os.getenv("CODE_SIM_VLLM_MODELS_TIMEOUT", "5"))
            log.debug("Querying vLLM /v1/models with timeout=%ss", timeout)
            r = requests.get(models_url, headers=self._headers(), timeout=timeout)
            if r.status_code == 401:
                log.warning("vLLM /v1/models returned 401 Unauthorized (check VLLM_API_KEY / server --api-key)")
                return
            r.raise_for_status()
            payload = r.json()
            ids = [m.get("id") for m in payload.get("data", []) if isinstance(m, dict)]
            if ids:
                preview = ", ".join(ids[:5]) + ("..." if len(ids) > 5 else "")
                log.info("vLLM server reports %d model(s): %s", len(ids), preview)
                if self._model not in ids:
                    log.warning("Configured VLLM_MODEL=%s not found in /v1/models (still continuing)", self._model)
            else:
                log.info("vLLM /v1/models responded but returned no model ids")
        except Exception as e:
            log.warning("Could not query vLLM /v1/models at %s (%s)", models_url, e)

    def embed(self, texts: List[str]) -> List[List[float]]:
        url = f"{self._base_url}/v1/embeddings"
        payload: Dict[str, Any] = {"model": self._model, "input": texts}
        log.debug("vLLM embed request: url=%s model=%s n_texts=%d", url, self._model, len(texts))
        try:
            r = requests.post(url, headers=self._headers(), json=payload, timeout=self._timeout_s)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to reach vLLM embeddings endpoint at {url}: {e}") from e

        if r.status_code == 401:
            raise RuntimeError("vLLM returned 401 Unauthorized. Ensure your curl works with Authorization: Bearer $VLLM_API_KEY and that your code exports VLLM_API_KEY.")

        r.raise_for_status()
        data = r.json()
        resp_model = data.get("model")
        if resp_model:
            log.info("vLLM embeddings response model=%s", resp_model)
        items = data.get("data", [])
        if not isinstance(items, list) or not items:
            raise RuntimeError(f"Unexpected embeddings response format: {data}")
        embeddings = [item["embedding"] for item in items]
        if embeddings and isinstance(embeddings[0], list):
            log.debug("vLLM embed response: n=%d dim=%d", len(embeddings), len(embeddings[0]))
        return embeddings


class EmbeddingClient:
    """Unified embedding client. Chooses VoyageAI or vLLM based on environment."""
    def __init__(self):
        # Explicit override if you want to force a backend
        backend = (
            os.getenv("CODE_SIM_EMBEDDINGS_BACKEND")
            or os.getenv("EMBEDDINGS_BACKEND")
            or ""
        ).strip().lower()

        # Auto-detect backend if not specified
        if not backend:
            if os.getenv("VLLM_BASE_URL") or os.getenv("VLLM_API_KEY") or os.getenv("VLLM_MODEL"):
                backend = "vllm"
            elif os.getenv("VOYAGE_API_KEY"):
                backend = "voyage"
            else:
                backend = "voyage"  # keep previous default, but will error with a clearer message

        self.backend = backend
        self._impl: Any

        if backend == "vllm":
            base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
            api_key = os.getenv("VLLM_API_KEY")
            model = os.getenv("VLLM_MODEL", "Octen/Octen-Embedding-8B")
            timeout_s = float(os.getenv("VLLM_TIMEOUT_S", "60"))
            self._impl = VLLMEmbedder(base_url=base_url, api_key=api_key, model=model, timeout_s=timeout_s)
            log.info("Using vLLM embeddings backend")
        elif backend in {"voyage", "voyageai"}:
            api_key = os.getenv("VOYAGE_API_KEY")
            model = os.getenv("VOYAGE_MODEL", "voyage-code-2")
            input_type = os.getenv("VOYAGE_INPUT_TYPE", "document")
            self._impl = VoyageEmbedder(api_key=api_key, model=model, input_type=input_type)
            log.info("Using VoyageAI embeddings backend")
        else:
            raise ValueError(f"Unknown embeddings backend '{backend}'. Use CODE_SIM_EMBEDDINGS_BACKEND=vllm or voyage.")

    @property
    def model(self) -> str:
        return getattr(self._impl, "model", "unknown")

    def embed(self, texts: List[str]) -> List[List[float]]:
        log.info("Embedding request backend=%s model=%s n_texts=%d", self.backend, self.model, len(texts))
        return self._impl.embed(texts)

    # Compatibility with LangChain-style embedding interface used elsewhere in the code.
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Alias for embed(texts)."""
        return self.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string and return the vector."""
        vecs = self.embed([text])
        return vecs[0] if vecs else []

class CodeProcessor:
    def __init__(self, store: CodeVectorStore, embedder: EmbeddingClient):
        self.store = store
        self.embedder = embedder

    def analyze_modified_file(self, file_path: Path):
        """
        Compare HEAD vs INDEX and return:
          rel_for_git, new_elems, deleted_ids, deleted_names
        """
        rel_for_git = str(file_path.relative_to(REPO_ROOT))
        log.info(f"Processing modified file: {rel_for_git}")

        before = get_git_file_content('HEAD', rel_for_git)
        after  = get_git_file_content('',     rel_for_git)
        if not after:
            log.info(f"[skip] Cannot read staged content for: {rel_for_git}")
            return rel_for_git, [], [], []

        before_el = extract_code_elements(file_path, before) if before else []
        after_el  = extract_code_elements(file_path, after)

        # Attach per-file instance ids
        for el in before_el:
            el["id"] = make_instance_id(rel_for_git, el["hash"])
        for el in after_el:
            el["id"] = make_instance_id(rel_for_git, el["hash"])

        # Diff by content hash
        before_map = {e["hash"]: e for e in before_el}
        after_map  = {e["hash"]: e for e in after_el}

        deleted_hashes = list(set(before_map.keys()) - set(after_map.keys()))
        added_hashes   = list(set(after_map.keys())  - set(before_map.keys()))

        new_elems     = [after_map[h] for h in added_hashes]
        deleted_ids   = [before_map[h]["id"] for h in deleted_hashes]
        deleted_names = [before_map[h]["name"] for h in deleted_hashes]

        if not new_elems and not deleted_ids:
            log.info(f"[skip] No function-level changes in: {rel_for_git}")
            return rel_for_git, [], [], []

        return rel_for_git, new_elems, deleted_ids, deleted_names

    def process_deleted_file(self, rel_path: str):
        """Remove all DB entries for a deleted file path (repo-relative)."""
        log.info(f"File deleted. Removing DB entries for: {rel_path}")
        num = self.store.delete_by_file_path(rel_path)
        log.info(f"Deleted {num} function(s).")

    def _show_query_results(self, results: Dict, query_element: QueryElement):
        # Leading separator only; no trailing dashed line
        print("-" * 25)
        print(f"Query for code similar to '{query_element['name']}' in '{query_element['file_path']}':")
        if not results.get('ids') or not results['ids'][0]:
            print("  -> Query returned no results.\n")
            return

        ids   = results['ids'][0]
        dists = results['distances'][0]
        metas = results['metadatas'][0]
        if len(ids) < 2:
            print("  -> No other similar items found in the database.\n")
            return

        for i in range(1, len(ids)):  # skip exact self
            m = metas[i]
            print(f"\n  -> Found similar item (distance: {dists[i]:.4f})")
            print(f"     File: {m['file_path']}")
            print(f"     Function: {m['function_name']} (lines {m['start_line']}-{m['end_line']})")
        print()  # blank line between blocks


# -------- Entry point for pre-commit --------
def main():
    # Ensure DB exists
    DB_PATH.mkdir(parents=True, exist_ok=True)

    # Parse args (support --deleted and --with-deletions)
    args = sys.argv[1:]
    deleted_mode   = False
    with_deletions = False

    if args and args[0] == "--deleted":
        deleted_mode = True
        args = args[1:]
    elif args and args[0] == "--with-deletions":
        with_deletions = True
        args = args[1:]

    # From pre-commit (argv)
    from_precommit = [Path(p).resolve() for p in args]
    # From git staged AM
    staged_am_git  = get_staged_added_modified()

    matcher = load_ignore_file(REPO_ROOT)

    # Union the two sets, apply filters + ignore
    union: List[Path] = []
    seen: Set[Path] = set()
    for p in from_precommit + staged_am_git:
        try:
            p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if ".git" in p.parts:
            continue
        if not matcher.allows(p, is_dir=False):
            continue
        if p not in seen:
            seen.add(p)
            union.append(p)

    store = CodeVectorStore(path=str(DB_PATH), collection_name=COLLECTION_NAME, metric=METRIC)
    embedder = EmbeddingClient()
    proc = CodeProcessor(store, embedder)

    # Legacy deletion-only mode
    if deleted_mode:
        if not union:
            log.info("No relevant files to process after filtering.")
            sys.exit(0)
        for fp in union:
            rel = str(fp.relative_to(REPO_ROOT))
            proc.process_deleted_file(rel)
        sys.exit(0)

    # Analyze AM changes; batch embeddings across all files
    all_new: List[CodeElement] = []
    all_new_rel_paths: List[str] = []

    if union:
        log.info(f"Processing {len(union)} staged file(s)...")
        for fp in union:
            rel_for_git, new_elems, deleted_ids, deleted_names = proc.analyze_modified_file(fp)

            if deleted_ids:
                log.info("Detected changes, updating database...")
                for nm in deleted_names:
                    log.info(f"- Deleting old version of: {nm}")
                store.delete_by_ids(deleted_ids)

            if new_elems:
                for el in new_elems:
                    log.info(f"+ Adding new version of: {el['name']}")
                all_new.extend(new_elems)
                all_new_rel_paths.extend([rel_for_git] * len(new_elems))
    else:
        log.info("No relevant files to process after filtering.")

    # Single embed call for all new/changed functions
    if all_new:
        vectors = embedder.embed_documents([e["text"] for e in all_new])
        if vectors is None:
            log.error("Failed to get embeddings. Aborting update.")
            sys.exit(1)

        # Upsert grouped by file path
        from collections import defaultdict
        by_file_elems: Dict[str, List[CodeElement]] = defaultdict(list)
        by_file_vecs:  Dict[str, List[List[float]]] = defaultdict(list)
        for el, vec, rel in zip(all_new, vectors, all_new_rel_paths):
            by_file_elems[rel].append(el)
            by_file_vecs[rel].append(vec)

        for rel in by_file_elems:
            store.upsert_code_elements(by_file_elems[rel], by_file_vecs[rel], rel)
        log.info("Database updated successfully.")

        # Similarity queries (for user feedback)
        idx = 0
        for rel in by_file_elems:
            for el in by_file_elems[rel]:
                similar = store.query_by_embedding(vectors[idx], n_results=6)
                proc._show_query_results(similar, {"name": el['name'], "file_path": rel})
                idx += 1

    # Optionally also purge staged deletions of whole files
    if with_deletions:
        staged_deleted = get_staged_deletions()
        if staged_deleted:
            log.info("Processing deletions...")
            for fp in staged_deleted:
                try:
                    rel = str(fp.relative_to(REPO_ROOT))
                except ValueError:
                    continue
                abs_p = (REPO_ROOT / rel).resolve()
                if not matcher.allows(abs_p, is_dir=False):
                    continue
                proc.process_deleted_file(rel)

    sys.exit(0)


if __name__ == "__main__":
    main()
