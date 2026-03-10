#!/usr/bin/env python3
"""CLI entry point: code-sim-config

Display the resolved configuration for the code similarity tool.
Shows all settings with their current values and sources (env var or default).
"""
from __future__ import annotations

import os
from pathlib import Path

from ..core.runtime import find_repo_root, load_runtime_context
from .snapshot import _default_snapshots_dir


def _val(env_var: str, default: str) -> tuple[str, bool]:
    """Return (resolved_value, is_from_env)."""
    raw = os.getenv(env_var)
    if raw is not None:
        return raw.strip() or default, True
    return default, False


def _row(label: str, value: str, env_var: str, from_env: bool) -> str:
    source = env_var if from_env else f"{env_var} (default)"
    return f"  {label:<22} {value:<36} {source}"


def main() -> None:
    _B = "\033[1m"
    _C = "\033[96m"
    _D = "\033[2m"
    _R = "\033[0m"

    # Resolve repo context (best-effort)
    try:
        ctx = load_runtime_context()
        repo_root = str(ctx.repo_root)
        repo_name = ctx.repo_name
        org_id = ctx.org_id
        repo_id = ctx.repo_id[:12] + "..."
        db_path = str(ctx.db_path)
        priv_coll = ctx.private_collection_name
        pub_coll = ctx.public_collection_name
        metric = ctx.metric
    except Exception:
        repo_root = str(find_repo_root())
        repo_name = Path(repo_root).name
        org_id = os.getenv("CODE_SIM_ORG_ID", "local")
        repo_id = "?"
        db_path = os.getenv("CODE_SIM_DB_PATH", str(Path.home() / ".code-sim" / "chroma"))
        priv_coll = os.getenv("CODE_SIM_PRIVATE_COLLECTION", "project_code_private")
        pub_coll = os.getenv("CODE_SIM_PUBLIC_COLLECTION", "project_code_public")
        metric = os.getenv("CODE_SIM_METRIC", "cosine")

    try:
        snap_dir = str(_default_snapshots_dir())
    except Exception:
        snap_dir = "?"

    # Embedding settings
    max_chars_val, max_chars_env = _val("VLLM_MAX_CHARS", "0")
    long_text_val, long_text_env = _val("VLLM_LONG_TEXT_MODE", "chunk")
    overlap_val, overlap_env = _val("VLLM_CHUNK_OVERLAP", "512")
    model_val, model_env = _val("VLLM_MODEL", "Octen/Octen-Embedding-8B")
    base_url_val, base_url_env = _val("VLLM_BASE_URL", "http://127.0.0.1:8000")
    batch_val, batch_env = _val("VLLM_BATCH_SIZE", "64")
    timeout_val, timeout_env = _val("VLLM_TIMEOUT_S", "120")
    api_key_raw = os.getenv("VLLM_API_KEY")
    api_key_display = "***" if api_key_raw else "(not set)"
    api_key_env = api_key_raw is not None
    prefix_val, prefix_env = _val("VLLM_INPUT_PREFIX", "")
    prefix_display = repr(prefix_val) if prefix_val else "(none)"
    max_file_val, max_file_env = _val("CODE_SIM_MAX_FILE_BYTES", "250000")

    # Check if org/db are from env
    _, org_from_env = _val("CODE_SIM_ORG_ID", "local")
    _, db_from_env = _val("CODE_SIM_DB_PATH", str(Path.home() / ".code-sim" / "chroma"))
    _, metric_from_env = _val("CODE_SIM_METRIC", "cosine")

    _W = 90
    print()
    print(f"  {_C}Repository{_R}")
    print(f"  {'Root':<22} {_B}{repo_root}{_R}")
    print(f"  {'Name':<22} {_B}{repo_name}{_R}")

    print()
    print(f"  {_C}Database{_R}")
    print(_row("DB path", db_path, "CODE_SIM_DB_PATH", db_from_env))
    print(_row("Org ID", org_id, "CODE_SIM_ORG_ID", org_from_env))
    print(_row("Metric", metric, "CODE_SIM_METRIC", metric_from_env))
    print(f"  {'Private collection':<22} {priv_coll}")
    print(f"  {'Public collection':<22} {pub_coll}")
    print(f"  {'Snapshots dir':<22} {snap_dir}")

    print()
    print(f"  {_C}Embeddings{_R}")
    print(_row("Model", model_val, "VLLM_MODEL", model_env))
    print(_row("Base URL", base_url_val, "VLLM_BASE_URL", base_url_env))
    max_chars_display = f"{max_chars_val} (no limit)" if max_chars_val == "0" else max_chars_val
    print(_row("Max chars", max_chars_display, "VLLM_MAX_CHARS", max_chars_env))
    if max_chars_val != "0":
        print(_row("Long text mode", long_text_val, "VLLM_LONG_TEXT_MODE", long_text_env))
        print(_row("Chunk overlap", overlap_val, "VLLM_CHUNK_OVERLAP", overlap_env))
    else:
        print(f"  {'Long text mode':<22} {_D}n/a (max_chars=0){_R}")
        print(f"  {'Chunk overlap':<22} {_D}n/a (max_chars=0){_R}")
    print(_row("Batch size", batch_val, "VLLM_BATCH_SIZE", batch_env))
    print(_row("Timeout", f"{timeout_val}s", "VLLM_TIMEOUT_S", timeout_env))
    print(_row("API key", api_key_display, "VLLM_API_KEY", api_key_env))
    print(_row("Input prefix", prefix_display, "VLLM_INPUT_PREFIX", prefix_env))

    print()
    print(f"  {_C}Limits{_R}")
    print(_row("Max file bytes", max_file_val, "CODE_SIM_MAX_FILE_BYTES", max_file_env))

    print()


if __name__ == "__main__":
    main()
