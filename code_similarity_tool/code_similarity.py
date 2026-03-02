#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict

from .check_utils import (
    add_scope_argument,
    configure_logging,
    extract_hits,
    query_elements_from_args,
    validate_scope_args,
)
from .clients import CodeVectorStore
from .embeddings import EmbeddingClient
from .public_links import build_github_commit_url, build_public_commit_permalink, build_public_match_permalink


def _include_private_hit_factory(query_repo_id: str):
    def _include_hit(meta: Dict, query_rel: str, query_hash: str) -> bool:
        # Skip exact self match from same repo/file/content
        if (
            meta.get("repo_id") == query_repo_id
            and meta.get("file_path") == query_rel
            and meta.get("content_hash") == query_hash
        ):
            return False
        return True

    return _include_hit


def _include_all(meta: Dict, query_rel: str, query_hash: str) -> bool:
    return True


def _run_check(
    prog: str,
    description: str,
    check_private: bool,
    check_public: bool,
) -> None:
    log = configure_logging()

    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("paths", nargs="*", help="Optional paths (used by --scope staged/files).")
    add_scope_argument(parser)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Maximum allowed distance (inclusive). Lower values are more similar.",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        default=None,
        help="Write results as JSON to FILE.",
    )
    args = parser.parse_args()
    validate_scope_args(parser, args)

    top_k = max(1, args.top_k)

    ctx, rel_paths, query_elements = query_elements_from_args(args.paths, args.scope)

    if not rel_paths:
        if args.scope == "repo":
            log.info("No repository files matched the active scope and filters.")
        elif args.scope == "files":
            log.info("No provided files matched the active scope and filters.")
        else:
            log.info("No staged files matched the active scope and filters.")
        return

    if not query_elements:
        log.info("No queryable code elements found after filters and ignore rules.")
        return

    log.info(
        "Checking %d code element(s) from %d file(s) in scope '%s'.",
        len(query_elements),
        len(rel_paths),
        args.scope,
    )

    embedder = EmbeddingClient()
    vectors = embedder.embed_documents([element["text"] for _, element in query_elements])

    store = CodeVectorStore(
        path=str(ctx.db_path),
        private_collection_name=ctx.private_collection_name,
        public_collection_name=ctx.public_collection_name,
        metric=ctx.metric,
    )
    pool_size = max(top_k * 4, 20)

    include_private = _include_private_hit_factory(ctx.repo_id)
    total_private_hits = 0
    total_public_hits = 0
    query_results = []

    for (rel_path, element), vector in zip(query_elements, vectors):
        print("-" * 72)
        print(f"Query: {element['name']} ({rel_path}:{element['start_line']}-{element['end_line']})")

        query_record: Dict = {
            "name": element["name"],
            "file": rel_path,
            "start_line": element["start_line"],
            "end_line": element["end_line"],
        }

        # ── Private ──────────────────────────────────────────────────────────
        if check_private:
            private_results = store.query_private_by_embedding(
                vector, org_id=ctx.org_id, n_results=pool_size
            )
            private_hits = extract_hits(
                private_results,
                top_k=top_k,
                max_distance=args.max_distance,
                query_rel=rel_path,
                query_hash=element["hash"],
                include_hit=include_private,
            )
            print("  [Private]")
            if not private_hits:
                print("    No private-org similar items found.")
            else:
                total_private_hits += len(private_hits)
                for idx, (distance, meta) in enumerate(private_hits, start=1):
                    print(
                        f"    {idx}. distance={distance:.4f} "
                        f"repo={meta.get('repo_name', '<unknown>')} "
                        f"file={meta.get('file_path', '<unknown>')} "
                        f"function={meta.get('function_name', '<unknown>')} "
                        f"lines={meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
                    )
            query_record["private_hits"] = [
                {
                    "rank": idx,
                    "distance": round(distance, 6),
                    "repo": meta.get("repo_name", "<unknown>"),
                    "file": meta.get("file_path", "<unknown>"),
                    "function": meta.get("function_name", "<unknown>"),
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                }
                for idx, (distance, meta) in enumerate(private_hits if check_private else [], start=1)
            ]

        # ── Public ───────────────────────────────────────────────────────────
        if check_public:
            public_results = store.query_public_by_embedding(vector, n_results=pool_size)
            public_hits = extract_hits(
                public_results,
                top_k=top_k,
                max_distance=args.max_distance,
                query_rel=rel_path,
                query_hash=element["hash"],
                include_hit=_include_all,
            )
            print("  [Public]")
            if not public_hits:
                print("    No public-index similar items found.")
            else:
                total_public_hits += len(public_hits)
                for idx, (distance, meta) in enumerate(public_hits, start=1):
                    permalink = build_public_match_permalink(meta)
                    commit_url = None
                    if not permalink:
                        commit_url = meta.get("source_commit_url")
                        if not isinstance(commit_url, str) or not commit_url.strip():
                            commit_url = build_public_commit_permalink(meta)
                    license_display = meta.get("license")
                    if not isinstance(license_display, str) or not license_display.strip():
                        license_display = meta.get("license_spdx", "<unknown>")
                    print(
                        f"    {idx}. distance={distance:.4f} "
                        f"repo={meta.get('repo_name', '<unknown>')} "
                        f"file={meta.get('file_path', '<unknown>')} "
                        f"function={meta.get('function_name', '<unknown>')} "
                        f"license={license_display}"
                    )
                    file_commit = meta.get("source_file_commit", "")
                    file_commit_url = None
                    if isinstance(file_commit, str) and file_commit.strip():
                        file_commit_url = build_github_commit_url(
                            str(meta.get("source_url", "")), file_commit.strip()
                        )
                    if permalink:
                        print(f"       permalink={permalink}")
                    if file_commit_url:
                        print(f"       file_commit={file_commit_url}")
                    if commit_url:
                        print(f"       commit_url={commit_url}")

            public_hit_records = []
            for idx, (distance, meta) in enumerate(public_hits if check_public else [], start=1):
                permalink = build_public_match_permalink(meta)
                commit_url = None
                if not permalink:
                    commit_url = meta.get("source_commit_url")
                    if not isinstance(commit_url, str) or not commit_url.strip():
                        commit_url = build_public_commit_permalink(meta)
                license_display = meta.get("license")
                if not isinstance(license_display, str) or not license_display.strip():
                    license_display = meta.get("license_spdx", "<unknown>")
                file_commit = meta.get("source_file_commit", "")
                file_commit_url = None
                if isinstance(file_commit, str) and file_commit.strip():
                    file_commit_url = build_github_commit_url(
                        str(meta.get("source_url", "")), file_commit.strip()
                    )
                hit: Dict = {
                    "rank": idx,
                    "distance": round(distance, 6),
                    "repo": meta.get("repo_name", "<unknown>"),
                    "file": meta.get("file_path", "<unknown>"),
                    "function": meta.get("function_name", "<unknown>"),
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                    "license": license_display,
                    "commit": meta.get("source_commit", "<unknown>"),
                }
                if permalink:
                    hit["permalink"] = permalink
                if file_commit_url:
                    hit["file_commit_url"] = file_commit_url
                if commit_url:
                    hit["commit_url"] = commit_url
                public_hit_records.append(hit)
            query_record["public_hits"] = public_hit_records

        query_results.append(query_record)

    print("=" * 72)
    if check_private and check_public:
        print(
            f"Checked {len(query_elements)} code element(s); "
            f"found {total_private_hits} private match(es), {total_public_hits} public match(es)."
        )
    elif check_private:
        print(
            f"Checked {len(query_elements)} code element(s); "
            f"found {total_private_hits} private similar match(es) in org '{ctx.org_id}'."
        )
    else:
        print(
            f"Checked {len(query_elements)} code element(s); "
            f"found {total_public_hits} public similar match(es)."
        )

    if args.json:
        data: Dict = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command": prog,
            "scope": args.scope,
            "top_k": top_k,
            "max_distance": args.max_distance,
            "org_id": ctx.org_id,
            "queries": query_results,
            "summary": {
                "total_elements_checked": len(query_elements),
            },
        }
        if check_private:
            data["summary"]["total_private_hits"] = total_private_hits
        if check_public:
            data["summary"]["total_public_hits"] = total_public_hits
        Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nJSON written \u2192 {args.json}")


def main() -> None:
    _run_check(
        prog="code-sim-check",
        description=(
            "Read-only similarity check against both private org-scoped and public indexes "
            "(scope: staged/files/repo)."
        ),
        check_private=True,
        check_public=True,
    )


def main_private() -> None:
    _run_check(
        prog="code-sim-check-private",
        description=(
            "Read-only similarity check against private org-scoped index only "
            "(scope: staged/files/repo)."
        ),
        check_private=True,
        check_public=False,
    )


if __name__ == "__main__":
    main()
