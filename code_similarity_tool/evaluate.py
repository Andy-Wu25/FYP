#!/usr/bin/env python3
"""CLI entry point: code-sim-eval

Run the current repo through similarity checks against a snapshot and collect
structured evaluation metrics for FYP scalability analysis.

Usage:
    code-sim-eval --snapshot NAME --json FILE [--private] [--top-k N] [--no-queries]
                  [--relevance-threshold FLOAT]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .check_utils import (
    collect_full_file_query_elements,
    configure_logging,
    extract_hits,
)
from .clients import CodeVectorStore
from .embeddings import EmbeddingClient
from .ignore import load_ignore_file
from .runtime import iter_repo_source_files, load_runtime_context
from .snapshot import _META_FILE, _SNAPSHOTS_SUBDIR, _resolve_db_path, _snapshots_root
from .stats import _LANG_EXT, _compute_stats, _get_all_metadatas, _lang


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _include_any(meta: Dict, query_rel: str, query_hash: str) -> bool:
    """Accept all hits (no filtering)."""
    return True


def _include_skip_self(meta: Dict, query_rel: str, query_hash: str) -> bool:
    """Skip hits that match the query element's content hash."""
    return meta.get("content_hash") != query_hash


def _percentile(sorted_vals: List[float], p: float) -> float:
    """Simple percentile on a pre-sorted list (linear interpolation)."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    k = (n - 1) * p
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _compute_distance_histogram(distances: List[float], n_bins: int = 12) -> List[Dict[str, Any]]:
    """Create a histogram of distances with bins from 0.0 to 1.0."""
    bin_width = 1.0 / n_bins
    bins: List[Dict[str, Any]] = []
    for i in range(n_bins):
        lo = round(i * bin_width, 4)
        hi = round((i + 1) * bin_width, 4)
        count = sum(1 for d in distances if lo <= d < hi) if i < n_bins - 1 else sum(1 for d in distances if lo <= d <= hi)
        bins.append({"bin_low": lo, "bin_high": hi, "count": count})
    return bins


def _precision_at_k(
    per_query_distances: List[List[float]], k: int, threshold: float
) -> float:
    """Fraction of top-k results that are relevant (distance <= threshold), averaged."""
    if not per_query_distances:
        return 0.0
    scores = []
    for dists in per_query_distances:
        top = dists[:k]
        if not top:
            scores.append(0.0)
            continue
        relevant = sum(1 for d in top if d <= threshold)
        scores.append(relevant / len(top))
    return statistics.mean(scores)


def _mrr(per_query_distances: List[List[float]], threshold: float) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant hit, averaged."""
    if not per_query_distances:
        return 0.0
    rrs = []
    for dists in per_query_distances:
        rr = 0.0
        for rank, d in enumerate(dists, start=1):
            if d <= threshold:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return statistics.mean(rrs)


def _dcg(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        total += rel / math.log2(i + 2)  # i+2 because rank starts at 1
    return total


def _ndcg_at_k(per_query_distances: List[List[float]], k: int) -> float:
    """Normalized DCG at k, with rel = max(0, 1 - distance)."""
    if not per_query_distances:
        return 0.0
    scores = []
    for dists in per_query_distances:
        rels = [max(0.0, 1.0 - d) for d in dists[:k]]
        dcg = _dcg(rels, k)
        ideal = _dcg(sorted(rels, reverse=True), k)
        scores.append(dcg / ideal if ideal > 0 else 0.0)
    return statistics.mean(scores)


def _compute_distance_cdf(distances: List[float], n_points: int = 21) -> List[Dict[str, float]]:
    """CDF of distances at evenly-spaced x values from 0.0 to 1.0."""
    n = len(distances)
    if n == 0:
        return [{"x": round(i / (n_points - 1), 2), "cdf": 0.0} for i in range(n_points)]
    sorted_d = sorted(distances)
    points = []
    for i in range(n_points):
        x = round(i / (n_points - 1), 2)
        count = sum(1 for d in sorted_d if d <= x)
        points.append({"x": x, "cdf": round(count / n, 6)})
    return points


def _breakdown_stats(values: List[float]) -> Dict[str, Any]:
    """Mean/median for a group of distances."""
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
    }


def _resolve_snapshot(snapshot: str) -> Path:
    """Resolve snapshot argument to a directory path.

    If it's an existing directory path, use it directly.
    Otherwise treat it as a name and look it up in the default snapshot location.
    """
    candidate = Path(snapshot).expanduser().resolve()
    if candidate.is_dir():
        return candidate

    # Look up by name in default location
    db_path = _resolve_db_path(None)
    snap_dir = _snapshots_root(db_path) / snapshot
    if snap_dir.is_dir():
        return snap_dir

    print(f"Error: snapshot '{snapshot}' not found.", file=sys.stderr)
    print(f"  Checked path: {candidate}", file=sys.stderr)
    print(f"  Checked name: {snap_dir}", file=sys.stderr)
    sys.exit(1)


def run_evaluation(
    snapshot: str,
    json_path: str,
    *,
    use_private: bool = False,
    top_k: int = 5,
    include_queries: bool = True,
    relevance_threshold: float = 0.30,
) -> None:
    log = configure_logging()

    # Resolve snapshot: either a path or a name
    snap_dir = _resolve_snapshot(snapshot)
    snapshot_name = snap_dir.name

    # Load snapshot metadata
    snap_meta: Dict[str, Any] = {}
    meta_path = snap_dir / _META_FILE
    if meta_path.exists():
        try:
            snap_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Resolve current repo
    ctx = load_runtime_context()
    log.info("Repo: %s (%s)", ctx.repo_name, ctx.repo_root)
    log.info("Snapshot: %s (%s)", snapshot_name, snap_dir)

    # Open the snapshot DB read-only (point CodeVectorStore at snapshot)
    store = CodeVectorStore(
        path=str(snap_dir),
        private_collection_name=ctx.private_collection_name,
        public_collection_name=ctx.public_collection_name,
        metric=ctx.metric,
    )
    collection = store.private_collection if use_private else store.public_collection
    collection_label = "private" if use_private else "public"

    # Collect index stats
    log.info("Collecting %s index statistics ...", collection_label)
    all_metas = _get_all_metadatas(collection)
    index_stats = _compute_stats(all_metas)
    index_size_bytes = _dir_size(snap_dir)

    total_elements = index_stats["total"]
    total_repos = len(index_stats["repos"])
    total_files = index_stats["files"]

    log.info("Index: %d elements, %d repos, %d files", total_elements, total_repos, total_files)

    if total_elements == 0:
        print(f"Error: {collection_label} index in snapshot '{snapshot_name}' is empty.", file=sys.stderr)
        sys.exit(1)

    # Collect query elements from current repo
    log.info("Extracting code elements from current repo ...")
    matcher = load_ignore_file(ctx.repo_root)
    rel_paths = [str(p.relative_to(ctx.repo_root)) for p in iter_repo_source_files(ctx.repo_root, matcher)]
    query_elements = collect_full_file_query_elements(ctx.repo_root, rel_paths)

    if not query_elements:
        print("Error: no queryable code elements found in current repo.", file=sys.stderr)
        sys.exit(1)

    log.info("Extracted %d query elements from %d files.", len(query_elements), len(rel_paths))

    # Embed query elements
    log.info("Embedding %d elements ...", len(query_elements))
    texts = [element["text"] for _, element in query_elements]
    labels = [f"{rel}:{el['name']}" for rel, el in query_elements]

    t0_embed = time.monotonic()
    vectors = EmbeddingClient().embed_documents(texts, labels=labels)
    t_embed = time.monotonic() - t0_embed

    embed_ms_per_element = (t_embed * 1000) / len(query_elements) if query_elements else 0

    # Query each element against the snapshot
    log.info("Querying %d elements against snapshot (top_k=%d) ...", len(query_elements), top_k)
    pool_size = max(top_k * 4, 20)
    hit_filter = _include_skip_self if use_private else _include_any

    all_top1_distances: List[float] = []
    all_top1_gaps: List[float] = []
    all_hit_distances: List[float] = []
    all_result_repos: set = set()
    all_result_licenses: List[str] = []
    query_latencies_ms: List[float] = []
    queries_with_results: List[int] = []  # count of results per query
    per_query_data: List[Dict[str, Any]] = []
    per_query_hit_distances: List[List[float]] = []  # all distances per query
    per_query_unique_repos: List[int] = []  # unique repos per query
    kind_distances: Dict[str, List[float]] = {}  # kind -> top-1 distances
    lang_distances: Dict[str, List[float]] = {}  # language -> top-1 distances

    for i, ((rel_path, element), vector) in enumerate(zip(query_elements, vectors)):
        t0_q = time.monotonic()
        if use_private:
            results = store.query_private_by_embedding(vector, org_id=ctx.org_id, n_results=pool_size)
        else:
            results = store.query_public_by_embedding(vector, n_results=pool_size)
        t_q = time.monotonic() - t0_q
        query_latencies_ms.append(t_q * 1000)

        hits = extract_hits(
            results,
            top_k=top_k,
            max_distance=None,
            query_rel=rel_path,
            query_hash=element["hash"],
            include_hit=hit_filter,
        )

        n_hits = len(hits)
        queries_with_results.append(n_hits)

        # Collect per-query hit distances and unique repos
        query_dists = [dist for dist, _ in hits]
        per_query_hit_distances.append(query_dists)
        query_repos = {meta.get("repo_name", "<unknown>") for _, meta in hits}
        per_query_unique_repos.append(len(query_repos))

        if hits:
            top1_dist = hits[0][0]
            all_top1_distances.append(top1_dist)

            if len(hits) >= 2:
                gap = hits[1][0] - hits[0][0]
                all_top1_gaps.append(gap)

            for dist, meta in hits:
                all_hit_distances.append(dist)
                repo = meta.get("repo_name", "<unknown>")
                all_result_repos.add(repo)
                lic = (meta.get("license") or meta.get("license_spdx") or "Unknown").upper()
                all_result_licenses.append(lic)

            # Per-kind breakdown (top-1 distance)
            kind = element["kind"]
            kind_distances.setdefault(kind, []).append(top1_dist)

            # Per-language breakdown (top-1 distance)
            lang = _lang(rel_path)
            lang_distances.setdefault(lang, []).append(top1_dist)

        if include_queries:
            hit_records = []
            for rank, (dist, meta) in enumerate(hits, start=1):
                hit_records.append({
                    "rank": rank,
                    "distance": round(dist, 6),
                    "repo": meta.get("repo_name", "<unknown>"),
                    "file": meta.get("file_path", "<unknown>"),
                    "function": meta.get("function_name", "<unknown>"),
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                    "license": (meta.get("license") or meta.get("license_spdx") or "Unknown"),
                })
            per_query_data.append({
                "name": element["name"],
                "file": rel_path,
                "kind": element["kind"],
                "start_line": element["start_line"],
                "end_line": element["end_line"],
                "n_hits": n_hits,
                "top1_distance": round(hits[0][0], 6) if hits else None,
                "hits": hit_records,
            })

        if (i + 1) % 100 == 0:
            log.info("  queried %d / %d elements ...", i + 1, len(query_elements))

    # Compute aggregate metrics
    log.info("Computing metrics ...")

    sorted_top1 = sorted(all_top1_distances)
    n_queries = len(query_elements)

    # Distance stats
    if sorted_top1:
        dist_stats = {
            "mean": round(statistics.mean(sorted_top1), 6),
            "median": round(statistics.median(sorted_top1), 6),
            "stdev": round(statistics.stdev(sorted_top1), 6) if len(sorted_top1) > 1 else 0.0,
            "min": round(sorted_top1[0], 6),
            "max": round(sorted_top1[-1], 6),
            "p25": round(_percentile(sorted_top1, 0.25), 6),
            "p75": round(_percentile(sorted_top1, 0.75), 6),
        }
    else:
        dist_stats = {"mean": None, "median": None, "stdev": None, "min": None, "max": None, "p25": None, "p75": None}

    # Hit rate at thresholds
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    hit_rates: Dict[str, float] = {}
    for t in thresholds:
        count = sum(1 for d in all_top1_distances if d <= t)
        hit_rates[str(t)] = round(count / n_queries * 100, 2) if n_queries > 0 else 0.0

    # Top-k coverage
    coverage = {
        "gte_1": round(sum(1 for c in queries_with_results if c >= 1) / n_queries * 100, 2) if n_queries > 0 else 0.0,
        "gte_3": round(sum(1 for c in queries_with_results if c >= 3) / n_queries * 100, 2) if n_queries > 0 else 0.0,
        "gte_5": round(sum(1 for c in queries_with_results if c >= 5) / n_queries * 100, 2) if n_queries > 0 else 0.0,
    }

    # Score gap
    gap_stats: Dict[str, Any] = {}
    if all_top1_gaps:
        gap_stats = {
            "mean": round(statistics.mean(all_top1_gaps), 6),
            "median": round(statistics.median(all_top1_gaps), 6),
        }

    # IR metrics
    p_at_1 = round(_precision_at_k(per_query_hit_distances, 1, relevance_threshold), 6)
    p_at_3 = round(_precision_at_k(per_query_hit_distances, 3, relevance_threshold), 6)
    p_at_5 = round(_precision_at_k(per_query_hit_distances, 5, relevance_threshold), 6)
    mrr_val = round(_mrr(per_query_hit_distances, relevance_threshold), 6)
    ndcg_1 = round(_ndcg_at_k(per_query_hit_distances, 1), 6)
    ndcg_3 = round(_ndcg_at_k(per_query_hit_distances, 3), 6)
    ndcg_5 = round(_ndcg_at_k(per_query_hit_distances, 5), 6)

    # Per-kind breakdown
    by_kind: Dict[str, Dict[str, Any]] = {}
    for kind, dists in sorted(kind_distances.items()):
        by_kind[kind] = _breakdown_stats(dists)

    # Per-language breakdown
    by_language: Dict[str, Dict[str, Any]] = {}
    for lang, dists in sorted(lang_distances.items()):
        by_language[lang] = _breakdown_stats(dists)

    # Distribution metrics
    distance_histogram = _compute_distance_histogram(all_hit_distances)
    distance_cdf = _compute_distance_cdf(all_top1_distances)
    license_dist = dict(Counter(all_result_licenses).most_common())

    # Result diversity
    diversity: Dict[str, Any] = {}
    if per_query_unique_repos:
        diversity = {
            "mean_unique_repos_per_query": round(statistics.mean(per_query_unique_repos), 4),
            "median_unique_repos_per_query": round(statistics.median(per_query_unique_repos), 4),
        }

    # Latency stats
    latency_stats: Dict[str, Any] = {}
    if query_latencies_ms:
        latency_stats = {
            "mean_ms": round(statistics.mean(query_latencies_ms), 2),
            "median_ms": round(statistics.median(query_latencies_ms), 2),
            "total_ms": round(sum(query_latencies_ms), 2),
        }

    # Console summary
    _W = 66
    _B = "\033[1m"
    _C = "\033[96m"
    _G = "\033[92m"
    _D = "\033[2m"
    _R = "\033[0m"

    print()
    print("\u2554" + "\u2550" * (_W - 2) + "\u2557")
    title = "EVALUATION RESULTS"
    pad_l = (_W - 2 - len(title)) // 2
    pad_r = _W - 2 - len(title) - pad_l
    print("\u2551" + " " * pad_l + _B + title + _R + " " * pad_r + "\u2551")
    print("\u2560" + "\u2550" * (_W - 2) + "\u2563")
    print("\u2551  " + f"Snapshot   {_B}{snapshot_name}{_R}".ljust(_W - 4 + len(_B) + len(_R)) + "  \u2551")
    print("\u2551  " + f"Collection {collection_label}".ljust(_W - 4) + "  \u2551")
    print("\u255a" + "\u2550" * (_W - 2) + "\u255d")

    # Index overview
    print()
    print(f"  {_C}Index{_R}")
    print(f"    Elements   {_B}{total_elements:,}{_R}")
    print(f"    Repos      {_B}{total_repos}{_R}")
    print(f"    Files      {_B}{total_files:,}{_R}")

    # Distance stats
    print()
    print(f"  {_C}Top-1 Distance{_R}")
    if dist_stats["mean"] is not None:
        print(f"    Mean       {_B}{dist_stats['mean']:.4f}{_R}        p25        {_B}{dist_stats['p25']:.4f}{_R}")
        print(f"    Median     {_B}{dist_stats['median']:.4f}{_R}        p75        {_B}{dist_stats['p75']:.4f}{_R}")
        print(f"    Std        {_B}{dist_stats['stdev']:.4f}{_R}        Range      {_B}{dist_stats['min']:.4f} \u2013 {dist_stats['max']:.4f}{_R}")
    else:
        print(f"    {_D}No hits found{_R}")

    # IR Metrics
    print()
    print(f"  {_C}IR Metrics{_R}  {_D}(threshold={relevance_threshold}){_R}")
    print(f"    MRR        {_B}{mrr_val:.4f}{_R}")
    print(f"    P@1        {_B}{p_at_1:.4f}{_R}        P@3        {_B}{p_at_3:.4f}{_R}        P@5        {_B}{p_at_5:.4f}{_R}")
    print(f"    nDCG@1     {_B}{ndcg_1:.4f}{_R}        nDCG@3     {_B}{ndcg_3:.4f}{_R}        nDCG@5     {_B}{ndcg_5:.4f}{_R}")

    # Hit rates
    print()
    print(f"  {_C}Hit Rate at Threshold{_R}")
    if dist_stats["mean"] is not None:
        row1 = "   "
        row2 = "   "
        for t in thresholds:
            rate = hit_rates[str(t)]
            row1 += f" {t:<8}"
            row2 += f" {_B}{rate:>5.1f}%{_R}  "
        print(row1)
        print(row2)

    # Performance
    print()
    print(f"  {_C}Performance{_R}")
    print(f"    Queries    {_B}{n_queries}{_R}")
    print(f"    Embed      {_B}{embed_ms_per_element:.1f}{_R} ms/el")
    print(f"    Query      {_B}{latency_stats.get('mean_ms', 0):.1f}{_R} ms/el")

    print()
    print("\u2501" * _W)

    # Build output JSON
    output: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot_name,
        "snapshot_meta": snap_meta,
        "collection": collection_label,
        "repo": ctx.repo_name,
        "top_k": top_k,
        "relevance_threshold": relevance_threshold,
        "index": {
            "total_elements": total_elements,
            "total_repos": total_repos,
            "total_files": total_files,
            "size_bytes": index_size_bytes,
            "by_repo": dict(index_stats["repos"].most_common()),
            "by_license": dict(index_stats["licenses"].most_common()),
        },
        "query_summary": {
            "total_queries": n_queries,
            "queries_with_hits": sum(1 for c in queries_with_results if c >= 1),
        },
        "retrieval_quality": {
            "top1_distance": dist_stats,
            "hit_rate_at_threshold": hit_rates,
            "top_k_coverage": coverage,
            "top1_top2_gap": gap_stats,
            "precision_at_k": {
                "relevance_threshold": relevance_threshold,
                "p_at_1": p_at_1,
                "p_at_3": p_at_3,
                "p_at_5": p_at_5,
            },
            "mrr": mrr_val,
            "ndcg_at_k": {
                "ndcg_at_1": ndcg_1,
                "ndcg_at_3": ndcg_3,
                "ndcg_at_5": ndcg_5,
            },
        },
        "breakdowns": {
            "by_kind": by_kind,
            "by_language": by_language,
        },
        "scalability": {
            "embed_ms_per_element": round(embed_ms_per_element, 2),
            "query_latency": latency_stats,
            "index_size_bytes": index_size_bytes,
        },
        "distribution": {
            "distance_histogram": distance_histogram,
            "distance_cdf": distance_cdf,
            "result_diversity": diversity,
            "unique_repos_in_results": sorted(all_result_repos),
            "license_distribution": license_dist,
        },
    }

    if include_queries:
        output["queries"] = per_query_data

    Path(json_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"JSON written -> {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-eval",
        description="Evaluate code similarity against a snapshot and collect structured metrics.",
    )
    parser.add_argument("--snapshot", required=True, help="Snapshot name or path to snapshot directory")
    parser.add_argument("--json", required=True, metavar="FILE", help="Output JSON file path")
    parser.add_argument("--private", action="store_true", help="Query private index instead of public")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results per query (default: 5)")
    parser.add_argument("--no-queries", action="store_true", help="Omit per-query raw data from JSON output")
    parser.add_argument("--relevance-threshold", type=float, default=0.30, help="Distance threshold for Precision@k and MRR (default: 0.30)")
    args = parser.parse_args()

    run_evaluation(
        snapshot=args.snapshot,
        json_path=args.json,
        use_private=args.private,
        top_k=max(1, args.top_k),
        include_queries=not args.no_queries,
        relevance_threshold=args.relevance_threshold,
    )


if __name__ == "__main__":
    main()
