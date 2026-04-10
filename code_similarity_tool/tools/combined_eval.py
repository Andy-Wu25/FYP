#!/usr/bin/env python3
"""Batch evaluation: public embedding-based retrieval across all snapshots.

Runs embedding evaluation against every snapshot found in the snapshots
directory, producing per-snapshot JSON files and a unified summary table
suitable for academic reporting.

Metric computation is repeated for every relevance threshold so that
embedding and querying (the expensive steps) happen only once per snapshot.

Usage:
    code-sim-batch-eval [--output-dir DIR] [--top-k N] [--relevance-thresholds 0.25 0.30 ...]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluate import run_evaluation
from .snapshot import _META_FILE, _resolve_snapshots_dir


def _bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(values) < 2:
        v = statistics.mean(values) if values else 0.0
        return {"lower": round(v, 6), "upper": round(v, 6)}
    rng = random.Random(42)
    boot_stats = sorted(
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(n_bootstrap)
    )
    alpha = (1 - ci) / 2
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = min(int((1 - alpha) * n_bootstrap) - 1, len(boot_stats) - 1)
    return {"lower": round(boot_stats[lo_idx], 6), "upper": round(boot_stats[hi_idx], 6)}

DEFAULT_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]


def _discover_snapshots(snapshot_dir: Optional[str] = None) -> List[Path]:
    """Find all snapshot directories, sorted by numeric prefix."""
    snap_root = _resolve_snapshots_dir(snapshot_dir)
    if not snap_root.is_dir():
        return []
    dirs = [d for d in snap_root.iterdir() if d.is_dir()]

    def _sort_key(p: Path) -> int:
        # Extract the repo-count number from names like "128repos" or
        # "trunc32768-8repos" — always the last numeric sequence.
        import re
        nums = re.findall(r"\d+", p.name)
        return int(nums[-1]) if nums else 0

    return sorted(dirs, key=_sort_key)


def _recompute_metrics_at_threshold(
    queries: List[Dict[str, Any]], threshold: float,
) -> Dict[str, Any]:
    """Recompute all threshold-dependent metrics from per-query hit distances."""
    n = len(queries)
    empty_rt5 = {
        "distribution": {i: 0 for i in range(6)},
        "mean_relevant": 0.0,
        "queries_with_hit": 0,
        "queries_with_hit_pct": 0.0,
    }
    if n == 0:
        return {
            "mrr": 0.0, "map": 0.0,
            "p_at_1": 0.0, "p_at_5": 0.0,
            "ci_mrr": {"lower": 0.0, "upper": 0.0},
            "ci_map": {"lower": 0.0, "upper": 0.0},
            "ci_p_at_1": {"lower": 0.0, "upper": 0.0},
            "ci_p_at_5": {"lower": 0.0, "upper": 0.0},
            "relevant_in_top5": empty_rt5,
        }

    rrs: List[float] = []
    aps: List[float] = []
    p1s: List[float] = []
    p5s: List[float] = []

    rel_dist = {i: 0 for i in range(6)}
    total_relevant = 0
    queries_with_hit = 0

    for q in queries:
        dists = [h["distance"] for h in q.get("hits", [])]

        # MRR
        rr = 0.0
        for rank, d in enumerate(dists, start=1):
            if d <= threshold:
                rr = 1.0 / rank
                break
        rrs.append(rr)

        # MAP (average precision)
        n_rel = 0
        prec_sum = 0.0
        for rank, d in enumerate(dists, start=1):
            if d <= threshold:
                n_rel += 1
                prec_sum += n_rel / rank
        aps.append(prec_sum / n_rel if n_rel > 0 else 0.0)

        # P@1
        top1 = dists[:1]
        p1s.append(
            sum(1 for d in top1 if d <= threshold) / len(top1) if top1 else 0.0
        )

        # P@5
        top5 = dists[:5]
        p5s.append(
            sum(1 for d in top5 if d <= threshold) / len(top5) if top5 else 0.0
        )

        # relevant_in_top5
        relevant = min(sum(1 for d in dists if d <= threshold), 5)
        rel_dist[relevant] += 1
        total_relevant += relevant
        if relevant > 0:
            queries_with_hit += 1

    return {
        "mrr": round(statistics.mean(rrs), 6),
        "map": round(statistics.mean(aps), 6),
        "p_at_1": round(statistics.mean(p1s), 6),
        "p_at_5": round(statistics.mean(p5s), 6),
        "ci_mrr": _bootstrap_ci(rrs),
        "ci_map": _bootstrap_ci(aps),
        "ci_p_at_1": _bootstrap_ci(p1s),
        "ci_p_at_5": _bootstrap_ci(p5s),
        "relevant_in_top5": {
            "distribution": rel_dist,
            "mean_relevant": round(total_relevant / n, 4),
            "queries_with_hit": queries_with_hit,
            "queries_with_hit_pct": round(queries_with_hit / n * 100, 2),
        },
    }


def _extract_metrics(
    json_path: Path, thresholds: List[float],
) -> Dict[str, Any]:
    """Extract metrics from an evaluation JSON file for every threshold."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    idx = data.get("index", {})
    qs = data.get("query_summary", {})
    sc = data.get("scalability", {})
    ci_raw = data.get("retrieval_quality", {}).get("confidence_intervals", {})
    ql = sc.get("query_latency", {})
    rq = data.get("retrieval_quality", {})

    base: Dict[str, Any] = {
        "corpus_repos": idx.get("total_repos", 0),
        "corpus_elements": idx.get("total_elements", 0),
        "queries": qs.get("total_queries", 0),
        "mean_top1_dist": rq.get("top1_distance", {}).get("mean", 0.0),
        "query_latency_ms": ql.get("mean_ms", 0.0),
        "query_latency_median_ms": ql.get("median_ms", 0.0),
        "index_size_bytes": sc.get("index_size_bytes", 0),
        "embed_ms_per_element": sc.get("embed_ms_per_element", 0.0),
        "embed_elements_per_second": sc.get("embed_elements_per_second", 0.0),
    }

    queries = data.get("queries", [])
    per_threshold: Dict[str, Dict[str, Any]] = {}
    for t in thresholds:
        per_threshold[str(t)] = _recompute_metrics_at_threshold(queries, t)

    base["per_threshold"] = per_threshold
    return base


def _print_table(
    rows: List[Dict[str, Any]], thresholds: List[float],
) -> None:
    """Print a results table for each threshold."""
    for t in thresholds:
        t_key = str(t)
        print(f"\n  Threshold = {t}")
        header = (
            f"  {'Snapshot':<12} {'Repos':>5} {'Elements':>8} "
            f"{'MRR':>6} {'P@1':>6} {'P@5':>6} "
            f"{'Top1 Dist':>9} {'Size GB':>8} "
            f"{'Lat ms':>7} {'Lat med':>8}"
        )
        sep = "  " + "-" * (len(header) - 2)
        print(sep)
        print(header)
        print(sep)

        for r in rows:
            tm = r.get("per_threshold", {}).get(t_key, {})
            size_gb = r.get("index_size_bytes", 0) / 1e9
            print(
                f"  {r['snapshot']:<12} {r['repos']:>5} {r['elements']:>8} "
                f"{tm.get('mrr', 0):>6.3f} {tm.get('p_at_1', 0):>6.3f} {tm.get('p_at_5', 0):>6.3f} "
                f"{r['mean_top1_dist']:>9.4f} {size_gb:>8.2f} "
                f"{r.get('query_latency_ms', 0):>7.1f} {r.get('query_latency_median_ms', 0):>8.1f}"
            )

        print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-batch-eval",
        description="Run public embedding evaluation across all snapshots.",
    )
    parser.add_argument(
        "--output-dir", default="eval_results",
        help="Directory for output JSON files (default: eval_results/)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Results per query (default: 5)")
    parser.add_argument(
        "--relevance-thresholds", type=float, nargs="+",
        default=DEFAULT_THRESHOLDS,
        help=f"Distance thresholds for relevance (default: {DEFAULT_THRESHOLDS})",
    )
    parser.add_argument("--no-queries", action="store_true", help="Omit per-query data from JSON")
    parser.add_argument(
        "--snapshot-dir",
        help="Directory containing snapshots. Default: <CODE_SIM_DB_PATH>/../snapshots/",
    )
    args = parser.parse_args()

    thresholds = sorted(args.relevance_thresholds)

    # Discover snapshots
    snapshots = _discover_snapshots(args.snapshot_dir)
    if not snapshots:
        print("Error: no snapshots found.", file=sys.stderr)
        sys.exit(1)

    snap_names = [s.name for s in snapshots]
    print(f"Found {len(snapshots)} snapshots: {', '.join(snap_names)}")
    print(f"Thresholds: {thresholds}")
    print()

    # Prepare output dir
    out_root = Path(args.output_dir)
    emb_dir = out_root / "embedding"
    emb_dir.mkdir(parents=True, exist_ok=True)

    include_queries = not args.no_queries
    table_rows: List[Dict[str, Any]] = []
    results: Dict[str, Dict[str, Any]] = {}
    summary_snapshots: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    # Use the first threshold for run_evaluation (only affects the metrics
    # it prints to console; we recompute all thresholds from per-query data).
    eval_threshold = thresholds[0]

    # ── Embedding evaluation (all snapshots) ───────────────────
    print("=" * 60)
    print("Public embedding evaluation")
    print("=" * 60)
    for i, snap_path in enumerate(snapshots, 1):
        name = snap_path.name
        snap_abs = str(snap_path)
        emb_json = emb_dir / f"{name}.json"

        print(f"\n[{i}/{len(snapshots)}] {name}: Embedding evaluation ...", flush=True)
        try:
            run_evaluation(
                snapshot=snap_abs,
                json_path=str(emb_json),
                check_private=False,
                check_public=True,
                top_k=args.top_k,
                include_queries=include_queries,
                relevance_threshold=eval_threshold,
            )
            results[name] = _extract_metrics(emb_json, thresholds)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    # ── Collect table rows ─────────────────────────────────────
    for snap_path in snapshots:
        name = snap_path.name
        snap_entry: Dict[str, Any] = {"name": name}
        metrics = results.get(name)

        if metrics:
            snap_entry["embedding"] = metrics
            table_rows.append({
                "snapshot": name,
                "repos": metrics["corpus_repos"],
                "elements": metrics["corpus_elements"],
                "mean_top1_dist": metrics["mean_top1_dist"],
                "index_size_bytes": metrics.get("index_size_bytes", 0),
                "query_latency_ms": metrics.get("query_latency_ms", 0.0),
                "query_latency_median_ms": metrics.get("query_latency_median_ms", 0.0),
                "per_threshold": metrics.get("per_threshold", {}),
            })

        summary_snapshots.append(snap_entry)

    elapsed = time.monotonic() - t0

    # ── Print results table ────────────────────────────────────
    print()
    print("BATCH EVALUATION RESULTS")
    print(f"Thresholds: {thresholds}    Top-k: {args.top_k}")
    _print_table(table_rows, thresholds)

    # ── Write summary JSON ─────────────────────────────────────
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_config": {
            "top_k": args.top_k,
            "relevance_thresholds": thresholds,
            "collection": "public",
        },
        "wall_time_min": round(elapsed / 60, 1),
        "snapshots": summary_snapshots,
    }
    summary_path = out_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary -> {summary_path}")
    print(f"Details -> {emb_dir}/")
    print(f"Wall time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
