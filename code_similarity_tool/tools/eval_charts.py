#!/usr/bin/env python3
"""CLI entry point: code-sim-eval-charts

Generate publication-ready charts from one or more eval JSON files produced by
code-sim-eval.

Usage:
    code-sim-eval-charts FILE [FILE ...] [--out-dir DIR] [--format png|pdf|svg] [--dpi N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Lazy-loaded matplotlib references (set by _ensure_matplotlib)
plt = None
matplotlib = None


def _ensure_matplotlib():
    """Import matplotlib on first use — avoids module-level import issues."""
    global plt, matplotlib
    if plt is not None:
        return
    try:
        import matplotlib as _mpl

        _mpl.use("Agg")  # non-interactive backend — works headless
        import matplotlib.pyplot as _plt

        matplotlib = _mpl
        plt = _plt
    except ImportError as exc:
        print(
            f"Error: matplotlib is required for chart generation ({exc}).\n"
            '  Install with: pip install "code-similarity-tool[charts]"',
            file=sys.stderr,
        )
        sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _label(data: Dict[str, Any]) -> str:
    """Derive a short label from an eval JSON (e.g. '5 repos')."""
    n_repos = data.get("index", {}).get("total_repos", "?")
    snap = data.get("snapshot", "")
    return snap if snap else f"{n_repos} repos"


def _sorted_by_repos(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort datasets by number of repos for consistent x-axis ordering."""
    return sorted(datasets, key=lambda d: d.get("index", {}).get("total_repos", 0))


# ── Chart generators ─────────────────────────────────────────────────────────
# Each returns (fig, ax) or None if the required data is missing.


def chart_01_top1_distance_trend(
    datasets: List[Dict[str, Any]],
) -> Optional[Any]:
    """Line chart: mean/median top-1 distance vs corpus size."""
    ds = _sorted_by_repos(datasets)
    labels = [_label(d) for d in ds]
    means = []
    medians = []
    for d in ds:
        t1 = d.get("retrieval_quality", {}).get("top1_distance", {})
        if t1.get("mean") is None:
            return None
        means.append(t1["mean"])
        medians.append(t1["median"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(labels))
    ax.plot(x, means, "o-", label="Mean", linewidth=2, markersize=7)
    ax.plot(x, medians, "s--", label="Median", linewidth=2, markersize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Top-1 Distance")
    ax.set_xlabel("Corpus Size")
    ax.set_title("Top-1 Distance vs Corpus Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def chart_02_top1_distance_box(
    datasets: List[Dict[str, Any]],
) -> Optional[Any]:
    """Box plot: top-1 distance distribution from per-query data."""
    ds = _sorted_by_repos(datasets)
    all_dists = []
    labels = []
    for d in ds:
        queries = d.get("queries", [])
        dists = [q["top1_distance"] for q in queries if q.get("top1_distance") is not None]
        if not dists:
            # Fall back to summary stats for a synthetic box
            t1 = d.get("retrieval_quality", {}).get("top1_distance", {})
            if t1.get("mean") is None:
                continue
            dists = [t1["mean"]]  # degenerate single-point
        all_dists.append(dists)
        labels.append(_label(d))

    if not all_dists:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(all_dists, labels=labels, patch_artist=True)
    colors = plt.cm.Blues([0.3 + 0.5 * i / max(len(all_dists) - 1, 1) for i in range(len(all_dists))])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Top-1 Distance")
    ax.set_xlabel("Corpus Size")
    ax.set_title("Top-1 Distance Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def chart_03_hit_rate_bars(
    datasets: List[Dict[str, Any]],
) -> Optional[Any]:
    """Grouped bar chart: hit rate at each threshold across corpus sizes."""
    ds = _sorted_by_repos(datasets)
    labels = [_label(d) for d in ds]

    # Collect all thresholds from first dataset
    first_hr = ds[0].get("retrieval_quality", {}).get("hit_rate_at_threshold", {})
    if not first_hr:
        return None
    thresholds = sorted(first_hr.keys(), key=float)

    import numpy as np

    n_groups = len(thresholds)
    n_bars = len(ds)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, d in enumerate(ds):
        hr = d.get("retrieval_quality", {}).get("hit_rate_at_threshold", {})
        vals = [hr.get(t, 0.0) for t in thresholds]
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=_label(d))

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"\u2264{t}" for t in thresholds])
    ax.set_ylabel("Hit Rate (%)")
    ax.set_xlabel("Distance Threshold")
    ax.set_title("Hit Rate at Distance Thresholds")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def chart_04_distance_cdf(
    datasets: List[Dict[str, Any]],
) -> Optional[Any]:
    """Overlaid line chart: distance CDF comparison."""
    ds = _sorted_by_repos(datasets)
    has_cdf = False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in ds:
        cdf_data = d.get("distribution", {}).get("distance_cdf", [])
        if not cdf_data:
            continue
        has_cdf = True
        xs = [p["x"] for p in cdf_data]
        ys = [p["cdf"] * 100 for p in cdf_data]
        ax.plot(xs, ys, "o-", label=_label(d), linewidth=2, markersize=4)

    if not has_cdf:
        plt.close(fig)
        return None

    ax.set_xlabel("Distance")
    ax.set_ylabel("Cumulative % of Queries")
    ax.set_title("Distance CDF Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return fig


def chart_05_query_latency(
    datasets: List[Dict[str, Any]],
) -> Optional[Any]:
    """Line chart: query latency vs corpus size."""
    ds = _sorted_by_repos(datasets)
    labels = [_label(d) for d in ds]
    means = []
    medians = []
    for d in ds:
        lat = d.get("scalability", {}).get("query_latency", {})
        if not lat:
            return None
        means.append(lat.get("mean_ms", 0))
        medians.append(lat.get("median_ms", 0))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(labels))
    ax.plot(x, means, "o-", label="Mean", linewidth=2, markersize=7)
    ax.plot(x, medians, "s--", label="Median", linewidth=2, markersize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Corpus Size")
    ax.set_title("Query Latency vs Corpus Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def chart_06_index_size(
    datasets: List[Dict[str, Any]],
) -> Optional[Any]:
    """Line chart: index size (MB) vs corpus size."""
    ds = _sorted_by_repos(datasets)
    labels = [_label(d) for d in ds]
    sizes_mb = []
    for d in ds:
        size_bytes = d.get("scalability", {}).get("index_size_bytes")
        if size_bytes is None:
            size_bytes = d.get("index", {}).get("size_bytes")
        if size_bytes is None:
            return None
        sizes_mb.append(size_bytes / (1024 * 1024))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(labels))
    ax.plot(x, sizes_mb, "o-", color="tab:green", linewidth=2, markersize=7)
    ax.fill_between(x, sizes_mb, alpha=0.15, color="tab:green")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Index Size (MB)")
    ax.set_xlabel("Corpus Size")
    ax.set_title("Index Size vs Corpus Size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def chart_07_per_kind_metrics(
    datasets: List[Dict[str, Any]],
) -> Optional[Any]:
    """Grouped bar chart: mean top-1 distance per element kind across corpus sizes."""
    ds = _sorted_by_repos(datasets)

    # Collect all kinds across datasets
    all_kinds: set = set()
    for d in ds:
        by_kind = d.get("breakdowns", {}).get("by_kind", {})
        all_kinds.update(by_kind.keys())

    if not all_kinds:
        return None

    kinds = sorted(all_kinds)

    import numpy as np

    n_groups = len(kinds)
    n_bars = len(ds)
    width = 0.8 / max(n_bars, 1)
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, d in enumerate(ds):
        by_kind = d.get("breakdowns", {}).get("by_kind", {})
        vals = [by_kind.get(k, {}).get("mean", 0) or 0 for k in kinds]
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=_label(d))

    ax.set_xticks(list(x))
    ax.set_xticklabels(kinds)
    ax.set_ylabel("Mean Top-1 Distance")
    ax.set_xlabel("Element Kind")
    ax.set_title("Per-Element-Kind Match Quality")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def chart_08_summary_table(
    datasets: List[Dict[str, Any]],
) -> Optional[Any]:
    """Table image: all key metrics side-by-side."""
    ds = _sorted_by_repos(datasets)
    if not ds:
        return None

    col_labels = [_label(d) for d in ds]
    row_labels = [
        "Total Elements",
        "Total Repos",
        "Mean Top-1 Dist",
        "Median Top-1 Dist",
        "MRR",
        "P@1",
        "P@5",
        "nDCG@5",
        "Hit Rate @0.3",
        "Query Latency (ms)",
        "Index Size (MB)",
    ]

    cell_data = []
    for d in ds:
        t1 = d.get("retrieval_quality", {}).get("top1_distance", {})
        pak = d.get("retrieval_quality", {}).get("precision_at_k", {})
        ndcg = d.get("retrieval_quality", {}).get("ndcg_at_k", {})
        hr = d.get("retrieval_quality", {}).get("hit_rate_at_threshold", {})
        lat = d.get("scalability", {}).get("query_latency", {})
        size_bytes = d.get("scalability", {}).get("index_size_bytes", d.get("index", {}).get("size_bytes", 0))

        col = [
            f"{d.get('index', {}).get('total_elements', '?'):,}",
            str(d.get("index", {}).get("total_repos", "?")),
            f"{t1.get('mean', 0):.4f}" if t1.get("mean") is not None else "\u2014",
            f"{t1.get('median', 0):.4f}" if t1.get("median") is not None else "\u2014",
            f"{d.get('retrieval_quality', {}).get('mrr', 0):.4f}",
            f"{pak.get('p_at_1', 0):.4f}",
            f"{pak.get('p_at_5', 0):.4f}",
            f"{ndcg.get('ndcg_at_5', 0):.4f}",
            f"{hr.get('0.3', 0):.1f}%",
            f"{lat.get('mean_ms', 0):.1f}",
            f"{(size_bytes or 0) / (1024 * 1024):.1f}",
        ]
        cell_data.append(col)

    # Transpose: rows x cols
    table_data = [[cell_data[c][r] for c in range(len(ds))] for r in range(len(row_labels))]

    fig_height = 0.5 + 0.35 * len(row_labels)
    fig_width = 2.5 + 1.5 * len(ds)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title("Evaluation Summary", fontsize=13, fontweight="bold", pad=12)

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")
    # Style row labels
    for i in range(len(row_labels)):
        cell = table[i + 1, -1]
        cell.set_facecolor("#D9E2F3")

    fig.tight_layout()
    return fig


# ── Chart registry ────────────────────────────────────────────────────────────

CHARTS = [
    ("01_top1_distance_trend", chart_01_top1_distance_trend),
    ("02_top1_distance_distribution", chart_02_top1_distance_box),
    ("03_hit_rate_at_thresholds", chart_03_hit_rate_bars),
    ("04_distance_cdf_comparison", chart_04_distance_cdf),
    ("05_query_latency", chart_05_query_latency),
    ("06_index_size", chart_06_index_size),
    ("07_per_kind_metrics", chart_07_per_kind_metrics),
    ("08_summary_table", chart_08_summary_table),
]


# ── CLI ───────────────────────────────────────────────────────────────────────


def generate_charts(
    json_files: List[str],
    out_dir: str = "eval_charts",
    fmt: str = "png",
    dpi: int = 300,
) -> List[str]:
    """Generate all charts and return list of output file paths."""
    _ensure_matplotlib()

    datasets: List[Dict[str, Any]] = []
    for fpath in json_files:
        with open(fpath, encoding="utf-8") as f:
            datasets.append(json.load(f))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generated: List[str] = []
    for name, chart_fn in CHARTS:
        fig = chart_fn(datasets)
        if fig is None:
            print(f"  Skipped {name} (missing data)")
            continue
        fname = f"{name}.{fmt}"
        dest = out_path / fname
        fig.savefig(str(dest), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(dest))
        print(f"  {fname}")

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-eval-charts",
        description="Generate publication-ready charts from eval JSON files.",
    )
    parser.add_argument("files", nargs="+", metavar="FILE", help="Eval JSON file(s)")
    parser.add_argument("--out-dir", default="eval_charts", help="Output directory (default: eval_charts)")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png", help="Image format (default: png)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster output (default: 300)")
    args = parser.parse_args()

    print(f"Generating charts from {len(args.files)} file(s) ...")
    generated = generate_charts(args.files, out_dir=args.out_dir, fmt=args.format, dpi=args.dpi)
    print(f"\n{len(generated)} chart(s) written to {args.out_dir}/")


if __name__ == "__main__":
    main()
