"""
charts.py
=========
Generates presentation-ready publication charts for Assignment 1B deliverables.

Visualizations:
1. Corpus Cleaning Progression (before/after cleaning stages)
2. Decoding Strategies Throughput Comparison
3. Speculative Decoding Latency & Speedup
4. Production Economics (Throughput vs VRAM vs Cost / 1M Tokens)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # Headless backend for scripts
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_corpus_cleaning(
    stats_json: Path | str = "results/corpus_statistics.json",
    output_path: Path | str = "results/charts/corpus_cleaning.png",
) -> None:
    """Plot reduction of corpus size across data cleaning pipeline stages."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    stages = [
        "Raw Extracted",
        "Length Filter\n(>=100 chars)",
        "Exact Dedup\n(SHA-256)",
        "Language Filter\n(English)",
    ]
    chars_millions = [
        data.get("total_chars_before_cleaning", 0) / 1e6,
        data.get("total_chars_after_length_filter", 0) / 1e6,
        data.get("total_chars_after_dedup", 0) / 1e6,
        data.get("total_chars_after_language_filter", 0) / 1e6,
    ]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(stages, chars_millions, color=["#1f77b4", "#aec7e8", "#2ca02c", "#2ca02c"], width=0.55, edgecolor="black", linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.2f}M chars", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Corpus Volume (Million Characters)", fontsize=11, fontweight="bold")
    ax.set_title("Part A Step 1: Domain Corpus Cleaning & Filtering Progression", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylim(0, max(chars_millions) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_p)
    plt.close()
    logger.info("Saved corpus cleaning plot to %s", out_p)


def plot_decoding_comparison(
    decoding_json: Path | str = "results/decoding/decoding_results.json",
    output_path: Path | str = "results/charts/decoding_comparison.png",
) -> None:
    """Plot tokens per second across all 7 decoding configurations."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(decoding_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    matrix = data.get("matrix", [])
    if not matrix:
        return

    strategies = ["greedy", "beam_search", "top_k", "top_p", "temp_0_2", "temp_0_7", "temp_1_0"]
    labels = ["Greedy", "Beam (4)", "Top-K (50)", "Top-P (0.9)", "Temp 0.2", "Temp 0.7", "Temp 1.0"]

    avg_speeds = []
    for s in strategies:
        vals = [row[s]["tok_per_sec"] for row in matrix if s in row]
        avg_speeds.append(np.mean(vals) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#17becf", "#9467bd", "#8c564b", "#e377c2"]
    bars = ax.bar(labels, avg_speeds, color=colors, width=0.6, edgecolor="black", linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.1f} t/s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Inference Throughput (Tokens / Second)", fontsize=11, fontweight="bold")
    ax.set_title("Part C Step 1: Throughput Across 5 Decoding Strategies (7 Configurations)", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylim(0, max(avg_speeds) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_p)
    plt.close()
    logger.info("Saved decoding comparison plot to %s", out_p)


def plot_speculative_decoding(
    spec_json: Path | str = "results/speculative/speculative_results.json",
    output_path: Path | str = "results/charts/speculative_speedup.png",
) -> None:
    """Plot baseline vs speculative throughput and latency."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(spec_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not runs:
        return

    prompts = [r["prompt_id"].upper() for r in runs]
    std_toks = [r["standard"]["tok_per_sec"] for r in runs]
    spec_toks = [r["speculative"]["tok_per_sec"] for r in runs]

    x = np.arange(len(prompts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    b1 = ax.bar(x - width / 2, std_toks, width, label="Standard Autoregressive (1.5B)", color="#1f77b4", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width / 2, spec_toks, width, label="Speculative Assisted (1.5B + 0.5B Draft)", color="#2ca02c", edgecolor="black", linewidth=0.8)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(prompts, fontsize=11, fontweight="bold")
    ax.set_ylabel("Throughput (Tokens / Second)", fontsize=11, fontweight="bold")
    ax.set_title(f"Part C Step 2: Speculative Decoding Acceleration (Avg +{data['metadata'].get('average_speedup_pct', 0)}%)", fontsize=12, fontweight="bold", pad=15)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_p)
    plt.close()
    logger.info("Saved speculative speedup plot to %s", out_p)


def plot_production_economics(
    cost_json: Path | str = "results/quantization/quantization_cost_results.json",
    output_path: Path | str = "results/charts/production_economics.png",
) -> None:
    """Plot VRAM, throughput, and Cost / 1M tokens across the 3 production configurations."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(cost_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfgs = data.get("configurations", [])
    if not cfgs:
        return

    names = ["BF16 Baseline", "4-bit NF4", "4-bit NF4 +\nSpeculative"]
    vram = [c["peak_vram_gb"] for c in cfgs]
    throughput = [c["throughput_tok_s"] for c in cfgs]
    cost_t4 = [c["cost_t4_inr"] for c in cfgs]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # 1. Peak VRAM
    b1 = ax1.bar(names, vram, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black", linewidth=0.8)
    ax1.set_title("Peak VRAM / Memory (GB)", fontweight="bold")
    ax1.set_ylabel("GB", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f}G", ha="center", va="bottom", fontweight="bold")

    # 2. Throughput
    b2 = ax2.bar(names, throughput, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black", linewidth=0.8)
    ax2.set_title("Inference Speed (Tokens / Sec)", fontweight="bold")
    ax2.set_ylabel("tok/s", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontweight="bold")

    # 3. Cost / 1M Tokens (T4)
    b3 = ax3.bar(names, cost_t4, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black", linewidth=0.8)
    ax3.set_title("Cost per 1M Tokens (T4 @ ₹3.50/hr)", fontweight="bold")
    ax3.set_ylabel("Cost (INR ₹)", fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in b3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"₹{h:.2f}", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Part C Step 3: Production Serving Optimization & Economics", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_p)
    plt.close()
    logger.info("Saved production economics plot to %s", out_p)


def generate_all_charts():
    """Generate all figures in results/charts."""
    plot_corpus_cleaning()
    try:
        plot_decoding_comparison()
    except Exception as e:
        logger.warning("Decoding chart failed: %s", e)
    try:
        plot_speculative_decoding()
    except Exception as e:
        logger.warning("Speculative chart failed: %s", e)
    try:
        plot_production_economics()
    except Exception as e:
        logger.warning("Production economics chart failed: %s", e)


if __name__ == "__main__":
    generate_all_charts()
