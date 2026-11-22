"""
charts.py
=========
Generates presentation-ready publication charts for Assignment 1B deliverables.

Visualizations:
1. Corpus Cleaning Progression (`corpus_cleaning.png`)
2. QLoRA / LoRA Training & Evaluation Loss Convergence (`training_loss.png`)
3. Decoding Strategies Throughput Comparison (`decoding_comparison.png`)
4. Speculative Decoding Throughput Comparison (`speculative_speedup.png`)
5. Production Economics: VRAM, Throughput & Cost / 1M Tokens (`production_economics.png`)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
    data = json.loads(Path(stats_json).read_text(encoding="utf-8"))

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

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    bars = ax.bar(stages, chars_millions, color=["#1f77b4", "#aec7e8", "#2ca02c", "#2ca02c"], width=0.55, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.2f}M chars", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_ylabel("Corpus Volume (Million Characters)", fontsize=10.5, fontweight="bold")
    ax.set_title("Part A Step 1: Domain Corpus Cleaning & Filtering Progression", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, max(chars_millions) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_p)
    plt.close()


def plot_training_loss(
    history_json: Path | str = "results/training/training_history.json",
    output_path: Path | str = "results/charts/training_loss.png",
) -> None:
    """Plot training and evaluation loss progression across steps."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(Path(history_json).read_text(encoding="utf-8"))

    t_steps = [e["step"] for e in history if "loss" in e]
    t_loss = [e["loss"] for e in history if "loss" in e]
    e_steps = [e["step"] for e in history if "eval_loss" in e]
    e_loss = [e["eval_loss"] for e in history if "eval_loss" in e]

    if not t_steps and not e_steps:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    if t_steps:
        ax.plot(t_steps, t_loss, marker="o", linewidth=2, color="#1f77b4", label="Training Loss")
    if e_steps:
        ax.plot(e_steps, e_loss, marker="s", linewidth=2.2, linestyle="--", color="#d62728", label="Evaluation Loss")
        for s, val in zip(e_steps, e_loss):
            ax.annotate(f"{val:.3f}", (s, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#d62728")

    ax.set_xlabel("Optimization Step", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10.5, fontweight="bold")
    ax.set_title("Part B Step 2: Adapter B (r=16, α=32) Training & Eval Loss Convergence", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_p)
    plt.close()


def plot_decoding_comparison(
    decoding_json: Path | str = "results/decoding/decoding_results.json",
    output_path: Path | str = "results/charts/decoding_comparison.png",
) -> None:
    """Plot tokens per second across all 7 decoding configurations."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(decoding_json).read_text(encoding="utf-8"))
    matrix = data.get("matrix", [])
    if not matrix:
        return

    strategies = ["greedy", "beam_search", "top_k", "top_p", "temp_0_2", "temp_0_7", "temp_1_0"]
    labels = ["Greedy", "Beam (4)", "Top-K (50)", "Top-P (0.9)", "Temp 0.2", "Temp 0.7", "Temp 1.0"]
    avg_speeds = [float(np.mean([row[s]["tok_per_sec"] for row in matrix if s in row])) for s in strategies]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#17becf", "#9467bd", "#8c564b", "#e377c2"]
    bars = ax.bar(labels, avg_speeds, color=colors, width=0.6, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f} t/s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Inference Throughput (Tokens / Second)", fontsize=10.5, fontweight="bold")
    ax.set_title("Part C Step 1: Throughput Across 5 Decoding Strategies (7 Configurations)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, max(avg_speeds) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_p)
    plt.close()


def plot_speculative_decoding(
    spec_json: Path | str = "results/speculative/speculative_results.json",
    output_path: Path | str = "results/charts/speculative_speedup.png",
) -> None:
    """Plot standard vs speculative throughput across the 3 canonical prompts."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(spec_json).read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    if not runs:
        return

    prompts = [r["prompt_id"].upper() for r in runs]
    std_toks = [r["standard"]["tok_per_sec"] for r in runs]
    spec_toks = [r["speculative"]["tok_per_sec"] for r in runs]
    avg_spd = float(data["metadata"].get("average_speedup_pct", 0.0))
    dev = data["metadata"].get("device", "device")

    x = np.arange(len(prompts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    b1 = ax.bar(x - width / 2, std_toks, width, label="Standard Autoregressive (1.5B)", color="#1f77b4", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width / 2, spec_toks, width, label="Speculative Assisted (1.5B + 0.5B Draft)", color="#2ca02c", edgecolor="black", linewidth=0.8)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(prompts, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Throughput (Tokens / Second)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max(std_toks + spec_toks) * 1.22)
    ax.set_title(f"Part C Step 2: Speculative Decoding on {dev.upper()} (Avg {avg_spd:+0.1f}%)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_p)
    plt.close()


def plot_production_economics(
    cost_json: Path | str = "results/quantization/quantization_cost_results.json",
    output_path: Path | str = "results/charts/production_economics.png",
) -> None:
    """Plot empirical VRAM, throughput, and Cost / 1M tokens for measured configurations."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(cost_json).read_text(encoding="utf-8"))
    cfgs = data.get("configurations", [])
    if not cfgs:
        return

    measured = [c for c in cfgs if c.get("measured", True) and c.get("throughput_tok_s") is not None]
    if not measured:
        return

    short_names = [
        c["configuration"].replace(" (Part A baseline)", "\n(Baseline)").replace(" (Step 2)", "").replace(" + ", "\n+ ")
        for c in measured
    ]
    vram = [c["peak_vram_gb"] for c in measured]
    throughput = [c["throughput_tok_s"] for c in measured]
    cost_t4 = [c["cost_t4_inr"] for c in measured]
    cost_a100 = [c["cost_a100_inr"] for c in measured]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"][: len(measured)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.5), dpi=200)

    b1 = ax1.bar(short_names, vram, color=palette, width=0.5, edgecolor="black", linewidth=0.8)
    ax1.set_title("Peak Memory / VRAM (GB)", fontweight="bold")
    ax1.set_ylabel("GB", fontweight="bold")
    ax1.set_ylim(0, max(vram) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + max(vram) * 0.03, f"{h:.2f} GB", ha="center", va="bottom", fontweight="bold")

    b2 = ax2.bar(short_names, throughput, color=palette, width=0.5, edgecolor="black", linewidth=0.8)
    ax2.set_title("Empirical Throughput (tok/s)", fontweight="bold")
    ax2.set_ylabel("tok/s", fontweight="bold")
    ax2.set_ylim(0, max(throughput) * 1.25)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + max(throughput) * 0.03, f"{h:.2f}", ha="center", va="bottom", fontweight="bold")

    x = np.arange(len(short_names))
    w = 0.35
    b3a = ax3.bar(x - w / 2, cost_t4, width=w, label="T4 (₹3.50/hr)", color="#2ca02c", edgecolor="black", linewidth=0.8)
    b3b = ax3.bar(x + w / 2, cost_a100, width=w, label="A100 (₹12.00/hr)", color="#d62728", edgecolor="black", linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names)
    ax3.set_title("Cost per 1M Tokens (INR ₹)", fontweight="bold")
    ax3.set_ylabel("Cost (₹)", fontweight="bold")
    ax3.set_ylim(0, max(cost_a100) * 1.25)
    ax3.legend(fontsize=8.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in list(b3a) + list(b3b):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + max(cost_a100) * 0.02, f"₹{h:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    suffix = "CUDA Measured" if data["metadata"].get("cuda_4bit_measured") else "Local MPS Baseline (4-bit Pending Colab T4)"
    plt.suptitle(f"Part C Step 3: Empirical Production Economics — {suffix}", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()


def generate_all_charts() -> None:
    """Generate all assignment charts cleanly."""
    for fn in [
        plot_corpus_cleaning,
        plot_training_loss,
        plot_decoding_comparison,
        plot_speculative_decoding,
        plot_production_economics,
    ]:
        try:
            fn()
        except Exception as e:
            logger.warning("Chart %s skipped: %s", fn.__name__, e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_charts()
