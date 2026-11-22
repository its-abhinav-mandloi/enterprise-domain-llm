"""
metrics.py
==========
Standardized metric calculations and cost estimation functions for Assignment 1B.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import torch


def compute_throughput(num_tokens: int, elapsed_seconds: float) -> float:
    """Compute tokens per second."""
    if elapsed_seconds <= 0:
        return 0.0
    return round(float(num_tokens) / elapsed_seconds, 2)


def compute_speedup_percentage(baseline_speed: float, optimized_speed: float) -> float:
    """Compute percentage speedup between two throughput or latency values."""
    if baseline_speed <= 0:
        return 0.0
    return round(((optimized_speed - baseline_speed) / baseline_speed) * 100.0, 1)


def compute_cost_per_million_tokens(throughput_tokens_per_sec: float, hourly_gpu_rate_inr: float) -> float:
    """Calculate serving cost per 1 million tokens in INR.

    Formula from Assignment 1B specification:
    Cost per 1M tokens = (1,000,000 / throughput_tok_per_sec) / 3600 * hourly_rate
    """
    if throughput_tokens_per_sec <= 0:
        return 0.0
    return round((1_000_000.0 / throughput_tokens_per_sec) / 3600.0 * hourly_gpu_rate_inr, 2)


def get_device_memory_stats() -> Dict[str, float]:
    """Get active device memory allocation in Megabytes and Gigabytes."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        return {
            "device": "cuda",
            "allocated_mb": round(allocated, 2),
            "peak_mb": round(peak, 2),
            "allocated_gb": round(allocated / 1024.0, 2),
            "peak_gb": round(peak / 1024.0, 2),
        }
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            allocated = torch.mps.current_allocated_memory() / (1024**2)
            driver = torch.mps.driver_allocated_memory() / (1024**2)
            return {
                "device": "mps",
                "allocated_mb": round(allocated, 2),
                "peak_mb": round(driver, 2),
                "allocated_gb": round(allocated / 1024.0, 2),
                "peak_gb": round(driver / 1024.0, 2),
            }
        except Exception:
            return {"device": "mps", "allocated_mb": 0.0, "peak_mb": 0.0, "allocated_gb": 0.0, "peak_gb": 0.0}
    else:
        return {"device": "cpu", "allocated_mb": 0.0, "peak_mb": 0.0, "allocated_gb": 0.0, "peak_gb": 0.0}
