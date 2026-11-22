"""
test_model_and_metrics.py
=========================
Unit tests for model loading configuration, metrics computation, and cost formulas.
"""

from pathlib import Path
import pytest
from src.models.model_loader import detect_device, get_torch_dtype
from src.reporting.metrics import compute_throughput, compute_speedup_percentage, compute_cost_per_million_tokens


def test_device_detection():
    """Verify that detect_device returns a valid platform string."""
    device = detect_device()
    assert device in ("cuda", "mps", "cpu"), f"Invalid device detected: {device}"


def test_torch_dtype_selection():
    """Verify bfloat16 is chosen for mps/cuda, float32 for cpu."""
    import torch
    assert get_torch_dtype("cuda") == torch.bfloat16
    assert get_torch_dtype("mps") == torch.bfloat16
    assert get_torch_dtype("cpu") == torch.float32


def test_throughput_calculation():
    """Verify token generation speed formula."""
    # 150 tokens in 5.0 seconds = 30.0 tok/s
    speed = compute_throughput(150, 5.0)
    assert speed == 30.0

    # Zero elapsed seconds should return 0.0 safely
    assert compute_throughput(150, 0.0) == 0.0


def test_speedup_percentage():
    """Verify speedup calculation formula."""
    # Baseline 30 tok/s -> Speculative 45 tok/s = +50.0% speedup
    speedup = compute_speedup_percentage(30.0, 45.0)
    assert speedup == 50.0


def test_cost_per_million_tokens_formula():
    """Verify Assignment 1B official cost formula:
    Cost / 1M tokens = (1,000,000 / throughput_tok_s) / 3600 * hourly_rate
    """
    # 50 tok/s on T4 (₹3.50/hr):
    # (1,000,000 / 50) = 20,000 seconds
    # (20,000 / 3600) * 3.50 = 5.555... * 3.50 = 19.44
    cost_t4 = compute_cost_per_million_tokens(50.0, 3.50)
    assert abs(cost_t4 - 19.44) <= 0.02

    # 100 tok/s on A100 (₹12.00/hr):
    # (1,000,000 / 100) / 3600 * 12.00 = 10,000 / 3600 * 12 = 33.33
    cost_a100 = compute_cost_per_million_tokens(100.0, 12.00)
    assert abs(cost_a100 - 33.33) <= 0.02


def test_canonical_prompts_configured():
    """Verify the 3 canonical domain prompts are defined in configs/model_config.yaml."""
    import yaml
    base_dir = Path(__file__).resolve().parents[1]
    with open(base_dir / "configs" / "model_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    prompts = cfg.get("evaluation_prompts", [])
    assert len(prompts) == 3, f"Expected 3 canonical prompts, found {len(prompts)}"
    assert "NTFS" in prompts[0]["text"]
    assert "CPU" in prompts[1]["text"]
    assert "EC2" in prompts[2]["text"]
