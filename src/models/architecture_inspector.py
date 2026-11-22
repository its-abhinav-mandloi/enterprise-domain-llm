"""
architecture_inspector.py
==========================
Inspect and display the architecture details of a loaded HuggingFace model.

Provides:
- Layer-by-layer parameter counts
- Module type breakdown
- Trainable vs frozen parameter ratio
- Summary table (printed and returned as a dict)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def count_parameters(model: PreTrainedModel) -> dict[str, int]:
    """Count total, trainable, and frozen parameters.

    Returns
    -------
    Dict with keys ``total``, ``trainable``, ``frozen``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def get_layer_summary(model: PreTrainedModel) -> list[dict[str, Any]]:
    """Return a per-module parameter summary.

    Each entry has ``name``, ``type``, ``params``, ``trainable``.
    """
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not list(module.children()):  # leaf module
            params = sum(p.numel() for p in module.parameters(recurse=False))
            trainable = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )
            if params > 0:
                rows.append(
                    {
                        "name": name,
                        "type": type(module).__name__,
                        "params": params,
                        "trainable": trainable,
                    }
                )
    return rows


def get_module_type_breakdown(model: PreTrainedModel) -> dict[str, int]:
    """Aggregate parameter counts by module class name.

    Returns
    -------
    Dict mapping class name → total parameter count.
    """
    breakdown: dict[str, int] = defaultdict(int)
    for module in model.modules():
        if not list(module.children()):
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                breakdown[type(module).__name__] += params
    return dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))


def inspect_model(
    model: PreTrainedModel,
    print_layers: bool = True,
    max_layers: int = 40,
) -> dict[str, Any]:
    """Inspect a model and print a formatted summary.

    Parameters
    ----------
    model:
        The loaded HuggingFace model.
    print_layers:
        Whether to print the per-layer table.
    max_layers:
        Maximum number of leaf layers to print (avoids overwhelming output).

    Returns
    -------
    Dict with ``parameter_counts``, ``module_breakdown``, ``config_summary``,
    ``layers`` keys.
    """
    param_counts = count_parameters(model)
    module_breakdown = get_module_type_breakdown(model)
    layers = get_layer_summary(model)

    config_summary: dict[str, Any] = {}
    if hasattr(model, "config"):
        cfg = model.config
        for attr in [
            "model_type", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "intermediate_size",
            "vocab_size", "max_position_embeddings",
        ]:
            if hasattr(cfg, attr):
                config_summary[attr] = getattr(cfg, attr)

    print("\n" + "=" * 60)
    print(f"Model Architecture: {type(model).__name__}")
    print("=" * 60)

    print(f"\n{'Parameter Counts':}")
    print(f"  Total     : {param_counts['total']:>12,}")
    print(f"  Trainable : {param_counts['trainable']:>12,}  ({100 * param_counts['trainable'] / max(param_counts['total'], 1):.1f}%)")
    print(f"  Frozen    : {param_counts['frozen']:>12,}")

    if config_summary:
        print("\nConfig Summary:")
        for k, v in config_summary.items():
            print(f"  {k:<30}: {v}")

    print("\nModule Type Breakdown (top 10):")
    for cls, cnt in list(module_breakdown.items())[:10]:
        print(f"  {cls:<30}: {cnt:>12,} params")

    if print_layers:
        print(f"\nLeaf Layer Table (first {min(max_layers, len(layers))} of {len(layers)}):")
        print(f"  {'Name':<50} {'Type':<25} {'Params':>10}")
        print("  " + "-" * 87)
        for row in layers[:max_layers]:
            print(f"  {row['name'][:50]:<50} {row['type']:<25} {row['params']:>10,}")

    print("=" * 60 + "\n")

    return {
        "parameter_counts": param_counts,
        "module_breakdown": module_breakdown,
        "config_summary": config_summary,
        "layers": layers,
    }


if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)

    from src.models.model_loader import load_model_and_tokenizer

    model_name = sys.argv[1] if len(sys.argv) > 1 else "distilgpt2"
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    result = inspect_model(model)
    logger.info("Inspection complete. Total params: %d", result["parameter_counts"]["total"])
