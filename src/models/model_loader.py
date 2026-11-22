"""
model_loader.py
===============
Load a causal language model and tokenizer with device-aware configuration.

Device priority: CUDA → MPS (Apple Silicon) → CPU.
BF16 is used when supported (MPS / CUDA); FP32 fallback for CPU.

bitsandbytes 4-bit quantization is CUDA-only and raises a clear error on MPS.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Return the best available device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    """Return the preferred dtype for the given device.

    - CUDA and MPS → bfloat16 (stable, wide range)
    - CPU → float32
    """
    if device in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


def build_bnb_config(
    use_4bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
    use_double_quant: bool = True,
) -> BitsAndBytesConfig | None:
    """Build a BitsAndBytesConfig for quantized loading.

    Returns ``None`` when *use_4bit* is False.

    .. warning::
        4-bit NF4 quantisation requires CUDA.  Calling this on MPS or CPU
        will fail at model load time with a bitsandbytes error.
    """
    if not use_4bit:
        return None
    device = detect_device()
    if device != "cuda":
        raise RuntimeError(
            "4-bit quantization via bitsandbytes is only supported on CUDA devices. "
            f"Current device: '{device}'. "
            "On Apple MPS, use full BF16 (use_4bit=False). "
            "For 4-bit experiments, use Google Colab with a GPU runtime."
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
    )


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerBase:
    """Load a tokenizer from HuggingFace Hub or local path.

    Sets ``pad_token`` to ``eos_token`` if not already defined (common for
    decoder-only models such as LLaMA).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Tokenizer loaded: %s (vocab size %d)", model_name, len(tokenizer))
    return tokenizer


def load_model(
    model_name: str,
    device: str | None = None,
    use_4bit: bool = False,
    trust_remote_code: bool = False,
    low_cpu_mem_usage: bool = True,
) -> tuple[PreTrainedModel, str]:
    """Load a causal LM with appropriate dtype and optional quantization.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier or local directory path.
    device:
        Target device.  Auto-detected if ``None``.
    use_4bit:
        Load in 4-bit NF4 (CUDA only).  Raises on non-CUDA hardware.
    trust_remote_code:
        Allow executing custom modelling code from the Hub.
    low_cpu_mem_usage:
        Reduce peak RAM during loading.

    Returns
    -------
    Tuple ``(model, device)`` where model is moved to the target device.
    """
    if device is None:
        device = detect_device()
    logger.info("Loading model '%s' on device='%s'", model_name, device)

    dtype = get_torch_dtype(device)
    bnb_config = build_bnb_config(use_4bit=use_4bit)

    kwargs: dict = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Move to device if device_map was not used
    if bnb_config is None:
        model = model.to(device)

    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded: %.1fM parameters | dtype=%s | device=%s",
        param_count / 1e6,
        dtype,
        device,
    )
    return model, device


def load_model_and_tokenizer(
    model_name: str,
    device: str | None = None,
    use_4bit: bool = False,
    trust_remote_code: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    """Convenience wrapper: load both model and tokenizer.

    Returns
    -------
    Tuple ``(model, tokenizer, device)``.
    """
    tokenizer = load_tokenizer(model_name, trust_remote_code)
    model, device = load_model(model_name, device, use_4bit, trust_remote_code)
    return model, tokenizer, device


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, tokenizer, dev = load_model_and_tokenizer("distilgpt2")
    logger.info("Device: %s", dev)
    logger.info("Model class: %s", type(model).__name__)
