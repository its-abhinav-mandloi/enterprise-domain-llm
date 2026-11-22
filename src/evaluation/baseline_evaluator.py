"""
baseline_evaluator.py
=====================
Runs pre-adaptation baseline generation on the canonical domain-specific prompts.

Assignment 1B — Part A Step 2:
- Load chosen base model in bfloat16
- Run on 3 domain-specific prompts (no system prompt)
- Save generated outputs and architecture metadata to results/baseline/
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

CANONICAL_PROMPTS = [
    {
        "id": "prompt_1",
        "category": "linux_storage",
        "prompt": "How do I troubleshoot an Ubuntu server that cannot mount an NTFS drive?",
    },
    {
        "id": "prompt_2",
        "category": "linux_performance",
        "prompt": "How do I investigate high disk I/O and CPU usage on a Linux production server?",
    },
    {
        "id": "prompt_3",
        "category": "cloud_networking",
        "prompt": "How do I troubleshoot AWS EC2 instance connectivity and SSH access issues?",
    },
]


def run_baseline_evaluation(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: Path | str = "results/baseline",
    max_new_tokens: int = 150,
    device: str | None = None,
) -> Dict[str, Any]:
    """Execute the baseline evaluation and save results.

    Returns dictionary containing architecture details and output generations.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    logger.info("Loading baseline model: %s on %s with dtype %s", model_name, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device != "mps" else None,
        trust_remote_code=False,
    )
    if device == "mps":
        model = model.to("mps")
    model.eval()

    # Model architecture metadata
    cfg = model.config
    param_count = sum(p.numel() for p in model.parameters())
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
    hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    vocab_size = getattr(cfg, "vocab_size", len(tokenizer))

    metadata = {
        "model_identifier": model_name,
        "parameter_count": param_count,
        "parameter_count_formatted": f"{param_count / 1e9:.2f}B" if param_count >= 1e9 else f"{param_count / 1e6:.1f}M",
        "num_decoder_layers": num_layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "tokenizer": type(tokenizer).__name__,
        "device": device,
        "precision": str(dtype),
        "generation_settings": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "system_prompt": None,
        },
    }

    # Run prompts
    results = []
    print("\n" + "=" * 70)
    print("PART A STEP 2 — BASELINE MODEL GENERATION (NO SYSTEM PROMPT)")
    print("=" * 70)

    for item in CANONICAL_PROMPTS:
        p_id = item["id"]
        prompt_text = item["prompt"]
        category = item["category"]

        # Tokenize (pure prompt, no system prompt as mandated)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        # Warmup and synchronize
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        output_tokens = outputs[0][input_len:]
        num_generated = len(output_tokens)
        tok_per_sec = round(num_generated / max(elapsed, 1e-4), 2)
        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

        print(f"\n--- [{p_id}] ({category}) ---")
        print(f"Prompt: {prompt_text}")
        print(f"Latency: {elapsed:.2f}s | Generated: {num_generated} tok | Speed: {tok_per_sec} tok/s")
        print(f"Output:\n{generated_text[:300]}...")

        results.append({
            "id": p_id,
            "category": category,
            "prompt": prompt_text,
            "generated_text": generated_text,
            "generation_time_sec": round(elapsed, 3),
            "generated_tokens": num_generated,
            "tokens_per_second": tok_per_sec,
        })

    payload = {
        "metadata": metadata,
        "baseline_runs": results,
    }

    # Save JSON
    json_path = out_path / "baseline_outputs.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Save Markdown summary table
    md_path = out_path / "baseline_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Baseline Model Output Report (Part A Step 2)\n\n")
        f.write(f"- **Model**: `{metadata['model_identifier']}`\n")
        f.write(f"- **Parameters**: `{metadata['parameter_count_formatted']}` ({metadata['parameter_count']:,})\n")
        f.write(f"- **Decoder Layers**: `{metadata['num_decoder_layers']}`\n")
        f.write(f"- **Hidden Size**: `{metadata['hidden_size']}`\n")
        f.write(f"- **Vocabulary Size**: `{metadata['vocab_size']:,}`\n")
        f.write(f"- **Precision**: `{metadata['precision']}`\n")
        f.write(f"- **Device**: `{metadata['device']}`\n\n")
        f.write("## Baseline Generations\n\n")
        for r in results:
            f.write(f"### {r['id'].upper()} ({r['category']})\n\n")
            f.write(f"**Prompt:** {r['prompt']}\n\n")
            f.write(f"**Metrics:** {r['generation_time_sec']}s, {r['generated_tokens']} tokens, {r['tokens_per_second']} tok/s\n\n")
            f.write(f"**Generated Output:**\n\n```text\n{r['generated_text']}\n```\n\n")

    logger.info("Saved baseline outputs to %s and %s", json_path, md_path)
    return payload


if __name__ == "__main__":
    import sys
    m_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
    run_baseline_evaluation(model_name=m_name)
