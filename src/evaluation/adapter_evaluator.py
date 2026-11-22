"""
adapter_evaluator.py
====================
Runs evaluation of the fine-tuned LoRA adapter on the canonical domain-specific prompts.

Assignment 1B — Part B Step 3:
- Load base model + LoRA adapter
- Run on the same 3 domain-specific prompts
- Record latency, tokens, tokens/sec, and full generated text
- Save results to results/evaluation/adapter_outputs.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.baseline_evaluator import CANONICAL_PROMPTS

logger = logging.getLogger(__name__)


def run_adapter_evaluation(
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: Path | str = "models/lora_adapter",
    output_dir: Path | str = "results/evaluation",
    max_new_tokens: int = 150,
    device: str | None = None,
) -> Dict[str, Any]:
    """Execute evaluation with the fine-tuned adapter."""
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

    logger.info("Loading base model: %s with adapter: %s on %s", base_model_name, adapter_path, device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device if device != "mps" else None,
        trust_remote_code=False,
    )
    if device == "mps":
        base_model = base_model.to("mps")

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    results = []
    print("\n" + "=" * 70)
    print("PART B STEP 3 — FINE-TUNED ADAPTER MODEL GENERATION")
    print("=" * 70)

    for item in CANONICAL_PROMPTS:
        p_id = item["id"]
        prompt_text = item["prompt"]
        category = item["category"]

        # Apply chat template if supported, or format cleanly
        messages = [{"role": "user", "content": prompt_text}]
        try:
            formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_input = prompt_text

        inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

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
        "base_model": base_model_name,
        "adapter_path": str(adapter_path),
        "device": device,
        "adapter_runs": results,
    }

    json_path = out_path / "adapter_outputs.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved adapter outputs to %s", json_path)
    return payload


if __name__ == "__main__":
    import sys
    b_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
    run_adapter_evaluation(base_model_name=b_name)
