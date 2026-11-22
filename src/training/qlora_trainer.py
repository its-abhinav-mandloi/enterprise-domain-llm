"""
qlora_trainer.py
================
QLoRA / PEFT fine-tuning using Hugging Face PEFT + TRL SFTTrainer.

Adapter B Configuration:
- LoRA rank = 16, alpha = 32
- Target modules: q_proj, v_proj
- bfloat16 precision on Apple Silicon MPS
- SFTTrainer with SFTConfig and chat template formatting
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from src.models.model_loader import detect_device, get_torch_dtype

logger = logging.getLogger(__name__)


def load_qlora_config(config_path: Path) -> dict:
    """Load QLoRA configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_lora_config(lora_cfg: dict) -> LoraConfig:
    """Build a PEFT LoraConfig for Adapter B (Rank=16, Alpha=32, target: q_proj, v_proj)."""
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def format_dataset_with_chat_template(dataset: Dataset, tokenizer: Any) -> Dataset:
    """Format instruction and response pairs using model's chat template."""
    def _format(example):
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            formatted = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
        return {"text": formatted}

    return dataset.map(_format, desc="Applying chat template")


def run_qlora_training(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    dataset_path: Path | str | None = None,
    output_dir: Path | str = "models/lora_adapter",
    config_path: Path | str = "configs/qlora_config.yaml",
) -> None:
    """Run LoRA fine-tuning for Adapter B on domain instruction dataset."""
    base_dir = Path(__file__).resolve().parents[2]
    out_path = base_dir / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    cfg_path = base_dir / config_path if not Path(config_path).is_absolute() else Path(config_path)
    data_dir = base_dir / "data"

    out_path.mkdir(parents=True, exist_ok=True)
    results_dir = base_dir / "results" / "training"
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_qlora_config(cfg_path)
    device = detect_device()
    dtype = get_torch_dtype(device)
    print("\n" + "=" * 70)
    print("PART B STEP 2 — QLoRA FINE-TUNING (ADAPTER B)")
    print("=" * 70)
    print(f"Model: {model_name} | Device: {device} | DType: {dtype}")
    print(f"LoRA Config: Rank={cfg['lora']['r']}, Alpha={cfg['lora']['lora_alpha']}, Targets={cfg['lora']['target_modules']}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Base Model in BF16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device != "mps" else None,
        trust_remote_code=False,
    )
    if device == "mps":
        model = model.to("mps")

    # Apply LoRA Adapter B
    lora_config = build_lora_config(cfg["lora"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load Train and Eval Datasets
    train_file = data_dir / "train.jsonl"
    eval_file = data_dir / "eval.jsonl"
    train_ds = load_dataset("json", data_files=str(train_file), split="train")
    eval_ds = load_dataset("json", data_files=str(eval_file), split="train")

    print(f"Loaded {len(train_ds)} train examples and {len(eval_ds)} eval examples.")
    train_ds = format_dataset_with_chat_template(train_ds, tokenizer)
    eval_ds = format_dataset_with_chat_template(eval_ds, tokenizer)

    # Training Arguments using SFTConfig
    training_args = SFTConfig(
        output_dir=str(results_dir / "checkpoints"),
        dataset_text_field="text",
        max_length=512,
        num_train_epochs=1,
        max_steps=25,  # 25 optimization steps for effective convergence and fast turnaround on M4 MPS
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        weight_decay=0.001,
        bf16=(device in ("cuda", "mps")),
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_steps=25,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("\nStarting LoRA Fine-Tuning on Apple MPS...")
    train_result = trainer.train()

    # Save trained LoRA adapter
    print(f"\nSaving LoRA adapter to {out_path}...")
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    # Save training log history
    log_history = trainer.state.log_history
    metrics_path = results_dir / "training_history.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    # Save training summary markdown
    summary_md = results_dir / "training_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Part B Step 2: QLoRA Fine-Tuning Report\n\n")
        f.write(f"- **Base Model**: `{model_name}`\n")
        f.write(f"- **Adapter Configuration**: Adapter B (Balanced)\n")
        f.write(f"- **LoRA Rank ($r$)**: `16`\n")
        f.write(f"- **LoRA Alpha ($\\alpha$)**: `32` (scaling ratio = 2.0)\n")
        f.write(f"- **Target Modules**: `[\"q_proj\", \"v_proj\"]`\n")
        f.write(f"- **Training Examples**: {len(train_ds)}\n")
        f.write(f"- **Evaluation Examples**: {len(eval_ds)}\n")
        f.write(f"- **Effective Batch Size**: 8 (batch_size=2 * grad_accum=4)\n")
        f.write(f"- **Learning Rate**: `2e-4` (Cosine schedule)\n")
        f.write(f"- **Precision**: `bfloat16` on Apple Silicon MPS\n\n")
        f.write("## Training Loss Progression\n\n")
        f.write("| Step | Training Loss | Eval Loss | Learning Rate |\n")
        f.write("| :---: | :---: | :---: | :---: |\n")
        for log in log_history:
            step = log.get("step", "-")
            loss = log.get("loss", log.get("train_loss", "-"))
            eval_loss = log.get("eval_loss", "-")
            lr = log.get("learning_rate", "-")
            if loss != "-" or eval_loss != "-":
                f.write(f"| {step} | {loss if isinstance(loss, str) else round(loss, 4)} | {eval_loss if isinstance(eval_loss, str) else round(eval_loss, 4)} | {lr if isinstance(lr, str) else f'{lr:.2e}'} |\n")

    print(f"Training completed successfully! Saved to {out_path} and {metrics_path}")


if __name__ == "__main__":
    run_qlora_training()
