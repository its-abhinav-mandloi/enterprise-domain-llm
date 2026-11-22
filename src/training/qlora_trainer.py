"""
qlora_trainer.py
================
QLoRA / LoRA fine-tuning using Hugging Face Transformers, PEFT, BitsAndBytes,
and TRL SFTTrainer.

Assignment 1B — Part B Step 2 (2 Marks):
- Adapter B (Balanced): LoRA Rank r=16, Alpha=32, target_modules=["q_proj", "v_proj"]
- Uses Qwen2.5 chat template via tokenizer.apply_chat_template
- On CUDA: true 4-bit NF4 QLoRA via BitsAndBytesConfig + prepare_model_for_kbit_training
- On Apple Silicon MPS: native bfloat16 LoRA (bitsandbytes 4-bit kernels require CUDA)
- Evaluates pre-training (step 0) and per-epoch eval loss for verifiable convergence
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

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
    """Format instruction and response pairs using the model's native chat template."""
    def _format(example: Dict[str, Any]) -> Dict[str, str]:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            formatted = (
                f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['response']}<|im_end|>"
            )
        return {"text": formatted}

    return dataset.map(_format, desc="Applying chat template")


def run_qlora_training(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: Path | str = "models/lora_adapter",
    config_path: Path | str = "configs/qlora_config.yaml",
    num_train_epochs: int = 2,
    max_length: int = 384,
) -> Dict[str, Any]:
    """Run QLoRA (CUDA 4-bit NF4) or LoRA (MPS bfloat16) fine-tuning for Adapter B."""
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

    use_4bit_qlora = False
    cuda_bf16_supported = bool(device == "cuda" and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if (device == "mps" or cuda_bf16_supported) else torch.float16

    print("\n" + "=" * 70)
    print("PART B STEP 2 — QLoRA FINE-TUNING (ADAPTER B)")
    print("=" * 70)
    print(f"Model: {model_name} | Device: {device} | Compute DType: {compute_dtype}")
    print(
        f"LoRA Config: Rank={cfg['lora']['r']}, Alpha={cfg['lora']['lora_alpha']}, "
        f"Targets={cfg['lora']['target_modules']}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model: true 4-bit NF4 QLoRA on CUDA; native bfloat16 LoRA on Apple MPS
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
        )
        model = prepare_model_for_kbit_training(model)
        use_4bit_qlora = True
        precision_label = f"4-bit NF4 QLoRA (compute: {str(compute_dtype).replace('torch.', '')}) on CUDA"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=False,
        )
        if device == "mps":
            model = model.to("mps")
        precision_label = f"{str(dtype).replace('torch.', '')} LoRA on {device.upper()} (4-bit NF4 requires CUDA)"

    model.config.use_cache = False

    # Apply LoRA Adapter B
    lora_config = build_lora_config(cfg["lora"])
    model = get_peft_model(model, lora_config)
    trainable_params, total_params = model.get_nb_trainable_parameters()
    trainable_pct = round(100.0 * trainable_params / max(total_params, 1), 4)
    model.print_trainable_parameters()

    # Load clean 80/20 Train & Eval datasets
    train_file = data_dir / "train.jsonl"
    eval_file = data_dir / "eval.jsonl"
    train_ds = load_dataset("json", data_files=str(train_file), split="train")
    eval_ds = load_dataset("json", data_files=str(eval_file), split="train")

    print(f"Loaded {len(train_ds)} train examples and {len(eval_ds)} eval examples.")
    train_ds = format_dataset_with_chat_template(train_ds, tokenizer)
    eval_ds = format_dataset_with_chat_template(eval_ds, tokenizer)

    per_device_batch = 4 if device in ("cuda", "mps") else 2
    grad_accum = 4
    effective_batch = per_device_batch * grad_accum
    lr = 2e-4

    training_args = SFTConfig(
        output_dir=str(results_dir / "checkpoints"),
        dataset_text_field="text",
        max_length=max_length,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        weight_decay=0.01,
        optim="paged_adamw_8bit" if use_4bit_qlora else "adamw_torch",
        bf16=(device == "mps" or cuda_bf16_supported),
        fp16=(device == "cuda" and not cuda_bf16_supported),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # Measure initial pre-training eval loss (Step 0) for honest convergence proof
    print("\nEvaluating Step-0 pre-training baseline loss on eval split...")
    init_eval = trainer.evaluate()
    init_eval_loss = round(float(init_eval["eval_loss"]), 4)
    print(f"Step 0 Eval Loss: {init_eval_loss}")

    print(f"\nStarting fine-tuning ({num_train_epochs} epochs, {precision_label})...")
    train_result = trainer.train()

    # Save trained adapter and tokenizer
    print(f"\nSaving adapter to {out_path}...")
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    # Persist full log history (prepend step-0 eval entry if not already present)
    log_history = list(trainer.state.log_history)
    if not any(entry.get("step") == 0 and "eval_loss" in entry for entry in log_history):
        log_history.insert(0, {"step": 0, "epoch": 0.0, "eval_loss": init_eval_loss})

    metrics_path = results_dir / "training_history.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    # Extract key summary numbers directly from log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    final_train_loss = round(float(train_losses[-1]), 4) if train_losses else round(float(train_result.training_loss), 4)
    avg_train_loss = round(float(train_result.training_loss), 4)
    final_eval_loss = round(float(eval_losses[-1]), 4) if eval_losses else init_eval_loss

    meta = {
        "model_name": model_name,
        "adapter_config": "Adapter B (Balanced)",
        "lora_r": cfg["lora"]["r"],
        "lora_alpha": cfg["lora"]["lora_alpha"],
        "target_modules": cfg["lora"]["target_modules"],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_pct,
        "use_4bit_qlora": use_4bit_qlora,
        "precision": precision_label,
        "device": device,
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "num_train_epochs": num_train_epochs,
        "total_steps": int(trainer.state.global_step),
        "max_length": max_length,
        "per_device_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch,
        "learning_rate": lr,
        "initial_eval_loss": init_eval_loss,
        "final_eval_loss": final_eval_loss,
        "final_train_loss": final_train_loss,
        "average_train_loss": avg_train_loss,
        "train_runtime_sec": round(float(train_result.metrics.get("train_runtime", 0.0)), 2),
    }

    meta_path = results_dir / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save training summary markdown
    summary_md = results_dir / "training_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Part B Step 2: QLoRA / LoRA Fine-Tuning Report\n\n")
        f.write(f"- **Base Model**: `{model_name}`\n")
        f.write(f"- **Adapter Configuration**: Adapter B (Balanced)\n")
        f.write(f"- **LoRA Rank ($r$)**: `{cfg['lora']['r']}` | **Alpha ($\\alpha$)**: `{cfg['lora']['lora_alpha']}` (scaling = 2.0)\n")
        f.write(f"- **Target Modules**: `{cfg['lora']['target_modules']}`\n")
        f.write(f"- **Trainable Parameters**: `{trainable_params:,}` / `{total_params:,}` (`{trainable_pct}%`)\n")
        f.write(f"- **Precision / Backend**: {precision_label}\n")
        f.write(f"- **Dataset Split**: `{len(train_ds)}` train / `{len(eval_ds)}` eval (80/20, seed=42)\n")
        f.write(f"- **Hyperparameters**: `epochs={num_train_epochs}`, `steps={meta['total_steps']}`, `max_length={max_length}`, `effective_batch_size={effective_batch}`, `lr={lr}` (cosine)\n")
        f.write(f"- **Convergence**: Step-0 Eval Loss `{init_eval_loss}` → Final Eval Loss `{final_eval_loss}` (Final Train Loss `{final_train_loss}`)\n\n")
        f.write("## Training & Evaluation Loss Progression\n\n")
        f.write("| Step | Epoch | Training Loss | Eval Loss | Learning Rate |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: |\n")
        for log in log_history:
            if "loss" in log or "eval_loss" in log:
                step = log.get("step", "-")
                ep = round(log["epoch"], 2) if "epoch" in log else "-"
                tl = round(log["loss"], 4) if "loss" in log else "-"
                el = round(log["eval_loss"], 4) if "eval_loss" in log else "-"
                lr_v = f"{log['learning_rate']:.2e}" if "learning_rate" in log else "-"
                f.write(f"| {step} | {ep} | {tl} | {el} | {lr_v} |\n")

    print(f"Training completed! Eval loss: {init_eval_loss} -> {final_eval_loss}. Saved to {out_path}")
    return meta


if __name__ == "__main__":
    run_qlora_training()
