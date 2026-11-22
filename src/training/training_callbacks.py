"""
training_callbacks.py
=====================
Custom callbacks for tracking training loss, evaluation loss, learning rate,
and memory during QLoRA fine-tuning.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class MetricsLoggingCallback(TrainerCallback):
    """Callback to record step-by-step training metrics to JSON for plotting."""

    def __init__(self, log_file: Path | str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Any] | None = None,
        **kwargs,
    ):
        if logs is None:
            return

        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            "elapsed_seconds": round(time.time() - self.start_time, 2),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in logs.items()},
        }

        # Record device memory if MPS is available
        if torch.backends.mps.is_available():
            try:
                alloc = torch.mps.current_allocated_memory() / (1024**2)
                entry["mps_allocated_mb"] = round(alloc, 2)
            except Exception:
                pass
        elif torch.cuda.is_available():
            entry["cuda_allocated_mb"] = round(
                torch.cuda.memory_allocated() / (1024**2), 2
            )

        self.history.append(entry)

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)


class EarlyStoppingOnLossPlateau(TrainerCallback):
    """Simple early stopping callback based on evaluation loss."""

    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.stagnant_steps = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, Any] | None = None,
        **kwargs,
    ):
        if metrics is None or "eval_loss" not in metrics:
            return

        eval_loss = metrics["eval_loss"]
        if eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1
            if self.stagnant_steps >= self.patience:
                logger.info(
                    "Early stopping triggered: eval_loss did not improve for %d evals.",
                    self.patience,
                )
                control.should_training_stop = True
