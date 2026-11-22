#!/usr/bin/env python3
"""
run_experiments.py
==================
Master execution script for Assignment 1B experiments:
- Baseline Evaluation (Part A2)
- LoRA Adapter Training & Evaluation (Part B2 & B3)
- Comparative Analysis (Part B3)
- Decoding Strategies (Part C1)
- Speculative Decoding (Part C2)
- 4-bit Quantization & Production Economics (Part C3)
- Chart Generation
- Jupyter Notebook Execution & HTML Export
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

def step_baseline():
    logger.info("=== STEP 1: BASELINE EVALUATION (Part A2) ===")
    from src.evaluation.baseline_evaluator import run_baseline_evaluation
    return run_baseline_evaluation(model_name="Qwen/Qwen2.5-1.5B-Instruct")

def step_training():
    logger.info("=== STEP 2: QLoRA / PEFT TRAINING (Part B2) ===")
    from src.training.qlora_trainer import run_qlora_training
    dataset_file = BASE_DIR / "data" / "train.jsonl"
    adapter_out = BASE_DIR / "models" / "lora_adapter"
    cfg_file = BASE_DIR / "configs" / "qlora_config.yaml"
    run_qlora_training(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_path=dataset_file,
        output_dir=adapter_out,
        config_path=cfg_file,
    )

def step_adapter_eval():
    logger.info("=== STEP 3: ADAPTER EVALUATION (Part B3) ===")
    from src.evaluation.adapter_evaluator import run_adapter_evaluation
    return run_adapter_evaluation(base_model_name="Qwen/Qwen2.5-1.5B-Instruct")

def step_comparative_analysis():
    logger.info("=== STEP 4: COMPARATIVE ANALYSIS (Part B3) ===")
    from src.evaluation.quality_evaluator import generate_comparative_analysis
    return generate_comparative_analysis()

def step_decoding():
    logger.info("=== STEP 5: DECODING STRATEGIES (Part C1) ===")
    from src.optimization.decoding import run_decoding_experiments
    return run_decoding_experiments(model_name="Qwen/Qwen2.5-1.5B-Instruct")

def step_speculative():
    logger.info("=== STEP 6: SPECULATIVE DECODING (Part C2) ===")
    from src.optimization.speculative_decoding import run_speculative_decoding_benchmark
    return run_speculative_decoding_benchmark(
        target_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        draft_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    )

def step_quantization():
    logger.info("=== STEP 7: QUANTIZATION & PRODUCTION ECONOMICS (Part C3) ===")
    from src.optimization.quantization import run_production_benchmarks
    return run_production_benchmarks(model_name="Qwen/Qwen2.5-1.5B-Instruct")

def step_charts():
    logger.info("=== STEP 8: GENERATING CHARTS ===")
    from src.reporting.charts import generate_all_charts
    generate_all_charts()

if __name__ == "__main__":
    t_start = time.time()
    try:
        step_baseline()
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}")

    try:
        step_training()
    except Exception as e:
        logger.error(f"Training failed: {e}")

    try:
        step_adapter_eval()
    except Exception as e:
        logger.error(f"Adapter evaluation failed: {e}")

    try:
        step_comparative_analysis()
    except Exception as e:
        logger.error(f"Comparative analysis failed: {e}")

    try:
        step_decoding()
    except Exception as e:
        logger.error(f"Decoding benchmark failed: {e}")

    try:
        step_speculative()
    except Exception as e:
        logger.error(f"Speculative benchmark failed: {e}")

    try:
        step_quantization()
    except Exception as e:
        logger.error(f"Quantization benchmark failed: {e}")

    try:
        step_charts()
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")

    logger.info(f"Master pipeline completed in {time.time() - t_start:.2f} seconds.")
