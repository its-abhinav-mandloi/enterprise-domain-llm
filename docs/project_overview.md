# Project Overview: Enterprise IT Helpdesk & Incident Resolution Assistant

## Executive Summary
This project delivers a complete enterprise-grade Large Language Model (LLM) domain adaptation and production serving pipeline for **Assignment 1B: Domain LLM Adaptation & Production Optimization** (Variant V5: IT Helpdesk & Incident Resolution Assistant).

The system adapts **Qwen/Qwen2.5-1.5B-Instruct** using Low-Rank Adaptation (LoRA / QLoRA) on official enterprise IT system documentation (AWS, Ubuntu Server, Red Hat Enterprise Linux, GNU Bash, PostgreSQL, and GCC). The adapted model is systematically evaluated against the unadapted baseline, optimized using 5 decoding strategies and speculative decoding (with a 0.5B draft model), and benchmarked for production deployment economics under 4-bit NF4 quantization on cloud infrastructure (NVIDIA T4 and A100 GPUs).

---

## Key Academic Accomplishments

1. **Part A (Data Collection & Baseline)**:
   - Ingested 6 official domain PDF manuals comprising **9,239 pages** and **18.2 million characters**.
   - Implemented an automated cleaning pipeline (length filtering, SHA-256 + prefix near-deduplication, language filtering), preserving 18,040,053 page characters (18,058,435 chars on disk with page separators).
   - Evaluated the pre-adaptation baseline on 3 canonical enterprise IT prompts in pure `bfloat16` (no system prompt).

2. **Part B (Instruction Fine-Tuning with QLoRA)**:
   - Synthesized **581 grounded instruction-response pairs** partitioned into an 80/20 train/eval split (465 training examples, 116 evaluation examples, `seed=42`).
   - Trained PEFT LoRA **Adapter B** (Rank=16, Alpha=32, target modules: `q_proj`, `v_proj`, dropout=0.05) locally in `bfloat16` on MPS (`use_4bit_qlora=false`), reducing held-out eval loss by 36.8% (`3.1321` → `1.9787`); true 4-bit NF4 QLoRA execution on CUDA is packaged in `notebooks/assignment_1B_colab_t4.ipynb`.
   - Conducted an evidence-backed comparative analysis (`results/evaluation/comparative_analysis.md`) documenting exact output differences and honest limitations.

3. **Part C (Inference Optimization & Production Metrics)**:
   - Benchmarked **5 decoding strategies across 7 configurations** (`max_new_tokens=150`) with exact 3-prompt empirical throughput/latency measurements (Greedy mean: **19.86 tok/s**, Beam mean: **10.89 tok/s**).
   - Implemented **speculative decoding** (`Qwen2.5-1.5B` target + `Qwen2.5-0.5B` draft) on MPS (`-7.0%` avg delta due to Metal dispatch overhead on a 3× model ratio) with CUDA verification ready in Colab.
   - Measured the 10-prompt BF16 baseline (**27.42 tok/s**, **2.90 GB**, **₹35.46 / 1M T4**, **₹121.57 / 1M A100**); CUDA-only 4-bit NF4 and 4-bit + Speculative benchmarks await execution of `notebooks/assignment_1B_colab_t4.ipynb`.
