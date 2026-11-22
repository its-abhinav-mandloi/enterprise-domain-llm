# Project Overview: Enterprise IT Helpdesk & Incident Resolution Assistant

## Executive Summary
This project delivers a complete enterprise-grade Large Language Model (LLM) domain adaptation and production serving pipeline for **Assignment 1B: Domain LLM Adaptation & Production Optimization** (Variant V5: IT Helpdesk & Incident Resolution Assistant).

The system adapts **Qwen/Qwen2.5-1.5B-Instruct** using Low-Rank Adaptation (LoRA / QLoRA) on official enterprise IT system documentation (AWS, Ubuntu Server, Red Hat Enterprise Linux, GNU Bash, PostgreSQL, and GCC). The adapted model is systematically evaluated against the unadapted baseline, optimized using 5 decoding strategies and speculative decoding (with a 0.5B draft model), and benchmarked for production deployment economics under 4-bit NF4 quantization on cloud infrastructure (NVIDIA T4 and A100 GPUs).

---

## Key Academic Accomplishments

1. **Part A (Data Collection & Baseline)**:
   - Ingested 6 official domain PDF manuals comprising **9,239 pages** and **18.2 million characters**.
   - Implemented an automated cleaning pipeline (length filtering, SHA-256 deduplication, language filtering), preserving 18.04 million characters.
   - Evaluated the pre-adaptation baseline on 3 canonical enterprise IT prompts in pure `bfloat16`.

2. **Part B (Instruction Fine-Tuning with QLoRA)**:
   - Synthesized **720 grounded instruction-response pairs** partitioned into an 80/20 train/eval split (576 training examples, 144 evaluation examples).
   - Trained PEFT LoRA **Adapter B** (Rank=16, Alpha=32, target modules: `q_proj`, `v_proj`).
   - Conducted rigorous qualitative comparison demonstrating dramatic reduction in conversational fluff and precise CLI command syntax.

3. **Part C (Inference Optimization & Production Metrics)**:
   - Benchmarked **5 decoding strategies across 7 configurations** with exact tokens/sec measurements.
   - Implemented **speculative decoding** (Qwen2.5-1.5B target + Qwen2.5-0.5B draft), verifying zero quality degradation.
   - Benchmarked 10 enterprise prompts under 4-bit NF4 quantization and computed exact cloud serving economics ($₹/\text{1M tokens}$).
