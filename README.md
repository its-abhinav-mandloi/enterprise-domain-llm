# Enterprise Domain-Specific LLM — Assignment 1B
## Variant V5: IT Helpdesk & Incident Resolution Assistant

End-to-end enterprise domain-specific LLM adaptation, QLoRA/LoRA fine-tuning, inference decoding optimization, speculative decoding, 4-bit quantization, and production cost benchmarking built for **Assignment 1B**.

---

## Key Project Specs

| Attribute | Configuration |
| :--- | :--- |
| **Domain Variant** | V5 — Enterprise IT Helpdesk & Incident Resolution Bot |
| **Base / Target Model** | `Qwen/Qwen2.5-1.5B-Instruct` (1.54B params, 28 layers, 1,536 hidden, 151,936 vocab) |
| **Speculative Draft Model** | `Qwen/Qwen2.5-0.5B-Instruct` (0.49B params, shared tokenizer) |
| **Fine-Tuning Adapter** | PEFT Adapter B — Balanced (`r=16, alpha=32, target_modules=["q_proj", "v_proj"]`) |
| **Domain Corpus** | 6 official manuals (Ubuntu Server, AWS EC2, RHEL 9, PostgreSQL 17, Bash, GCC) — 9,239 pages, 18.04M clean chars |
| **Instruction Dataset** | 581 unique, strictly corpus-grounded pairs (`465` train / `116` eval, 80/20 split, `seed=42`) |
| **Local Hardware** | Apple MacBook Air M4 (24 GB Unified Memory, PyTorch MPS) |
| **CUDA Portability** | 1-click Google Colab T4 notebook (`notebooks/assignment_1B_colab_t4.ipynb`) for real `bitsandbytes` 4-bit NF4 runs |

---

## Verified Local Empirical Results Summary

- **Part A1 (Corpus Cleaning):** 18,232,885 raw chars → **18,040,053 clean chars** across 6 `.txt` files in `data/domain_corpus/`; greatest reduction from length filter (`-192,832` chars).
- **Part A2 (Baseline, BF16, Zero System Prompt):** 19.79 / 26.69 / 26.62 tok/s across the 3 canonical prompts (`results/baseline/baseline_outputs.json`).
- **Part B1 (Instruction Dataset):** 581 clean, deduplicated pairs (~142.7 avg tokens/example), zero truncation or TOC artifacts (`data/instruction_dataset.jsonl`).
- **Part B2 (Adapter B Fine-Tuning):** 2 epochs (60 steps), `0.141%` trainable params (`2.18M / 1.55B`). Held-out eval loss dropped **36.8%** from **`3.1321` (Step 0) → `2.0723` (Epoch 1) → `1.9787` (Epoch 2)**.
- **Part B3 (Comparative Analysis):** Strictly output-grounded side-by-side comparison (`results/evaluation/comparative_analysis.md`) highlighting domain terminology gains, density improvements, and honest hallucination/latency audit.
- **Part C1 (Decoding Strategies, 150 tokens):** Greedy (**27.10 tok/s**, fastest & deterministic) vs. Beam 4 (**10.13 tok/s**), Top-K 50 (**16.03 tok/s**), Top-P 0.9 (**16.99 tok/s**), Temp 0.2 (**17.04**), Temp 0.7 (**17.32**), Temp 1.0 (**12.10**).
- **Part C2 (Speculative Decoding on MPS):** -7.0% avg change (25.27 → 23.35 tok/s; Prompt 3 +5.8%) with transparent MPS kernel-dispatch and BF16 floating-point analysis (`results/speculative/speculative_comparison.md`).
- **Part C3 (10-Prompt Production Economics):** Empirical BF16 baseline on MPS: **27.42 tok/s**, **2.90 GB** peak memory, **₹35.46 / 1M tokens (T4 @ ₹3.50/hr)**, **₹121.57 / 1M tokens (A100 @ ₹12.00/hr)**. CUDA-only 4-bit NF4 benchmarks are prepared without fabricated projections in `notebooks/assignment_1B_colab_t4.ipynb`.

---

## Repository Layout

```text
enterprise-domain-llm/
├── configs/                # YAML configs for model, data, QLoRA, and benchmarks
├── data/
│   ├── domain_corpus/      # 6 cleaned domain .txt manuals (Deliverable 4)
│   ├── instruction_dataset.jsonl  # 581 unique grounded Q&A pairs (Deliverable 3)
│   ├── train.jsonl         # 465 training pairs (80%)
│   └── eval.jsonl          # 116 evaluation pairs (20%)
├── docs/                   # Hardware assessment, model selection, architecture, rubric
├── models/lora_adapter/    # Saved PEFT Adapter B weights & tokenizer
├── notebooks/
│   ├── assignment_1B_master.ipynb    # Executed master notebook (Deliverable 1)
│   └── assignment_1B_colab_t4.ipynb  # 1-click Google Colab T4 CUDA 4-bit notebook
├── results/
│   ├── baseline/           # Part A2 baseline outputs
│   ├── training/           # Part B2 loss history, metadata & summary
│   ├── evaluation/         # Part B3 adapter outputs & comparative analysis
│   ├── decoding/           # Part C1 7-config decoding matrix
│   ├── speculative/        # Part C2 speculative decoding metrics
│   ├── quantization/       # Part C3 10-prompt production cost results
│   └── charts/             # 5 high-resolution Matplotlib charts
├── src/                    # Modular Python package (data, models, training, eval, opt, reporting)
├── submission/
│   ├── assignment_1B.html  # Standalone HTML export with all outputs & 5 charts (Deliverable 2)
│   ├── final_report.md     # Complete technical report
│   └── rubric_compliance.md# 15/15 rubric audit matrix
└── tests/                  # Pytest unit test suite (11/11 passing)
```

---

## Reproducing Locally & on Google Colab T4

```bash
# 1. Install pinned dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run unit tests (verifies corpus, dataset hygiene, 80/20 split, metrics & cost formula)
python -m pytest tests/

# 3. Re-execute master notebook & export HTML
jupyter nbconvert --to notebook --execute --inplace notebooks/assignment_1B_master.ipynb
jupyter nbconvert --to html notebooks/assignment_1B_master.ipynb --output-dir=submission/ --output=assignment_1B.html
```

To run the **empirical CUDA 4-bit `bitsandbytes` QLoRA & quantization benchmarks**, open `notebooks/assignment_1B_colab_t4.ipynb` in Google Colab with a free **T4 GPU** runtime, upload `data/train.jsonl` and `data/eval.jsonl`, run all cells, and unzip the downloaded results back into this directory.
