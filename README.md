# Enterprise Domain-Specific LLM — Assignment 1B
## Variant V5: IT Helpdesk & Incident Resolution Assistant

**BITS Pilani — Group 92**

| Team Member | BITS ID |
| :--- | :--- |
| Ravi | `2025aa05029` |
| Pritish | `2025aa05686` |
| Siddarth | `2025aa05446` |
| Abhinav | `2025aa05473` |

End-to-end enterprise domain-specific LLM adaptation, LoRA/QLoRA fine-tuning, inference decoding optimization, speculative decoding, 4-bit quantization, and production cost benchmarking built for **Assignment 1B**.

---

## Key Project Specs

| Attribute | Configuration |
| :--- | :--- |
| **Domain Variant** | V5 — Enterprise IT Helpdesk & Incident Resolution Bot |
| **Base / Target Model** | `Qwen/Qwen2.5-1.5B-Instruct` (`1,543,714,304` params, 28 layers, 1,536 hidden, 151,936 vocab) |
| **Speculative Draft Model** | `Qwen/Qwen2.5-0.5B-Instruct` (`494,032,768` params, shared tokenizer) |
| **Fine-Tuning Adapter** | PEFT Adapter B — Balanced (`r=16, alpha=32, target_modules=["q_proj", "v_proj"], dropout=0.05`) |
| **Domain Corpus** | 6 official manuals (Ubuntu Server, AWS EC2, RHEL 9, PostgreSQL 17, Bash, GCC) — 9,239 pages, 18,040,053 clean page chars (`18,058,435` on disk) |
| **Instruction Dataset** | 581 unique, strictly corpus-grounded pairs (`465` train / `116` eval, 80/20 split, `seed=42`) |
| **Local Hardware** | Apple MacBook Air M4 (24 GB Unified Memory, PyTorch MPS) |
| **CUDA Portability** | Google Colab T4 notebook (`notebooks/assignment_1B_colab_t4.ipynb`) for real `bitsandbytes` 4-bit NF4 QLoRA & Part C3 benchmarks |

---

## Verified Empirical Status (Local M4 MPS vs. Pending CUDA T4)

- **Part A1 (Corpus Cleaning — Done Locally):** 18,232,885 raw chars → **18,040,053 clean page chars** (`18,058,435` chars on disk with `\n\n` page separators) across 6 `.txt` files in `data/domain_corpus/`; greatest reduction from length filter (`-192,832` chars). *(Note: near-dedup uses 5,000-char prefixes and large-doc `langdetect` samples first 5 pages.)*
- **Part A2 (Baseline, BF16, Zero System Prompt — Done Locally):** Prompt 1: 19.79 tok/s (7.58s), Prompt 2: 26.69 tok/s (5.62s), Prompt 3: 26.62 tok/s (5.64s); architecture printed from `results/baseline/baseline_outputs.json`.
- **Part B1 (Instruction Dataset — Done Locally):** 581 clean, deduplicated pairs (~142.7 avg tokens/example), zero truncation or TOC artifacts (`data/instruction_dataset.jsonl`, `train.jsonl`, `eval.jsonl`).
- **Part B2 (Adapter B Fine-Tuning — BF16 LoRA Done Locally; True 4-bit NF4 QLoRA Pending Colab T4):** Local run trained on MPS in `bfloat16` (`use_4bit_qlora: false`, 2 epochs, 60 steps, `0.141%` trainable params = `2,179,072`). Held-out eval loss dropped **36.8%** (`3.1321` → `2.0723` → `1.9787`). Genuine 4-bit NF4 QLoRA via `bitsandbytes` requires CUDA and is ready in `notebooks/assignment_1B_colab_t4.ipynb`.
- **Part B3 (Comparative Analysis — Done for Local Adapter; Re-run After Colab QLoRA):** Evidence-backed comparison (`results/evaluation/comparative_analysis.md`) transparently noting raw-prompt baseline vs. chat-template adapter framing and observed command/hallucination differences.
- **Part C1 (Decoding Strategies, 150 tokens — Done Locally):** 3-prompt empirical means: Greedy (**19.86 tok/s**, 8.01s), Beam Search k=4 (**10.89 tok/s**, 13.84s), Top-K 50 (**15.24 tok/s**, 10.02s), Top-P 0.9 (**18.27 tok/s**, 8.34s), Temp 0.2 (**18.21 tok/s**, 8.31s), Temp 0.7 (**18.87 tok/s**, 8.05s), Temp 1.0 (**17.09 tok/s**, 9.33s).
- **Part C2 (Speculative Decoding on MPS — Done Locally; CUDA Pending Colab):** `-7.0%` avg throughput delta on MPS (`25.27 → 23.35 tok/s`; Prompt 3 `+5.8%`), 0/3 exact full matches (`54.0` tok avg common prefix), acceptance rate unmeasured locally (`results/speculative/speculative_comparison.md`).
- **Part C3 (10-Prompt Production Economics — BF16 Measured Locally; 4-bit NF4 Pending Colab T4):** Measured BF16 baseline on MPS: **27.42 tok/s**, **2.90 GB**, **₹35.46 / 1M tokens (T4 @ ₹3.50/hr)**, **₹121.57 / 1M tokens (A100 @ ₹12.00/hr)**. Both 4-bit CUDA configurations are marked `Pending CUDA (Colab T4)` with zero fabricated projections.

---

## Repository Layout

```text
enterprise-domain-llm/
├── configs/                        # YAML configs for model, data, QLoRA, benchmarks
├── data/
│   ├── domain_corpus/              # 6 cleaned domain .txt manuals
│   ├── instruction_dataset.jsonl   # 581 unique grounded Q&A pairs
│   ├── train.jsonl                 # 465 training pairs (80%)
│   └── eval.jsonl                  # 116 evaluation pairs (20%)
├── docs/                           # Hardware assessment, model selection, architecture, rubric
├── models/lora_adapter/            # Saved PEFT Adapter B weights & tokenizer
├── notebooks/
│   ├── assignment_1B_master.ipynb  # Consolidated master notebook (with embedded outputs/charts)
│   └── assignment_1B_colab_t4.ipynb# Colab T4 CUDA notebook for 4-bit QLoRA & Part C3
├── results/                        # Empirical JSON/CSV/MD metrics & 5 PNG charts
├── src/                            # Modular Python package (data, models, training, eval, opt, reporting)
├── submission/
│   ├── assignment_1B.html          # Exported HTML report with outputs & 5 charts
│   ├── final_report.md             # Consolidated technical report
│   └── rubric_compliance.md        # Rubric audit matrix
└── tests/                          # Pytest unit test suite (11/11 passing)
```

---

## Reproducing Locally & on Google Colab T4

```bash
# 1. Install pinned dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run unit tests
python -m pytest tests/

# 3. Re-execute master notebook & export HTML
jupyter nbconvert --to notebook --execute --inplace notebooks/assignment_1B_master.ipynb
jupyter nbconvert --to html notebooks/assignment_1B_master.ipynb --output-dir=submission/ --output=assignment_1B.html
```

To run the **CUDA 4-bit `bitsandbytes` QLoRA & Part C3 quantization benchmarks**, upload `notebooks/assignment_1B_colab_t4.ipynb`, `data/train.jsonl`, and `data/eval.jsonl` to Google Colab on a **T4 GPU** runtime, run all cells, and unzip the downloaded archives back into this directory.
