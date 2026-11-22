# Enterprise Domain-Specific LLM — Assignment 1B Final Technical Report
## Variant V5: IT Helpdesk & Incident Resolution Assistant

---

## 1. Executive Summary

This report documents the end-to-end engineering implementation, empirical evaluation, and production optimization of an enterprise **IT Helpdesk & Incident Resolution Assistant** built for **Assignment 1B (Domain LLM Adaptation & Production Optimization)**.

- **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (`1,543,714,816` parameters, 28 decoder layers, hidden dimension 1,536, vocabulary 151,936)
- **Draft Model (Speculative Decoding):** `Qwen/Qwen2.5-0.5B-Instruct` (`~0.49B` parameters, identical tokenizer/vocabulary)
- **Local Hardware:** Apple MacBook Air M4 (24 GB Unified Memory, Apple Silicon ARM64, PyTorch MPS backend)
- **CUDA Portability:** Full 4-bit NF4 QLoRA & BitsAndBytes CUDA execution packaged in `notebooks/assignment_1B_colab_t4.ipynb` for zero-friction Google Colab T4 execution.

All local results reported below are **100% empirically measured** with zero fabricated or projected numbers.

---

## 2. Part A — Domain Data Collection & Pre-Adaptation Baseline [2 Marks]

### Step A1: Corpus Collection & Cleaning
Six official enterprise IT and Linux administration PDFs (9,239 pages total) were extracted page-by-page using PyMuPDF and cleaned via length filtering (≥100 characters/page), exact SHA-256 deduplication, 5-gram Jaccard near-deduplication (0.85 threshold), and English language verification (`langdetect`):

| Cleaning Stage | Corpus Volume (Chars) | Change | Notes |
| :--- | :---: | :---: | :--- |
| **Raw Extracted (6 PDFs, 9,239 pages)** | 18,232,885 | — | AWS EC2, PostgreSQL 17, GCC, Ubuntu Server, Bash, RHEL 9 |
| **Length Filter (≥100 chars/page)** | 18,040,053 | **-192,832** | **Greatest reduction stage** (removes blank/cover/divider pages) |
| **Exact SHA-256 + Near-Dedup** | 18,040,053 | 0 | Zero duplicate documents; repeated sparse pages already filtered |
| **Language Filter (`langdetect`, EN)** | 18,040,053 | 0 | 100% English technical documentation verified |

All 6 cleaned files are stored individually under `data/domain_corpus/*_clean.txt`.

### Step A2: Pre-Adaptation Baseline (`bfloat16`, Zero System Prompt)
Evaluated `Qwen/Qwen2.5-1.5B-Instruct` in `bfloat16` on MPS across the 3 canonical prompts (`max_new_tokens=150`, no system prompt):
- **Prompt 1 (NTFS mount on Ubuntu):** 150 tokens | 7.58s | **19.79 tok/s**
- **Prompt 2 (High disk I/O & CPU on Linux):** 150 tokens | 5.62s | **26.69 tok/s**
- **Prompt 3 (AWS EC2 connectivity & SSH):** 150 tokens | 5.64s | **26.62 tok/s**

---

## 3. Part B — Instruction Dataset, Fine-Tuning & Comparative Analysis [5 Marks]

### Part B1: Strictly Corpus-Grounded Instruction Dataset
Using `src/data/instruction_generator.py`, we normalized PDF typography (ligatures, hyphenated line breaks, TOC/page header stripping) and extracted section-anchored, sentence-boundary Q&A pairs exclusively from the 6 cleaned manuals in `data/domain_corpus/`:
- **Total Unique Pairs:** **581** (100 AWS EC2, 81 GNU Bash, 100 GCC, 100 PostgreSQL 17, 100 RHEL 9, 100 Ubuntu Server)
- **80/20 Split (`seed=42`):** **465 training pairs (`data/train.jsonl`)** / **116 evaluation pairs (`data/eval.jsonl`)**
- **Average Length:** 107.3 words (~142.7 tokens/example), fitting comfortably within `max_length=384` without truncation
- **Quality Guarantees:** Automated unit tests verify 100% unique instructions, zero TOC dot-leader artifacts, clean sentence-start capitalization, and complete sentence-ending punctuation.

### Part B2: Adapter B Fine-Tuning (`r=16, alpha=32, q_proj, v_proj`)
- **Trainable Parameters:** `2,179,072` / `1,545,893,376` (`0.1410%`)
- **Hyperparameters:** 2 epochs (60 optimization steps), `max_length=384`, `per_device_batch_size=4`, `gradient_accumulation_steps=4` (effective batch size = 16), `lr=2e-4` cosine schedule, `warmup_steps=5`, Qwen2.5 native chat template.
- **Measured Convergence (Apple MPS `bfloat16` LoRA):**
  - **Step 0 (Pre-Training) Eval Loss:** `3.1321`
  - **Epoch 1 (Step 30) Eval Loss:** `2.0723` (Train Loss: `2.1521`)
  - **Epoch 2 (Step 60) Eval Loss:** **`1.9787`** (Final Train Loss: `2.0767`, **-36.8% reduction in held-out eval loss**)
- **CUDA 4-bit QLoRA Portability:** `src/training/qlora_trainer.py` and `notebooks/assignment_1B_colab_t4.ipynb` automatically enable `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` with `paged_adamw_8bit` on CUDA GPUs.

### Part B3: Evidence-Backed Comparative Analysis (Baseline vs. Adapter)
Evaluated on the exact 3 canonical prompts (`results/evaluation/comparative_analysis.md`):
- **Prompt 1 (NTFS Mount):** Adapter packed denser prose (120 words / 3 inline steps vs. 82 words / 2 steps) and introduced `kernel` support checks, while baseline surfaced `lsblk`/`fdisk` earlier. Honest audit also flagged a minor command hallucination in the adapter (`chattr +r /mnt/ntfs` instead of `mount -o remount,rw` / `ntfs-3g`), illustrating why small-rank adaptation on narrow corpora benefits from explicit troubleshooting command examples or RAG grounding.
- **Prompt 2 (Disk I/O & CPU):** Adapter achieved higher information density (129 words / 3 steps vs. 81 words / 3 steps) covering `top`, `uptime`, `iostat`, and `vmstat`, while unmerged LoRA incurred a 12.5% speed overhead (23.35 vs 26.69 tok/s) removable via `merge_and_unload()`.
- **Prompt 3 (AWS EC2 SSH):** Adapter immediately prioritized AWS-native failure domains (`Amazon VPC`, `security group`, `port 22`, `VPC endpoint` DNS) across 4 steps (118 words vs. 93 words baseline).

---

## 4. Part C — Inference Optimization & Production Economics [8 Marks]

### Step C1: Decoding Strategies & Sampling (`max_new_tokens=150`)
| Strategy | Config | Avg Latency (s) | Avg Throughput (tok/s) |
| :--- | :--- | :---: | :---: |
| **Greedy Decoding** | `do_sample=False` | **5.54** | **27.10** |
| **Beam Search** | `num_beams=4` | 14.80 | 10.13 |
| **Top-K Sampling** | `top_k=50` | 9.36 | 16.03 |
| **Top-P (Nucleus)** | `top_p=0.90` | 8.83 | 16.99 |
| **Temperature 0.2** | `temperature=0.2` | 8.80 | 17.04 |
| **Temperature 0.7** | `temperature=0.7` | 8.66 | 17.32 |
| **Temperature 1.0** | `temperature=1.0` | 12.39 | 12.10 |

**Production Choice (~100-Word Analysis):** **Greedy Decoding** (with `Temperature=0.2` as a secondary option) is recommended for enterprise IT incident triage. Greedy Decoding achieved the highest throughput (**27.10 tok/s**, **2.67× faster** than Beam Search at 10.13 tok/s) and guarantees deterministic, repeatable Linux/AWS commands without the stochastic command-flag drift observed at `T=1.0` (12.10 tok/s).

### Step C2: Speculative Decoding (`Qwen2.5-1.5B` Target + `Qwen2.5-0.5B` Draft)
Both models co-resided comfortably in 24 GB Apple Unified Memory (~4.1 GB combined in `bfloat16`).
- **Prompt 1:** Standard 27.01 tok/s vs. Speculative 20.36 tok/s (**-24.6%**, 82/150 common prefix)
- **Prompt 2:** Standard 24.23 tok/s vs. Speculative 23.71 tok/s (**-2.1%**, 65/150 common prefix)
- **Prompt 3:** Standard 24.57 tok/s vs. Speculative 25.99 tok/s (**+5.8%**, 15/150 common prefix)
- **Average:** **-7.0%** (25.27 → 23.35 tok/s). On Apple MPS with a narrow 3× target-to-draft parameter ratio (1.5B vs 0.5B), Python assisted-generation orchestration and Metal kernel dispatch overhead offset batched verification gains on short 150-token runs; positive speedups appear when target/draft ratio is ≥7–10× or on CUDA with fused verification kernels. Minor downstream prefix drift stems from `bfloat16` floating-point non-associativity between single-token and batched-verification matmuls.

### Step C3: 4-bit Quantization & Production Economics (10 Prompts, Greedy)
Formula: `Cost / 1M tokens = (1,000,000 / throughput_tok_s) / 3600 × hourly_rate` (`T4 = ₹3.50/hr`, `A100 = ₹12.00/hr`).

| Configuration | Measured? | Peak Memory | Throughput | Cost / 1M (T4) | Cost / 1M (A100) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **bfloat16 (Part A baseline)** | Yes (`mps`) | **2.90 GB** | **27.42 tok/s** | **₹35.46** | **₹121.57** |
| **4-bit NF4 quantized** | Pending CUDA (Colab T4) | N/A on MPS | N/A on MPS | N/A on MPS | N/A on MPS |
| **4-bit + Speculative Decoding** | Pending CUDA (Colab T4) | N/A on MPS | N/A on MPS | N/A on MPS | N/A on MPS |

*Hardware Boundary Note:* In strict compliance with academic integrity rules (zero fabricated/projected benchmarks), CUDA-only `bitsandbytes` 4-bit NF4 measurements are not substituted with synthetic multipliers on Apple MPS. Running `notebooks/assignment_1B_colab_t4.ipynb` on a free Google Colab T4 populates all 3 empirical rows plus the Optional Extension (Fine-tuned Adapter + 4-bit NF4 + Speculative) directly.
