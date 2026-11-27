# Enterprise Domain-Specific LLM — Assignment 1B Final Technical Report
## Variant V5: IT Helpdesk & Incident Resolution Assistant

**Institution:** BITS Pilani  
**Group Number:** **92**

| Team Member | BITS Student ID |
| :--- | :--- |
| Ravi | `2025aa05029` |
| Pritish | `2025aa05686` |
| Siddarth | `2025aa05446` |
| Abhinav | `2025aa05473` |

---

## 1. Executive Summary

This report documents the end-to-end engineering implementation, empirical evaluation, and production optimization of an enterprise **IT Helpdesk & Incident Resolution Assistant** built for **Assignment 1B (Domain LLM Adaptation & Production Optimization)**.

- **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (`1,543,714,304` parameters, 28 decoder layers, hidden dimension 1,536, vocabulary 151,936)
- **Draft Model (Speculative Decoding):** `Qwen/Qwen2.5-0.5B-Instruct` (`494,032,768` parameters, identical tokenizer/vocabulary)
- **Local Hardware:** Apple MacBook Air M4 (24 GB Unified Memory, Apple Silicon ARM64, PyTorch MPS backend)
- **CUDA Portability:** Full 4-bit NF4 QLoRA & BitsAndBytes CUDA execution packaged in `notebooks/assignment_1B_colab_t4.ipynb` for Google Colab T4 execution.

All local results reported below are **empirically measured** on Apple M4 MPS with zero fabricated or projected numbers. Components requiring NVIDIA CUDA (`bitsandbytes` 4-bit NF4 QLoRA training in Part B2 and 4-bit NF4 benchmarks in Part C3) are explicitly marked **Pending CUDA (Colab T4)** until `notebooks/assignment_1B_colab_t4.ipynb` is executed.

---

## 2. Part A — Domain Data Collection & Pre-Adaptation Baseline [2 Marks]

### Step A1: Corpus Collection & Cleaning
Six official enterprise IT and Linux administration PDFs (9,239 pages total) were extracted page-by-page using PyMuPDF and cleaned via length filtering (≥100 characters/page), SHA-256 exact page deduplication, 5-gram Jaccard near-deduplication (0.85 threshold on 5,000-character prefixes), and English language verification (`langdetect`, sampled on the first 5 pages for large manuals >200 pages and page-by-page otherwise):

| Cleaning Stage | Corpus Volume (Page Chars) | Change | Notes |
| :--- | :---: | :---: | :--- |
| **Raw Extracted (6 PDFs, 9,239 pages)** | 18,232,885 | — | AWS EC2, PostgreSQL 17, GCC, Ubuntu Server, Bash, RHEL 9 |
| **Length Filter (≥100 chars/page)** | 18,040,053 | **-192,832** | **Greatest reduction stage** (removes blank/cover/divider pages) |
| **Exact SHA-256 + Prefix Near-Dedup** | 18,040,053 | 0 | No full-page/prefix duplicates remained |
| **Language Filter (`langdetect`, EN)** | 18,040,053 | 0 | Verified English on sampled/per-page checks |

All 6 cleaned files are saved individually under `data/domain_corpus/*_clean.txt` (`18,058,435` total characters on disk including `\n\n` page-join separators).

### Step A2: Pre-Adaptation Baseline (`bfloat16`, Zero System Prompt)
Evaluated `Qwen/Qwen2.5-1.5B-Instruct` in `bfloat16` on MPS across the 3 canonical prompts (`max_new_tokens=150`, `do_sample=False`, `system_prompt=None`):
- **Prompt 1 (NTFS mount on Ubuntu):** 150 tokens | 7.581s | **19.79 tok/s**
- **Prompt 2 (High disk I/O & CPU on Linux):** 150 tokens | 5.619s | **26.69 tok/s**
- **Prompt 3 (AWS EC2 connectivity & SSH):** 150 tokens | 5.635s | **26.62 tok/s**

---

## 3. Part B — Instruction Dataset, LoRA/QLoRA Fine-Tuning & Comparative Analysis [5 Marks]

### Part B1: Strictly Corpus-Grounded Instruction Dataset
Using `src/data/instruction_generator.py`, we normalized PDF typography (ligatures, hyphenated line breaks, TOC/page header stripping) and extracted section-anchored, sentence-boundary Q&A pairs exclusively from the 6 cleaned manuals in `data/domain_corpus/`:
- **Total Unique Pairs:** **581** (100 AWS EC2, 81 GNU Bash, 100 GCC, 100 PostgreSQL 17, 100 RHEL 9, 100 Ubuntu Server)
- **80/20 Split (`seed=42`):** **465 training pairs (`data/train.jsonl`, 80.03%)** / **116 evaluation pairs (`data/eval.jsonl`, 19.97%)**
- **Average Length:** 107.3 words (~142.7 estimated tokens/example), fitting comfortably within `max_length=384` without truncation
- **Quality Guarantees:** Automated unit tests verify 0 train/eval overlap, 100% unique instructions, zero TOC dot-leader artifacts, clean sentence-start capitalization, and complete sentence-ending punctuation.

### Part B2: Adapter B Fine-Tuning (`r=16, alpha=32, q_proj, v_proj, dropout=0.05`)
- **Trainable Parameters:** `2,179,072` / `1,545,893,376` (`0.1410%`)
- **Hyperparameters:** 2 epochs (60 optimization steps), `max_length=384`, `per_device_batch_size=4`, `gradient_accumulation_steps=4` (effective batch size = 16), `lr=2e-4` cosine schedule, `warmup_steps=5`, Qwen2.5 native chat template.
- **Local Convergence (Apple MPS `bfloat16` LoRA — `use_4bit_qlora=false`):**
  - **Step 0 (Pre-Training) Eval Loss:** `3.1321`
  - **Epoch 1 (Step 30) Eval Loss:** `2.0723` (Train Loss: `2.1521`)
  - **Epoch 2 (Step 60) Eval Loss:** **`1.9787`** (Final Train Loss: `2.0767`, **-36.8% reduction in held-out eval loss**)
- **Genuine 4-bit NF4 QLoRA Status (Pending Colab T4):** Because `bitsandbytes` 4-bit NF4 requires CUDA, the local run executed `bfloat16` LoRA on MPS rather than 4-bit QLoRA. Genuine 4-bit NF4 QLoRA (`BitsAndBytesConfig` + `prepare_model_for_kbit_training` + `paged_adamw_8bit`) is implemented in `src/training/qlora_trainer.py` and `notebooks/assignment_1B_colab_t4.ipynb` (Note: T4 GPUs lack native BF16 tensor cores and compute in `float16`).

### Part B3: Evidence-Backed Comparative Analysis (Baseline vs. Adapter)
Evaluated on the exact 3 canonical prompts (`results/evaluation/comparative_analysis.md`). Note that per A2 the baseline used raw prompt text (`system_prompt=None`), whereas `adapter_evaluator.py` applied `apply_chat_template()` to match SFT formatting:
- **Prompt 1 (NTFS Mount):** Adapter packed denser prose (120 words / 3 inline steps vs. 82 words / 2 steps) and introduced `kernel` support checks, while baseline surfaced `lsblk`/`fdisk` earlier. Honest audit flagged a minor command hallucination in the adapter (`chattr +r /mnt/ntfs` instead of `mount -o remount,rw` / `ntfs-3g`).
- **Prompt 2 (Disk I/O & CPU):** Adapter achieved higher information density (129 words / 3 steps vs. 81 words / 3 steps) covering `top`, `uptime`, `iostat`, and `vmstat`, while unmerged LoRA incurred a 12.5% speed overhead (23.35 vs 26.69 tok/s) removable via `merge_and_unload()`.
- **Prompt 3 (AWS EC2 SSH):** Adapter immediately prioritized AWS-native failure domains (`Amazon VPC`, `security group`, `port 22`, `VPC endpoint` DNS) across 4 steps (118 words vs. 93 words baseline).

---

## 4. Part C — Inference Optimization & Production Economics [8 Marks]

### Step C1: Decoding Strategies & Sampling (`max_new_tokens=150`)
| Strategy | Config | Avg Latency (s) | Avg Throughput (tok/s) |
| :--- | :--- | :---: | :---: |
| **Greedy Decoding** | `do_sample=False, num_beams=1` | **8.01** | **19.86** |
| **Beam Search** | `num_beams=4` | 13.84 | 10.89 |
| **Top-K Sampling** | `top_k=50` | 10.02 | 15.24 |
| **Top-P (Nucleus)** | `top_p=0.90` | 8.34 | 18.27 |
| **Temperature 0.2** | `temperature=0.2` | 8.31 | 18.21 |
| **Temperature 0.7** | `temperature=0.7` | 8.05 | 18.87 |
| **Temperature 1.0** | `temperature=1.0` | 9.33 | 17.09 |

**Production Choice (~100-Word Analysis):** For an enterprise IT Helpdesk & Incident Resolution Assistant (V5), Greedy Decoding (or low Temperature `T=0.2`) is the optimal production choice. Across all three prompts, Greedy Decoding achieved the highest mean throughput (**19.86 tok/s**, **8.01s** avg latency) and deterministic commands (`lsblk -f`, `fdisk -l`, `iostat -x 1`, `aws ec2 describe-instances`). By contrast, Beam Search (`k=4`) slowed generation to **10.89 tok/s** (**13.84s**, **~1.73× latency**) with no gain in actionable CLI depth. High-entropy sampling introduced factual errors, such as Top-K suggesting `vfat` in `/etc/fstab` for an NTFS volume and `T=1.0` (**17.09 tok/s**) hallucinating `blkid` kernel checks and forum-style prose.

### Step C2: Speculative Decoding (`Qwen2.5-1.5B` Target + `Qwen2.5-0.5B` Draft)
Both models co-resided in 24 GB Apple Unified Memory (`~4.08 GB` weight footprint in `bfloat16`).
- **Prompt 1:** Standard 27.01 tok/s vs. Speculative 20.36 tok/s (**-24.6%**, 82/150 common prefix tokens)
- **Prompt 2:** Standard 24.23 tok/s vs. Speculative 23.71 tok/s (**-2.1%**, 65/150 common prefix tokens)
- **Prompt 3:** Standard 24.57 tok/s vs. Speculative 25.99 tok/s (**+5.8%**, 15/150 common prefix tokens)
- **Average:** **-7.0%** (25.27 → 23.35 tok/s). **Acceptance Rate:** Unmeasured locally (not exposed by default by `transformers.generate(assistant_model=...)`). On Apple MPS with a narrow 3× target-to-draft parameter ratio (`1.5B` vs `0.5B`), Python assisted-generation orchestration and Metal kernel dispatch overhead offset batched verification gains on 150-token runs. Exact sequence agreement was 0/3 (average 54.0 common prefix tokens), reflecting `bfloat16` floating-point non-associativity and default logits-processor handling; CUDA verification is ready in Colab.

### Step C3: 4-bit Quantization & Production Economics (10 Prompts, Greedy)
Formula: `Cost / 1M tokens = (1,000,000 / throughput_tok_s) / 3600 × hourly_rate` (`T4 = ₹3.50/hr`, `A100 = ₹12.00/hr`).

| Configuration | Measured? | Peak Memory | Throughput | Cost / 1M (T4) | Cost / 1M (A100) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **bfloat16 (Part A baseline)** | Yes (`mps`) | **2.90 GB** | **27.42 tok/s** | **₹35.46** | **₹121.57** |
| **4-bit NF4 quantized** | Pending CUDA (Colab T4) | Pending CUDA | Pending CUDA | Pending CUDA | Pending CUDA |
| **4-bit + Speculative Decoding** | Pending CUDA (Colab T4) | Pending CUDA | Pending CUDA | Pending CUDA | Pending CUDA |

*Hardware Boundary Note:* In strict compliance with academic integrity rules (zero fabricated or extrapolated benchmarks), CUDA-only `bitsandbytes` 4-bit NF4 measurements are not substituted with synthetic multipliers on Apple MPS. Running `notebooks/assignment_1B_colab_t4.ipynb` on a free Google Colab T4 populates all 3 empirical rows plus the Optional Extension (Fine-tuned Adapter + 4-bit NF4 + Speculative) directly.
