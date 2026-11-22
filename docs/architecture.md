# System Architecture & Technical Flow

## 1. End-to-End Enterprise LLM Pipeline

```mermaid
flowchart TD
    subgraph PartA["Part A: Data Collection & Baseline [2 Marks]"]
        A1["6 Domain PDFs<br>(AWS, Ubuntu, RHEL, Bash, Postgres, GCC)"] --> A2["PyMuPDF Page-by-Page Extraction"]
        A2 --> A3["Corpus Cleaning:<br>• Length filter (>=100 chars)<br>• SHA-256 Deduplication<br>• LangDetect English Filter"]
        A3 --> A4["Clean Domain Corpus<br>(data/domain_corpus/*.txt)"]
        A5["Qwen2.5-1.5B-Instruct<br>(bfloat16 Baseline)"] --> A6["3 Canonical Domain Prompts<br>(Zero System Prompt)"]
        A6 --> A7["Baseline Output Report<br>(results/baseline/)"]
    end

    subgraph PartB["Part B: QLoRA Fine-Tuning [5 Marks]"]
        A4 --> B1["Instruction Dataset Generator<br>(720 Pairs, 80/20 Train/Eval Split)"]
        B1 --> B2["PEFT LoRA Adapter B<br>(r=16, alpha=32, target: q_proj, v_proj)"]
        B2 --> B3["SFT Training (bfloat16 MPS / 4-bit CUDA)"]
        B3 --> B4["Saved Adapter<br>(models/lora_adapter/)"]
        B4 --> B5["Comparative Analysis<br>(Baseline vs Adapter on 3 Prompts)"]
    end

    subgraph PartC["Part C: Inference Optimization & Production Metrics [8 Marks]"]
        C1["Step C1: 5 Decoding Strategies<br>(Greedy, Beam, Top-K, Top-P, Temp 0.2/0.7/1.0)"]
        C2["Step C2: Speculative Decoding<br>(Target 1.5B + Draft 0.5B)"]
        C3["Step C3: 4-bit Quantization & Economics<br>(10 Prompts, T4 @ ₹3.50/hr, A100 @ ₹12.00/hr)"]
    end

    PartA --> PartB
    PartB --> PartC
    PartC --> Sub["Submission Deliverables & Final Report"]
```

## 2. Directory & Component Design

- **`configs/`**: Strict YAML separation of model parameters, training hyperparameters, and benchmark configurations.
- **`data/`**: Structured data flow: `raw_pdfs/` → `extracted_text/` → `domain_corpus/` → `train.jsonl` / `eval.jsonl`.
- **`src/`**: Modular Python modules adhering to PEP 8, type hinting, and robust exception handling.
- **`notebooks/`**: Educational, reproducible notebooks tracking each academic step with inline visualizations.
- **`results/`**: Structured JSON, CSV, and PNG outputs for reproducible grading.
- **`submission/`**: Consolidated HTML exports, executive reports, and rubric compliance tables.
