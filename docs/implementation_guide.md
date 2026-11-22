# Implementation Guide & Reproduction Manual

## 1. Environment Setup

### Local macOS Execution (Apple Silicon M4)
```bash
# Clone the private repository
git clone https://github.com/abhinavmandloi/enterprise-domain-llm.git
cd enterprise-domain-llm

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install reproducible dependencies
pip install -r requirements.txt
```

---

## 2. Pipeline Execution Steps

### Step 1: Data Ingestion & Cleaning (Part A1 & B1)
```bash
python data_pipeline.py
```
*Outputs: `data/domain_corpus/*.txt`, `results/corpus_statistics.json`, `data/train.jsonl`, `data/eval.jsonl`.*

### Step 2: Baseline Model Execution (Part A2)
```bash
python -m src.evaluation.baseline_evaluator
```
*Outputs: `results/baseline/baseline_outputs.json`.*

### Step 3: QLoRA Fine-Tuning (Part B2)
```bash
python -m src.training.qlora_trainer
```
*Outputs: `models/lora_adapter/`.*

### Step 4: Adapter Evaluation & Comparative Analysis (Part B3)
```bash
python -m src.evaluation.adapter_evaluator
python -m src.evaluation.quality_evaluator
```
*Outputs: `results/evaluation/comparative_analysis.md`.*

### Step 5: Decoding Strategies Benchmark (Part C1)
```bash
python -m src.optimization.decoding
```
*Outputs: `results/decoding/decoding_benchmark.csv`.*

### Step 6: Speculative Decoding Benchmark (Part C2)
```bash
python -m src.optimization.speculative_decoding
```
*Outputs: `results/speculative/speculative_benchmark.csv`.*

### Step 7: 4-bit Quantization & Production Economics (Part C3)
```bash
python -m src.optimization.quantization
```
*Outputs: `results/quantization/quantization_cost_results.json`.*

### Step 8: Visualization Generation
```bash
python -m src.reporting.charts
```
*Outputs: `results/charts/*.png`.*
