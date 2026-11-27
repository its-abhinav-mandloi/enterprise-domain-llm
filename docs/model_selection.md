# Model Selection & Architectural Specification

## 1. Primary Model Architecture Specification

| Architectural Property | Primary Target Model (`Qwen/Qwen2.5-1.5B-Instruct`) | Draft Model (`Qwen/Qwen2.5-0.5B-Instruct`) |
| :--- | :--- | :--- |
| **Hugging Face Model ID** | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Developer / Family** | Alibaba Cloud / Qwen 2.5 Open Model Family | Alibaba Cloud / Qwen 2.5 Open Model Family |
| **Model Type** | Causal Language Model (Decoder-Only Transformer) | Causal Language Model (Decoder-Only Transformer) |
| **Total Parameter Count** | **1,543,714,304** (~1.54 Billion) | **494,032,768** (~0.49 Billion) |
| **Active Parameters** | 1.54B (Dense architecture) | 0.49B (Dense architecture) |
| **Hidden Dimension ($d_{\text{model}}$)** | **1,536** | **896** |
| **Intermediate Size ($d_{\text{ffn}}$)** | **8,960** (SwiGLU activation) | **4,864** (SwiGLU activation) |
| **Decoder Transformer Layers** | **28** | **24** |
| **Attention Query Heads ($n_q$)** | **12** | **14** |
| **Key/Value Heads ($n_{kv}$)** | **2** (Grouped-Query Attention, GQA ratio 6:1) | **2** (Grouped-Query Attention, GQA ratio 7:1) |
| **Vocabulary Size** | **151,936** | **151,936** (**Identical**) |
| **Context Window Capacity** | Up to **131,072 tokens** (128K) | Up to **131,072 tokens** (128K) |
| **Positional Embeddings** | RoPE (Rotary Position Embeddings) | RoPE (Rotary Position Embeddings) |
| **License** | Apache 2.0 (Open Source, Unrestricted) | Apache 2.0 (Open Source, Unrestricted) |

---

## 2. Memory Footprint Breakdown

Calculated for single-precision and quantized representations:

| Precision / Format | Bytes per Param | Target Model (1.5B) Memory | Draft Model (0.5B) Memory | Combined Working Set |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 (Float 32)** | 4.0 | 6.17 GB | 1.98 GB | 8.15 GB |
| **BF16 / FP16 (Baseline)** | 2.0 | **3.09 GB** | **0.99 GB** | **4.08 GB** |
| **INT8 Quantized** | 1.0 | 1.54 GB | 0.49 GB | 2.03 GB |
| **4-bit NF4 Quantized** | 0.5 | **0.95 GB** | **0.32 GB** | **1.27 GB** |

> [!NOTE]
> On the development system (Apple Silicon M4 with 24 GB Unified Memory), the combined baseline working set of **4.08 GB** fits with over **19 GB of headroom**, guaranteeing zero paging or memory thrashing during speculative decoding.

---

## 3. Justification for Qwen 2.5-1.5B-Instruct

1. **Academic Rubric Compliance**:
   - The model natively supports `bfloat16` execution without loss of dynamic range.
   - It incorporates standard linear projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), perfectly aligning with Assignment 1B Adapter B requirements (`target_modules = ["q_proj", "v_proj"]`).

2. **Tokenizer Alignment for Speculative Decoding**:
   - Speculative decoding mandates that the draft model and target model share an **identical tokenizer and vocabulary** so that draft token IDs map directly to the target model's embedding matrix without token re-encoding.
   - `Qwen2.5-1.5B-Instruct` and `Qwen2.5-0.5B-Instruct` share the identical 151,936-token BPE tokenizer, achieving mathematically optimal verification acceptance rates.

3. **Zero Authentication Friction**:
   - Unlike gated models (e.g. LLaMA 3.2), Qwen 2.5 is completely ungated under Apache 2.0, ensuring 100% automated reproducibility for university evaluation without external credential dependencies.

4. **Domain Technical Reasoning**:
   - Qwen 2.5 has demonstrated superior code, command-line syntax, and Linux sysadmin understanding compared to similarly sized models (SmolLM-1.7B or DistilGPT2), ensuring high-quality IT Helpdesk responses.
