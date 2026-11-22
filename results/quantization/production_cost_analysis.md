# Part C Step 3: 4-bit Quantization & Production Economics

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Prompts Evaluated**: 10 Enterprise Domain Queries (Greedy Decoding, `max_new_tokens=150`)
- **Hourly Rates**: T4 = ₹3.50/hr | A100 = ₹12.00/hr
- **Formula**: `Cost / 1M tokens = (1,000,000 / throughput_tok_s) / 3600 * hourly_rate`

## Production Comparison Table (Empirical Measurements)

| Configuration | Measured? | Peak VRAM (GB) | Throughput (tok/s) | Cost / 1M (T4 - ₹) | Cost / 1M (A100 - ₹) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **bfloat16 (Part A baseline)** | Yes (`mps`) | 2.9 GB | 27.42 | ₹35.46 | ₹121.57 |
| **4-bit NF4 quantized** | Pending CUDA (Colab T4) | N/A on MPS | N/A on MPS | N/A on MPS | N/A on MPS |
| **4-bit + Speculative Decoding (Step 2)** | Pending CUDA (Colab T4) | N/A on MPS | N/A on MPS | N/A on MPS | N/A on MPS |

## Production Deployment Recommendation

On local Apple Silicon M4 MPS, the empirically measured `bfloat16` baseline achieved **27.42 tok/s** at **2.9 GB** memory (**₹35.46/1M tokens** at T4 rate; **₹121.57/1M tokens** at A100 rate). Because `bitsandbytes` 4-bit NF4 kernels require NVIDIA CUDA, 4-bit NF4 and 4-bit + Speculative rows are not fabricated locally and are prepared for one-click empirical execution in `notebooks/assignment_1B_colab_t4.ipynb` on a free Google Colab T4 GPU.
