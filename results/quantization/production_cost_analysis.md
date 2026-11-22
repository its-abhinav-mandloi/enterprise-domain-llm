# Part C Step 3: 4-bit Quantization & Production Economics

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Prompts Evaluated**: 10 Enterprise Domain Queries
- **Hourly Rates**: T4 = ₹3.50/hr | A100 = ₹12.00/hr

## Production Comparison Table (Assignment 1B Rubric)

| Configuration | Peak VRAM (GB) | Throughput (tok/s) | Cost / 1M tokens (T4 - ₹) | Cost / 1M tokens (A100 - ₹) |
| :--- | :---: | :---: | :---: | :---: |
| **bfloat16 (Part A baseline)** | 2.9 GB | 23.22 | ₹41.87 | ₹143.55 |
| **4-bit NF4 quantized** | 1.22 GB | 31.35 | ₹31.01 | ₹106.33 |
| **4-bit + Speculative Decoding (Step 2)** | 1.57 GB | 41.38 | ₹23.49 | ₹80.55 |

## Production Deployment Recommendation

> We recommend deploying the **4-bit NF4 + Speculative Decoding** configuration on an NVIDIA T4 GPU instance for production enterprise IT helpdesk operations. This architecture achieves the lowest serving cost of ₹23.49 per 1 million tokens while consuming only 1.57 GB of VRAM (well within the T4's 16 GB boundary). Because speculative verification guarantees mathematical equivalence to the target model's greedy output, this setup delivers maximum generation throughput without compromising technical troubleshooting accuracy or command fidelity.

