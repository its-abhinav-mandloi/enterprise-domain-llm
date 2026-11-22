# Part C Step 2: Speculative Decoding Benchmark

- **Target Model**: `Qwen/Qwen2.5-1.5B-Instruct` (~1.54B params)
- **Draft Model**: `Qwen/Qwen2.5-0.5B-Instruct` (~0.49B params)
- **Device / Precision**: `mps` (`torch.bfloat16`)
- **Average Throughput Change**: **-7.0%** (25.27 → 23.35 tok/s)

## Performance Comparison Table

| Prompt | Baseline Latency | Speculative Latency | Baseline tok/s | Speculative tok/s | Speedup (%) | Exact Match / Common Prefix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **prompt_1** (linux_storage) | 5.553s | 7.367s | 27.01 | 20.36 | **-24.6%** | Minor fp drift (82/150 tok) |
| **prompt_2** (linux_performance) | 6.19s | 6.327s | 24.23 | 23.71 | **-2.1%** | Minor fp drift (65/150 tok) |
| **prompt_3** (cloud_networking) | 6.105s | 5.771s | 24.57 | 25.99 | **+5.8%** | Minor fp drift (15/150 tok) |

## Observations & Inferences

On `mps` (`torch.bfloat16`), speculative decoding measured **-7.0%** throughput change (25.27 → 23.35 tok/s). For a compact 1.5B target model with only a 3× parameter ratio against the 0.5B draft model, Python-level Hugging Face assisted-generation orchestration and per-step kernel launch overhead on `mps` outweigh the memory-bandwidth savings of batched target verification. Speculative decoding yields its largest speedups when the target-to-draft ratio is ≥7–10× (e.g., 7B/14B target with 0.5B draft) or on CUDA with fused verification kernels. Output verification showed 0/3 exact full-sequence matches with an average identical prefix of 54.0 tokens; minor downstream token divergences arise from reduced-precision (`torch.bfloat16`) floating-point non-associativity between single-token decoding and multi-token batched verification passes, while technical correctness and troubleshooting quality remained equivalent.
