# Part C Step 2: Speculative Decoding Benchmark

- **Target Model**: `Qwen/Qwen2.5-1.5B-Instruct` (1.54B params)
- **Draft Model**: `Qwen/Qwen2.5-0.5B-Instruct` (0.49B params)
- **Device / Precision**: `mps` (`torch.bfloat16`)
- **Average Throughput Change**: **-7.0%** (25.27 -> 23.35 tok/s)
- **Draft Acceptance Rate**: Unmeasured (not directly exposed by `transformers.generate(assistant_model=...)`)

## Performance Comparison Table

| Prompt | Baseline Latency | Speculative Latency | Baseline tok/s | Speculative tok/s | Speedup (%) | Exact Match / Common Prefix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **prompt_1** (linux_storage) | 5.553s | 7.367s | 27.01 | 20.36 | **-24.6%** | Diverged (82/150 tok) |
| **prompt_2** (linux_performance) | 6.19s | 6.327s | 24.23 | 23.71 | **-2.1%** | Diverged (65/150 tok) |
| **prompt_3** (cloud_networking) | 6.105s | 5.771s | 24.57 | 25.99 | **+5.8%** | Diverged (15/150 tok) |

## Observations & Inferences

On `mps` (`torch.bfloat16`), speculative decoding measured **-7.0%** throughput change (25.27 -> 23.35 tok/s). For a compact 1.5B target model with only a 3x parameter ratio against the 0.5B draft model, Python-level Hugging Face assisted-generation orchestration and per-step Metal kernel launch overhead on `mps` outweigh batched target verification savings. Output comparison showed 0/3 exact full-sequence matches with an average identical prefix of 54.0 tokens; downstream divergences reflect `torch.bfloat16` floating-point non-associativity between single-token decoding and batched verification passes (alongside default logits-processor interaction), while both runs produced relevant troubleshooting steps upon inspection. Per-step draft acceptance rate is not exposed directly by `transformers.generate()` and is unmeasured.
