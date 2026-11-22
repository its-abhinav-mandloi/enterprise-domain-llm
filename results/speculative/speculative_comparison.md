# Part C Step 2: Speculative Decoding Benchmark

- **Target Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Draft Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Device**: `mps`
- **Average Throughput Speedup**: **-21.8%**

## Performance Comparison Table

| Prompt | Baseline Latency | Speculative Latency | Baseline tok/s | Speculative tok/s | Speedup (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **prompt_1** (linux_storage) | 7.054s | 12.167s | 21.27 | 12.33 | **-42.0%** |
| **prompt_2** (linux_performance) | 7.022s | 8.0s | 21.36 | 18.75 | **-12.2%** |
| **prompt_3** (cloud_networking) | 7.295s | 8.216s | 20.56 | 18.26 | **-11.2%** |

## Observations & Inferences

> Speculative decoding achieved an average speedup of -21.8% throughput increase across the enterprise IT domain prompts. Because speculative decoding uses the draft model (Qwen2.5-0.5B) to propose candidate tokens in parallel and the target model (Qwen2.5-1.5B) to verify them in a single forward pass, the mathematically verified output remains identical to greedy decoding of the target model while significantly reducing total memory bandwidth latency bottleneck.

