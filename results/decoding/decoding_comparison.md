# Part C Step 1: Decoding Strategies & Sampling Benchmark

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Device**: `mps` (`torch.bfloat16`)
- **Max New Tokens**: `150`

## Throughput Comparison Matrix (Tokens / Sec)

| Prompt | Greedy | Beam Search | Top-K (50) | Top-P (0.9) | Temp 0.2 | Temp 0.7 | Temp 1.0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **prompt_1** (linux_storage) | 27.1 | 10.13 | 16.03 | 16.99 | 17.04 | 17.32 | 12.1 |
| **prompt_2** (linux_performance) | 15.64 | 11.95 | 17.13 | 16.26 | 16.87 | 17.39 | 16.98 |
| **prompt_3** (cloud_networking) | 16.84 | 10.6 | 12.55 | 21.55 | 20.73 | 21.9 | 22.18 |
| **3-Prompt Mean (tok/s)** | **19.86** | **10.89** | **15.24** | **18.27** | **18.21** | **18.87** | **17.09** |

## Production Recommendation & 100-Word Analysis

> For an enterprise IT Helpdesk & Incident Resolution Assistant (V5), Greedy Decoding (or low Temperature T=0.2) is the optimal production choice. Across all three prompts, Greedy Decoding achieved the highest mean throughput (19.86 tok/s, 8.01s avg latency) and deterministic commands (`lsblk -f`, `fdisk -l`, `iostat -x 1`, `aws ec2 describe-instances`). By contrast, Beam Search (k=4) slowed generation to 10.89 tok/s (13.84s, ~1.73x latency) with no gain in actionable CLI depth. High-entropy sampling introduced factual errors, such as Top-K suggesting `vfat` in `/etc/fstab` for an NTFS volume and T=1.0 (17.09 tok/s) hallucinating `blkid` kernel checks and forum-style prose.
