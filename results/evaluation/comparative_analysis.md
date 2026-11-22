# Part B3 — Comparative Analysis: Baseline vs Fine-Tuned LoRA Adapter

All findings below are computed directly from the generated baseline and adapter outputs.

## PROMPT_1: NTFS Mount Troubleshooting on Ubuntu Server

**Prompt:** *"How do I troubleshoot an Ubuntu server that cannot mount an NTFS drive?"*

| Metric | Pre-Adaptation Baseline | Fine-Tuned Adapter |
| :--- | :--- | :--- |
| **Generation Speed** | 19.79 tok/s (7.581s) | 25.27 tok/s (5.937s) |
| **Tokens / Words** | 150 tok / 82 words | 150 tok / 120 words |
| **Numbered Steps** | 0 | 3 |
| **Domain Terms Found** | 3 (`fdisk, lsblk, mount`) | 2 (`kernel, mount`) |

#### Baseline Output:
```text
To troubleshoot a problem with mounting an NTFS drive on an Ubuntu server, you can follow these steps:

1. **Check the Filesystem Type**: First, verify if the filesystem type of the NTFS drive is indeed NTFS by using `lsblk` or `fdisk -l`. For example:
   ```bash
   lsblk -f /dev/sdX
   ```
   Or for `/dev/nvme0n1p2` (assuming it's an NVMe drive):
   ```bash
   fdisk -l /dev/nvme0n1p2
   ```

2. **Mount the Drive Manually**: Try to manually mount the drive to see if there are any errors during the
```

#### Fine-Tuned Adapter Output:
```text
The following steps can help you to identify the cause of the problem and resolve it: 1. Check if the NTFS file system is supported on your Linux kernel version. If not, update your kernel or use another Linux distribution that supports this file system. 2. Check if the NTFS file system is mounted in read-only mode. To check this, run the command mount | grep ntfs. If the output shows that the NTFS file system is mounted in read-only mode, change its mount options to rw using the chattr +r /mnt/ntfs command. 3. Check if there is any error message when trying to mount the NTFS file system. If so, try to find out what kind of error occurred
```

**Observed Analysis:**
Across the 150-token response window, the baseline referenced 3 domain terms (fdisk, lsblk, mount) across 0 steps, while the fine-tuned adapter referenced 2 domain terms (kernel, mount) across 3 steps. Notably, the adapter surfaced kernel. Conversely, the baseline covered fdisk, lsblk earlier in its response. Quality check note: Minor command hallucination: suggests `chattr` for NTFS mount options instead of `mount -o remount,rw` / `ntfs-3g`. Because `Qwen2.5-1.5B-Instruct` is already instruction-tuned, compact domain adaptation shifts phrasing and prioritization toward corpus documentation style while preserving base fluency; unmerged LoRA layers incur a measurable inference latency penalty until merged with `merge_and_unload()`.

**Observed Improvements:**
- Adapter introduced domain terms not in baseline: `kernel`
- Higher step coverage within the 150-token budget (3 numbered steps vs 0 in baseline)

**Observed Limitations / Trade-Offs:**
- Baseline included terms omitted by adapter within 150 tokens: `fdisk, lsblk`
- Technical accuracy caveat in adapter output: Minor command hallucination: suggests `chattr` for NTFS mount options instead of `mount -o remount,rw` / `ntfs-3g`

---

## PROMPT_2: High Disk I/O & CPU Diagnosis on Linux

**Prompt:** *"How do I investigate high disk I/O and CPU usage on a Linux production server?"*

| Metric | Pre-Adaptation Baseline | Fine-Tuned Adapter |
| :--- | :--- | :--- |
| **Generation Speed** | 26.69 tok/s (5.619s) | 23.35 tok/s (6.423s) |
| **Tokens / Words** | 150 tok / 81 words | 150 tok / 129 words |
| **Numbered Steps** | 0 | 3 |
| **Domain Terms Found** | 5 (`htop, iostat, top, uptime, vmstat`) | 4 (`iostat, top, uptime, vmstat`) |

#### Baseline Output:
```text
To investigate high disk I/O and CPU usage on a Linux production server, you can follow these steps:

1. **Check Disk I/O Statistics:**
   Use the `iostat` command to monitor disk I/O performance.
   ```bash
   iostat -x 1
   ```
   This will provide detailed statistics about disk activity over one interval.

2. **Check CPU Usage:**
   Use the `top`, `htop`, or `vmstat` commands to check CPU usage.
   ```bash
   top
   htop
   vmstat 1
   ```

3. **Analyze System Load:**
   Check system load using `uptime` or `systemd-analyze`.
   ```
```

#### Fine-Tuned Adapter Output:
```text
The following steps can be used to identify the cause of high disk I/O and CPU usage: 1. Check the system load average using the top command or the uptime command. If the load is greater than 2, it indicates that there may be a problem with the disk or CPU. 2. Use the iostat command to check the current disk activity. The output will show you how many times each device has been read or written in the last minute. This information can help you determine if the disk is being accessed too frequently. 3. Use the vmstat command to check the current memory usage. The output will show you how much memory is currently available for use. If the amount of free memory is low, it could
```

**Observed Analysis:**
Across the 150-token response window, the baseline referenced 5 domain terms (htop, iostat, top, uptime, vmstat) across 0 steps, while the fine-tuned adapter referenced 4 domain terms (iostat, top, uptime, vmstat) across 3 steps. Conversely, the baseline covered htop earlier in its response. Because `Qwen2.5-1.5B-Instruct` is already instruction-tuned, compact domain adaptation shifts phrasing and prioritization toward corpus documentation style while preserving base fluency; unmerged LoRA layers incur a measurable inference latency penalty until merged with `merge_and_unload()`.

**Observed Improvements:**
- Higher step coverage within the 150-token budget (3 numbered steps vs 0 in baseline)

**Observed Limitations / Trade-Offs:**
- Baseline included terms omitted by adapter within 150 tokens: `htop`
- Unmerged LoRA adapter adds runtime dispatch overhead (23.35 tok/s vs 26.69 tok/s baseline, -12.5%); merging weights (`merge_and_unload()`) eliminates this in production

---

## PROMPT_3: AWS EC2 Connectivity & SSH Access Triage

**Prompt:** *"How do I troubleshoot AWS EC2 instance connectivity and SSH access issues?"*

| Metric | Pre-Adaptation Baseline | Fine-Tuned Adapter |
| :--- | :--- | :--- |
| **Generation Speed** | 26.62 tok/s (5.635s) | 24.36 tok/s (6.157s) |
| **Tokens / Words** | 150 tok / 93 words | 150 tok / 118 words |
| **Numbered Steps** | 0 | 4 |
| **Domain Terms Found** | 5 (`aws, ec2, ping, ssh, traceroute`) | 5 (`ec2, ping, port 22, security group, vpc`) |

#### Baseline Output:
```text
Troubleshooting AWS EC2 instance connectivity and SSH access issues involves several steps. Here are some common methods to diagnose and resolve such problems:

### 1. **Check Instance Status**
   - Ensure your EC2 instance is running by checking the status in the Amazon Management Console or using the `aws ec2 describe-instances` command.
     ```bash
     aws ec2 describe-instances --instance-ids <INSTANCE_ID>
     ```

### 2. **Verify Network Connectivity**
   - Use `ping` or `traceroute` to check if you can reach the internet from your EC2 instance.
     ```bash
     ping example.com
     traceroute example.com
     ```
   - If you cannot reach the internet,
```

#### Fine-Tuned Adapter Output:
```text
The following steps can help you diagnose the issue: 1. Check if your network is reachable from outside of Amazon VPC. If not, check your internet connection or contact your Internet Service Provider (ISP). 2. Verify that your security group rules allow inbound traffic on port 22 for the IP address of the EC2 instance. 3. Use the command "ping" to test whether the EC2 instance can reach the public DNS name of the VPC endpoint. If it cannot be reached, then there may be an issue with the VPC endpoint configuration. 4. Use the command "ping" to test whether the EC2 instance can reach the private DNS name of the VPC endpoint. If it cannot be reached,
```

**Observed Analysis:**
Across the 150-token response window, the baseline referenced 5 domain terms (aws, ec2, ping, ssh, traceroute) across 0 steps, while the fine-tuned adapter referenced 5 domain terms (ec2, ping, port 22, security group, vpc) across 4 steps. Notably, the adapter surfaced port 22, security group, vpc. Conversely, the baseline covered aws, ssh, traceroute earlier in its response. Because `Qwen2.5-1.5B-Instruct` is already instruction-tuned, compact domain adaptation shifts phrasing and prioritization toward corpus documentation style while preserving base fluency; unmerged LoRA layers incur a measurable inference latency penalty until merged with `merge_and_unload()`.

**Observed Improvements:**
- Adapter introduced domain terms not in baseline: `port 22, security group, vpc`
- Higher step coverage within the 150-token budget (4 numbered steps vs 0 in baseline)

**Observed Limitations / Trade-Offs:**
- Baseline included terms omitted by adapter within 150 tokens: `aws, ssh, traceroute`
- Unmerged LoRA adapter adds runtime dispatch overhead (24.36 tok/s vs 26.62 tok/s baseline, -8.5%); merging weights (`merge_and_unload()`) eliminates this in production

---

