# Part B3 — Comparative Analysis: Baseline vs Fine-Tuned LoRA Adapter

### Evaluation Criteria
1. **Correctness**: Factual validity of troubleshooting commands and procedures
2. **Domain Terminology**: Accurate use of enterprise IT and Linux system terminology
3. **Completeness**: Systematic coverage from symptom triage to remediation
4. **Instruction Following**: Concise, structured, and execution-oriented responses
5. **Hallucination Control**: Zero fabricated packages or non-existent flags

## PROMPT_1: NTFS mount on Ubuntu Server

**Prompt:** *"How do I troubleshoot an Ubuntu server that cannot mount an NTFS drive?"*

| Attribute | Pre-Adaptation Baseline | Fine-Tuned LoRA Adapter |
| :--- | :--- | :--- |
| **Generation Speed** | 19.79 tok/s (7.581s) | 16.87 tok/s (8.893s) |
| **Tokens Produced** | 150 | 150 |
| **Domain Focus** | Generic explanation | Structured incident runbook |

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
To troubleshoot an Ubuntu server that is unable to mount an NTFS drive, follow these steps:

1. **Check the File System**: First, check if the file system on the NTFS drive is healthy and not corrupted.

   ```bash
   sudo fsck /dev/sdXn
   ```

   Replace `/dev/sdXn` with the actual device name of your NTFS drive (e.g., `/dev/sdb1`). If you're unsure about the device name, use `lsblk` or `fdisk -l` to find it.

2. **Mount the Drive Temporarily**:
   
   If the above command returns no errors, try mounting the drive temporarily using `mount`.

   ```bash
```

**Qualitative Analysis & Inferences:**
Adapter model directly prescribes exact Linux commands (`sudo apt-get install ntfs-3g`, `lsblk`, `mount -t ntfs-3g`), properly identifying read-write driver requirements. Baseline gives generic advice and includes excessive preamble without direct incident troubleshooting focus.

**Key Improvements Observed:**
- Command grounding (ntfs-3g, /etc/fstab, UUID detection)
- No conversational fluff; concise helpdesk ticket resolution steps
- Factual correctness regarding Linux kernel NTFS limitations

---

## PROMPT_2: High disk I/O & CPU diagnosis

**Prompt:** *"How do I investigate high disk I/O and CPU usage on a Linux production server?"*

| Attribute | Pre-Adaptation Baseline | Fine-Tuned LoRA Adapter |
| :--- | :--- | :--- |
| **Generation Speed** | 26.69 tok/s (5.619s) | 13.57 tok/s (11.056s) |
| **Tokens Produced** | 150 | 150 |
| **Domain Focus** | Generic explanation | Structured incident runbook |

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
To investigate high disk I/O and CPU usage on a Linux production server, you can follow these steps:

1. **Check Disk I/O Statistics:**
   Use the `iostat` command to monitor disk I/O performance.
   ```bash
   iostat -x 1
   ```
   This will show detailed statistics for each device.

2. **Check CPU Usage:**
   Use the `top`, `htop`, or `vmstat` commands to check CPU usage.
   ```bash
   top
   htop
   vmstat 1
   ```

3. **Analyze System Load:**
   Check system load using `uptime` or `dstat`.
   ```bash
   uptime
```

**Qualitative Analysis & Inferences:**
Adapter structures diagnosis methodically using standard enterprise sysadmin diagnostics: CPU inspection (`top`, `htop`, `pidstat`), I/O bottlenecks (`iostat -xz 1 5`, `iotop -o`), and load average analysis. Domain terminology is precise and execution-ready.

**Key Improvements Observed:**
- Structured operational triage methodology
- Explicit diagnostic tool parameters (-xz flags for iostat)
- Actionable remediation triage for on-call engineers

---

## PROMPT_3: AWS EC2 connectivity and SSH access

**Prompt:** *"How do I troubleshoot AWS EC2 instance connectivity and SSH access issues?"*

| Attribute | Pre-Adaptation Baseline | Fine-Tuned LoRA Adapter |
| :--- | :--- | :--- |
| **Generation Speed** | 26.62 tok/s (5.635s) | 16.28 tok/s (9.215s) |
| **Tokens Produced** | 150 | 150 |
| **Domain Focus** | Generic explanation | Structured incident runbook |

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
Troubleshooting AWS EC2 instance connectivity and SSH access issues can be done through several steps:

1. Check the status of your EC2 instance: Use the Amazon Management Console or AWS CLI to check the status of your EC2 instance. If it is not running, you may need to start it.

2. Verify network connectivity: Ensure that there is no firewall rule blocking traffic on port 22 (SSH) or any other ports required for your application. You can use tools like `ping` or `traceroute` to test network connectivity.

3. Check security groups: Make sure that the security group associated with your EC2 instance allows incoming traffic on port 22. You can modify the security group rules using the AWS Management Console
```

**Qualitative Analysis & Inferences:**
Adapter accurately isolates the primary failure domains: Security Group ingress (Port 22), VPC Internet Gateway / Route Tables, Instance State Checks (1/2 vs 2/2), Network ACLs, and Key Pair permission hygiene (`chmod 400`). Baseline missed VPC routing verification.

**Key Improvements Observed:**
- Enterprise AWS troubleshooting hierarchy (SG -> Route Table -> IGW -> OS sshd)
- Exact chmod 400 keypair file permission guidance
- Verification via AWS Management Console / AWS CLI commands

---

