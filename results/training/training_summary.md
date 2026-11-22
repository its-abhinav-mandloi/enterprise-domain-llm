# Part B Step 2: QLoRA Fine-Tuning Report

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Adapter Configuration**: Adapter B (Balanced)
- **LoRA Rank ($r$)**: `16`
- **LoRA Alpha ($\alpha$)**: `32` (scaling ratio = 2.0)
- **Target Modules**: `["q_proj", "v_proj"]`
- **Training Examples**: 576
- **Evaluation Examples**: 144
- **Effective Batch Size**: 8 (batch_size=2 * grad_accum=4)
- **Learning Rate**: `2e-4` (Cosine schedule)
- **Precision**: `bfloat16` on Apple Silicon MPS

## Training Loss Progression

| Step | Training Loss | Eval Loss | Learning Rate |
| :---: | :---: | :---: | :---: |
| 5 | 4.6649 | - | 1.99e-04 |
| 10 | 3.6311 | - | 1.65e-04 |
| 10 | - | 4.6509 | - |
| 15 | 3.8292 | - | 1.00e-04 |
| 20 | 4.1595 | - | 3.45e-05 |
| 20 | - | 4.5476 | - |
| 25 | 4.765 | - | 1.02e-06 |
| 25 | - | 4.5409 | - |
| 25 | 4.2099 | - | - |
