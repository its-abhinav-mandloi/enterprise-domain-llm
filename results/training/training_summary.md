# Part B Step 2: QLoRA / LoRA Fine-Tuning Report

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Adapter Configuration**: Adapter B (Balanced)
- **LoRA Rank ($r$)**: `16` | **Alpha ($\alpha$)**: `32` (scaling = 2.0)
- **Target Modules**: `['q_proj', 'v_proj']`
- **Trainable Parameters**: `2,179,072` / `1,545,893,376` (`0.141%`)
- **Precision / Backend**: bfloat16 LoRA on MPS (4-bit NF4 requires CUDA)
- **Dataset Split**: `465` train / `116` eval (80/20, seed=42)
- **Hyperparameters**: `epochs=2`, `steps=60`, `max_length=384`, `effective_batch_size=16`, `lr=0.0002` (cosine)
- **Convergence**: Step-0 Eval Loss `3.1321` → Final Eval Loss `1.9787` (Final Train Loss `2.0767`)

## Training & Evaluation Loss Progression

| Step | Epoch | Training Loss | Eval Loss | Learning Rate |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0.0 | - | 3.1321 | - |
| 5 | 0.17 | 3.1058 | - | 1.60e-04 |
| 10 | 0.34 | 2.6674 | - | 1.97e-04 |
| 15 | 0.51 | 2.4731 | - | 1.87e-04 |
| 20 | 0.68 | 2.2907 | - | 1.70e-04 |
| 25 | 0.85 | 2.2903 | - | 1.47e-04 |
| 30 | 1.0 | 2.1521 | - | 1.20e-04 |
| 30 | 1.0 | - | 2.0723 | - |
| 35 | 1.17 | 2.0606 | - | 9.14e-05 |
| 40 | 1.34 | 2.0574 | - | 6.37e-05 |
| 45 | 1.51 | 2.0461 | - | 3.89e-05 |
| 50 | 1.68 | 2.0651 | - | 1.91e-05 |
| 55 | 1.85 | 2.0009 | - | 5.82e-06 |
| 60 | 2.0 | 2.0767 | - | 1.63e-07 |
| 60 | 2.0 | - | 1.9787 | - |
