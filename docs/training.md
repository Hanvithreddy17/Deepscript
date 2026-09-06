# DeepScript Training Guide & Transfer Learning Strategy

This document outlines the two-stage transfer learning strategy, optimization parameters, learning rate schedules, and step-by-step reproduction instructions for training DeepScript.

---

## 1. Two-Stage Transfer Learning Strategy

```
┌────────────────────────────────────────────────────────┐
│ STAGE 1: HEAD WARMUP                                   │
│ • ViT Backbone: FROZEN (85.8M parameters)              │
│ • Projector + Cosine Head: TRAINABLE (804.8K params)   │
│ • Head LR: 1e-3 | Backbone LR: 0.0                     │
│ • Optimizer: AdamW | Scheduler: CosineAnnealingLR      │
│ • Goal: Initialize metric projector & class weights    │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: FINE-TUNING                                   │
│ • ViT Backbone: Unfreeze Last 2 Blocks (Layers 10 & 11)│
│ • Trainable Params: 14.98M parameters                  │
│ • Differential LR: Backbone LR = 2e-5, Head LR = 2e-4  │
│ • Optimizer: AdamW | Scheduler: CosineAnnealingLR      │
│ • Goal: Adapt attention maps to inscription strokes    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Training Hyperparameters

| Hyperparameter | Stage 1 (Warmup) | Stage 2 (Fine-Tuning) |
| :--- | :--- | :--- |
| **Epochs** | 2 Epochs | 2 Epochs |
| **Batch Size** | 32 | 32 |
| **Optimizer** | AdamW | AdamW |
| **Head Learning Rate** | $1.0 \times 10^{-3}$ | $2.0 \times 10^{-4}$ |
| **Backbone Learning Rate** | $0.0$ (Frozen) | $2.0 \times 10^{-5}$ |
| **Weight Decay** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ |
| **LR Scheduler** | `CosineAnnealingLR` | `CosineAnnealingLR` |
| **Gradient Clipping Norm** | $1.0$ | $1.0$ |
| **Loss Function** | `CrossEntropyLoss` | `CrossEntropyLoss` |

---

## 3. Actual Observed Training History

| Epoch | Stage | Train Loss | Train Acc (Top-1) | Val Loss | Val Acc (Top-1) | Val Acc (Top-3) | Head LR | Duration |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Stage 1: Warmup | 1.6299 | 55.61% | 0.7376 | 77.08% | 94.32% | $1.0 \times 10^{-3}$ | 1,759.0s |
| **2** | Stage 1: Warmup | 0.4221 | 86.55% | 0.4403 | 85.70% | 97.55% | $5.0 \times 10^{-4}$ | 2,627.5s |
| **3** | Stage 2: Fine-Tuning | 0.2317 | 92.53% | 0.3236 | 89.32% | 98.63% | $2.0 \times 10^{-4}$ | 2,669.6s |
| **4** | Stage 2: Fine-Tuning | **0.1373** | **96.00%** | **0.2591** | **92.26%** | **98.73%** | $1.0 \times 10^{-4}$ | 1,824.5s |

---

## 4. How to Reproduce Training

### Standard Training Command (Using YAML Config)
```bash
python training/train.py --config configs/training_config.yaml
```

### Custom CLI Override Example
```bash
python training/train.py \
  --warmup_epochs 2 \
  --finetune_epochs 2 \
  --batch_size 32 \
  --head_lr 0.001 \
  --backbone_lr 0.00002 \
  --unfreeze_blocks 2 \
  --checkpoint_dir checkpoints \
  --seed 42
```

### Outputs Generated
- **Best Model Checkpoint:** `checkpoints/best_vit_model.pth`
- **Latest Checkpoint:** `checkpoints/latest_vit_model.pth`
- **Training Telemetry CSV:** `results/training/training_history.csv`
- **Training Metadata JSON:** `checkpoints/training_metadata.json`
