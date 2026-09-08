# Experiment & Training Configurations

This directory contains configuration files specifying hyperparameter settings, dataset parameters, model architecture choices, and training schedules for DeepScript.

---

## Configuration Files

### `training_config.yaml`
Specifies the master parameters utilized in the full multi-epoch transfer learning pipeline:

- **Model Backbone:** `vit_b_16` (ImageNet-1K pretrained)
- **Input Resolution:** `224 x 224` (Bicubic interpolation, ImageNet normalized)
- **Classification Classes:** 62 script classes
- **Embedding Space:** 256 dimensions (L2 normalized unit hypersphere)
- **Classification Head:** `CosineSimilarityHead` ($s=16.0$)
- **Batch Size:** 32 (149 training batches, 32 validation batches per epoch)
- **Two-Stage Schedule:**
  - **Stage 1 (Head Warmup):** 2 epochs, Backbone frozen, Head LR $= 1.0 \times 10^{-3}$
  - **Stage 2 (Fine-Tuning):** 2 epochs, Last 2 blocks unfrozen, Backbone LR $= 2.0 \times 10^{-5}$, Head LR $= 2.0 \times 10^{-4}$
- **Reproducibility Seed:** 42

---

## Reproducing the Experiment

### Full Multi-Epoch Training Run:
```bash
python training/train.py --config configs/training_config.yaml
```

### Or using CLI Parameter Flags:
```bash
python training/train.py \
  --warmup_epochs 2 \
  --finetune_epochs 2 \
  --batch_size 32 \
  --head_lr 0.001 \
  --backbone_lr 0.00002 \
  --unfreeze_blocks 2 \
  --seed 42
```

