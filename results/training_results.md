# DeepScript: Full Multi-Epoch Training & Fine-Tuning Results

This document records the **ACTUAL observed training results** from the full multi-epoch supervised transfer learning experiment executed on the DeepScript dataset.

---

## 1. Experiment Summary & Configuration

- **Model Architecture:** Vision Transformer (`ViT-B/16`) Backbone + 2-layer MLP Projector ($768 \to 512 \to 256$) + `CosineSimilarityHead` (256-dim unit hypersphere to 62 classes)
- **Dataset Size:** 6,792 total inscription character images across 62 script classes
- **Partitioning:** Stratified 70/15/15 split ($N_{\text{train}} = 4,751$, $N_{\text{val}} = 1,021$, $N_{\text{test}} = 1,020$)
- **Batch Size:** 32 (149 training batches per epoch, 32 validation batches per epoch)
- **Optimizer:** AdamW with `CosineAnnealingLR` scheduler
- **Loss Criterion:** `CrossEntropyLoss` with gradient clipping ($\text{max\_norm} = 1.0$)
- **Execution Device:** CPU (12 parallel PyTorch threads)
- **Total Duration:** 8,891.5 seconds (~2.47 hours)

---

## 2. Parameter Distribution by Stage

| Component | Stage 1 (Head Warmup) | Stage 2 (Fine-Tuning) |
| :--- | :--- | :--- |
| **ViT Backbone Layers 0–9** | Frozen (71.6M params) | Frozen (71.6M params) |
| **ViT Backbone Layers 10–11** | Frozen (14.2M params) | **Trainable (14.2M params)** |
| **Embedding Projector (MLP: 768 $\to$ 512 $\to$ 256)** | **Trainable (525.1K params)** | **Trainable (525.1K params)** |
| **Cosine Classification Head (256 $\to$ 62)** | **Trainable (279.7K params)** | **Trainable (279.7K params)** |
| **Total Model Parameters** | 86,603,521 | 86,603,521 |
| **Total Trainable Parameters** | **804,865** | **14,982,145** |
| **Total Frozen Parameters** | **85,798,656** | **71,621,376** |

---

## 3. Actual Observed Epoch-by-Epoch Metrics

| Epoch | Stage | Train Loss | Train Acc (Top-1) | Val Loss | Val Acc (Top-1) | Val Acc (Top-3) | Head LR | Backbone LR | Duration |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Stage 1: Head Warmup | 1.6299 | 55.61% | 0.7376 | 77.08% | 94.32% | $1.0 \times 10^{-3}$ | $0.0$ | 1,759.0s |
| **2** | Stage 1: Head Warmup | 0.4221 | 86.55% | 0.4403 | 85.70% | 97.55% | $5.0 \times 10^{-4}$ | $0.0$ | 2,627.5s |
| **3** | Stage 2: Fine-Tuning | 0.2317 | 92.53% | 0.3236 | 89.32% | 98.63% | $2.0 \times 10^{-4}$ | $2.0 \times 10^{-5}$ | 2,669.6s |
| **4** | Stage 2: Fine-Tuning | **0.1373** | **96.00%** | **0.2591** | **92.26%** | **98.73%** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-5}$ | 1,824.5s |

---

## 4. Key Findings & Convergence Insights

1. **Rapid Head Warmup Convergence:** In Stage 1, training only the projector and head (804.8K parameters) while freezing the ViT backbone raised validation Top-1 accuracy from baseline to **85.70%** (Top-3: **97.55%**) by Epoch 2.
2. **Domain Adaptation via Fine-Tuning:** Unfreezing the top 2 transformer encoder blocks in Stage 2 allowed the self-attention heads to specialize on historical inscription stroke topologies, pushing validation Top-1 accuracy to **92.26%** and Top-3 accuracy to **98.73%**.
3. **Overfitting Resistance:** Validation loss declined smoothly alongside training loss ($0.7376 \to 0.4403 \to 0.3236 \to 0.2591$), indicating that the combination of ImageNet initialization, subtle augmentations, dropout (0.1), and L2 unit-norm embedding normalization effectively prevented overfitting.
4. **Checkpoint Preservation:** The optimal model checkpoint was saved to `checkpoints/best_vit_model.pth` corresponding to Epoch 4 validation performance.
