# DeepScript: Full Test Set Evaluation Results

This document records the **ACTUAL observed evaluation metrics** when evaluating the full trained ViT-B/16 checkpoint (`checkpoints/best_vit_model.pth`) on the held-out, completely isolated test partition.

---

## 1. Test Protocol & Isolation Verification

- **Checkpoint Evaluated:** `checkpoints/best_vit_model.pth` (Epoch 4, Validation Top-1: 92.26%)
- **Evaluation Script:** `evaluation/evaluate.py`
- **Dataset Partition:** Held-out test split ($N = 1,020$ samples across 62 script classes)
- **Data Leakage Check:** Zero overlap verified ($\text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$)
- **Batch Size:** 32 (32 mini-batches total)
- **Random Seed:** 42 (deterministic split reconstruction)

---

## 2. Summary Test Metrics

| Metric Name | Observed Full Training Value | Preliminary Smoke-Test Value | Uniform Random Baseline (62 Classes) |
| :--- | :---: | :---: | :---: |
| **Evaluated Test Samples** | **1,020 images** | 1,020 images | — |
| **Cross-Entropy Test Loss** | **0.2944** | 3.9030 | $\approx 4.1271$ ($\ln(62)$) |
| **Top-1 Test Accuracy** | **89.71%** | 14.41% | $\approx 1.61\%$ ($1/62$) |
| **Top-3 Test Accuracy** | **98.33%** | 28.53% | $\approx 4.84\%$ ($3/62$) |
| **Macro Precision** | **90.46%** | 8.58% | — |
| **Macro Recall** | **89.60%** | 13.72% | — |
| **Macro F1-Score** | **89.61%** | 7.81% | — |
| **Weighted F1-Score** | **89.69%** | 8.20% | — |

---

## 3. Analysis & Epigraphic Interpretation

1. **Massive Improvement over Baseline:**
   - On a complex 62-class script identification task with weathered inscription samples, purely uniform random guessing achieves $1.61\%$ Top-1 accuracy.
   - The trained ViT-B/16 model achieves **89.71% Top-1 Accuracy** ($55.7\times$ above random baseline) and **98.33% Top-3 Accuracy** ($20.3\times$ above random baseline).
2. **Metric Embedding Quality:**
   - The 256-dimensional unit hypersphere projection head ($\|\mathbf{z}\|_2 = 1.0$) separates fine-grained orthographic script characters, yielding high harmonic balance across classes ($\text{Macro F1} = 89.61\%$).
3. **Artifact Availability:**
   - Full per-class precision, recall, and F1 breakdown: `results/evaluation/classification_report.csv`
   - Normalized 62-class confusion matrix: `results/evaluation/confusion_matrix.png`
   - Loss, accuracy, and learning rate curves: `results/training/training_curves.png`
   - Machine-readable summary: `results/evaluation/metrics.json`
