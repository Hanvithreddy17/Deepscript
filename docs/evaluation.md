# DeepScript Test Evaluation & Benchmark Suite

This document outlines the test set evaluation protocol, metric computation formulas, data isolation guarantees, and observed held-out test performance.

---

## 1. Zero Data-Leakage Protocol

To ensure strict scientific integrity:
1. **Isolated Test Partition:** $N = 1,020$ samples across 62 script classes (15.02% of the corpus) were held out entirely during training, validation, and hyperparameter tuning.
2. **Disjoint Sets:** Zero overlapping files between splits:
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
3. **Deterministic Evaluation:** Fixed `random_seed = 42` for exact reproducibility.

---

## 2. Final Test Set Metrics Summary

| Evaluation Metric | Observed Full Model Value | Preliminary Smoke Test | Uniform Random Baseline (62 Classes) |
| :--- | :---: | :---: | :---: |
| **Evaluated Test Samples** | **1,020 images** | 1,020 images | — |
| **Cross-Entropy Test Loss** | **0.2944** | 3.9030 | $\approx 4.1271$ |
| **Top-1 Test Accuracy** | **89.71%** | 14.41% | $\approx 1.61\%$ ($1/62$) |
| **Top-3 Test Accuracy** | **98.33%** | 28.53% | $\approx 4.84\%$ ($3/62$) |
| **Macro Precision** | **90.46%** | 8.58% | — |
| **Macro Recall** | **89.60%** | 13.72% | — |
| **Weighted Precision** | **90.45%** | 8.24% | — |
| **Weighted Recall** | **89.71%** | 14.41% | — |
| **Weighted F1-Score** | **89.69%** | 8.20% | — |

---

## 3. Few-Shot Prototypical Learning Benchmark

Episodic few-shot recognition on the held-out test partition using the trained 256-D metric projector:

| Configuration | Mean Top-1 Accuracy | 95% Confidence Interval | Std Dev |
| :--- | :---: | :---: | :---: |
| **5-Way 1-Shot** | **96.87%** | $\pm 1.00\%$ | 5.12% |
| **5-Way 5-Shot** | **99.07%** | $\pm 0.55\%$ | 2.83% |
| **10-Way 1-Shot** | **93.40%** | $\pm 1.16\%$ | 5.93% |
| **10-Way 5-Shot** | **97.50%** | $\pm 0.61\%$ | 3.10% |
| **20-Way 1-Shot** | **89.18%** | $\pm 1.00\%$ | 5.10% |
| **20-Way 5-Shot** | **95.23%** | $\pm 0.52\%$ | 2.68% |

---

## 4. How to Run Test Evaluation

### Run Supervised Test Evaluation CLI
```bash
python evaluation/evaluate.py --checkpoint checkpoints/best_vit_model.pth --batch_size 32
```

### Run Few-Shot Prototypical Benchmark
```bash
python evaluation/few_shot_eval.py --checkpoint checkpoints/best_vit_model.pth --episodes 100 --seed 42
```

### Generated Artifacts
- **Metrics Summary JSON & CSV:** `results/evaluation/metrics.json`, `results/evaluation/metrics.csv`
- **Per-Class Metrics CSV:** `results/evaluation/classification_report.csv`
- **Confusion Matrix Heatmap:** `results/evaluation/confusion_matrix.png`
- **Few-Shot Results JSON & CSV:** `results/evaluation/few_shot_results.json`, `results/evaluation/few_shot_results.csv`
- **Training Curves:** `results/training/training_curves.png`

