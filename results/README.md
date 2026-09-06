# DeepScript: Results & Experiment Artifacts Directory

This directory contains the training metrics, held-out test evaluation outputs, confusion matrices, and experiment artifacts for the DeepScript project.

---

## Directory Structure

```
results/
├── training/
│   ├── training_history.csv       # Epoch-by-epoch loss, accuracy, and learning rate telemetry
│   ├── training_metadata.json      # Experiment hyperparameters, sample counts, and duration
│   └── training_curves.png         # Multi-panel visualization of loss, accuracy, and LR schedules
├── evaluation/
│   ├── metrics.json                # Summary test metrics (Top-1, Top-3, Loss, Macro/Weighted F1)
│   ├── classification_report.csv   # Per-class precision, recall, f1-score, and support counts
│   └── confusion_matrix.png        # High-resolution 62-class normalized confusion matrix heatmap
├── training_results.md             # Detailed breakdown of training stages and convergence
├── test_results.md                 # Detailed breakdown of held-out test set evaluation
└── README.md                       # This directory index
```

---

## Key Performance Summary (Full Model Run)

| Metric | Full Training Result | Smoke-Test Baseline | Random Baseline |
| :--- | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | **89.71%** | 14.41% | 1.61% |
| **Top-3 Test Accuracy** | **98.33%** | 28.53% | 4.84% |
| **Cross-Entropy Test Loss** | **0.2944** | 3.9030 | 4.1271 |
| **Macro F1-Score** | **89.61%** | 7.81% | — |
| **Weighted F1-Score** | **89.69%** | 8.20% | — |
