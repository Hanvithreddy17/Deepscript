# DeepScript Experiment Log & Version Changelog

This log tracks model experiments, configuration changes, and performance evolutions across the DeepScript project lifecycle.

---

## Experiment 3: Production FastAPI Inference Backend & Live Integration (Current Milestone)
- **Date:** September 8, 2026
- **Status:** **Completed & Verified**
- **Architecture:** `FastAPI` + `Uvicorn` serving `ViT-B/16` (`checkpoints/best_vit_model.pth`)
- **Key Features:**
  - Automated engine lifespan management with warm-up dummy pass.
  - Image validation and on-the-fly bicubic resizing $(224 \times 224)$ + ImageNet normalization.
  - RESTful endpoints: `POST /predict`, `GET /health`, `GET /classes`.
  - Sub-second inference latency on multi-threaded CPU ($\approx 40-70\text{ms}$ forward pass).
  - 62-class phonetic and epigraphic dictionary integration in frontend.
  - Live proxy and visual connection indicator in React UI.
- **Verification:** Automated backend integration test suite (`backend/test_backend.py`) fully passed.

---

## Experiment 2: Full Multi-Epoch Supervised Transfer Learning
- **Date:** September 6, 2026
- **Status:** **Completed & Verified**
- **Backbone:** Pretrained `ViT-B/16` (ImageNet-1K)
- **Embedding Projector:** MLP ($768 \to 512 \to 256$) with L2 unit-norm normalization ($\|\mathbf{z}\|_2 = 1.0$)
- **Classification Head:** `CosineSimilarityHead` with learnable scale $s$
- **Training Strategy:**
  - Stage 1 Warmup: 2 epochs (ViT Backbone frozen, Head LR $= 1.0 \times 10^{-3}$)
  - Stage 2 Fine-Tuning: 2 epochs (Unfreezing last 2 ViT blocks, Backbone LR $= 2.0 \times 10^{-5}$, Head LR $= 2.0 \times 10^{-4}$)
- **Dataset Partition:** Stratified 70/15/15 ($N_{\text{train}} = 4,751$, $N_{\text{val}} = 1,021$, $N_{\text{test}} = 1,020$) across 62 classes
- **Observed Metrics:**
  - **Best Validation Top-1 Accuracy:** **92.26%** (Top-3: **98.73%**, Val Loss: **0.2591**)
  - **Held-Out Test Top-1 Accuracy:** **89.71%** (Top-3: **98.33%**, Test Loss: **0.2944**)
  - **Macro F1-Score:** **89.61%**
  - **Weighted F1-Score:** **89.69%**
- **Artifacts Saved:**
  - Checkpoint: `checkpoints/best_vit_model.pth`
  - Training History: `results/training/training_history.csv`
  - Metrics JSON: `results/evaluation/metrics.json`
  - Classification Report: `results/evaluation/classification_report.csv`
  - Confusion Matrix: `results/evaluation/confusion_matrix.png`
  - Training Curves: `results/training/training_curves.png`

---

## Experiment 1: Preliminary Transfer Learning Smoke Test (Baseline Milestone)
- **Date:** September 6, 2026
- **Status:** Completed (Verification Run)
- **Scope:** Rapid verification run with capped batch counts (3 training batches, 2 val batches) to verify pipeline execution, gradient backpropagation, checkpoint serialization, and evaluation metrics.
- **Observed Metrics:**
  - **Validation Top-1 Accuracy:** 10.42% (Top-3: 62.50%)
  - **Held-Out Test Top-1 Accuracy:** 14.41% (Top-3: 28.53%)
  - **Macro F1-Score:** 7.81%
- **Outcome:** Successfully proved pipeline correctness prior to full uncapped multi-epoch execution.

---

## Next Milestone: Few-Shot Episodic Training & Metric Prototyping
- **Scope:** Implement $N$-way $K$-shot episodic sampling using `PrototypicalHead` on top of the trained `ViT-B/16` metric representations for fine-grained low-sample character generalization.
