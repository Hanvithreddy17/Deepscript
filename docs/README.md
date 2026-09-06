# Project Documentation Directory

This directory contains the formal architectural specifications, training guidelines, benchmark evaluations, and experiment logs for the DeepScript project.

---

## Documentation Index

1. [System Architecture Guide](architecture.md)
   - Deep learning vision architecture overview
   - Vision Transformer (ViT-B/16) backbone and patch extraction
   - Metric Embedding Projector ($768 \to 512 \to 256$) with unit L2 hypersphere normalization
   - Classification heads: Cosine Similarity, Linear, and Prototypical Centroid heads
   - Parameter breakdown and layer freezing strategy

2. [Training & Transfer Learning Guide](training.md)
   - Two-stage transfer learning procedure (Stage 1 Warmup + Stage 2 Fine-Tuning)
   - Hyperparameter configurations and differential learning rate optimization
   - Observed convergence history and loss progression
   - Step-by-step training reproduction guide

3. [Test Evaluation & Benchmark Guide](evaluation.md)
   - Zero data leakage validation protocol
   - Held-out test set evaluation benchmarks on 1,020 isolated samples
   - Metric formulas (Top-1, Top-3, Macro/Weighted Precision, Recall, F1)
   - Classification report and confusion matrix interpretation

4. [Experiment Log & Version History](experiment_log.md)
   - Chronological log of model iterations and experiment runs
   - Detailed comparison between preliminary smoke tests and full multi-epoch training
   - Next milestone roadmap (Few-Shot Prototypical learning)
