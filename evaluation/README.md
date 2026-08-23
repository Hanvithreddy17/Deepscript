# Evaluation Directory

## Directory Purpose

This directory will store evaluation routines, performance testing scripts, metric computation modules, and result visualization tools.

> **Status Notice:** Evaluation scripts and metric reports are not yet implemented. No synthetic performance results or fake accuracy numbers are included.

---

## Planned Metrics & Evaluation Criteria

Evaluating the performance of DeepScript on ancient Indian script classification will rely on a comprehensive set of quantitative metrics:

1. **Overall Accuracy:** Percentage of total correctly predicted script samples.
2. **Precision:** Macro and weighted precision to measure prediction exactness per script class.
3. **Recall:** Sensitivity measurement to capture the model's ability to identify all instances of a given script.
4. **F1-Score:** Harmonic mean of precision and recall, serving as a primary metric for overall system reliability.
5. **Per-Class Performance:** Individual precision, recall, and F1 metrics for each script class.
   > **Note on Class Imbalance:** Because physical inscription datasets frequently suffer from uneven sample distribution across historical scripts, per-class performance metrics are especially critical to ensure rarer scripts are not obscured by high overall accuracy on dominant classes.
6. **Confusion Matrix Analysis:** Visual mapping of true vs. predicted script classifications to highlight specific pairs of visually similar scripts prone to confusion.
