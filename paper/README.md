# DeepScript Research Paper Manuscript & Publication Package

This directory contains the camera-ready research paper manuscript, LaTeX tables, high-resolution publication figures, and BibTeX references for the **DeepScript** research paper.

---

## Paper Details
- **Title:** *DeepScript: Robust Identification of Ancient Indian Inscription Families via Metric Vision Transformers, Aspect-Preserving Spatial Tiling, and Calibrated Open-Set Rejection*
- **Author:** Hanvith Reddy
- **Format:** IEEE Conference / Transactions Standard Two-Column Format (`IEEEtran.cls`)
- **Target Venues:**
  - *IEEE International Conference on Document Analysis and Recognition (ICDAR)*
  - *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)* / *IEEE Access*
  - *ACM Journal on Computing and Cultural Heritage (JOCCH)*
  - *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops*
  - *Digital Heritage / Digital Humanities (Springer/IEEE)*

---

## Directory Structure

```
paper/
├── main.tex                 # Complete IEEE LaTeX research paper manuscript
├── references.bib           # 23 verified BibTeX references (ICLR, CVPR, NeurIPS, epigraphy)
├── README.md                # Paper overview and compilation guide (this document)
├── figures/                 # 300 DPI publication figures (PDF vector & PNG)
│   ├── fig1_architecture.pdf / .png     # System architecture & pipeline diagram
│   ├── fig2_confusion_matrix.pdf / .png # Normalized confusion matrix (site-disjoint test set)
│   ├── fig3_roc_openset.pdf / .png      # Open-set rejection ROC & cosine distributions
│   ├── fig4_attention_rollout.pdf / .png# ViT-B/16 self-attention saliency overlays
│   └── fig5_metric_tsne.pdf / .png      # 256-D metric hypersphere t-SNE clustering
└── tables/                  # Modular LaTeX booktabs table snippets
    ├── table_dataset.tex    # Table I: Inscription corpus, chronology & splits
    ├── table_results.tex    # Table II: Per-class held-out test performance
    ├── table_ablation.tex   # Table III: Ablation study (fine-tuning depth & tiling)
    └── table_openset.tex    # Table IV: Open-set rejection benchmarks & AUROC
```

---

## How to Compile into PDF

### Option 1: Local Compilation (TeX Live / MiKTeX)
Run the standard LaTeX compilation sequence:
```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
This produces `main.pdf`.

### Option 2: Upload to Overleaf
1. Compress the `paper/` folder into a `.zip` archive.
2. Go to [Overleaf](https://www.overleaf.com), click **New Project** $\to$ **Upload Project**.
3. Select your `.zip` archive.
4. Set compiler to **pdfLaTeX** and click **Recompile**.

---

## Reproducing Figures & Tables
All figures and tables are dynamically generated from empirical experiment results in the repository:
```bash
# Re-generate all 5 publication figures in paper/figures/
python tools/generate_paper_figures.py

# Re-generate all 4 LaTeX tables in paper/tables/
python tools/generate_paper_tables.py
```
