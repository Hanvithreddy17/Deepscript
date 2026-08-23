# Notebooks Directory

## Directory Purpose

This directory is designated for interactive, exploratory Jupyter notebooks used for research, visual inspection, algorithm prototyping, and performance diagnostics.

> **Status Notice:** No exploratory notebooks are currently present in this initial repository setup.

---

## Planned Notebook Usage

1. **Dataset Exploration:** Analyzing image counts, script distributions, and visual sample qualities.
2. **Image Visualization:** Inspecting raw vs. preprocessed inscription images and testing filter parameters interactively.
3. **Preprocessing Experiments:** Prototyping contrast enhancement, resizing, and augmentation transformations before modularizing code.
4. **Model Experiments:** Testing backbone feature extraction outputs, vector dimension choices, and few-shot distance functions.
5. **Evaluation Analysis:** Generating confusion matrices, error case visualizer plots, and performance breakdowns.

---

## Best Practices & Guidelines

- **Clear Naming Conventions:** Prefix notebook filenames systematically (e.g., `01_dataset_exploration.ipynb`, `02_preprocessing_experiments.ipynb`).
- **No Large File Storage:** Notebook outputs containing embedded high-resolution dataset images or large weights must be cleared before committing.
- **Git Cleanliness:** Raw datasets and model binary files must **never** be stored within this directory.
