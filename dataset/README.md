# Dataset Directory

## Directory Purpose

This directory will eventually serve as the central repository for dataset metadata, schema specifications, data partitioning guidelines, and source references.

> **Status Notice:** No raw or processed datasets are stored in this repository currently. Actual dataset files, binary images, and large archives should be managed locally or stored in remote data storage and excluded from Git version control (as configured in `.gitignore`).

---

## Target Scope

The initial phase targets approximately **3 to 5 distinct ancient Indian script classes**, subject to the availability of verified, high-quality labeled epigraphic data.

---

## Planned Dataset Considerations

1. **Ancient Indian Script Classes:** Selection of target script categories based on historical significance and availability of published inscriptions.
2. **Image Quality & Resolution:** Handling images of varying visual clarity, illumination, perspective distortion, and stone/surface erosion.
3. **Class Imbalance:** Managing variations in sample availability across different script types through balanced sampling or class-weighted loss strategies.
4. **Limited Labeled Examples:** Addressing scarcity of labeled epigraphic artifacts using few-shot classification techniques.
5. **Source & Provenance:** Ensuring authentic, documented origins for all inscription imagery (e.g., archaeological surveys, museums, academic archives).
6. **Licensing & Usage:** Verifying copyright and fair-use licensing permissions for academic and research utilization.
7. **Train / Validation / Test Partitioning:** Structuring reproducible, disjoint data splits to ensure reliable evaluation.
8. **Avoiding Data Leakage:** Strict separation of images originating from the same physical monument or inscription source across train/val/test splits.
