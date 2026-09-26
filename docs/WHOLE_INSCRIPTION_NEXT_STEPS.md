# Whole-Inscription Pipeline: Next Steps & Ground-Truth Roadmap

## 1. Current Usability Summary

| Category | Count | Usability Status |
| :--- | :---: | :--- |
| **Currently Usable Whole-Inscription Images** | **0** | **0%** (Cannot train supervised models without verified ground truth) |
| **Unlabeled / Unverified Whole-Inscription Images** | **237** | **Needs Verification** (Must not guess or infer from appearance) |
| **Isolated Brahmi Character Crops** | 23,164 | Usable for character-level classification only (`dataset/dataset/`) |
| **Unrelated Datasets in Root** | 3,446 | Non-epigraphic Thai line OCR crops (`a001`–`a140`); excluded from pipeline |

---

## 2. Information & Sources Required for Ground Truth

To convert the 237 unlabelled whole-inscription images into training/evaluation sets:

1. **Source Catalog / Thesis Index Recovery:**
   - Retrieve `shilalekhfeature.xlsx` or the master index corresponding to the MATLAB scripts ([`noisefinal.m`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/noisefinal.m), [`noisem.m`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/noisem.m)).
2. **Epigraphic Reference Matching:**
   - Cross-reference field images (`I_1`–`I_46`, `T01`–`T27`, `N01`–`N09`, `J01`–`J03`) against cataloged archaeological publications:
     - Archaeological Survey of India (ASI) *Epigraphia Indica* volumes
     - *Corpus Inscriptionum Indicarum* (CII)
     - South Indian Inscriptions (SII) corpus
     - Curated museum / university epigraph registries
3. **Structured Entry into [`dataset/whole_inscription_metadata.csv`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/whole_inscription_metadata.csv):**
   - Populate `script`, `location`, `period`, `source`, `evidence`, and set `status = VERIFIED` for every verified specimen.

---

## 3. Mandatory Steps BEFORE Training ViT + Few-Shot Learning

Do **NOT** initiate ViT training or fine-tuning until the following workflow is executed:

```
[Phase 1: Catalog Verification]
  ├── Verify ≥5–10 whole-inscription images per target script class (Brahmi, Gupta, Kharosthi, Grantha, Kadamba, Tamil-Brahmi)
  └── Update dataset/whole_inscription_metadata.csv with verified citations

[Phase 2: Dataset Cleansing & Partitioning]
  ├── Filter out exact duplicate images (OriginalImage38/55, 39/58, 40/64, 50/84)
  ├── Remove corrupted files (N03.JPG) and synthetic filter duplicates (HFI/VFI)
  └── Isolate unrelated datasets (Thai line crops) into a separate directory

[Phase 3: Stratified Epigraphic Data Splitting]
  ├── Ensure strict split by monument/site to prevent data leakage
  └── Structure N-way K-shot support and query episode pools

[Phase 4: ViT-B/16 Metric Learning Training]
  ├── Fine-tune ViT backbone with episodic prototypical / contrastive loss
  └── Benchmark on unseen test inscriptions
```
