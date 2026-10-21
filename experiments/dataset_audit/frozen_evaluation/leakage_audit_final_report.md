# DeepScript: Final Pre-Evaluation Leakage Audit & Zero-Leakage Test Manifest

**Target Checkpoint:** [`checkpoints/six_class_vit/best_vit_model.pth`](file:///C:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/checkpoints/six_class_vit/best_vit_model.pth)  
**Expected SHA-256:** `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c`  
**Verified Pre-Audit SHA-256:** `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c` (**MATCH CONFIRMED: 100% UNTOUCHED**)  
**Audit Date:** October 2026  
**Status:** Pre-Evaluation Audit Completed — **Evaluation Held for User Approval**  

---

## 1. Resolution of Pre-Evaluation Leakage Issues

### A. Proof of Actual Training Exposure & Kongarpuliyankulam Exclusion
1. **The Discrepancy:** In `dataset/metadata.csv` (updated on Oct 18), `TBSI_029_original.jpg` (Kongarpuliyankulam cavern) was labeled as `split = test`.
2. **Saved Training Split Records:**
   - Saved metadata in [`checkpoints/six_class_vit/training_metadata.json`](file:///C:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/checkpoints/six_class_vit/training_metadata.json) records that `best_vit_model.pth` was trained on October 16, 2026, on 42 train samples, 11 validation samples, and 8 test samples.
   - Saved evaluation logs in [`results/six_class_vit/evaluation/classification_report.csv`](file:///C:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/results/six_class_vit/evaluation/classification_report.csv) record that the held-out test split contained **only 2 Tamil-Brahmi samples** (`support = 2`).
   - Re-running the deterministic stratified partition algorithm (`create_stratified_splits(ds_61, random_seed=42)`) identical to `training/train.py` proved that the 2 test samples on October 16 were `TBSI_002_original.jpg` (Alagarmalai) and `TBSI_035_original.jpg` (Mamandur).
   - In that same run, **`TBSI_029_original.jpg` was assigned to `train_dataset`** alongside 7 other Tamil-Brahmi specimens (`TBSI_068`, `TBSI_001`, `TBSI_022`, `TBSI_020`, `TBSI_026`, `TBSI_036`, `TBSI_025`).
3. **Audit Verdict:** **`TBSI_029_original.jpg` (Kongarpuliyankulam) is contaminated by prior training exposure.** It is strictly excluded from the zero-leakage evaluation benchmark.

---

### B. Hash, Similarity, Inscription Identity & Angle Audit against Train/Val Images
An algorithmic similarity comparison was conducted against all **53 training and validation images** of `checkpoints/six_class_vit/best_vit_model.pth` across all 6 script classes:
* **Bitwise Collision Check (SHA-256):** Zero candidate test image matches any of the 53 train/val image hashes.
* **Perceptual Difference Hash Check (dhash, 64-bit):**
  - In-Repository Sittanavasal captures: Minimum Hamming distance to any train/val image is **20 to 23 bits** (zero near-duplicates, zero shared rock faces).
  - External TBSI archive candidates: Minimum Hamming distance is **13 to 24 bits** (zero near-duplicates).
* **Archaeological Site Separation:** All candidate sites (*Sittanavasal, Vikkiramangalam, Varichiyur, Tondur, Marukaltalai, Mudalaikulam*) have **0 images in the training set and 0 images in the validation set**.

---

### C. Site Count Correction & Monument Identity Grouping
* **Total Independent Archaeological Sites:** Exactly **6 unique sites** across the entire candidate pool:
  1. *Sittanavasal (Sittannavasal)* — Pudukkottai District
  2. *Vikkiramangalam* — Madurai District
  3. *Varichiyur* — Madurai District
  4. *Tondur* — Villupuram District
  5. *Marukaltalai* — Tirunelveli District
  6. *Mudalaikulam* — Madurai District
* **Monument Overlap Resolution:**
  - The 4 in-repository Sittanavasal photographs (`Sittanavasal_Rock_Bed_Inscription.jpg`, `Stone_bed_with_inscription_in_sittanavasal.JPG`, `Chithannavasal_cave_inscription_1.JPG`, `Chithannavasal_cave_inscription_2.JPG`) and the 1 archive photo (`TBSI_075_original.jpg`) all capture the **exact same physical monument: Bed #1 at Ezhadippattam cavern**.
  - In reporting, they are explicitly grouped together as **1 physical monument / 1 inscription at 1 site**, and will never be counted as multiple independent generalization samples.
* **Distinct Physical Inscriptions:** Across the 6 independent sites, there are exactly **13 distinct physical inscriptions** represented by **17 photographic captures**.

---

## 2. Benchmark Characterization & Evaluation Protocol

1. **Benchmark Scope:**  
   This evaluation is explicitly characterized as a **Tamil-Brahmi Sensitivity / Recall Benchmark on Authenticated Unseen Stone Monuments**.
2. **No False Multi-Class Claims:**  
   Because all candidate specimens belong to the `tamil_brahmi` ground-truth class, the benchmark reports:
   - **Capture-Level Sensitivity:** Percentage of stone captures correctly classified as `tamil_brahmi`.
   - **Monument-Grouped Sensitivity:** Percentage of physical inscriptions correctly recognized across all available angles.
   - **Confusion Matrix Breakdown:** Distribution of false-negative misclassifications into the 5 competing script classes (`brahmi`, `grantha`, `gupta`, `kadamba`, `kharosthi`).
   - Overall 6-class macro accuracy will **not** be claimed from a single-class test set.
3. **Production Engine Pipeline:**  
   Evaluation is executed via `backend.vit_inference.vit_inference_engine.predict`, utilizing the exact production preprocessing:
   - Aspect-preserving tile extraction (`extract_aspect_preserving_tiles`, $AR > 1.8$ or $AR < 0.55$).
   - Standard ImageNet channel normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
   - ViT-B/16 feature extraction (768-d) $\rightarrow$ MLP projection (256-d bottleneck, $\|\mathbf{z}\|_2 = 1.0$) $\rightarrow$ learned cosine head ($s = 15.9930$).
   - Logit-averaging across tiles to preserve stroke morphology.

---

## 3. The Corrected Evaluation Manifest

The manifest is formalized in [`experiments/dataset_audit/frozen_evaluation/corrected_evaluation_manifest.json`](file:///C:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/experiments/dataset_audit/frozen_evaluation/corrected_evaluation_manifest.json):

### Set A: In-Repository Sittanavasal Stone Captures (4 Captures, 1 Physical Inscription, 1 Site)
*All images reside directly in `dataset/whole_inscriptions/tamil_brahmi/`:*

| Specimen ID | File Path | Site & District | Monument & Epigraphic Content | Dimensions | Epigraphic Reference | Ground Truth | Training Exposure |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **SET_A_01** | `Sittanavasal_Rock_Bed_Inscription.jpg` | Sittanavasal, Pudukkottai | Ezhadippattam Cavern, Bed #1 (Frontal view) | 2304×3072 | I. Mahadevan (2003) No. 8; TNDA p. 32 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_A_02** | `Stone_bed_with_inscription_in_sittanavasal.JPG` | Sittanavasal, Pudukkottai | Ezhadippattam Cavern, Bed #1 (Headrest view) | 4000×3000 | I. Mahadevan (2003) No. 8; TNDA p. 32 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_A_03** | `Chithannavasal_cave_inscription_1.JPG` | Sittanavasal, Pudukkottai | Ezhadippattam Cavern, Bed #1 (Angle 1) | 4000×3000 | I. Mahadevan (2003) No. 8; TNDA p. 32 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_A_04** | `Chithannavasal_cave_inscription_2.JPG` | Sittanavasal, Pudukkottai | Ezhadippattam Cavern, Bed #1 (Angle 2) | 4000×3000 | I. Mahadevan (2003) No. 8; TNDA p. 32 | `tamil_brahmi` | **0% (Unseen)** |

---

### Set B: TBSI Archive Unseen Stone Benchmark (13 Captures, 13 Inscriptions, 6 Sites)
*Sourced from `C:\Users\hanvi\Downloads\Tamil Brahmi Stone Inscription.zip` (Original Images/):*

| Specimen ID | Archive Filename | Site & District | Monument Grouping | Length & Lines | Epigraphic Reference | Ground Truth | Training Exposure |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **SET_B_01** | `TBSI_075_original.jpg` | Sittannavasal, Pudukkottai | **Ezhadippattam Bed #1 (Same as Set A)** | 215 cm, 1 line | I. Mahadevan No. 8 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_02** | `TBSI_088_original.jpg` | Vikkiramangalam, Madurai | Vikkiramangalam Cavern, Inscription 1 | 67 cm, 1 line | I. Mahadevan Nos. 35-40 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_03** | `TBSI_089_original.jpg` | Vikkiramangalam, Madurai | Vikkiramangalam Cavern, Inscription 2 | 173 cm, 1 line | I. Mahadevan Nos. 35-40 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_04** | `TBSI_090_original.jpg` | Vikkiramangalam, Madurai | Vikkiramangalam Cavern, Inscription 3 | 102 cm, 1 line | I. Mahadevan Nos. 35-40 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_05** | `TBSI_091_original.jpg` | Vikkiramangalam, Madurai | Vikkiramangalam Cavern, Inscription 4 | 36 cm, 1 line | I. Mahadevan Nos. 35-40 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_06** | `TBSI_092_original.jpg` | Vikkiramangalam, Madurai | Vikkiramangalam Cavern, Inscription 5 | 32 cm, 1 line | I. Mahadevan Nos. 35-40 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_07** | `TBSI_093_original.jpg` | Vikkiramangalam, Madurai | Vikkiramangalam Cavern, Inscription 6 | 24 cm, 1 line | I. Mahadevan Nos. 35-40 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_08** | `TBSI_085_original.jpg` | Varichiyur, Madurai | Varichiyur Cavern, Inscription 1 | 54 cm, 1 line | I. Mahadevan Nos. 28-30 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_09** | `TBSI_086_original.jpg` | Varichiyur, Madurai | Varichiyur Cavern, Inscription 2 | 158 cm, 1 line | I. Mahadevan Nos. 28-30 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_10** | `TBSI_087_original.jpg` | Varichiyur, Madurai | Varichiyur Cavern, Inscription 3 | 162 cm, 1 line | I. Mahadevan Nos. 28-30 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_11** | `TBSI_084_original.jpg` | Tondur, Villupuram | Tondur Rock Face Epigraph | 111, 102 cm, 2 lines | I. Mahadevan No. 78 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_12** | `TBSI_044_original.jpg` | Marukaltalai, Tirunelveli | Marukaltalai Cavern Drip Ledge | 426 cm, 1 line | I. Mahadevan No. 1 | `tamil_brahmi` | **0% (Unseen)** |
| **SET_B_13** | `TBSI_055_original.jpg` | Mudalaikulam, Madurai | Mudalaikulam Cavern Epigraph | 164 cm, 1 line | I. Mahadevan No. 34 | `tamil_brahmi` | **0% (Unseen)** |

---

## 4. Remaining Epigraphic & Technical Limitations

1. **Aspect Ratio Squashing on Single-Line Inscriptions:**  
   Single-line drip ledge inscriptions (e.g. `TBSI_044` at Marukaltalai: 2501×650 px, $AR = 3.85:1$; `TBSI_086`: 2498×765 px, $AR = 3.27:1$) trigger the production tiling engine ($AR > 1.8$). However, because tiles step across width, characters sliced at tile boundaries may lose continuous stroke context.
2. **Weathering & Natural Erosion:**  
   Unlike chalk-highlighted survey photos, authentic natural stone surfaces exhibit low contrast between chiseled strokes and background granite. This reflects genuine real-world field conditions.
3. **Sample Size:**  
   The candidate set comprises 17 total photographic captures across 13 distinct physical inscriptions and 6 independent sites. While this represents the largest defensible zero-leakage stone test set possible from current archives, metrics should be interpreted as rigorous epigraphic validation rather than large-scale asymptotic statistics.
