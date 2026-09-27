# DeepScript Dataset Validation & Audit Report

**Audit Execution Timestamp:** `2026-09-27T15:38:53.681379`  
**Dataset Directory Audited:** `C:\Users\hanvi\OneDrive\Desktop\Deepscript\Deepscript\dataset\whole_inscriptions`  
**Audit Status:** `CLEAN / PASS`

---

## 1. Executive Summary

- **Total Whole-Inscription Images Found:** `0`
- **Valid Images:** `0`
- **Corrupted / Unreadable Images:** `0`
- **Empty Script Classes:** `6 / 6`
- **Exact Duplicate Files:** `0`
- **Cross-Class Duplicates:** `0`

> [!NOTE]
> **Dataset structure detected.** No whole-inscription images have been added yet.
> All 6 script family directories are properly structured and ready to receive verified epigraphic specimens.

---

## 2. Script Family Statistics

| Script Family | Folder Status | Total Images | Valid | Corrupted | Resolution Range (Min – Max) | Formats |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`brahmi`** | Empty (Ready) | 0 | 0 | 0 | N/A | None |
| **`gupta`** | Empty (Ready) | 0 | 0 | 0 | N/A | None |
| **`kharosthi`** | Empty (Ready) | 0 | 0 | 0 | N/A | None |
| **`grantha`** | Empty (Ready) | 0 | 0 | 0 | N/A | None |
| **`kadamba`** | Empty (Ready) | 0 | 0 | 0 | N/A | None |
| **`tamil_brahmi`** | Empty (Ready) | 0 | 0 | 0 | N/A | None |

---

## 3. Duplicate & Leakage Analysis

- **Cross-Class Duplicates:** None detected (0).  
- **Exact Duplicates:** None detected (0).

---

## 4. Metadata Validation (`dataset/metadata.csv`)

- **Metadata Header Status:** `VALID`
- **Total Metadata Records:** `0`
- **Valid Metadata Rows:** `0`
- **Images on Disk Missing Metadata:** `0`
- **Metadata Paths Missing on Disk:** `0`
- **Invalid Script Entries:** `0`
- **Invalid Verified Flags:** `0`
- **Duplicate Metadata Rows:** `0`

---

## 5. Recommended Actions & Next Steps

1. **Preserve Clean Data Acquisition:** Only add whole-inscription images with verified source citations.
2. **Maintain Provenance in `metadata.csv`:** Populate `source`, `source_url`, `site`, `inscription_type`, and set `verified = TRUE` upon verification.
3. **Enforce Monument-Level Splitting:** When partitioning data into train/val/test splits, ensure images from the same physical monument or inscription panel remain within the same split to avoid data leakage.
4. **No Guesswork Policy:** Any image lacking historical confirmation must remain quarantined with `verified = FALSE` and excluded from model training.
