# Preprocessing Directory

## Directory Purpose

This directory contains modular Python components responsible for dataset loading, character image cleaning, standardization, transformation pipelines, and reproducible train/val/test partitioning for the DeepScript project.

---

## Implemented Modules

### 1. `preprocessing.dataset` (`AncientScriptDataset`)
- **Dynamic Class Discovery:** Scans class subdirectories, excludes empty directories, and assigns integer labels `0` to `C-1` in deterministic alphabetical order.
- **Color Mode Standardization:** Converts all incoming image modes (`RGBA`, `Grayscale`, `Palette`, etc.) to 3-channel RGB.
- **Corrupted File Detection:** Validates image integrity and handles unreadable images gracefully.
- **Metadata Inspection:** Supports sample inspection (`get_sample_info`), class mapping introspection (`get_class_to_idx`, `get_idx_to_class`), and distribution queries (`get_class_distribution`).

### 2. `preprocessing.transforms`
- **`get_train_transforms(image_size=(224, 224))`:**
  - Resize to `(224, 224)` via bicubic interpolation.
  - Historical-safe mild augmentations: subtle rotations ($\pm 8^\circ$), slight affine translation/scaling ($\pm 4\%$, $0.95\times - 1.05\times$), mild brightness/contrast jitter ($10\%$).
  - Normalization using standard ImageNet mean (`[0.485, 0.456, 0.406]`) and std (`[0.229, 0.224, 0.225]`).
- **`get_eval_transforms(image_size=(224, 224))`:**
  - Deterministic resize and ImageNet normalization for validation, testing, and inference.
- **`denormalize_tensor(tensor)`:**
  - Inverts ImageNet normalization for visualization and inspection.

### 3. `preprocessing.splits`
- **`create_stratified_splits(...)`:**
  - Partitions the dataset into Train ($70\%$), Validation ($15\%$), and Test ($15\%$) sets.
  - Stratified per-class distribution with a fixed random seed (`42`).
  - Attaches training augmentations to the train split and deterministic transforms to validation/test splits.
- **`verify_split_disjointness(...)`:**
  - Formally asserts zero image overlap (zero data leakage) between splits.
- **`create_dataloaders(...)`:**
  - Constructs standard PyTorch `DataLoader` instances with configurable batch sizes and worker counts.

---

## Verification

To run the complete automated test and verification suite:

```bash
python verify_preprocessing.py
```
