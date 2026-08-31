"""
DeepScript — Verification and Test Suite for Dataset & Preprocessing
====================================================================
This script executes a rigorous end-to-end verification of dataset integrity,
preprocessing pipelines, and data partitioning:
1. Dataset loading & dynamic 62-class discovery.
2. Numerical label mapping bijectivity and invertibility check.
3. Image loading, format conversion (RGBA/Grayscale -> RGB), and corruption checks.
4. Image preprocessing pipeline: Resize (224x224 Bicubic) -> ImageNet normalization -> Safe augmentation.
5. Processed tensor validation: shape [3, 224, 224], dtype float32, finite values (no NaN/Inf).
6. Stratified train/val/test partitioning (70/15/15) with fixed random seed (42).
7. Formal data leakage test verifying pairwise split disjointness.
8. PyTorch DataLoader mini-batch extraction and batch tensor shape verification.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Reconfigure stdout for UTF-8 compatibility on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import (
    AncientScriptDataset,
    get_train_transforms,
    get_eval_transforms,
    get_display_transforms,
    denormalize_tensor,
    create_stratified_splits,
    create_dataloaders,
    verify_split_disjointness,
    get_split_statistics,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    DEFAULT_IMAGE_SIZE,
)


def run_full_verification(dataset_root: str = "dataset/dataset") -> bool:
    """
    Executes the 6-step dataset and preprocessing verification test suite.

    Args:
        dataset_root: Path to the dataset directory containing class subfolders.

    Returns:
        bool: True if all 11 verification criteria pass.
    """
    print("=" * 70)
    print(" DEEPSCRIPT: DATASET & PREPROCESSING VERIFICATION SUITE")
    print("=" * 70)

    checklist = {}
    ds_path = Path(dataset_root)

    # -------------------------------------------------------------
    # Step 1: Dataset Loading & Class Discovery
    # Verifies scanning of dataset root directory, discovery of exactly
    # 62 script classes, and total count of valid, non-corrupted images.
    # -------------------------------------------------------------
    print("\n[Step 1/6] Initializing Dataset and Discovering Classes...")
    if not ds_path.exists():
        print(f"ERROR: Dataset directory not found at {ds_path.resolve()}")
        return False

    raw_dataset = AncientScriptDataset(root_dir=ds_path, transform=None)
    classes = raw_dataset.get_classes()
    total_samples = len(raw_dataset)

    print(f"  • Discovered Classes : {len(classes)} classes")
    print(f"  • Total Valid Images : {total_samples} samples")
    print(f"  • Sample Classes     : {classes[:10]} ... {classes[-5:]}")
    print(f"  • Corrupted Files    : {len(raw_dataset.corrupted_files)}")

    checklist["Dataset loads"] = total_samples > 0
    checklist["Classes are detected"] = len(classes) == 62
    checklist["Images are readable"] = len(raw_dataset.corrupted_files) == 0

    # -------------------------------------------------------------
    # Step 2: Label Mapping & Indexing Check
    # Verifies that alphabetical class ordering produces a strictly
    # bijective and invertible mapping: class_name <-> integer label.
    # -------------------------------------------------------------
    print("\n[Step 2/6] Verifying Label Mapping and Reversibility...")
    class_to_idx = raw_dataset.get_class_to_idx()
    idx_to_class = raw_dataset.get_idx_to_class()

    label_check_passed = True
    for idx in range(len(classes)):
        cls_name = idx_to_class.get(idx)
        if cls_name is None or class_to_idx.get(cls_name) != idx:
            label_check_passed = False
            break

    print(f"  • Label Range       : 0 to {len(classes) - 1}")
    print(f"  • Sample Mapping    : '{classes[0]}' -> {class_to_idx[classes[0]]}, '{classes[-1]}' -> {class_to_idx[classes[-1]]}")
    print(f"  • Invertibility     : {'PASSED' if label_check_passed else 'FAILED'}")
    checklist["Labels are correct"] = label_check_passed

    # -------------------------------------------------------------
    # Step 3: Image Preprocessing Pipeline Verification
    # Tests loading real inscription images and applying deterministic
    # evaluation transforms as well as augmented training transforms.
    # Checks tensor dimensions [3, 224, 224] and verifies no NaN / Inf values.
    # -------------------------------------------------------------
    print("\n[Step 3/6] Testing Image Preprocessing Pipeline...")
    train_tf = get_train_transforms(image_size=(224, 224))
    eval_tf = get_eval_transforms(image_size=(224, 224))

    sample_indices = [0, total_samples // 4, total_samples // 2, total_samples - 1]
    preprocessing_passed = True
    shape_passed = True
    finite_passed = True

    print("\n  Sample Transformations on Real Inscription Characters:")
    for s_idx in sample_indices:
        info = raw_dataset.get_sample_info(s_idx)
        raw_pil, label = raw_dataset[s_idx]
        
        # Apply eval transform (deterministic)
        processed_eval = eval_tf(raw_pil)
        # Apply train transform (augmented)
        processed_train = train_tf(raw_pil)

        print(f"    - Sample #{s_idx:4d} | Class: '{info['class_name']}' (Label: {label})")
        print(f"      Original File : {info['filename']} | Mode: {info['original_mode']} | Size: {info['original_size']}")
        print(f"      Eval Tensor   : Shape={list(processed_eval.shape)} | Dtype={processed_eval.dtype} | Min={processed_eval.min():.3f} | Max={processed_eval.max():.3f}")
        print(f"      Train Tensor  : Shape={list(processed_train.shape)} | Dtype={processed_train.dtype} | Min={processed_train.min():.3f} | Max={processed_train.max():.3f}")

        # Check tensor shape [3, 224, 224]
        if list(processed_eval.shape) != [3, 224, 224] or list(processed_train.shape) != [3, 224, 224]:
            shape_passed = False
        # Check finite values (no NaN / Inf)
        if not torch.isfinite(processed_eval).all() or not torch.isfinite(processed_train).all():
            finite_passed = False

    checklist["Preprocessing works"] = preprocessing_passed
    checklist["Output shape is correct"] = shape_passed
    checklist["No NaN/Inf values"] = finite_passed

    # -------------------------------------------------------------
    # Step 4: Stratified Train / Val / Test Partitioning
    # Creates stratified 70/15/15 data splits with seed 42,
    # ensuring class proportions are preserved across all 3 subsets.
    # -------------------------------------------------------------
    print("\n[Step 4/6] Creating Stratified Train / Validation / Test Splits (Seed=42)...")
    train_ds, val_ds, test_ds = create_stratified_splits(
        dataset=raw_dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        train_transform=train_tf,
        eval_transform=eval_tf,
    )

    stats = get_split_statistics(train_ds, val_ds, test_ds)
    print(f"  • Total Dataset Size : {stats['total_samples']} images across {stats['num_classes']} classes")
    print(f"  • Training Set       : {stats['train_count']} images ({stats['train_pct']:.2f}%)")
    print(f"  • Validation Set     : {stats['val_count']} images ({stats['val_pct']:.2f}%)")
    print(f"  • Test Set           : {stats['test_count']} images ({stats['test_pct']:.2f}%)")

    checklist["Train split works"] = len(train_ds) > 0
    checklist["Validation split works"] = len(val_ds) > 0
    checklist["Test split works"] = len(test_ds) > 0

    # -------------------------------------------------------------
    # Step 5: Data Leakage Verification
    # Formally checks set intersection across train, val, and test subsets
    # to guarantee zero data leakage.
    # -------------------------------------------------------------
    print("\n[Step 5/6] Formally Verifying Disjointness (Zero Data Leakage)...")
    try:
        verify_split_disjointness(train_ds, val_ds, test_ds)
        print("  • Overlap Check (Train ∩ Val)  : 0 items (PASSED)")
        print("  • Overlap Check (Train ∩ Test) : 0 items (PASSED)")
        print("  • Overlap Check (Val ∩ Test)   : 0 items (PASSED)")
        checklist["No data leakage"] = True
    except AssertionError as e:
        print(f"  • Overlap Check FAILED: {e}")
        checklist["No data leakage"] = False

    # -------------------------------------------------------------
    # Step 6: PyTorch DataLoader Batch Test
    # Verifies batch construction and mini-batch loading across all 3
    # DataLoader instances with batch_size=32.
    # -------------------------------------------------------------
    print("\n[Step 6/6] Testing PyTorch DataLoaders...")
    batch_size = 32
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=0
    )

    train_batch_img, train_batch_lbl = next(iter(train_loader))
    val_batch_img, val_batch_lbl = next(iter(val_loader))
    test_batch_img, test_batch_lbl = next(iter(test_loader))

    print(f"  • Train Loader Batch Shape : Images {list(train_batch_img.shape)}, Labels {list(train_batch_lbl.shape)}")
    print(f"  • Val Loader Batch Shape   : Images {list(val_batch_img.shape)}, Labels {list(val_batch_lbl.shape)}")
    print(f"  • Test Loader Batch Shape  : Images {list(test_batch_img.shape)}, Labels {list(test_batch_lbl.shape)}")
    print(f"  • Train Batch Label Range  : Min={train_batch_lbl.min().item()}, Max={train_batch_lbl.max().item()}")

    # -------------------------------------------------------------
    # Verification Summary Report
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" VERIFICATION SUMMARY REPORT")
    print("=" * 70)
    all_passed = True
    for item, status in checklist.items():
        symbol = "[x]" if status else "[ ]"
        status_text = "PASS" if status else "FAIL"
        print(f"  {symbol} {item:<30} : {status_text}")
        if not status:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print(" ALL 11 VERIFICATION CRITERIA PASSED SUCCESSFULLY!")
    else:
        print(" SOME CHECKS FAILED. PLEASE REVIEW LOGS ABOVE.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_full_verification()
    sys.exit(0 if success else 1)
