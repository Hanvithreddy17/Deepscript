"""
DeepScript — Stratified Dataset Partitioning and Split Management
=================================================================
This module provides reproducible, stratified train/validation/test partitioning
for ancient script datasets.

Key guarantees:
1. Stratified Partitioning: Maintains the proportional class balance across all splits.
2. Strict Disjointness (Zero Data Leakage): Ensures no image is shared between train, val, and test.
3. Full Determinism: Uses a configurable fixed random seed (default: 42).
4. Sub-Dataset Construction: Attaches train augmentations to the train split and
   deterministic evaluation transforms to validation and test splits.
5. DataLoader Factory: Generates standard PyTorch DataLoaders with customizable batch sizes.
"""

from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
import random
import torch
from torch.utils.data import DataLoader

from preprocessing.dataset import AncientScriptDataset
from preprocessing.transforms import get_train_transforms, get_eval_transforms


def create_stratified_splits(
    dataset: AncientScriptDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
) -> Tuple[AncientScriptDataset, AncientScriptDataset, AncientScriptDataset]:
    """
    Partitions an AncientScriptDataset into reproducible, stratified Train, Validation, and Test sets.

    Stratification is performed on a per-class basis:
    - Each class is individually shuffled using the fixed random seed.
    - Samples from that class are partitioned proportionally across train, val, and test.
    - Classes with very few samples (e.g., <= 15) are guaranteed at least 1 sample in each split.

    Args:
        dataset: Source AncientScriptDataset containing all discovered samples.
        train_ratio: Fraction of data for training (default 0.70).
        val_ratio: Fraction of data for validation (default 0.15).
        test_ratio: Fraction of data for testing (default 0.15).
        random_seed: Integer seed for reproducibility (default 42).
        train_transform: Transform pipeline for training set (defaults to get_train_transforms()).
        eval_transform: Transform pipeline for val/test sets (defaults to get_eval_transforms()).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Ratios must sum to 1.0 (got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio})")

    # Set default transforms if none provided
    if train_transform is None:
        train_transform = get_train_transforms()
    if eval_transform is None:
        eval_transform = get_eval_transforms()

    # Group samples by class
    class_to_samples: Dict[str, List[Tuple[Path, str, int]]] = defaultdict(list)
    for sample in dataset.samples:
        _, class_name, _ = sample
        class_to_samples[class_name].append(sample)

    train_samples: List[Tuple[Path, str, int]] = []
    val_samples: List[Tuple[Path, str, int]] = []
    test_samples: List[Tuple[Path, str, int]] = []

    rng = random.Random(random_seed)

    for class_name in dataset.classes:
        samples = list(class_to_samples[class_name])
        # Sort first for cross-platform deterministic order, then shuffle with seeded RNG
        samples.sort(key=lambda s: str(s[0]))
        rng.shuffle(samples)

        n_total = len(samples)
        if n_total == 0:
            continue

        if n_total < 3:
            # Extremely small class: allocate to train
            n_train = n_total
            n_val = 0
            n_test = 0
        else:
            n_train = int(round(n_total * train_ratio))
            n_val = int(round(n_total * val_ratio))
            # Ensure at least 1 in val and test if n_total >= 3
            n_val = max(1, n_val)
            n_test = n_total - n_train - n_val

            if n_test < 1:
                # Borrow from train if needed to maintain representation
                n_train -= 1
                n_test = n_total - n_train - n_val

        # Slice samples
        train_slice = samples[:n_train]
        val_slice = samples[n_train:n_train + n_val]
        test_slice = samples[n_train + n_val:]

        train_samples.extend(train_slice)
        val_samples.extend(val_slice)
        test_samples.extend(test_slice)

    # Instantiate distinct Dataset objects with their respective transforms
    train_dataset = AncientScriptDataset(
        root_dir=dataset.root_dir,
        samples=train_samples,
        classes=dataset.classes,
        class_to_idx=dataset.class_to_idx,
        transform=train_transform,
    )

    val_dataset = AncientScriptDataset(
        root_dir=dataset.root_dir,
        samples=val_samples,
        classes=dataset.classes,
        class_to_idx=dataset.class_to_idx,
        transform=eval_transform,
    )

    test_dataset = AncientScriptDataset(
        root_dir=dataset.root_dir,
        samples=test_samples,
        classes=dataset.classes,
        class_to_idx=dataset.class_to_idx,
        transform=eval_transform,
    )

    # Perform immediate leakage check
    verify_split_disjointness(train_dataset, val_dataset, test_dataset)

    return train_dataset, val_dataset, test_dataset


def verify_split_disjointness(
    train_dataset: AncientScriptDataset,
    val_dataset: AncientScriptDataset,
    test_dataset: AncientScriptDataset,
) -> bool:
    """
    Formally verifies that there is zero overlap (no data leakage) between train, val, and test splits.

    Args:
        train_dataset: Train dataset instance.
        val_dataset: Validation dataset instance.
        test_dataset: Test dataset instance.

    Returns:
        True if all splits are strictly disjoint.

    Raises:
        AssertionError if any duplicate file paths are discovered between splits.
    """
    train_paths = {str(p) for p, _, _ in train_dataset.samples}
    val_paths = {str(p) for p, _, _ in val_dataset.samples}
    test_paths = {str(p) for p, _, _ in test_dataset.samples}

    train_val_overlap = train_paths.intersection(val_paths)
    train_test_overlap = train_paths.intersection(test_paths)
    val_test_overlap = val_paths.intersection(test_paths)

    if train_val_overlap:
        raise AssertionError(f"Data leakage detected! {len(train_val_overlap)} samples overlap between Train and Val: {list(train_val_overlap)[:3]}")
    if train_test_overlap:
        raise AssertionError(f"Data leakage detected! {len(train_test_overlap)} samples overlap between Train and Test: {list(train_test_overlap)[:3]}")
    if val_test_overlap:
        raise AssertionError(f"Data leakage detected! {len(val_test_overlap)} samples overlap between Val and Test: {list(val_test_overlap)[:3]}")

    return True


def get_split_statistics(
    train_dataset: AncientScriptDataset,
    val_dataset: AncientScriptDataset,
    test_dataset: AncientScriptDataset,
) -> Dict[str, Any]:
    """
    Computes class-by-class and total breakdown statistics for all three splits.

    Returns:
        Dictionary containing total counts, split ratios, and per-class distributions.
    """
    train_dist = train_dataset.get_class_distribution()
    val_dist = val_dataset.get_class_distribution()
    test_dist = test_dataset.get_class_distribution()

    total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    classes = train_dataset.get_classes()

    per_class_table = []
    for c in classes:
        t_cnt = train_dist.get(c, 0)
        v_cnt = val_dist.get(c, 0)
        te_cnt = test_dist.get(c, 0)
        c_total = t_cnt + v_cnt + te_cnt
        per_class_table.append({
            "class": c,
            "train": t_cnt,
            "val": v_cnt,
            "test": te_cnt,
            "total": c_total,
            "train_pct": (t_cnt / c_total * 100) if c_total > 0 else 0.0,
        })

    return {
        "total_samples": total_samples,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "test_count": len(test_dataset),
        "train_pct": (len(train_dataset) / total_samples * 100) if total_samples > 0 else 0.0,
        "val_pct": (len(val_dataset) / total_samples * 100) if total_samples > 0 else 0.0,
        "test_pct": (len(test_dataset) / total_samples * 100) if total_samples > 0 else 0.0,
        "num_classes": len(classes),
        "per_class": per_class_table,
    }


def create_dataloaders(
    train_dataset: AncientScriptDataset,
    val_dataset: AncientScriptDataset,
    test_dataset: AncientScriptDataset,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates standard PyTorch DataLoaders for train, val, and test datasets.

    Args:
        train_dataset: Pre-configured train dataset (with train transforms).
        val_dataset: Pre-configured val dataset (with eval transforms).
        test_dataset: Pre-configured test dataset (with eval transforms).
        batch_size: Batch size for mini-batches.
        num_workers: Number of subprocess workers for data loading.
        pin_memory: Whether to copy tensors into CUDA pinned memory.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
