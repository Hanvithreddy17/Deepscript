"""
DeepScript — Preprocessing and Dataset Module
=============================================
Provides clean, modular dataset loaders, image transforms, and partitioning tools
for ancient script recognition with PyTorch and Vision Transformers.
"""

from preprocessing.transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    DEFAULT_IMAGE_SIZE,
    get_train_transforms,
    get_eval_transforms,
    get_display_transforms,
    denormalize_tensor,
)
from preprocessing.dataset import (
    AncientScriptDataset,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from preprocessing.splits import (
    create_stratified_splits,
    create_dataloaders,
    verify_split_disjointness,
    get_split_statistics,
)
from preprocessing.visualize import (
    generate_comparison_figure,
)

__all__ = [
    "IMAGENET_DEFAULT_MEAN",
    "IMAGENET_DEFAULT_STD",
    "DEFAULT_IMAGE_SIZE",
    "get_train_transforms",
    "get_eval_transforms",
    "get_display_transforms",
    "denormalize_tensor",
    "AncientScriptDataset",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "create_stratified_splits",
    "create_dataloaders",
    "verify_split_disjointness",
    "get_split_statistics",
    "generate_comparison_figure",
]
