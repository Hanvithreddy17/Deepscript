"""
DeepScript — PyTorch Dataset for Ancient Script Recognition
===========================================================
This module provides a clean, robust, and extensible PyTorch Dataset implementation
for loading ancient script character images and inscriptions.

Key features:
1. Dynamic Class Discovery: Automatically identifies non-empty class subdirectories.
2. Deterministic Numerical Encoding: Maps class names to integer labels (0 to C-1) alphabetically.
3. Image Mode Standardization: Loads images and safely converts RGBA/Grayscale to 3-channel RGB.
4. Corrupted Image Resilience: Validates image integrity and reports unreadable files gracefully.
5. Modular Transform Integration: Works seamlessly with torchvision transforms and DataLoader.
"""

from typing import List, Tuple, Dict, Optional, Callable, Any, Union
from pathlib import Path
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset

# Setup logger
logger = logging.getLogger(__name__)

# Standard image extensions supported by Pillow / DeepScript
SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".gif"
)


class AncientScriptDataset(Dataset):
    """
    PyTorch Dataset for Ancient Script Images.

    Each class is represented by a folder inside `root_dir`.
    Folders with no valid image files (e.g., empty placeholders) are automatically skipped.

    Attributes:
        root_dir (Path): Base directory containing class subdirectories.
        classes (List[str]): Alphabetically sorted list of distinct class names.
        class_to_idx (Dict[str, int]): Mapping from class name to numerical index.
        idx_to_class (Dict[int, str]): Mapping from numerical index to class name.
        samples (List[Tuple[Path, str, int]]): List of (image_path, class_name, label_idx).
        transform (Optional[Callable]): Transform pipeline to apply to images.
        target_transform (Optional[Callable]): Transform pipeline to apply to labels.
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        samples: Optional[List[Tuple[Path, str, int]]] = None,
        classes: Optional[List[str]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        supported_extensions: Tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS,
    ) -> None:
        """
        Initialize the AncientScriptDataset.

        Args:
            root_dir: Path to directory containing class subfolders (e.g., 'dataset/dataset').
            transform: Optional callable transform for PIL Images.
            target_transform: Optional callable transform for integer labels.
            samples: Pre-constructed sample list (used when creating train/val/test split subsets).
            classes: Optional explicit list of class names.
            class_to_idx: Optional explicit mapping from class name to integer label.
            supported_extensions: Allowed file extensions.
        """
        self.transform = transform
        self.target_transform = target_transform
        self.supported_extensions = supported_extensions

        if samples is not None:
            # Direct initialization from existing sample slice (e.g. from a split)
            self.root_dir = Path(root_dir) if root_dir else Path(".")
            self.samples = samples
            if classes is not None and class_to_idx is not None:
                self.classes = classes
                self.class_to_idx = class_to_idx
            else:
                self.classes = sorted(list({c for _, c, _ in self.samples}))
                self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            self.corrupted_files: List[Tuple[str, str]] = []
        else:
            # Auto-discovery from root_dir
            if root_dir is None:
                raise ValueError("Either 'root_dir' or 'samples' must be provided.")
            self.root_dir = Path(root_dir)
            if not self.root_dir.exists():
                raise FileNotFoundError(f"Dataset root directory does not exist: {self.root_dir}")
            if not self.root_dir.is_dir():
                raise NotADirectoryError(f"Dataset root is not a directory: {self.root_dir}")

            self.classes, self.class_to_idx, self.idx_to_class, self.samples, self.corrupted_files = (
                self._discover_dataset()
            )

    def _discover_dataset(
        self,
    ) -> Tuple[
        List[str],
        Dict[str, int],
        Dict[int, str],
        List[Tuple[Path, str, int]],
        List[Tuple[str, str]],
    ]:
        """
        Scans the root directory to find all non-empty class folders and valid image files.

        Returns:
            Tuple of (classes, class_to_idx, idx_to_class, samples, corrupted_files)
        """
        candidate_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        discovered_classes: List[str] = []
        class_files: Dict[str, List[Path]] = {}
        corrupted: List[Tuple[str, str]] = []

        # Find valid images in each folder
        for folder in candidate_dirs:
            imgs = [
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in self.supported_extensions
            ]
            if len(imgs) > 0:
                discovered_classes.append(folder.name)
                class_files[folder.name] = imgs

        # Sort classes alphabetically for deterministic label assignment across all runs
        classes = sorted(discovered_classes)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

        # Build list of samples
        samples: List[Tuple[Path, str, int]] = []
        for cls_name in classes:
            idx = class_to_idx[cls_name]
            for img_path in class_files[cls_name]:
                samples.append((img_path, cls_name, idx))

        if len(samples) == 0:
            logger.warning(
                f"No valid image files found in {self.root_dir} matching extensions {self.supported_extensions}"
            )

        return classes, class_to_idx, idx_to_class, samples, corrupted

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Union[torch.Tensor, Image.Image], int]:
        """
        Retrieves the preprocessed image tensor and numerical label for a given index.

        Args:
            index: Integer index in [0, len(self) - 1].

        Returns:
            Tuple of (image, label_idx):
                - image: torch.Tensor (or PIL.Image if transform is None) of shape [3, H, W]
                - label_idx: int representing class index
        """
        img_path, class_name, label_idx = self.samples[index]

        try:
            # Open image and standardize color mode to 3-channel RGB
            with Image.open(img_path) as raw_img:
                # Convert RGBA/Grayscale/Palette to RGB
                image = raw_img.convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise RuntimeError(f"Corrupted or unreadable image at {img_path}: {e}") from e

        # Apply image transform pipeline if provided
        if self.transform is not None:
            image = self.transform(image)

        # Apply target label transform if provided
        target = label_idx
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def get_sample_info(self, index: int) -> Dict[str, Any]:
        """
        Returns rich metadata for a given sample without applying transforms.

        Args:
            index: Sample index.

        Returns:
            Dict containing path, class_name, label_idx, file_size_bytes, image_mode, and size.
        """
        img_path, class_name, label_idx = self.samples[index]
        with Image.open(img_path) as img:
            mode = img.mode
            size = img.size

        return {
            "index": index,
            "path": str(img_path),
            "filename": img_path.name,
            "class_name": class_name,
            "label_idx": label_idx,
            "file_size_bytes": img_path.stat().st_size,
            "original_mode": mode,
            "original_size": size,
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """Returns a dictionary mapping class names to sample counts in this dataset instance."""
        distribution: Dict[str, int] = {c: 0 for c in self.classes}
        for _, class_name, _ in self.samples:
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution

    def get_classes(self) -> List[str]:
        """Returns the list of discovered class names."""
        return list(self.classes)

    def get_class_to_idx(self) -> Dict[str, int]:
        """Returns the class_to_idx dictionary."""
        return dict(self.class_to_idx)

    def get_idx_to_class(self) -> Dict[int, str]:
        """Returns the idx_to_class dictionary."""
        return dict(self.idx_to_class)

    def __repr__(self) -> str:
        return (
            f"AncientScriptDataset(num_samples={len(self.samples)}, "
            f"num_classes={len(self.classes)}, "
            f"has_transform={self.transform is not None})"
        )
