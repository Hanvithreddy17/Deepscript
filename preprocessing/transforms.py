"""
DeepScript — Image Preprocessing and Transformation Pipelines
============================================================
This module defines modular image transformation pipelines for ancient script recognition.
Designed for PyTorch vision models (e.g., Vision Transformers / ViT) with ImageNet normalization.

Key design principles:
1. Standardization: Converts all incoming images (Grayscale, RGBA, etc.) to 3-channel RGB.
2. ViT-Compatible Spatial Dimensions: Resizes images to standard (224, 224).
3. ImageNet Normalization: Standard mean and std for transfer learning compatibility.
4. Historical Integrity: Training augmentations are carefully calibrated to avoid distorting
   crucial character stroke geometry or epigraphic glyph features.
5. Extensibility: Modular architecture allows easy integration of future filters (CLAHE,
   denoising, binarization, estampage inversion).
"""

from typing import Tuple, List, Optional
import torch
import torchvision.transforms as T
from PIL import Image


# Standard ImageNet normalization parameters (compatible with ViT and standard backbones)
IMAGENET_DEFAULT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Standard spatial input dimensions for Vision Transformers
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)


def get_train_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
    enable_augmentation: bool = True,
) -> T.Compose:
    """
    Constructs the image preprocessing and augmentation pipeline for the TRAINING split.
    
    Augmentations are designed to be safe for ancient script characters:
    - Subtle rotation (-8 to +8 degrees) to simulate camera/surface angle variation.
    - Slight translation (up to 4%) and scaling (0.95 to 1.05) to simulate framing changes.
    - Mild brightness & contrast jitter (10%) to simulate illumination/weathering differences.
    - NO severe shears, perspective warps, or horizontal flips that could invert script direction.

    Args:
        image_size: Target (height, width) tuple, default (224, 224).
        mean: Normalization mean per RGB channel.
        std: Normalization std per RGB channel.
        enable_augmentation: Whether to include subtle training augmentations.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transform_list: List[object] = [
        # Resize image to target spatial dimensions (Bicubic interpolation recommended for ViT)
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    ]

    if enable_augmentation:
        transform_list.extend([
            # Subtle rotation: simulates natural alignment variance without glyph distortion
            T.RandomRotation(degrees=(-8, 8), interpolation=T.InterpolationMode.BILINEAR),
            # Slight translation and scaling
            T.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            # Mild brightness and contrast variations for varying stone/ink lighting
            T.ColorJitter(brightness=0.10, contrast=0.10),
        ])

    # Convert PIL Image to PyTorch Tensor [C, H, W] in range [0.0, 1.0]
    transform_list.append(T.ToTensor())

    # Normalize channels: (x - mean) / std
    transform_list.append(T.Normalize(mean=list(mean), std=list(std)))

    return T.Compose(transform_list)


def get_eval_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> T.Compose:
    """
    Constructs the deterministic preprocessing pipeline for VALIDATION, TEST, and INFERENCE.
    
    No stochastic augmentations are applied.
    
    Args:
        image_size: Target (height, width) tuple, default (224, 224).
        mean: Normalization mean per RGB channel.
        std: Normalization std per RGB channel.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        # Resize to fixed input dimensions
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        # Convert PIL Image to Tensor [C, H, W] in [0.0, 1.0]
        T.ToTensor(),
        # Normalize with ImageNet mean/std
        T.Normalize(mean=list(mean), std=list(std)),
    ])


def get_display_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> T.Compose:
    """
    Constructs a transform pipeline for raw image visualization / comparison.
    Resizes and converts to [0, 1] tensor WITHOUT normalization.
    """
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
    ])


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> torch.Tensor:
    """
    Reverses ImageNet normalization on a [C, H, W] or [B, C, H, W] tensor for visual inspection.
    Clamps the output values to the valid image range [0.0, 1.0].

    Args:
        tensor: Normalized PyTorch tensor of shape (3, H, W) or (B, 3, H, W).
        mean: Mean tuple used during normalization.
        std: Std tuple used during normalization.

    Returns:
        Denormalized tensor with values in [0.0, 1.0].
    """
    t = tensor.clone()
    if t.ndim == 3:
        if t.shape[0] != 3:
            raise ValueError(f"Expected 3 channels at dimension 0 for [C, H, W] tensor, got shape {tensor.shape}")
        for c in range(3):
            t[c] = t[c] * std[c] + mean[c]
    elif t.ndim == 4:
        if t.shape[1] != 3:
            raise ValueError(f"Expected 3 channels at dimension 1 for [B, C, H, W] tensor, got shape {tensor.shape}")
        for c in range(3):
            t[:, c] = t[:, c] * std[c] + mean[c]
    else:
        raise ValueError(f"Expected 3D [C, H, W] or 4D [B, C, H, W] tensor, got shape {tensor.shape}")
    
    return torch.clamp(t, 0.0, 1.0)
