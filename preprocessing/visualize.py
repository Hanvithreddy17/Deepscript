"""
DeepScript — Image Preprocessing Visualizer
===========================================
Generates visual comparison figures showing original character images alongside
preprocessed (evaluated/denormalized) and augmented training pipeline outputs.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from PIL import Image

from preprocessing.dataset import AncientScriptDataset
from preprocessing.transforms import (
    get_train_transforms,
    get_eval_transforms,
    denormalize_tensor,
)


def generate_comparison_figure(
    dataset_root: str = "dataset/dataset",
    output_path: str = "preprocessing_comparison.png",
    sample_classes: tuple = ("a", "ka", "ma", "three"),
) -> str:
    """
    Generates and saves a grid comparing raw images against evaluated and augmented pipelines.

    Args:
        dataset_root: Path to the character dataset directory.
        output_path: File path to save the output PNG.
        sample_classes: Tuple of class names to showcase.

    Returns:
        Absolute path to the saved figure.
    """
    ds = AncientScriptDataset(dataset_root)
    eval_tf = get_eval_transforms()
    train_tf = get_train_transforms(enable_augmentation=True)

    selected_indices = []
    for cls in sample_classes:
        for idx, (path, c_name, _) in enumerate(ds.samples):
            if c_name == cls:
                selected_indices.append(idx)
                break

    fig, axes = plt.subplots(len(selected_indices), 4, figsize=(14, 3.5 * len(selected_indices)))
    plt.suptitle("DeepScript: Original vs. Preprocessed & Augmented Pipeline Comparison", fontsize=14, y=0.98)

    col_titles = [
        "1. Original Image (Raw)",
        "2. Eval Pipeline [224x224 RGB]",
        "3. Train Augmentation (Var A)",
        "4. Train Augmentation (Var B)",
    ]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for row, s_idx in enumerate(selected_indices):
        info = ds.get_sample_info(s_idx)
        raw_pil, label = ds[s_idx]

        # 1. Raw image
        axes[row, 0].imshow(raw_pil)
        axes[row, 0].set_ylabel(f"Class: {info['class_name']}\n(Label: {label})", fontsize=11, fontweight="bold")
        axes[row, 0].set_xlabel(f"Size: {info['original_size']} | Mode: {info['original_mode']}", fontsize=8)

        # 2. Eval transform (denormalized for display)
        eval_tensor = eval_tf(raw_pil)
        eval_disp = denormalize_tensor(eval_tensor).permute(1, 2, 0).numpy()
        axes[row, 1].imshow(eval_disp)
        axes[row, 1].set_xlabel(f"Tensor: {list(eval_tensor.shape)}", fontsize=8)

        # 3. Train augmentation variation A
        train_tensor_a = train_tf(raw_pil)
        train_disp_a = denormalize_tensor(train_tensor_a).permute(1, 2, 0).numpy()
        axes[row, 2].imshow(train_disp_a)
        axes[row, 2].set_xlabel("Mild rotation/jitter", fontsize=8)

        # 4. Train augmentation variation B
        train_tensor_b = train_tf(raw_pil)
        train_disp_b = denormalize_tensor(train_tensor_b).permute(1, 2, 0).numpy()
        axes[row, 3].imshow(train_disp_b)
        axes[row, 3].set_xlabel("Affine scale/brightness", fontsize=8)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    out = Path(output_path)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out.resolve())


if __name__ == "__main__":
    out_file = generate_comparison_figure()
    print(f"Comparison figure saved to: {out_file}")
