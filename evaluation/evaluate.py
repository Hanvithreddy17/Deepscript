"""
DeepScript — Test Set Evaluation & Metrics Suite
================================================
Evaluates a trained DeepScript model checkpoint on the held-out, isolated test partition.

Computes & Generates:
1. Overall Top-1 Test Accuracy & Top-3 Accuracy.
2. Cross-Entropy Test Loss.
3. Macro & Weighted Precision, Recall, and F1-Score via scikit-learn.
4. Per-class metrics exported to results/evaluation/classification_report.csv.
5. Summary metrics exported to results/evaluation/metrics.json.
6. Normalized confusion matrix heatmap saved to results/evaluation/confusion_matrix.png.
7. Multi-panel training & validation curves saved to results/training/training_curves.png.
"""

import sys
import os
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    top_k_accuracy_score,
    confusion_matrix,
)

# Reconfigure stdout for UTF-8 compatibility
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import (
    AncientScriptDataset,
    get_eval_transforms,
    create_stratified_splits,
    create_dataloaders,
)
from models import get_deepscript_model


def plot_training_curves(history_csv: str | Path, output_png: str | Path) -> None:
    """Generates multi-panel training and validation curve visualization."""
    csv_path = Path(history_csv)
    if not csv_path.exists():
        print(f"Warning: History CSV not found at {csv_path}, skipping curve plot.")
        return

    epochs = []
    stages = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_top3_acc = []
    lr_head = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            stages.append(row["stage"])
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]) * 100)
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]) * 100)
            val_top3_acc.append(float(row["val_top3_acc"]) * 100)
            lr_head.append(float(row["lr_head"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    plt.subplots_adjust(wspace=0.28)

    # Style theme
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Panel 1: Cross-Entropy Loss
    axes[0].plot(epochs, train_loss, marker='o', color='#2563eb', linewidth=2, label='Train Loss')
    axes[0].plot(epochs, val_loss, marker='s', color='#dc2626', linewidth=2, linestyle='--', label='Val Loss')
    axes[0].set_title('Cross-Entropy Loss vs. Epoch', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].legend(frameon=True, fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Accuracy
    axes[1].plot(epochs, train_acc, marker='o', color='#2563eb', linewidth=2, label='Train Top-1 Acc')
    axes[1].plot(epochs, val_acc, marker='s', color='#16a34a', linewidth=2, label='Val Top-1 Acc')
    axes[1].plot(epochs, val_top3_acc, marker='^', color='#d97706', linewidth=2, linestyle='--', label='Val Top-3 Acc')
    axes[1].set_title('Classification Accuracy (%) vs. Epoch', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].legend(frameon=True, fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Learning Rate
    axes[2].plot(epochs, lr_head, marker='d', color='#7c3aed', linewidth=2, label='Head Learning Rate')
    axes[2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold', pad=10)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Learning Rate', fontsize=11)
    axes[2].set_yscale('log')
    axes[2].legend(frameon=True, fontsize=10)
    axes[2].grid(True, linestyle=':', alpha=0.6)

    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"  >>> Training curves plot saved to: {out_file}", flush=True)


def plot_confusion_matrix_heatmap(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_png: str | Path,
) -> None:
    """Generates a high-resolution normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    cax = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Normalized Recall', rotation=-90, va="bottom", fontsize=10)

    num_classes = len(class_names)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)

    ax.set_title('DeepScript: 62-Class Normalized Confusion Matrix (Test Set)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Script Class', fontsize=11, labelpad=10)
    ax.set_ylabel('Ground Truth Class', fontsize=11, labelpad=10)

    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"  >>> Confusion matrix heatmap saved to: {out_file}", flush=True)


def evaluate_checkpoint(
    checkpoint_path: str | Path = "checkpoints/best_vit_model.pth",
    dataset_root: str = "dataset/dataset",
    batch_size: int = 32,
    random_seed: int = 42,
    results_dir: str | Path = "results",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Loads the trained model checkpoint and performs full evaluation on the isolated test set.
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_file.resolve()}")

    exec_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = Path(results_dir)
    eval_out_dir = res_dir / "evaluation"
    training_out_dir = res_dir / "training"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    training_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75, flush=True)
    print(" DEEPSCRIPT: TEST SET EVALUATION SUITE", flush=True)
    print(f" Loading Checkpoint : {ckpt_file.resolve()}", flush=True)
    print(f" Execution Device   : {exec_device}", flush=True)
    print(f" Evaluation Output  : {eval_out_dir.resolve()}", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Checkpoint State
    checkpoint = torch.load(ckpt_file, map_location=exec_device)
    classes = checkpoint.get("classes", [])
    class_to_idx = checkpoint.get("class_to_idx", {})
    epoch = checkpoint.get("epoch", "N/A")
    val_acc = checkpoint.get("val_acc", 0.0)

    print(f"\n[Step 1/4] Checkpoint Metadata:", flush=True)
    print(f"  • Trained Epochs     : {epoch}", flush=True)
    print(f"  • Validation Accuracy: {val_acc * 100:.2f}%", flush=True)
    print(f"  • Total Classes      : {len(classes)}", flush=True)

    # 2. Reconstruct Isolated Test Set
    print("\n[Step 2/4] Reconstructing Isolated Test Dataset Split (Seed=42)...", flush=True)
    raw_dataset = AncientScriptDataset(root_dir=dataset_root, transform=None)
    eval_tf = get_eval_transforms(image_size=(224, 224))

    train_ds, val_ds, test_ds = create_stratified_splits(
        dataset=raw_dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_seed,
        eval_transform=eval_tf,
    )

    _, _, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=0
    )

    num_classes = len(raw_dataset.get_classes())
    print(f"  • Isolated Test Size : {len(test_ds)} samples ({len(test_loader)} batches)", flush=True)

    # 3. Instantiate Model and Load Weights
    print("\n[Step 3/4] Instantiating ViT-B/16 Model & Loading Checkpoint Weights...", flush=True)
    model = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=num_classes,
        embedding_dim=256,
        head_type="cosine",
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(exec_device)
    model.eval()

    # 4. Evaluate on Test Set
    print("\n[Step 4/4] Executing inference over test batches...", flush=True)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(exec_device, non_blocking=True)
            targets = targets.to(exec_device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    total_samples = len(test_ds)
    avg_test_loss = total_loss / total_samples if total_samples > 0 else 0.0
    test_top1_acc = accuracy_score(all_targets, all_preds)
    test_top3_acc = top_k_accuracy_score(
        all_targets, all_probs, k=min(3, num_classes), labels=list(range(num_classes))
    )

    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=classes if len(classes) == num_classes else None,
        output_dict=True,
        zero_division=0,
    )

    # 5. Export Detailed CSV Classification Report
    report_csv_path = eval_out_dir / "classification_report.csv"
    with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision", "recall", "f1-score", "support"])
        for key, val in report_dict.items():
            if isinstance(val, dict):
                writer.writerow([
                    key,
                    round(val.get("precision", 0.0), 4),
                    round(val.get("recall", 0.0), 4),
                    round(val.get("f1-score", 0.0), 4),
                    int(val.get("support", 0)),
                ])
    print(f"  >>> Classification report CSV saved to: {report_csv_path}", flush=True)

    # 6. Export Metrics JSON
    metrics_summary = {
        "checkpoint_evaluated": str(ckpt_file),
        "total_test_samples": total_samples,
        "num_classes": num_classes,
        "test_loss": round(avg_test_loss, 4),
        "top1_accuracy": round(test_top1_acc, 4),
        "top3_accuracy": round(test_top3_acc, 4),
        "macro_precision": round(report_dict["macro avg"]["precision"], 4),
        "macro_recall": round(report_dict["macro avg"]["recall"], 4),
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
    }

    metrics_json_path = eval_out_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  >>> Summary metrics JSON saved to: {metrics_json_path}", flush=True)

    # 7. Generate Plots
    cm_png_path = eval_out_dir / "confusion_matrix.png"
    plot_confusion_matrix_heatmap(all_targets, all_preds, classes, cm_png_path)

    training_csv_path = training_out_dir / "training_history.csv"
    training_curves_path = training_out_dir / "training_curves.png"
    if training_csv_path.exists():
        plot_training_curves(training_csv_path, training_curves_path)

    # Print Summary Report
    print("\n" + "=" * 75, flush=True)
    print(" FINAL TEST EVALUATION RESULTS", flush=True)
    print("=" * 75, flush=True)
    print(f"  • Evaluated Samples   : {total_samples}", flush=True)
    print(f"  • Test Loss           : {avg_test_loss:.4f}", flush=True)
    print(f"  • Top-1 Test Accuracy : {test_top1_acc * 100:.2f}%", flush=True)
    print(f"  • Top-3 Test Accuracy : {test_top3_acc * 100:.2f}%", flush=True)
    print(f"  • Macro Precision     : {report_dict['macro avg']['precision'] * 100:.2f}%", flush=True)
    print(f"  • Macro Recall        : {report_dict['macro avg']['recall'] * 100:.2f}%", flush=True)
    print(f"  • Macro F1-Score      : {report_dict['macro avg']['f1-score'] * 100:.2f}%", flush=True)
    print(f"  • Weighted F1-Score   : {report_dict['weighted avg']['f1-score'] * 100:.2f}%", flush=True)
    print("=" * 75, flush=True)

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="DeepScript: Test Set Model Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_vit_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset/dataset",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for test evaluation",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root directory to save evaluation results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for test split reconstruction",
    )

    args = parser.parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
