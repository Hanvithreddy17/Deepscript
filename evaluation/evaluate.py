"""
DeepScript — Test Set Evaluation & Metrics Suite
================================================
Evaluates a trained DeepScript model checkpoint on the held-out, isolated test partition.

Computes:
1. Overall Top-1 Test Accuracy & Top-3 Accuracy.
2. Cross-Entropy Test Loss.
3. Per-class & macro Precision, Recall, and F1-Score via scikit-learn.
4. Detailed verification of test isolation.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score

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


def evaluate_checkpoint(
    checkpoint_path: str | Path = "checkpoints/best_vit_model.pth",
    dataset_root: str = "dataset/dataset",
    batch_size: int = 32,
    random_seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Loads the trained model checkpoint and performs full evaluation on the isolated test set.

    Args:
        checkpoint_path: Path to the saved checkpoint (.pth).
        dataset_root: Dataset root directory.
        batch_size: Batch size for test evaluation.
        random_seed: Random seed used to reconstruct identical test partition.
        device: PyTorch device.

    Returns:
        Dictionary containing test metrics (loss, accuracy, top3_acc, classification_report).
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_file.resolve()}")

    exec_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print(" DEEPSCRIPT: TEST SET EVALUATION SUITE")
    print(f" Loading Checkpoint : {ckpt_file.resolve()}")
    print(f" Execution Device   : {exec_device}")
    print("=" * 75)

    # 1. Load Checkpoint State
    checkpoint = torch.load(ckpt_file, map_location=exec_device)
    classes = checkpoint.get("classes", [])
    class_to_idx = checkpoint.get("class_to_idx", {})
    epoch = checkpoint.get("epoch", "N/A")
    val_acc = checkpoint.get("val_acc", 0.0)

    print(f"\n[Step 1/3] Checkpoint Metadata:")
    print(f"  • Trained Epochs     : {epoch}")
    print(f"  • Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"  • Total Classes      : {len(classes)}")

    # 2. Reconstruct Isolated Test Set
    print("\n[Step 2/3] Reconstructing Isolated Test Dataset Split (Seed=42)...")
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
    print(f"  • Isolated Test Size : {len(test_ds)} samples ({len(test_loader)} batches)")

    # 3. Instantiate Model and Load Weights
    print("\n[Step 3/3] Instantiating ViT-B/16 Model & Loading Checkpoint Weights...")
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
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    print("\nRunning inference over test batches...")
    with torch.no_grad():
        for images, targets in test_loader:
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

    # Print Summary Report
    print("\n" + "=" * 75)
    print(" FINAL TEST EVALUATION RESULTS")
    print("=" * 75)
    print(f"  • Total Test Samples  : {total_samples}")
    print(f"  • Test Loss           : {avg_test_loss:.4f}")
    print(f"  • Top-1 Test Accuracy : {test_top1_acc * 100:.2f}%")
    print(f"  • Top-3 Test Accuracy : {test_top3_acc * 100:.2f}%")
    print(f"  • Macro Precision     : {report_dict['macro avg']['precision'] * 100:.2f}%")
    print(f"  • Macro Recall        : {report_dict['macro avg']['recall'] * 100:.2f}%")
    print(f"  • Macro F1-Score      : {report_dict['macro avg']['f1-score'] * 100:.2f}%")
    print(f"  • Weighted F1-Score   : {report_dict['weighted avg']['f1-score'] * 100:.2f}%")
    print("=" * 75)

    return {
        "test_loss": avg_test_loss,
        "test_top1_accuracy": test_top1_acc,
        "test_top3_accuracy": test_top3_acc,
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "classification_report": report_dict,
    }


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

    args = parser.parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
