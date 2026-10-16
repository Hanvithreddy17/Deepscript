#!/usr/bin/env python3
"""
DeepScript — Six-Class ViT-B/16 Candidate Training & Comparative Evaluation
==========================================================================
Trains a new candidate checkpoint using the audited dataset with group-aware
site-level splitting, aspect-ratio-preserving tiling, and domain-specific augmentation.

Guarantees:
1. Zero Data Leakage: Strictly adheres to site-level disjoint splits from metadata.csv.
2. Aspect-Ratio Preservation: Inscription ribbons are tiled to prevent character squishing.
3. Architectural Continuity: Retains ViT-B/16 backbone, 256-dim metric projector, and cosine head.
4. Backup Safety: Never touches checkpoints/six_class_vit or checkpoints/whole_inscription_vit.
5. Rigorous Evaluation: Evaluates candidate vs existing checkpoint on held-out test split (N=13).
"""

import sys
import os
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_deepscript_model
from preprocessing.transforms import (
    get_train_transforms,
    get_eval_transforms,
    extract_aspect_preserving_tiles,
)

CLASSES = ["brahmi", "grantha", "gupta", "kadamba", "kharosthi", "tamil_brahmi"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "six_class_candidate"
TILES_CACHE_DIR = PROJECT_ROOT / "dataset" / "tiles_cache"
METADATA_CSV = PROJECT_ROOT / "dataset" / "metadata.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "candidate_evaluation"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class InscriptionDataset(Dataset):
    """PyTorch Dataset loading image paths with given transform."""
    def __init__(self, samples: List[Tuple[Path, str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, script, label = self.samples[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def prepare_datasets() -> Tuple[List[Tuple[Path, str, int]], List[Tuple[Path, str, int]], List[Tuple[Path, str, int]]]:
    """Loads group-aware split from metadata.csv and generates training tiles for wide inscriptions."""
    df = pd.read_csv(METADATA_CSV)
    
    TILES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_tiles_dir = TILES_CACHE_DIR / "train"
    train_tiles_dir.mkdir(parents=True, exist_ok=True)

    train_samples: List[Tuple[Path, str, int]] = []
    val_samples: List[Tuple[Path, str, int]] = []
    test_samples: List[Tuple[Path, str, int]] = []

    for _, row in df.iterrows():
        p = PROJECT_ROOT / row["image_path"]
        script = row["script"]
        label = CLASS_TO_IDX[script]
        split = row["split"]

        if not p.exists():
            print(f"Warning: file not found: {p}")
            continue

        if split == "train":
            # Add parent whole image
            train_samples.append((p, script, label))

            # If wide or tall, generate aspect-preserving tiles
            with Image.open(p) as img:
                w, h = img.size
                ar = w / float(max(1, h))

            if ar > 1.8 or ar < 0.55:
                with Image.open(p) as img:
                    tiles = extract_aspect_preserving_tiles(img.convert("RGB"), max_tiles=5, overlap=0.25)
                # If multiple tiles extracted, cache and add each tile
                if len(tiles) > 1:
                    base_name = p.stem
                    for t_idx, tile in enumerate(tiles):
                        tile_path = train_tiles_dir / f"{base_name}_tile{t_idx}.jpg"
                        if not tile_path.exists():
                            tile.save(tile_path, "JPEG", quality=95)
                        train_samples.append((tile_path, script, label))
        elif split == "val":
            val_samples.append((p, script, label))
        elif split == "test":
            test_samples.append((p, script, label))

    print(f"Prepared Dataset Samples:")
    print(f"  • Train samples: {len(train_samples)} (includes whole images + unwarped tiles)")
    print(f"  • Val samples  : {len(val_samples)} (whole images, strictly disjoint sites)")
    print(f"  • Test samples : {len(test_samples)} (whole images, strictly disjoint sites)")

    return train_samples, val_samples, test_samples


def train_candidate():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples, test_samples = prepare_datasets()

    train_tf = get_train_transforms(image_size=(224, 224), enable_augmentation=True)
    eval_tf = get_eval_transforms(image_size=(224, 224))

    train_ds = InscriptionDataset(train_samples, transform=train_tf)
    val_ds = InscriptionDataset(val_samples, transform=eval_tf)
    test_ds = InscriptionDataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    print("\nInitializing ViT-B/16 DeepScript Model (6 Classes, Cosine Similarity Head)...")
    model = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=6,
        embedding_dim=256,
        head_type="cosine",
        pretrained=True,
        freeze_backbone=True,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Stage 1: Warmup Projector & Head (Backbone Frozen)
    warmup_epochs = 4
    print(f"\n--- STAGE 1: HEAD WARMUP ({warmup_epochs} epochs, backbone frozen) ---")
    optimizer_s1 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=warmup_epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, warmup_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_s1.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_s1.step()

            train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler_s1.step()
        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                v_loss += loss.item() * len(labels)
                v_correct += (logits.argmax(dim=-1) == labels).sum().item()
                v_total += len(labels)

        v_loss /= v_total
        v_acc = v_correct / v_total
        print(f"Epoch {epoch:02d}/{warmup_epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.1f}%")

        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Stage 2: Fine-Tuning (Unfreeze Last 2 Blocks)
    finetune_epochs = 6
    print(f"\n--- STAGE 2: FINE-TUNING ({finetune_epochs} epochs, last 2 ViT blocks unfrozen) ---")
    model.unfreeze_last_n_blocks(n=2)

    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    optimizer_s2 = AdamW([
        {"params": backbone_params, "lr": 2e-5, "weight_decay": 1e-4},
        {"params": head_params, "lr": 2e-4, "weight_decay": 1e-4},
    ])
    scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=finetune_epochs, eta_min=1e-6)

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_s2.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_s2.step()

            train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler_s2.step()
        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                v_loss += loss.item() * len(labels)
                v_correct += (logits.argmax(dim=-1) == labels).sum().item()
                v_total += len(labels)

        v_loss /= v_total
        v_acc = v_correct / v_total
        print(f"Epoch {epoch:02d}/{finetune_epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.1f}%")

        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save Candidate Checkpoint
    checkpoint_file = CHECKPOINT_DIR / "best_vit_model.pth"
    save_payload = {
        "model_state_dict": best_state,
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "backbone_name": "vit_b_16",
        "embedding_dim": 256,
        "head_type": "cosine",
        "best_val_acc": best_val_acc,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(save_payload, checkpoint_file)
    print(f"\nSaved Candidate Checkpoint to: {checkpoint_file} (Best Val Acc: {best_val_acc*100:.1f}%)")

    # Load best weights back into model for evaluation
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Load Existing 6-Class Checkpoint for direct head-to-head comparison
    existing_ckpt_path = PROJECT_ROOT / "checkpoints" / "six_class_vit" / "best_vit_model.pth"
    existing_model = None
    if existing_ckpt_path.exists():
        print(f"\nLoading Existing 6-Class Checkpoint for head-to-head comparison: {existing_ckpt_path}")
        existing_model = get_deepscript_model(
            backbone_name="vit_b_16",
            num_classes=6,
            embedding_dim=256,
            head_type="cosine",
            pretrained=False,
            freeze_backbone=True,
        )
        existing_data = torch.load(existing_ckpt_path, map_location="cpu")
        existing_model.load_state_dict(existing_data["model_state_dict"])
        existing_model.to(device)
        existing_model.eval()

    # Comparative Evaluation on Held-Out Test Split (N=13)
    print("\n" + "=" * 70)
    print(" HELD-OUT TEST SPLIT EVALUATION (N=13 Isolated Samples, Zero Site Leakage)")
    print("=" * 70)

    results_report = run_comparative_evaluation(
        candidate_model=model,
        existing_model=existing_model,
        test_samples=test_samples,
        device=device,
        eval_tf=eval_tf,
    )

    report_path = RESULTS_DIR / "candidate_test_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=2)
    print(f"\nDetailed Evaluation Report saved to: {report_path}")


def evaluate_model_on_test(
    model: nn.Module,
    test_samples: List[Tuple[Path, str, int]],
    device: torch.device,
    eval_tf: Any,
    use_tiling: bool = False,
) -> Dict[str, Any]:
    """Runs detailed evaluation computing confusion matrix, per-class metrics, and confusion rates."""
    y_true: List[int] = []
    y_pred: List[int] = []
    sample_records: List[Dict[str, Any]] = []

    scale = getattr(model.head, "scale", torch.tensor(1.0)).item() if hasattr(model, "head") else 1.0

    with torch.no_grad():
        for path, script, true_label in test_samples:
            with Image.open(path) as img:
                raw_rgb = img.convert("RGB")

            if use_tiling:
                tiles = extract_aspect_preserving_tiles(raw_rgb, max_tiles=5, overlap=0.25)
                # Stack whole image + tiles
                t_list = [eval_tf(raw_rgb)] + [eval_tf(t) for t in tiles]
                batch_t = torch.stack(t_list).to(device)
                logits = model(batch_t)
                mean_logits = logits.mean(dim=0, keepdim=True)
                pred_label = mean_logits.argmax(dim=-1).item()
                cos_sims = (mean_logits[0] / scale).cpu().tolist()
            else:
                img_t = eval_tf(raw_rgb).unsqueeze(0).to(device)
                logits = model(img_t)
                pred_label = logits.argmax(dim=-1).item()
                cos_sims = (logits[0] / scale).cpu().tolist()

            y_true.append(true_label)
            y_pred.append(pred_label)

            sample_records.append({
                "file": path.name,
                "true_class": script,
                "predicted_class": IDX_TO_CLASS[pred_label],
                "correct": (true_label == pred_label),
                "top_cosine": round(cos_sims[pred_label], 4),
            })

    # Confusion Matrix (6x6)
    cm = np.zeros((6, 6), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Per-class precision, recall, f1
    per_class = {}
    for idx, c_name in enumerate(CLASSES):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        support = cm[idx, :].sum()

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[c_name] = {
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    total_correct = sum(cm[i, i] for i in range(6))
    total_samples = len(y_true)
    accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0

    # Key Confusion Rates:
    # Tamil-Brahmi (idx 5) classified as Ashokan Brahmi (idx 0)
    tb_total = cm[5, :].sum()
    tb_to_brahmi_cnt = cm[5, 0]
    tb_to_brahmi_rate = float(tb_to_brahmi_cnt / tb_total) if tb_total > 0 else 0.0

    # Ashokan Brahmi (idx 0) classified as Tamil-Brahmi (idx 5)
    b_total = cm[0, :].sum()
    b_to_tb_cnt = cm[0, 5]
    b_to_tb_rate = float(b_to_tb_cnt / b_total) if b_total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "total_samples": total_samples,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "confusion_rates": {
            "tamil_brahmi_to_ashokan_brahmi_count": int(tb_to_brahmi_cnt),
            "tamil_brahmi_to_ashokan_brahmi_rate": round(tb_to_brahmi_rate, 4),
            "ashokan_brahmi_to_tamil_brahmi_count": int(b_to_tb_cnt),
            "ashokan_brahmi_to_tamil_brahmi_rate": round(b_to_tb_rate, 4),
        },
        "sample_predictions": sample_records,
    }


def run_comparative_evaluation(
    candidate_model: nn.Module,
    existing_model: Optional[nn.Module],
    test_samples: List[Tuple[Path, str, int]],
    device: torch.device,
    eval_tf: Any,
) -> Dict[str, Any]:
    print("\n--- 1. Evaluating Candidate Model (Whole Images) ---")
    cand_whole = evaluate_model_on_test(candidate_model, test_samples, device, eval_tf, use_tiling=False)
    print(f"Candidate Whole Images Accuracy: {cand_whole['accuracy']*100:.1f}%")
    print(f"Tamil-Brahmi -> Ashokan Brahmi Confusion: {cand_whole['confusion_rates']['tamil_brahmi_to_ashokan_brahmi_count']} / {cand_whole['per_class']['tamil_brahmi']['support']}")

    print("\n--- 2. Evaluating Candidate Model (Tile-Aggregated Pipeline) ---")
    cand_tiled = evaluate_model_on_test(candidate_model, test_samples, device, eval_tf, use_tiling=True)
    print(f"Candidate Tile-Aggregated Accuracy: {cand_tiled['accuracy']*100:.1f}%")
    print(f"Tamil-Brahmi -> Ashokan Brahmi Confusion: {cand_tiled['confusion_rates']['tamil_brahmi_to_ashokan_brahmi_count']} / {cand_tiled['per_class']['tamil_brahmi']['support']}")

    exist_whole = None
    exist_tiled = None
    if existing_model is not None:
        print("\n--- 3. Evaluating Existing Six-Class Model (Whole Images) ---")
        exist_whole = evaluate_model_on_test(existing_model, test_samples, device, eval_tf, use_tiling=False)
        print(f"Existing Whole Images Accuracy: {exist_whole['accuracy']*100:.1f}%")
        print(f"Tamil-Brahmi -> Ashokan Brahmi Confusion: {exist_whole['confusion_rates']['tamil_brahmi_to_ashokan_brahmi_count']} / {exist_whole['per_class']['tamil_brahmi']['support']}")

        print("\n--- 4. Evaluating Existing Six-Class Model (Tile-Aggregated) ---")
        exist_tiled = evaluate_model_on_test(existing_model, test_samples, device, eval_tf, use_tiling=True)
        print(f"Existing Tile-Aggregated Accuracy: {exist_tiled['accuracy']*100:.1f}%")
        print(f"Tamil-Brahmi -> Ashokan Brahmi Confusion: {exist_tiled['confusion_rates']['tamil_brahmi_to_ashokan_brahmi_count']} / {exist_tiled['per_class']['tamil_brahmi']['support']}")

    return {
        "candidate_whole": cand_whole,
        "candidate_tiled": cand_tiled,
        "existing_whole": exist_whole,
        "existing_tiled": exist_tiled,
    }


if __name__ == "__main__":
    train_candidate()
