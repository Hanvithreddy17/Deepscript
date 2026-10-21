#!/usr/bin/env python3
"""
DeepScript — Isolated Tamil-Brahmi Fine-Tuning Experiment
========================================================
Fine-tunes the 6-class classifier on the augmented experimental manifest:
- Initializes from active checkpoint: checkpoints/six_class_vit/best_vit_model.pth
- ViT-B/16 backbone strictly frozen throughout all epochs.
- Fine-tunes 256-d projector and cosine similarity head.
- Balanced sampling via WeightedRandomSampler prevents class imbalance.
- Saves experimental checkpoint to experiments/tamil_brahmi_finetune/checkpoints/
- Strictly leaves production files and active checkpoint untouched.
"""

import sys
import os
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_deepscript_model
from preprocessing.transforms import get_train_transforms, get_eval_transforms

EXP_DIR = PROJECT_ROOT / "experiments" / "tamil_brahmi_finetune"
CHECKPOINT_DIR = EXP_DIR / "checkpoints"
LOG_DIR = EXP_DIR / "logs"
MANIFEST_CSV = EXP_DIR / "experimental_manifest.csv"
ACTIVE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "six_class_vit" / "best_vit_model.pth"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["brahmi", "grantha", "gupta", "kadamba", "kharosthi", "tamil_brahmi"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InscriptionDataset(Dataset):
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


def prepare_data() -> Tuple[List[Tuple[Path, str, int]], List[Tuple[Path, str, int]]]:
    df = pd.read_csv(MANIFEST_CSV)
    train_samples = []
    val_samples = []

    for _, row in df.iterrows():
        p = PROJECT_ROOT / row["image_path"]
        script = row["script"]
        label = CLASS_TO_IDX[script]
        split = row["split"]

        if not p.exists():
            print(f"Warning: file not found: {p}")
            continue

        if split == "train":
            train_samples.append((p, script, label))
        elif split == "val":
            val_samples.append((p, script, label))

    print(f"Loaded samples from {MANIFEST_CSV}:")
    print(f"  • Train samples: {len(train_samples)}")
    print(f"  • Val samples  : {len(val_samples)}")
    return train_samples, val_samples


def run_training():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"DeepScript — Isolated Tamil-Brahmi Fine-Tuning")
    print(f"Device: {device}")
    print(f"Initializing from: {ACTIVE_CHECKPOINT}")
    print(f"=======================================================\n")

    train_samples, val_samples = prepare_data()

    train_labels = [label for _, _, label in train_samples]
    class_counts = torch.bincount(torch.tensor(train_labels), minlength=len(CLASSES))
    print("\n--- Per-Class Training Sample Distribution ---")
    for idx, c_name in enumerate(CLASSES):
        count = class_counts[idx].item()
        print(f"  • {c_name:15s} (idx {idx}): {count:2d} samples ({count/len(train_labels)*100:.1f}%)")

    # WeightedRandomSampler for class balance
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    print("\nInitialized WeightedRandomSampler for uniform 6-class representation per batch.")

    train_tf = get_train_transforms(image_size=(224, 224), enable_augmentation=True)
    eval_tf = get_eval_transforms(image_size=(224, 224))

    train_ds = InscriptionDataset(train_samples, transform=train_tf)
    val_ds = InscriptionDataset(val_samples, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Initialize model matching exact 6-class architecture
    model = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=6,
        embedding_dim=256,
        head_type="cosine",
        pretrained=False,
        freeze_backbone=True,
    )

    # Load active checkpoint weights
    print(f"Loading weights from active checkpoint: {ACTIVE_CHECKPOINT}")
    ckpt = torch.load(ACTIVE_CHECKPOINT, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)

    # Freeze backbone completely
    model.freeze_backbone()
    trainable_backbone = [n for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    assert len(trainable_backbone) == 0, f"Backbone parameters are trainable: {trainable_backbone}"
    head_params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"Backbone verified: 100% FROZEN (0 trainable params).")
    print(f"Trainable parameters in Projector + Cosine Head: {sum(p.numel() for p in head_params):,}")

    criterion = nn.CrossEntropyLoss()

    total_epochs = 8
    lr = 2e-4
    optimizer = AdamW(head_params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    saved_ckpt_path = CHECKPOINT_DIR / "best_finetuned_model.pth"

    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "scale": [],
    }

    t0_start = time.time()

    for epoch in range(1, total_epochs + 1):
        t_epoch_start = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
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
        scale_val = getattr(model.head, "scale", torch.tensor(1.0)).item()
        epoch_sec = time.time() - t_epoch_start

        print(
            f"Epoch {epoch:02d}/{total_epochs:02d} ({epoch_sec:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.1f}% | Scale: {scale_val:.2f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_acc, 4))
        history["scale"].append(round(scale_val, 4))

        # Select model based on validation loss / accuracy
        if v_loss < best_val_loss or (v_loss == best_val_loss and v_acc >= best_val_acc):
            best_val_loss = v_loss
            best_val_acc = v_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_training_sec = time.time() - t0_start
    print(f"\nTraining completed in {total_training_sec:.1f}s.")
    print(f"Best Validation Epoch: {best_epoch:02d} (Loss: {best_val_loss:.4f}, Acc: {best_val_acc*100:.1f}%)")

    # Save isolated experimental checkpoint
    assert best_state is not None, "Error: No model state was saved."
    torch.save(
        {
            "epoch": best_epoch,
            "stage": "tamil_brahmi_finetune",
            "model_state_dict": best_state,
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
            "classes": CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "history": history,
            "training_time_sec": total_training_sec,
            "parent_checkpoint": str(ACTIVE_CHECKPOINT),
        },
        saved_ckpt_path,
    )
    print(f"Successfully saved experimental checkpoint to: {saved_ckpt_path}")

    # Save training log JSON
    with open(LOG_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training log to: {LOG_DIR / 'training_history.json'}")


if __name__ == "__main__":
    run_training()
