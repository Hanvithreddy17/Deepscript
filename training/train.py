"""
DeepScript — Vision Transformer Supervised Training & Fine-Tuning Script
========================================================================
Executes supervised training and transfer learning for ViT-B/16 on the
ancient script dataset (62 classes).

Training Strategy:
1. Stage 1 (Head Warmup): ViT backbone is frozen. Only embedding projector
   and classification head parameters are trained.
2. Stage 2 (Fine-Tuning): The last N transformer encoder blocks are unfrozen
   and trained with a smaller differential learning rate.
"""

import sys
import os
import random
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Reconfigure stdout for UTF-8 compatibility
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import (
    AncientScriptDataset,
    get_train_transforms,
    get_eval_transforms,
    create_stratified_splits,
    create_dataloaders,
)
from models import get_deepscript_model
from training.trainer import Trainer


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads YAML experiment configuration file if it exists."""
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_optimizer_and_scheduler(
    model: nn.Module,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    total_epochs: int = 10,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Builds an AdamW optimizer with differential learning rates for the
    backbone vs. head/projector, paired with a CosineAnnealingLR scheduler.
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "name": "backbone",
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
            "name": "head_and_projector",
        })

    optimizer = AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=1e-6)

    return optimizer, scheduler


def train_deepscript(
    dataset_root: str = "dataset/dataset",
    backbone_name: str = "vit_b_16",
    head_type: str = "cosine",
    embedding_dim: int = 256,
    batch_size: int = 32,
    warmup_epochs: int = 3,
    finetune_epochs: int = 3,
    head_lr: float = 1e-3,
    backbone_lr: float = 2e-5,
    weight_decay: float = 1e-4,
    unfreeze_blocks: int = 2,
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "results/training",
    random_seed: int = 42,
    device_name: str = "auto",
    num_workers: int = 0,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    print_freq: int = 25,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Executes the two-stage transfer learning pipeline:
    1. Warmup training on projector & head (backbone frozen).
    2. Fine-tuning on last N transformer blocks with differential learning rate.
    """
    # 0. Reproducibility & Device Initialization
    set_seed(random_seed)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    training_start_time = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 75, flush=True)
    print(" DEEPSCRIPT: ViT-B/16 TRANSFER LEARNING & FINE-TUNING PIPELINE", flush=True)
    print(f" Start Timestamp      : {start_timestamp}", flush=True)
    print(f" Execution Device     : {device}", flush=True)
    print(f" Random Seed          : {random_seed} (Deterministic)", flush=True)
    print(f" Backbone Architecture: {backbone_name}", flush=True)
    print(f" Head Type            : {head_type}", flush=True)
    print(f" Embedding Dimension  : {embedding_dim}", flush=True)
    print(f" Batch Size           : {batch_size}", flush=True)
    print(f" Warmup Epochs        : {warmup_epochs} (Backbone Frozen)", flush=True)
    print(f" Fine-Tuning Epochs   : {finetune_epochs} (Unfreezing last {unfreeze_blocks} ViT blocks)", flush=True)
    print(f" Checkpoint Output    : {checkpoint_dir}", flush=True)
    print(f" Results Output       : {output_dir}", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Dataset and Create Stratified Splits
    print("\n[Step 1/4] Initializing Dataset and Stratified DataLoaders...", flush=True)
    raw_dataset = AncientScriptDataset(root_dir=dataset_root, transform=None)
    classes = raw_dataset.get_classes()
    class_to_idx = raw_dataset.get_class_to_idx()
    num_classes = len(classes)

    train_tf = get_train_transforms(image_size=(224, 224), enable_augmentation=True)
    eval_tf = get_eval_transforms(image_size=(224, 224))

    train_ds, val_ds, test_ds = create_stratified_splits(
        dataset=raw_dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_seed,
        train_transform=train_tf,
        eval_transform=eval_tf,
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=num_workers
    )

    print(f"  • Total Classes   : {num_classes}", flush=True)
    print(f"  • Training Size   : {len(train_ds)} samples ({len(train_loader)} batches)", flush=True)
    print(f"  • Validation Size : {len(val_ds)} samples ({len(val_loader)} batches)", flush=True)
    print(f"  • Test Size       : {len(test_ds)} samples (Isolated for evaluation)", flush=True)

    # 2. Instantiate DeepScript Model
    print("\n[Step 2/4] Initializing DeepScript Model with Pretrained ViT-B/16...", flush=True)
    model = get_deepscript_model(
        backbone_name=backbone_name,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        head_type=head_type,
        pretrained=True,
        freeze_backbone=True,  # Initially frozen
    )

    summary = model.get_parameter_summary()
    print(f"  • Total Parameters    : {summary['total_parameters']:,}", flush=True)
    print(f"  • Trainable Parameters: {summary['trainable_parameters']:,} (Projector + Head)", flush=True)
    print(f"  • Frozen Parameters   : {summary['frozen_parameters']:,} (ViT Backbone)", flush=True)

    # 3. Master Training Loop
    criterion = nn.CrossEntropyLoss()
    total_history = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top3_acc": [],
        "lr_head": [],
        "lr_backbone": [],
        "epoch_time_sec": [],
    }

    best_val_acc = 0.0
    best_epoch = 0

    # Stage 1: Head Warmup (Backbone Frozen)
    if warmup_epochs > 0:
        print("\n" + "-" * 75, flush=True)
        print(f" STAGE 1: HEAD WARMUP ({warmup_epochs} Epochs, Backbone Frozen, Head LR={head_lr})", flush=True)
        print("-" * 75, flush=True)

        optimizer_s1, scheduler_s1 = build_optimizer_and_scheduler(
            model,
            head_lr=head_lr,
            backbone_lr=0.0,
            weight_decay=weight_decay,
            total_epochs=warmup_epochs,
        )

        trainer_stage1 = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_s1,
            scheduler=scheduler_s1,
            device=device,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name="best_vit_model.pth",
            classes=classes,
            class_to_idx=class_to_idx,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
            print_freq=print_freq,
            stage_name="Stage 1: Head Warmup",
        )

        history_s1 = trainer_stage1.fit(num_epochs=warmup_epochs, epoch_offset=0)
        for k in total_history:
            total_history[k].extend(history_s1[k])

        best_val_acc = trainer_stage1.best_val_acc
        best_epoch = trainer_stage1.best_epoch

    # Stage 2: Fine-Tuning (Unfreeze last N blocks)
    if finetune_epochs > 0:
        print("\n" + "-" * 75, flush=True)
        print(
            f" STAGE 2: FINE-TUNING ({finetune_epochs} Epochs, Unfreezing Last {unfreeze_blocks} Blocks, "
            f"Backbone LR={backbone_lr}, Head LR={head_lr / 5.0})",
            flush=True,
        )
        print("-" * 75, flush=True)

        # Unfreeze last N transformer blocks
        model.unfreeze_last_n_blocks(n=unfreeze_blocks)
        summary_ft = model.get_parameter_summary()
        print(f"  • Trainable Parameters: {summary_ft['trainable_parameters']:,}", flush=True)
        print(f"  • Frozen Parameters   : {summary_ft['frozen_parameters']:,}", flush=True)

        # Setup optimizer with differential LR
        optimizer_ft, scheduler_ft = build_optimizer_and_scheduler(
            model,
            head_lr=head_lr / 5.0,  # Scaled down for stable fine-tuning
            backbone_lr=backbone_lr,
            weight_decay=weight_decay,
            total_epochs=finetune_epochs,
        )

        trainer_stage2 = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_ft,
            scheduler=scheduler_ft,
            device=device,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name="best_vit_model.pth",
            classes=classes,
            class_to_idx=class_to_idx,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
            print_freq=print_freq,
            stage_name="Stage 2: Fine-Tuning",
        )

        # Carry over best val tracking from Stage 1
        if warmup_epochs > 0:
            trainer_stage2.best_val_acc = best_val_acc
            trainer_stage2.best_epoch = best_epoch

        history_s2 = trainer_stage2.fit(num_epochs=finetune_epochs, epoch_offset=warmup_epochs)
        for k in total_history:
            total_history[k].extend(history_s2[k])

        best_val_acc = trainer_stage2.best_val_acc
        best_epoch = trainer_stage2.best_epoch

    total_duration_sec = round(time.time() - training_start_time, 2)

    # 4. Export Training History and Metadata
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path = out_dir / "training_history.csv"

    # Write CSV history
    fieldnames = [
        "epoch",
        "stage",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_top3_acc",
        "lr_head",
        "lr_backbone",
        "epoch_time_sec",
    ]
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(len(total_history["epoch"])):
            writer.writerow([total_history[k][i] for k in fieldnames])
    print(f"\n  >>> Saved training history CSV to: {history_csv_path}", flush=True)

    # Write Training Metadata JSON
    metadata = {
        "experiment_name": "vit_b16_transfer_learning_full_run",
        "timestamp_start": start_timestamp,
        "total_duration_sec": total_duration_sec,
        "device": str(device),
        "random_seed": random_seed,
        "num_classes": num_classes,
        "samples": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "model": {
            "backbone": backbone_name,
            "feature_dim": 768,
            "embedding_dim": embedding_dim,
            "head": head_type,
            "unfreeze_blocks": unfreeze_blocks,
        },
        "hyperparameters": {
            "batch_size": batch_size,
            "warmup_epochs": warmup_epochs,
            "finetune_epochs": finetune_epochs,
            "head_lr": head_lr,
            "backbone_lr": backbone_lr,
            "weight_decay": weight_decay,
        },
        "best_results": {
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "best_checkpoint": str(Path(checkpoint_dir) / "best_vit_model.pth"),
        },
    }

    metadata_path = Path(checkpoint_dir) / "training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  >>> Saved training metadata to: {metadata_path}", flush=True)

    # Also duplicate metadata into results/training
    results_meta_path = out_dir / "training_metadata.json"
    with open(results_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 75, flush=True)
    print(" ALL TRAINING PHASES FINISHED SUCCESSFULLY!", flush=True)
    print(f" Total Duration            : {total_duration_sec:.1f}s", flush=True)
    print(f" Best Validation Top-1 Acc : {best_val_acc * 100:.2f}% (Epoch {best_epoch})", flush=True)
    print(f" Best Checkpoint Saved At  : {Path(checkpoint_dir) / 'best_vit_model.pth'}", flush=True)
    print("=" * 75, flush=True)

    return model, total_history


def main():
    parser = argparse.ArgumentParser(description="DeepScript: ViT-B/16 Supervised Training")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to YAML configuration")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to dataset root")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone vision model")
    parser.add_argument("--head", type=str, default=None, choices=["cosine", "linear"], help="Classifier head")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--warmup_epochs", type=int, default=None, help="Head warmup epochs (backbone frozen)")
    parser.add_argument("--finetune_epochs", type=int, default=None, help="Fine-tuning epochs (last N blocks unfrozen)")
    parser.add_argument("--head_lr", type=float, default=None, help="Head learning rate")
    parser.add_argument("--backbone_lr", type=float, default=None, help="Backbone fine-tuning learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--unfreeze_blocks", type=int, default=None, help="Number of transformer blocks to unfreeze")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Output checkpoint folder")
    parser.add_argument("--output_dir", type=str, default=None, help="Output training results folder")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"], help="Execution device")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for data splitting & init")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Cap training batches (for testing)")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Cap validation batches (for testing)")
    parser.add_argument("--print_freq", type=int, default=25, help="Batch print frequency")

    args = parser.parse_args()

    # Load configuration from YAML
    yaml_cfg = load_yaml_config(args.config) if args.config else {}

    # Extract defaults from YAML or fallbacks
    ds_root = args.dataset_root or yaml_cfg.get("dataset", {}).get("root_dir", "dataset/dataset")
    backbone = args.backbone or yaml_cfg.get("model", {}).get("backbone_name", "vit_b_16")
    head = args.head or yaml_cfg.get("model", {}).get("head", {}).get("type", "cosine")
    emb_dim = args.embedding_dim or yaml_cfg.get("model", {}).get("embedding_dim", 256)
    bs = args.batch_size or yaml_cfg.get("training", {}).get("batch_size", 32)

    w_epochs = args.warmup_epochs
    if w_epochs is None:
        w_epochs = yaml_cfg.get("training", {}).get("stage_1_warmup", {}).get("epochs", 3)

    ft_epochs = args.finetune_epochs
    if ft_epochs is None:
        ft_epochs = yaml_cfg.get("training", {}).get("stage_2_finetune", {}).get("epochs", 3)

    h_lr = args.head_lr
    if h_lr is None:
        h_lr = yaml_cfg.get("training", {}).get("stage_1_warmup", {}).get("head_lr", 1e-3)

    b_lr = args.backbone_lr
    if b_lr is None:
        b_lr = yaml_cfg.get("training", {}).get("stage_2_finetune", {}).get("backbone_lr", 2e-5)

    wd = args.weight_decay
    if wd is None:
        wd = yaml_cfg.get("training", {}).get("stage_1_warmup", {}).get("weight_decay", 1e-4)

    unfreeze = args.unfreeze_blocks
    if unfreeze is None:
        unfreeze = yaml_cfg.get("training", {}).get("stage_2_finetune", {}).get("unfreeze_blocks", 2)

    ckpt_dir = args.checkpoint_dir or yaml_cfg.get("checkpoint", {}).get("checkpoint_dir", "checkpoints")
    out_dir = args.output_dir or yaml_cfg.get("output", {}).get("training_output_dir", "results/training")
    dev = args.device or yaml_cfg.get("experiment", {}).get("device", "auto")
    seed = args.seed if args.seed is not None else yaml_cfg.get("experiment", {}).get("random_seed", 42)

    max_t_batches = args.max_train_batches
    if max_t_batches is None:
        max_t_batches = yaml_cfg.get("training", {}).get("stage_1_warmup", {}).get("max_train_batches", None)

    max_v_batches = args.max_val_batches
    if max_v_batches is None:
        max_v_batches = yaml_cfg.get("training", {}).get("stage_1_warmup", {}).get("max_val_batches", None)

    train_deepscript(
        dataset_root=ds_root,
        backbone_name=backbone,
        head_type=head,
        embedding_dim=emb_dim,
        batch_size=bs,
        warmup_epochs=w_epochs,
        finetune_epochs=ft_epochs,
        head_lr=h_lr,
        backbone_lr=b_lr,
        weight_decay=wd,
        unfreeze_blocks=unfreeze,
        checkpoint_dir=ckpt_dir,
        output_dir=out_dir,
        random_seed=seed,
        device_name=dev,
        max_train_batches=max_t_batches,
        max_val_batches=max_v_batches,
        print_freq=args.print_freq,
    )


if __name__ == "__main__":
    main()
