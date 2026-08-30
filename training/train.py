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
import argparse
from pathlib import Path
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


def build_optimizer_and_scheduler(
    model: nn.Module,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    total_epochs: int = 10,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
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
    warmup_epochs: int = 2,
    finetune_epochs: int = 2,
    head_lr: float = 1e-3,
    backbone_lr: float = 2e-5,
    unfreeze_blocks: int = 2,
    checkpoint_dir: str = "checkpoints",
    random_seed: int = 42,
    num_workers: int = 0,
    max_train_batches: int = None,
    max_val_batches: int = None,
    print_freq: int = 25,
) -> tuple[nn.Module, dict]:
    """
    Executes the two-stage transfer learning pipeline:
    1. Warmup training on projector & head (backbone frozen).
    2. Fine-tuning on last N transformer blocks with differential learning rate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75, flush=True)
    print(" DEEPSCRIPT: ViT-B/16 TRANSFER LEARNING & FINE-TUNING PIPELINE", flush=True)
    print(f" Execution Device     : {device}", flush=True)
    print(f" Backbone Architecture: {backbone_name}", flush=True)
    print(f" Head Type            : {head_type}", flush=True)
    print(f" Embedding Dimension  : {embedding_dim}", flush=True)
    print(f" Batch Size           : {batch_size}", flush=True)
    print(f" Warmup Epochs        : {warmup_epochs} (Backbone Frozen)", flush=True)
    print(f" Fine-Tuning Epochs   : {finetune_epochs} (Unfreezing last {unfreeze_blocks} ViT blocks)", flush=True)
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

    # 3. Stage 1: Head Warmup (Backbone Frozen)
    criterion = nn.CrossEntropyLoss()
    total_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if warmup_epochs > 0:
        print("\n" + "-" * 75, flush=True)
        print(f" STAGE 1: HEAD WARMUP ({warmup_epochs} Epochs, Backbone Frozen, Head LR={head_lr})", flush=True)
        print("-" * 75, flush=True)

        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            head_lr=head_lr,
            backbone_lr=0.0,
            total_epochs=warmup_epochs,
        )

        trainer_stage1 = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name="best_vit_model.pth",
            classes=classes,
            class_to_idx=class_to_idx,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
            print_freq=print_freq,
        )

        history_s1 = trainer_stage1.fit(num_epochs=warmup_epochs)
        for k in total_history:
            total_history[k].extend(history_s1[k])

    # 4. Stage 2: Fine-Tuning (Unfreeze last N blocks)
    if finetune_epochs > 0:
        print("\n" + "-" * 75, flush=True)
        print(
            f" STAGE 2: FINE-TUNING ({finetune_epochs} Epochs, Unfreezing Last {unfreeze_blocks} Blocks, "
            f"Backbone LR={backbone_lr}, Head LR={head_lr / 5})",
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
        )
        # Carry over best val acc
        if warmup_epochs > 0:
            trainer_stage2.best_val_acc = trainer_stage1.best_val_acc
            trainer_stage2.best_val_loss = trainer_stage1.best_val_loss
            trainer_stage2.best_epoch = trainer_stage1.best_epoch

        history_s2 = trainer_stage2.fit(num_epochs=finetune_epochs)
        for k in total_history:
            total_history[k].extend(history_s2[k])

    print("\n" + "=" * 75, flush=True)
    print(" ALL TRAINING PHASES FINISHED SUCCESSFULLY!", flush=True)
    print(f" Best model checkpoint saved to: {Path(checkpoint_dir) / 'best_vit_model.pth'}", flush=True)
    print("=" * 75, flush=True)

    return model, total_history


def main():
    parser = argparse.ArgumentParser(description="DeepScript: ViT-B/16 Supervised Training")
    parser.add_argument("--dataset_root", type=str, default="dataset/dataset", help="Path to dataset root")
    parser.add_argument("--backbone", type=str, default="vit_b_16", help="Backbone vision model")
    parser.add_argument("--head", type=str, default="cosine", choices=["cosine", "linear"], help="Classifier head")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Head warmup epochs (backbone frozen)")
    parser.add_argument("--finetune_epochs", type=int, default=2, help="Fine-tuning epochs (last N blocks unfrozen)")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="Head learning rate")
    parser.add_argument("--backbone_lr", type=float, default=2e-5, help="Backbone fine-tuning learning rate")
    parser.add_argument("--unfreeze_blocks", type=int, default=2, help="Number of transformer blocks to unfreeze")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Output checkpoint folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting & init")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Cap training batches (for quick testing)")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Cap validation batches (for quick testing)")
    parser.add_argument("--print_freq", type=int, default=25, help="Batch print frequency")

    args = parser.parse_args()

    train_deepscript(
        dataset_root=args.dataset_root,
        backbone_name=args.backbone,
        head_type=args.head,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        unfreeze_blocks=args.unfreeze_blocks,
        checkpoint_dir=args.checkpoint_dir,
        random_seed=args.seed,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        print_freq=args.print_freq,
    )


if __name__ == "__main__":
    main()
