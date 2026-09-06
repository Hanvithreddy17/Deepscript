"""
DeepScript — Model Trainer & Fine-Tuning Module
===============================================
Provides a structured, modular trainer for supervised transfer learning and
fine-tuning of Vision Transformer (ViT-B/16) models on ancient script inscriptions.

Key capabilities:
1. Two-stage and differential learning rate fine-tuning.
2. Robust metric tracking (Train/Val Loss, Top-1/Top-3 Accuracy).
3. Checkpoint management saving best models strictly based on validation performance.
4. Clean separation of concern: Test set is NEVER touched during training or selection.
5. Automatic history logging and CSV export.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import time
import csv
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger("deepscript.training.trainer")


class Trainer:
    """
    Supervised Fine-Tuning Trainer for DeepScript models.

    Manages forward/backward passes, metric calculations, learning rate
    scheduling, validation checks, and checkpoint serialization.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        checkpoint_name: str = "best_vit_model.pth",
        classes: Optional[List[str]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None,
        print_freq: int = 25,
        stage_name: str = "Training",
    ):
        """
        Args:
            model: DeepScriptModel instance.
            train_loader: PyTorch DataLoader for the training partition.
            val_loader: PyTorch DataLoader for the validation partition.
            criterion: Loss function (defaults to nn.CrossEntropyLoss()).
            optimizer: PyTorch optimizer (defaults to AdamW).
            scheduler: Learning rate scheduler (optional).
            device: Execution device (cuda / cpu).
            checkpoint_dir: Directory to save model checkpoints.
            checkpoint_name: Filename for the best checkpoint.
            classes: List of class names.
            class_to_idx: Dictionary mapping class names to indices.
            max_train_batches: Optional cap on batches per epoch for rapid verification.
            max_val_batches: Optional cap on val batches.
            print_freq: Frequency of intra-epoch batch progress logs.
            stage_name: Descriptive name for current training stage.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / checkpoint_name
        self.latest_checkpoint_path = self.checkpoint_dir / "latest_vit_model.pth"

        self.classes = classes or []
        self.class_to_idx = class_to_idx or {}
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.print_freq = print_freq
        self.stage_name = stage_name

        self.best_val_acc: float = 0.0
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0

        self.history: Dict[str, List[Any]] = {
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

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Executes one full training epoch over the training dataset.

        Args:
            epoch: Current epoch index.

        Returns:
            Tuple of (average_training_loss, training_accuracy).
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        num_batches = len(self.train_loader)
        if self.max_train_batches:
            num_batches = min(num_batches, self.max_train_batches)

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            if self.max_train_batches and batch_idx >= self.max_train_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass through DeepScript model
            logits = self.model(images)
            loss = self.criterion(logits, targets)

            # Backward pass & gradient optimization
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += batch_size

            if (batch_idx + 1) % self.print_freq == 0 or (batch_idx + 1) == num_batches:
                batch_acc = correct / total if total > 0 else 0.0
                batch_loss = running_loss / total if total > 0 else 0.0
                print(
                    f"    [{self.stage_name}] Batch [{batch_idx+1:03d}/{num_batches:03d}] | "
                    f"Loss: {batch_loss:.4f} | Acc: {batch_acc * 100:.2f}%",
                    flush=True,
                )

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """
        Executes validation on the held-out validation set.

        Args:
            epoch: Current epoch index.

        Returns:
            Tuple of (val_loss, val_top1_acc, val_top3_acc).
        """
        self.model.eval()
        running_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0
        total = 0

        for batch_idx, (images, targets) in enumerate(self.val_loader):
            if self.max_val_batches and batch_idx >= self.max_val_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            # Top-1 accuracy
            preds = logits.argmax(dim=-1)
            correct_top1 += (preds == targets).sum().item()

            # Top-3 accuracy
            k = min(3, logits.size(-1))
            _, top_k_preds = torch.topk(logits, k=k, dim=-1)
            correct_top3 += top_k_preds.eq(targets.view(-1, 1)).any(dim=-1).sum().item()

        val_loss = running_loss / total if total > 0 else 0.0
        val_top1 = correct_top1 / total if total > 0 else 0.0
        val_top3 = correct_top3 / total if total > 0 else 0.0

        return val_loss, val_top1, val_top3

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_acc: float,
        is_best: bool = False,
    ) -> None:
        """
        Serializes model parameters and training state to disk.

        Args:
            epoch: Current epoch number.
            val_loss: Validation loss at this epoch.
            val_acc: Validation accuracy at this epoch.
            is_best: Whether this checkpoint achieves the best validation accuracy so far.
        """
        checkpoint_data = {
            "epoch": epoch,
            "stage": self.stage_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": self.best_val_acc,
            "classes": self.classes,
            "class_to_idx": self.class_to_idx,
            "history": self.history,
        }

        # Always save latest checkpoint
        torch.save(checkpoint_data, self.latest_checkpoint_path)

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint_data, self.checkpoint_path)
            print(
                f"  >>> Saved new best checkpoint to {self.checkpoint_path} (Val Acc: {val_acc * 100:.2f}%)",
                flush=True,
            )

    def fit(
        self,
        num_epochs: int,
        epoch_offset: int = 0,
    ) -> Dict[str, List[Any]]:
        """
        Executes the full training and validation loop for `num_epochs`.

        Args:
            num_epochs: Number of epochs to train in this phase.
            epoch_offset: Starting epoch index offset.

        Returns:
            Dictionary containing historical metrics across all epochs.
        """
        print("=" * 75, flush=True)
        print(f" DEEPSCRIPT: STARTING {self.stage_name.upper()} ({num_epochs} Epochs)", flush=True)
        print(f" Device: {self.device} | Checkpoint Directory: {self.checkpoint_dir}", flush=True)
        print("=" * 75, flush=True)

        for i in range(1, num_epochs + 1):
            epoch = epoch_offset + i
            epoch_start = time.time()

            # Train one epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_top1, val_top3 = self.validate_epoch(epoch)

            # Get learning rates
            lr_backbone = 0.0
            lr_head = 0.0
            if self.optimizer is not None:
                for group in self.optimizer.param_groups:
                    if group.get("name") == "backbone":
                        lr_backbone = group["lr"]
                    else:
                        lr_head = group["lr"]

            # Learning rate scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Track history
            self.history["epoch"].append(epoch)
            self.history["stage"].append(self.stage_name)
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["train_acc"].append(round(train_acc, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_acc"].append(round(val_top1, 4))
            self.history["val_top3_acc"].append(round(val_top3, 4))
            self.history["lr_head"].append(lr_head)
            self.history["lr_backbone"].append(lr_backbone)
            self.history["epoch_time_sec"].append(round(epoch_time, 2))

            # Check if this is the best epoch on validation
            is_best = val_top1 > self.best_val_acc
            if is_best:
                self.best_val_acc = val_top1
                self.best_val_loss = val_loss
                self.best_epoch = epoch

            # Save checkpoints
            self.save_checkpoint(epoch, val_loss, val_top1, is_best=is_best)

            best_flag = "★ BEST" if is_best else ""

            print(
                f"Epoch [{epoch:02d}] ({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_top1 * 100:.2f}% (Top-3: {val_top3 * 100:.2f}%) | "
                f"Head LR: {lr_head:.1e} {best_flag}",
                flush=True,
            )

        print("=" * 75, flush=True)
        print(f" {self.stage_name.upper()} COMPLETE!", flush=True)
        print(f"  • Best Validation Accuracy: {self.best_val_acc * 100:.2f}% (Epoch {self.best_epoch})", flush=True)
        print(f"  • Best Model Saved At     : {self.checkpoint_path}", flush=True)
        print("=" * 75, flush=True)

        return self.history

    def export_history_csv(self, output_path: str | Path) -> None:
        """Exports the recorded training history to a CSV file."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

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

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            num_rows = len(self.history["epoch"])
            for i in range(num_rows):
                row = [self.history[k][i] for k in fieldnames]
                writer.writerow(row)
        print(f"  >>> Training history exported to: {out_file}", flush=True)
