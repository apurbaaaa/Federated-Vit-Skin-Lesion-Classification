#!/usr/bin/env python3
"""
ISIC 2019 Skin Lesion Classification Training Script.

Simplified pipeline with precomputed segmentation masks:
    - Two-stage training (head-only → full fine-tuning)
    - Layer-wise learning rate decay (LLRD)
    - MixUp + CutMix augmentation
    - Asymmetric Loss for class imbalance
    - Warmup + Cosine LR scheduler
    - Gradient clipping

Usage:
    python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data import (
    VALID_CLASSES,
    NUM_CLASSES,
    build_dataloaders,
    compute_class_weights,
    load_isic_data,
    print_class_distribution,
)
from model import build_model
from losses import build_classification_loss
from utils import (
    Mixup_Cutmix,
    WarmupCosineScheduler,
    clip_grad_norm,
    get_device,
    load_config,
    mixup_criterion,
    mps_empty_cache,
    mps_sync,
    save_checkpoint,
    seed_everything,
)


def setup_logging(log_file: Optional[str] = None) -> None:
    """Configure logging to stdout and optionally a file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )
    # Redirect print → logging so everything goes to the file too
    import builtins
    _original_print = builtins.print

    def _logged_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        logging.info(msg)

    builtins.print = _logged_print


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    mixup_cutmix: Optional[Mixup_Cutmix] = None,
) -> float:
    """Train for one epoch.
    
    Returns
    -------
    float
        Average loss.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(loader)
    
    train_cfg = config.get("training", {})
    max_grad_norm = float(train_cfg.get("grad_clip", 1.0))
    use_metadata = config.get("model", {}).get("metadata", {}).get("enabled", False)
    
    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        
        # Metadata
        metadata = None
        if use_metadata and "metadata" in batch:
            metadata = {
                "age": batch["metadata"]["age"].to(device),
                "sex": batch["metadata"]["sex"].to(device),
                "site": batch["metadata"]["site"].to(device),
            }
        
        # Apply MixUp/CutMix
        if mixup_cutmix is not None:
            images, labels_a, labels_b, lam = mixup_cutmix(images, labels)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images, metadata=metadata)
        logits = outputs["logits"]
        
        # Compute loss
        if mixup_cutmix is not None:
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
        else:
            loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm > 0:
            clip_grad_norm(model.parameters(), max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        running_loss += loss.item()
        
        # Logging
        if (batch_idx + 1) % 50 == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx + 1}/{num_batches}  loss={loss.item():.4f}")
        
        # MPS sync
        if (batch_idx + 1) % 25 == 0:
            mps_sync()
    
    mps_sync()
    mps_empty_cache()
    
    return running_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict,
) -> Dict:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    
    use_metadata = config.get("model", {}).get("metadata", {}).get("enabled", False)
    
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        
        metadata = None
        if use_metadata and "metadata" in batch:
            metadata = {
                "age": batch["metadata"]["age"].to(device),
                "sex": batch["metadata"]["sex"].to(device),
                "site": batch["metadata"]["site"].to(device),
            }
        
        outputs = model(images, metadata=metadata)
        logits = outputs["logits"]
        
        # Simple CE loss for evaluation
        loss = nn.functional.cross_entropy(logits, labels)
        running_loss += loss.item() * images.size(0)
        
        mps_sync()
        
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
    
    total = len(all_labels)
    avg_loss = running_loss / max(total, 1)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    # Confusion matrix (8x8)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    
    # Per-class recall
    per_class_recall = []
    for i in range(NUM_CLASSES):
        if cm[i].sum() > 0:
            recall = cm[i, i] / cm[i].sum()
        else:
            recall = 0.0
        per_class_recall.append(recall)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_recall": per_class_recall,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def print_metrics(metrics: Dict, split_name: str = "Val") -> None:
    """Print evaluation metrics."""
    print(f"\n[{split_name}] Metrics:")
    print(f"  Loss             : {metrics['loss']:.4f}")
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    
    print(f"\n[{split_name}] Per-class Recall:")
    for idx, name in enumerate(VALID_CLASSES):
        print(f"  {name:5s}: {metrics['per_class_recall'][idx]:.4f}")


def print_confusion_matrix(cm: np.ndarray) -> None:
    """Print confusion matrix."""
    header = "      " + " ".join(f"{name:>5s}" for name in VALID_CLASSES)
    print(f"\nConfusion Matrix (8x8):\n{header}")
    for i, name in enumerate(VALID_CLASSES):
        row = " ".join(f"{cm[i, j]:5d}" for j in range(NUM_CLASSES))
        print(f"  {name:5s}: {row}")


# ============================================================================
# Training
# ============================================================================

def train(config: dict, resume_path: Optional[str] = None) -> Dict:
    """Main training function.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    resume_path : str, optional
        Path to checkpoint to resume Stage 2 from.
    """
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    aug_cfg = config.get("augmentation", {})
    ckpt_cfg = config.get("checkpoint", {})
    
    # Device
    device = get_device(config.get("device", "auto"))
    print(f"\nDevice: {device}")
    
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    train_df, val_df, test_df = load_isic_data(
        isic_dir=data_cfg.get("isic_dir", "./ISIC"),
        val_ratio=float(train_cfg.get("val_ratio", 0.2)),
        random_state=config.get("seed", 42),
    )
    
    print_class_distribution(train_df, "Train")
    print_class_distribution(val_df, "Val")
    print_class_distribution(test_df, "Test")
    
    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, config
    )
    
    # Class weights
    class_weights = compute_class_weights(train_df["label"].tolist())
    class_weights = class_weights.to(device)
    
    print("\n[Data] Class weights:")
    for idx, name in enumerate(VALID_CLASSES):
        print(f"  {name:5s}: {class_weights[idx].item():.4f}")
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val  : {len(val_df):,}")
    print(f"  Test : {len(test_df):,}")
    
    # =========================================================================
    # Model Setup
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL SETUP")
    print("=" * 70)
    
    model = build_model(config).to(device)
    
    counts = model.count_parameters()
    print(f"Total parameters    : {counts['total']:,}")
    print(f"Backbone parameters : {counts['backbone']:,}")
    print(f"Classifier params   : {counts['classifier']:,}")
    if "metadata" in counts:
        print(f"Metadata params     : {counts['metadata']:,}")
    
    # =========================================================================
    # Loss Function (Asymmetric Loss only)
    # =========================================================================
    criterion = build_classification_loss(
        loss_type=loss_cfg.get("type", "asymmetric"),
        class_weights=class_weights if loss_cfg.get("class_weights", True) else None,
        gamma_neg=float(loss_cfg.get("asymmetric", {}).get("gamma_neg", 4)),
        gamma_pos=float(loss_cfg.get("asymmetric", {}).get("gamma_pos", 1)),
        clip=float(loss_cfg.get("asymmetric", {}).get("clip", 0.05)),
        focal_gamma=float(loss_cfg.get("focal", {}).get("gamma", 2.0)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
    )
    print(f"\nLoss: {loss_cfg.get('type', 'asymmetric')}")
    
    # =========================================================================
    # Augmentation
    # =========================================================================
    mixup_cutmix = None
    if aug_cfg.get("mixup", {}).get("enabled", True) or aug_cfg.get("cutmix", {}).get("enabled", True):
        mixup_cutmix = Mixup_Cutmix(
            mixup_alpha=float(aug_cfg.get("mixup", {}).get("alpha", 0.2)),
            cutmix_alpha=float(aug_cfg.get("cutmix", {}).get("alpha", 1.0)),
            cutmix_prob=float(aug_cfg.get("cutmix", {}).get("prob", 0.5)),
        )
        print(f"MixUp+CutMix: mixup_alpha={aug_cfg.get('mixup', {}).get('alpha', 0.2)}, "
              f"cutmix_alpha={aug_cfg.get('cutmix', {}).get('alpha', 1.0)}")
    
    # Best tracking
    best_val_metric = float("inf")
    best_epoch = 0
    resume_epoch = 0  # Stage-2-local epoch to resume from (0 = start fresh)
    
    # Checkpoint directory
    ckpt_dir = Path(ckpt_cfg.get("dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Resume from checkpoint (skip Stage 1, continue Stage 2)
    # =========================================================================
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_metric = ckpt.get("best_metric", float("inf"))
        best_epoch = ckpt.get("epoch", 0)
        s1_epochs_done = int(train_cfg.get("stage1", {}).get("epochs", 5))
        resume_epoch = best_epoch - s1_epochs_done  # local Stage 2 epoch
        print(f"\n[Resume] Loaded checkpoint from epoch {best_epoch} "
              f"(Stage 2 local epoch {resume_epoch})")
        print(f"[Resume] Best val_loss so far: {best_val_metric:.4f}")
        print(f"[Resume] Skipping Stage 1, resuming Stage 2 from epoch {resume_epoch + 1}")
    
    # =========================================================================
    # Stage 1: Head-Only Training
    # =========================================================================
    stage1_cfg = train_cfg.get("stage1", {})
    stage1_epochs = int(stage1_cfg.get("epochs", 5))
    
    if stage1_epochs > 0 and not resume_path:
        print("\n" + "=" * 70)
        print("STAGE 1: HEAD-ONLY TRAINING")
        print("=" * 70)
        
        model.freeze_backbone()
        
        head_params = model.get_head_parameters()
        optimizer_s1 = AdamW(
            head_params,
            lr=float(stage1_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        )
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[Stage 1] Trainable parameters: {trainable:,}")
        print(f"[Stage 1] LR: {float(stage1_cfg.get('lr', 1e-3))}")
        
        for epoch in range(1, stage1_epochs + 1):
            start_time = time.time()
            
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer_s1,
                device=device,
                epoch=epoch,
                config=config,
                mixup_cutmix=mixup_cutmix,
            )
            
            val_metrics = evaluate(model, val_loader, device, config)
            elapsed = time.time() - start_time
            
            print(f"\n[Stage 1] Epoch [{epoch}/{stage1_epochs}]  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_metrics['loss']:.4f}  "
                  f"val_acc={val_metrics['accuracy']:.4f}  "
                  f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}  "
                  f"time={elapsed:.1f}s")
            
            if val_metrics["loss"] < best_val_metric:
                best_val_metric = val_metrics["loss"]
                best_epoch = epoch
                
                save_checkpoint(
                    model, optimizer_s1, None, None, epoch, best_val_metric,
                    str(ckpt_dir / "best_model.pt"),
                    config=config,
                )
                print(f"  ✓ Saved best checkpoint (val_loss={best_val_metric:.4f})")
    
    # =========================================================================
    # Stage 2: Full Fine-Tuning
    # =========================================================================
    stage2_cfg = train_cfg.get("stage2", {})
    stage2_epochs = int(stage2_cfg.get("epochs", 25))
    
    print("\n" + "=" * 70)
    print("STAGE 2: FULL MODEL FINE-TUNING")
    print("=" * 70)
    
    model.unfreeze_backbone()
    
    # Layer-wise LR decay
    llrd_cfg = train_cfg.get("llrd", {})
    if llrd_cfg.get("enabled", True):
        param_groups = model.get_layerwise_lr_groups(
            base_lr=float(stage2_cfg.get("lr", 5e-5)),
            decay_rate=float(llrd_cfg.get("decay_rate", 0.75)),
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        )
        optimizer_s2 = AdamW(param_groups)
        print(f"\n[Stage 2] Layer-wise LR decay (rate={float(llrd_cfg.get('decay_rate', 0.75))})")
    else:
        optimizer_s2 = AdamW(
            model.parameters(),
            lr=float(stage2_cfg.get("lr", 5e-5)),
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        )
    
    # Scheduler
    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = WarmupCosineScheduler(
        optimizer_s2,
        warmup_epochs=int(sched_cfg.get("warmup_epochs", 3)),
        total_epochs=stage2_epochs,
        min_lr=float(sched_cfg.get("min_lr", 1e-7)),
    )
    
    # Fast-forward scheduler to resume point
    if resume_path and resume_epoch > 0:
        for _ in range(resume_epoch):
            scheduler.step()
        print(f"[Resume] Scheduler fast-forwarded {resume_epoch} steps  "
              f"(current lr={scheduler.get_last_lr()[0]:.2e})")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage 2] Trainable parameters: {trainable:,}")
    print(f"[Stage 2] Base LR: {float(stage2_cfg.get('lr', 5e-5))}")
    print(f"[Stage 2] Scheduler: Warmup({sched_cfg.get('warmup_epochs', 3)}) + Cosine")
    
    start_s2_epoch = resume_epoch + 1 if resume_path else 1
    for epoch in range(start_s2_epoch, stage2_epochs + 1):
        global_epoch = stage1_epochs + epoch
        start_time = time.time()
        
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer_s2,
            device=device,
            epoch=global_epoch,
            config=config,
            mixup_cutmix=mixup_cutmix,
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        val_metrics = evaluate(model, val_loader, device, config)
        elapsed = time.time() - start_time
        
        print(f"\n[Stage 2] Epoch [{epoch}/{stage2_epochs}] (Global {global_epoch})  "
              f"lr={current_lr:.2e}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  "
              f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}  "
              f"time={elapsed:.1f}s")
        
        if val_metrics["loss"] < best_val_metric:
            best_val_metric = val_metrics["loss"]
            best_epoch = global_epoch
            
            save_checkpoint(
                model, optimizer_s2, scheduler, None, global_epoch, best_val_metric,
                str(ckpt_dir / "best_model.pt"),
                config=config,
            )
            print(f"  ✓ Saved best checkpoint (val_loss={best_val_metric:.4f})")
    
    # =========================================================================
    # Final Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Load best checkpoint
    ckpt_path = ckpt_dir / "best_model.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
    
    # Standard evaluation
    test_metrics = evaluate(model, test_loader, device, config)
    
    print_metrics(test_metrics, "Test")
    print_confusion_matrix(test_metrics["confusion_matrix"])
    
    print("\n[Test] Classification Report:")
    print(classification_report(
        test_metrics["all_labels"],
        test_metrics["all_preds"],
        target_names=VALID_CLASSES,
        zero_division=0,
    ))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_metric:.4f} (epoch {best_epoch})")
    print(f"\nTest Results:")
    print(f"  Accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1         : {test_metrics['macro_f1']:.4f}")
    
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_metric,
        "test_metrics": test_metrics,
        "checkpoint_path": str(ckpt_path),
    }


# ============================================================================
# Main
# ============================================================================

def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    # Setup logging
    setup_logging(args.log)
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI args
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.epochs:
        config.setdefault("training", {}).setdefault("stage2", {})["epochs"] = args.epochs
    
    # Seed
    seed_everything(config.get("seed", 42))
    
    # Train
    results = train(config, resume_path=args.resume)
    
    print("\n\n✓ Training complete!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISIC 2019 Skin Lesion Classification"
    )
    
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override stage 2 epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path (skips Stage 1)")
    parser.add_argument("--log", type=str, default=None,
                        help="Log file path (logs to both stdout and file)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
