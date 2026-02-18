"""
ISIC 2019 Skin Lesion Classification Training Script.

Hybrid Segmentation-Guided Vision Transformer with:
    - Two-stage training (head-only → full fine-tuning)
    - Layer-wise learning rate decay (LLRD)
    - MixUp + CutMix augmentation
    - EMA
    - Asymmetric Loss for class imbalance
    - Warmup + Cosine LR scheduler
    - Gradient clipping
    - Test-time augmentation (TTA)
    - K-fold support

Usage:
    # Single run
    python train.py --config config.yaml
    
    # K-fold (5 folds)
    python train.py --config config.yaml --kfold --n_folds 5
"""

from __future__ import annotations

import argparse
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
    build_tta_dataloader,
    collate_fn,
    compute_class_weights,
    load_isic_data,
    print_class_distribution,
)
from model import HybridViT, build_model
from losses import build_classification_loss, BCEDiceLoss, JointLoss
from utils import (
    EMA,
    Mixup_Cutmix,
    WarmupCosineScheduler,
    clip_grad_norm,
    evaluate_with_tta,
    get_device,
    load_config,
    mixup_criterion,
    mps_empty_cache,
    mps_sync,
    save_checkpoint,
    seed_everything,
)


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
    ema: Optional[EMA] = None,
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Returns
    -------
    tuple
        (avg_loss, avg_cls_loss)
    """
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    num_batches = len(loader)
    
    train_cfg = config.get("training", {})
    max_grad_norm = train_cfg.get("grad_clip", 1.0)
    use_metadata = config.get("model", {}).get("metadata", {}).get("enabled", False)
    seg_enabled = config.get("model", {}).get("segmentation", {}).get("enabled", False)
    
    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=False)
        labels = batch["label"].to(device, non_blocking=False)
        
        # Metadata
        metadata = None
        if use_metadata and "metadata" in batch:
            metadata = {
                "age": batch["metadata"]["age"].to(device),
                "sex": batch["metadata"]["sex"].to(device),
                "site": batch["metadata"]["site"].to(device),
            }
        
        # Segmentation masks
        masks = None
        if seg_enabled and "mask" in batch:
            masks = batch["mask"].to(device)
        
        # Apply MixUp/CutMix
        if mixup_cutmix is not None:
            images, labels_a, labels_b, lam = mixup_cutmix(images, labels)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images, metadata=metadata)
        logits = outputs["logits"]
        
        # Compute loss
        if mixup_cutmix is not None:
            if isinstance(criterion, JointLoss):
                # For joint loss with mixup
                cls_loss = mixup_criterion(
                    criterion.cls_loss, logits, labels_a, labels_b, lam
                )
                if seg_enabled and "seg_mask" in outputs and masks is not None:
                    seg_loss = criterion.seg_loss(outputs["seg_mask"], masks)
                    loss = cls_loss + criterion.seg_weight * seg_loss
                else:
                    loss = cls_loss
            else:
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                cls_loss = loss
        else:
            if isinstance(criterion, JointLoss):
                seg_pred = outputs.get("seg_mask") if seg_enabled else None
                loss, cls_loss, seg_loss = criterion(logits, labels, seg_pred, masks)
            else:
                loss = criterion(logits, labels)
                cls_loss = loss
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm > 0:
            clip_grad_norm(model.parameters(), max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # EMA update
        if ema is not None:
            ema.update()
        
        running_loss += loss.item()
        running_cls_loss += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
        
        # Logging
        if (batch_idx + 1) % 50 == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx + 1}/{num_batches}  loss={loss.item():.4f}")
        
        # MPS sync
        if (batch_idx + 1) % 25 == 0:
            mps_sync()
    
    mps_sync()
    mps_empty_cache()
    
    return running_loss / num_batches, running_cls_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> Dict:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_logits: List[np.ndarray] = []
    
    use_metadata = config.get("model", {}).get("metadata", {}).get("enabled", False)
    
    for batch in loader:
        images = batch["image"].to(device, non_blocking=False)
        labels = batch["label"].to(device, non_blocking=False)
        
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
        all_logits.append(logits.cpu().numpy())
    
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
        "all_logits": np.concatenate(all_logits, axis=0),
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
# Training Functions
# ============================================================================

def train_single_fold(
    config: dict,
    fold: Optional[int] = None,
    n_folds: int = 5,
) -> Dict:
    """Train single fold.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    fold : Optional[int]
        Fold index (None for no K-fold).
    n_folds : int
        Number of folds.
    
    Returns
    -------
    dict
        Results with metrics and checkpoint path.
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
    
    fold_str = f"_fold{fold}" if fold is not None else ""
    print(f"\n{'='*70}")
    print(f"TRAINING{' FOLD ' + str(fold) if fold is not None else ''}")
    print(f"{'='*70}")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    train_df, val_df, test_df = load_isic_data(
        isic_dir=data_cfg.get("isic_dir", "./ISIC"),
        val_ratio=train_cfg.get("val_ratio", 0.2),
        random_state=config.get("seed", 42),
        fold=fold,
        n_folds=n_folds,
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
    if "segmentation" in counts:
        print(f"Segmentation params : {counts['segmentation']:,}")
    if "metadata" in counts:
        print(f"Metadata params     : {counts['metadata']:,}")
    
    # =========================================================================
    # Loss Function
    # =========================================================================
    seg_enabled = model_cfg.get("segmentation", {}).get("enabled", False)
    
    cls_loss = build_classification_loss(
        loss_type=loss_cfg.get("type", "asymmetric"),
        class_weights=class_weights if loss_cfg.get("class_weights", True) else None,
        gamma_neg=loss_cfg.get("asymmetric", {}).get("gamma_neg", 4),
        gamma_pos=loss_cfg.get("asymmetric", {}).get("gamma_pos", 1),
        clip=loss_cfg.get("asymmetric", {}).get("clip", 0.05),
        focal_gamma=loss_cfg.get("focal", {}).get("gamma", 2.0),
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
    )
    
    if seg_enabled:
        seg_loss = BCEDiceLoss()
        seg_weight = model_cfg.get("segmentation", {}).get("seg_loss_weight", 0.3)
        criterion = JointLoss(cls_loss, seg_loss, seg_weight)
        print(f"\nLoss: {loss_cfg.get('type', 'asymmetric')} + Segmentation (λ={seg_weight})")
    else:
        criterion = cls_loss
        print(f"\nLoss: {loss_cfg.get('type', 'asymmetric')}")
    
    # =========================================================================
    # Augmentation
    # =========================================================================
    mixup_cutmix = None
    if aug_cfg.get("mixup", {}).get("enabled", True) or aug_cfg.get("cutmix", {}).get("enabled", True):
        mixup_cutmix = Mixup_Cutmix(
            mixup_alpha=aug_cfg.get("mixup", {}).get("alpha", 0.2),
            cutmix_alpha=aug_cfg.get("cutmix", {}).get("alpha", 1.0),
            cutmix_prob=aug_cfg.get("cutmix", {}).get("prob", 0.5),
        )
        print(f"MixUp+CutMix: mixup_alpha={aug_cfg.get('mixup', {}).get('alpha', 0.2)}, "
              f"cutmix_alpha={aug_cfg.get('cutmix', {}).get('alpha', 1.0)}")
    
    # =========================================================================
    # EMA
    # =========================================================================
    ema = None
    if train_cfg.get("ema", {}).get("enabled", True):
        ema = EMA(model, decay=train_cfg.get("ema", {}).get("decay", 0.9998))
        print(f"EMA: decay={train_cfg.get('ema', {}).get('decay', 0.9998)}")
    
    # Best tracking
    best_val_metric = float("inf")
    best_epoch = 0
    
    # Checkpoint directory
    ckpt_dir = Path(ckpt_cfg.get("dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Stage 1: Head-Only Training
    # =========================================================================
    stage1_cfg = train_cfg.get("stage1", {})
    stage1_epochs = stage1_cfg.get("epochs", 5)
    
    if stage1_epochs > 0:
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
            
            train_loss, _ = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer_s1,
                device=device,
                epoch=epoch,
                config=config,
                mixup_cutmix=mixup_cutmix,
                ema=ema,
            )
            
            # Evaluate with EMA
            if ema:
                ema.apply_shadow()
            val_metrics = evaluate(model, val_loader, criterion, device, config)
            if ema:
                ema.restore()
            
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
                
                if ema:
                    ema.apply_shadow()
                save_checkpoint(
                    model, optimizer_s1, None, ema, epoch, best_val_metric,
                    str(ckpt_dir / f"best_model{fold_str}.pt"),
                    config=config,
                )
                if ema:
                    ema.restore()
                print(f"  ✓ Saved best checkpoint (val_loss={best_val_metric:.4f})")
    
    # =========================================================================
    # Stage 2: Full Fine-Tuning
    # =========================================================================
    stage2_cfg = train_cfg.get("stage2", {})
    stage2_epochs = stage2_cfg.get("epochs", 30)
    
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
    
    # Re-initialize EMA with all parameters
    if train_cfg.get("ema", {}).get("enabled", True):
        ema = EMA(model, decay=float(train_cfg.get("ema", {}).get("decay", 0.9998)))
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage 2] Trainable parameters: {trainable:,}")
    print(f"[Stage 2] Base LR: {float(stage2_cfg.get('lr', 5e-5))}")
    print(f"[Stage 2] Scheduler: Warmup({sched_cfg.get('warmup_epochs', 3)}) + Cosine")
    
    for epoch in range(1, stage2_epochs + 1):
        global_epoch = stage1_epochs + epoch
        start_time = time.time()
        
        train_loss, _ = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer_s2,
            device=device,
            epoch=global_epoch,
            config=config,
            mixup_cutmix=mixup_cutmix,
            ema=ema,
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Evaluate with EMA
        if ema:
            ema.apply_shadow()
        val_metrics = evaluate(model, val_loader, criterion, device, config)
        if ema:
            ema.restore()
        
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
            
            if ema:
                ema.apply_shadow()
            save_checkpoint(
                model, optimizer_s2, scheduler, ema, global_epoch, best_val_metric,
                str(ckpt_dir / f"best_model{fold_str}.pt"),
                config=config,
            )
            if ema:
                ema.restore()
            print(f"  ✓ Saved best checkpoint (val_loss={best_val_metric:.4f})")
    
    # =========================================================================
    # Final Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Load best checkpoint
    ckpt_path = ckpt_dir / f"best_model{fold_str}.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
    
    # Standard evaluation
    test_metrics = evaluate(model, test_loader, criterion, device, config)
    
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
    # TTA Evaluation
    # =========================================================================
    tta_cfg = config.get("tta", {})
    if tta_cfg.get("enabled", True):
        print("\n" + "=" * 70)
        print("TEST-TIME AUGMENTATION (TTA) EVALUATION")
        print("=" * 70)
        
        tta_loader = build_tta_dataloader(test_df, config)
        use_metadata = model_cfg.get("metadata", {}).get("enabled", False)
        
        tta_preds, tta_labels, tta_logits = evaluate_with_tta(
            model, tta_loader, device, use_metadata
        )
        
        tta_accuracy = accuracy_score(tta_labels, tta_preds)
        tta_balanced_acc = balanced_accuracy_score(tta_labels, tta_preds)
        tta_macro_f1 = f1_score(tta_labels, tta_preds, average="macro", zero_division=0)
        tta_cm = confusion_matrix(tta_labels, tta_preds, labels=list(range(NUM_CLASSES)))
        
        print(f"\n[TTA] Metrics:")
        print(f"  Accuracy         : {tta_accuracy:.4f}")
        print(f"  Balanced Accuracy: {tta_balanced_acc:.4f}")
        print(f"  Macro F1         : {tta_macro_f1:.4f}")
        
        # Update test metrics with TTA
        test_metrics["tta_accuracy"] = tta_accuracy
        test_metrics["tta_balanced_accuracy"] = tta_balanced_acc
        test_metrics["tta_macro_f1"] = tta_macro_f1
        test_metrics["tta_logits"] = tta_logits
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE{' (FOLD ' + str(fold) + ')' if fold is not None else ''}")
    print("=" * 70)
    print(f"Best validation loss: {best_val_metric:.4f} (epoch {best_epoch})")
    print(f"\nTest Results:")
    print(f"  Accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1         : {test_metrics['macro_f1']:.4f}")
    
    if "tta_accuracy" in test_metrics:
        print(f"\nTest Results (TTA):")
        print(f"  Accuracy         : {test_metrics['tta_accuracy']:.4f}")
        print(f"  Balanced Accuracy: {test_metrics['tta_balanced_accuracy']:.4f}")
        print(f"  Macro F1         : {test_metrics['tta_macro_f1']:.4f}")
    
    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_metric,
        "test_metrics": test_metrics,
        "checkpoint_path": str(ckpt_path),
    }


def train_kfold(config: dict, n_folds: int = 5) -> Dict:
    """Train with K-fold cross-validation.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    n_folds : int
        Number of folds.
    
    Returns
    -------
    dict
        Aggregated results across folds.
    """
    print("=" * 70)
    print(f"K-FOLD TRAINING ({n_folds} folds)")
    print("=" * 70)
    
    all_results = []
    all_logits = []
    
    for fold in range(n_folds):
        print(f"\n\n{'#' * 70}")
        print(f"# FOLD {fold + 1} / {n_folds}")
        print(f"{'#' * 70}")
        
        result = train_single_fold(config, fold=fold, n_folds=n_folds)
        all_results.append(result)
        
        if "tta_logits" in result["test_metrics"]:
            all_logits.append(result["test_metrics"]["tta_logits"])
    
    # Aggregate metrics
    print("\n\n" + "=" * 70)
    print("K-FOLD RESULTS SUMMARY")
    print("=" * 70)
    
    accuracies = [r["test_metrics"]["accuracy"] for r in all_results]
    balanced_accs = [r["test_metrics"]["balanced_accuracy"] for r in all_results]
    macro_f1s = [r["test_metrics"]["macro_f1"] for r in all_results]
    
    print(f"\nPer-fold results:")
    for i, r in enumerate(all_results):
        print(f"  Fold {i}: acc={r['test_metrics']['accuracy']:.4f}  "
              f"bal_acc={r['test_metrics']['balanced_accuracy']:.4f}  "
              f"f1={r['test_metrics']['macro_f1']:.4f}")
    
    print(f"\nAggregate (mean ± std):")
    print(f"  Accuracy         : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Balanced Accuracy: {np.mean(balanced_accs):.4f} ± {np.std(balanced_accs):.4f}")
    print(f"  Macro F1         : {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    
    # Ensemble predictions (average logits)
    if all_logits:
        print("\n[Ensemble] Averaging logits across folds...")
        ensemble_logits = np.mean(all_logits, axis=0)
        ensemble_preds = np.argmax(ensemble_logits, axis=1).tolist()
        ensemble_labels = all_results[0]["test_metrics"]["all_labels"]
        
        ensemble_acc = accuracy_score(ensemble_labels, ensemble_preds)
        ensemble_bal_acc = balanced_accuracy_score(ensemble_labels, ensemble_preds)
        ensemble_f1 = f1_score(ensemble_labels, ensemble_preds, average="macro", zero_division=0)
        
        print(f"\n[Ensemble] Results:")
        print(f"  Accuracy         : {ensemble_acc:.4f}")
        print(f"  Balanced Accuracy: {ensemble_bal_acc:.4f}")
        print(f"  Macro F1         : {ensemble_f1:.4f}")
    
    return {
        "fold_results": all_results,
        "mean_accuracy": np.mean(accuracies),
        "mean_balanced_accuracy": np.mean(balanced_accs),
        "mean_macro_f1": np.mean(macro_f1s),
    }


# ============================================================================
# Main
# ============================================================================

def main(args: argparse.Namespace) -> None:
    """Main entry point."""
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
    if args.kfold:
        results = train_kfold(config, n_folds=args.n_folds)
    else:
        results = train_single_fold(config)
    
    print("\n\n✓ Training complete!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISIC 2019 Skin Lesion Classification"
    )
    
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--kfold", action="store_true",
                        help="Enable K-fold training")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for K-fold")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override stage 2 epochs")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
