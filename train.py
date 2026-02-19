#!/usr/bin/env python3
"""
train.py — 5-Fold Stratified CV with SwinV2-Large ISIC classifier.

Features
────────
  • 5-Fold StratifiedKFold (or StratifiedGroupKFold when lesion_id available)
  • Mixed precision (AMP)
  • EMA (decay 0.9995)
  • MixUp / CutMix
  • Warmup + Cosine LR
  • Gradient clipping
  • TTA evaluation on test set
  • Auto batch-size probe
  • Fold-averaged test logits
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from tqdm import tqdm

from data import (
    VALID_CLASSES,
    NUM_CLASSES,
    load_isic_data,
    build_fold_loaders,
    build_tta_loader,
    build_test_loader,
    print_class_distribution,
)
from losses import build_loss
from model import build_model, count_parameters, get_layerwise_lr_groups
from utils import (
    EMA,
    MixupCutmix,
    WarmupCosineScheduler,
    auto_batch_size,
    clip_grad_norm,
    evaluate,
    evaluate_with_tta,
    get_device,
    load_checkpoint,
    load_config,
    mixup_criterion,
    save_checkpoint,
    seed_everything,
)


# ============================================================================
# Logging
# ============================================================================

def setup_logging(log_dir: str, fold: int = -1) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    tag = f"fold{fold}" if fold >= 0 else "main"
    logger = logging.getLogger(f"isic_{tag}")
    logger.setLevel(logging.INFO)
    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    # file
    fh = logging.FileHandler(os.path.join(log_dir, f"train_{tag}.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    return logger


# ============================================================================
# Training one epoch
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scheduler,             # stepped per-epoch (after this function)
    scaler,                # torch.amp.GradScaler
    ema: Optional[EMA],
    device: torch.device,
    config: dict,
    epoch: int,
    logger: logging.Logger,
) -> float:
    model.train()
    t_cfg = config.get("training", {})
    use_amp = t_cfg.get("use_amp", True) and device.type == "cuda"
    grad_clip = t_cfg.get("grad_clip", 1.0)
    accum_steps = max(1, t_cfg.get("gradient_accumulation_steps", 1))
    use_meta = config.get("model", {}).get("metadata", {}).get("enabled", True)

    aug_cfg = config.get("augmentation", {})
    mixer = None
    mixup_a = aug_cfg.get("mixup", {}).get("alpha", 0.0)
    cutmix_p = aug_cfg.get("cutmix", {}).get("prob", 0.0)
    if mixup_a > 0 or cutmix_p > 0:
        mixer = MixupCutmix(
            mixup_alpha=mixup_a,
            cutmix_alpha=aug_cfg.get("cutmix", {}).get("alpha", 1.0),
            cutmix_prob=cutmix_p,
        )

    running_loss = 0.0
    total = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"  Train E{epoch:02d}", leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        meta = batch.get("metadata")
        if meta is not None:
            meta = meta.to(device, non_blocking=True)
        bs = images.size(0)

        # MixUp / CutMix
        do_mix = mixer is not None
        if do_mix:
            images, labels_a, labels_b, lam = mixer(images, labels)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(images, metadata=meta if use_meta else None)
            logits = out["logits"]
            if do_mix:
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                loss = criterion(logits, labels)
            loss = loss / accum_steps  # scale for accumulation

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            clip_grad_norm(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update()

        running_loss += loss.item() * accum_steps * bs
        total += bs
        pbar.set_postfix(loss=f"{running_loss / total:.4f}")

    return running_loss / max(total, 1)


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    config: dict,
) -> dict:
    model.eval()
    use_amp = config.get("training", {}).get("use_amp", True) and device.type == "cuda"
    use_meta = config.get("model", {}).get("metadata", {}).get("enabled", True)

    running_loss = 0.0
    total = 0
    all_preds, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        meta = batch.get("metadata")
        if meta is not None:
            meta = meta.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, metadata=meta if use_meta else None)["logits"]
            loss = criterion(logits, labels)

        bs = images.size(0)
        running_loss += loss.item() * bs
        total += bs
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n = max(total, 1)
    return {
        "loss": running_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


# ============================================================================
# Single fold
# ============================================================================

def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    device: torch.device,
    log_dir: str,
) -> np.ndarray:
    """
    Train one fold end-to-end. Returns test-set logits (N, C).
    """
    logger = setup_logging(log_dir, fold=fold)
    logger.info(f"{'='*60}")
    logger.info(f" FOLD {fold}")
    logger.info(f"{'='*60}")
    logger.info(f" Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    t_cfg = config.get("training", {})
    epochs = t_cfg.get("epochs", 80)
    patience = t_cfg.get("early_stopping", {}).get("patience", 15)
    metric_name = t_cfg.get("early_stopping", {}).get("metric", "balanced_accuracy")

    # Model
    model = build_model(config).to(device)
    logger.info(f" Parameters: {count_parameters(model):,.0f}")

    # EMA
    ema_cfg = t_cfg.get("ema", {})
    ema = EMA(model, decay=ema_cfg.get("decay", 0.9995)) if ema_cfg.get("enabled", True) else None

    # Optimizer — with layer-wise LR
    opt_cfg = t_cfg.get("optimizer", {})
    llrd_cfg = t_cfg.get("llrd", {})
    lr_groups = get_layerwise_lr_groups(
        model,
        base_lr=opt_cfg.get("lr", 1e-4),
        decay_rate=llrd_cfg.get("decay_rate", 0.75) if llrd_cfg.get("enabled", True) else 1.0,
        weight_decay=opt_cfg.get("weight_decay", 1e-5),
    )
    optimizer = torch.optim.AdamW(lr_groups, weight_decay=opt_cfg.get("weight_decay", 1e-5))

    # Scheduler
    warmup = t_cfg.get("scheduler", {}).get("warmup_epochs", 5)
    min_lr = t_cfg.get("scheduler", {}).get("min_lr", 1e-6)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup, total_epochs=epochs, min_lr=min_lr)

    # AMP
    use_amp = t_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # Loss
    criterion = build_loss(config).to(device)

    # Dataloaders
    train_loader, val_loader = build_fold_loaders(train_df, val_df, config)

    # Checkpoint dir
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, ema, device, config, epoch, logger,
        )

        # Swap EMA for validation
        if ema is not None:
            ema.apply_shadow()

        val_metrics = validate(model, val_loader, criterion, device, config)

        if ema is not None:
            ema.restore()

        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[-1]["lr"]
        logger.info(
            f"  E{epoch:02d} | trn_loss {train_loss:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | val_acc {val_metrics['accuracy']:.4f} | "
            f"val_bal {val_metrics['balanced_accuracy']:.4f} | val_f1 {val_metrics['macro_f1']:.4f} | "
            f"lr {lr_now:.2e} | {elapsed:.1f}s"
        )

        # Best check
        metric_val = val_metrics[metric_name]
        if metric_val > best_metric:
            best_metric = metric_val
            epochs_without_improve = 0
            save_checkpoint(model, optimizer, scheduler, ema, epoch, best_metric, ckpt_path, config)
            logger.info(f"  >>> New best {metric_name}: {best_metric:.4f} — saved.")
        else:
            epochs_without_improve += 1
            if patience > 0 and epochs_without_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={patience}).")
                break

    # ── Load best & evaluate on test ──────────────────────────────────────
    logger.info(f"  Loading best checkpoint (best {metric_name}={best_metric:.4f})")
    load_checkpoint(ckpt_path, model, ema=ema, device=device)
    if ema is not None:
        ema.apply_shadow()

    use_meta = config.get("model", {}).get("metadata", {}).get("enabled", True)
    use_amp_val = t_cfg.get("use_amp", True)

    if len(test_df) == 0:
        logger.info("  No test data available — skipping test evaluation.")
        if ema is not None:
            ema.restore()
        return np.zeros((0, NUM_CLASSES))

    # TTA evaluation
    tta_cfg = t_cfg.get("tta", {})
    use_tta = tta_cfg.get("enabled", True)
    if use_tta:
        logger.info("  Running TTA on test set…")
        tta_loader = build_tta_loader(test_df, config)
        preds, labels, logits = evaluate_with_tta(
            model, tta_loader, device,
            use_metadata=use_meta,
            use_amp=use_amp_val,
        )
    else:
        logger.info("  Evaluating on test set (no TTA)…")
        test_loader = build_test_loader(test_df, config)
        result = evaluate(
            model, test_loader, device,
            use_metadata=use_meta,
            use_amp=use_amp_val,
        )
        preds, labels = result["all_preds"], result["all_labels"]
        logits = np.zeros((len(preds), NUM_CLASSES))
        for i, p in enumerate(preds):
            logits[i, p] = 1.0

    # Only compute metrics if we have ground truth (non-placeholder labels)
    if test_df["label"].nunique() > 1 or test_df["dx"].iloc[0] != "MEL":
        acc = accuracy_score(labels, preds)
        bal = balanced_accuracy_score(labels, preds)
        f1m = f1_score(labels, preds, average="macro", zero_division=0)
        logger.info(f"  Fold {fold} Test — acc: {acc:.4f} | bal_acc: {bal:.4f} | macro_f1: {f1m:.4f}")
        logger.info(f"\n{classification_report(labels, preds, target_names=VALID_CLASSES, digits=4)}")

    if ema is not None:
        ema.restore()

    return logits  # (N, C)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ISIC 2019 — 5-Fold CV Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--fold",   type=int, default=-1, help="Run a single fold (-1 = all)")
    parser.add_argument("--log",    type=str, default="logs")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    device = get_device()

    t_cfg = config.get("training", {})
    d_cfg = config.get("data", {})
    n_folds = t_cfg.get("cv", {}).get("n_splits", 5)

    print(f"\n{'='*60}")
    print(f"  ISIC 2019 Classifier — {n_folds}-Fold CV")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Load data
    train_full_df, test_df = load_isic_data(d_cfg.get("isic_dir", "./ISIC"))
    print_class_distribution(train_full_df, "Full Train")
    if len(test_df) > 0:
        print_class_distribution(test_df, "Test")

    accum  = t_cfg.get("gradient_accumulation_steps", 1)
    eff_bs = t_cfg.get("batch_size", 4) * accum
    print(f"\n  [Config] physical_bs={t_cfg.get('batch_size', 4)}, "
          f"accum={accum}, effective_bs={eff_bs}")

    # Auto batch-size (optional — disabled by default for SwinV2-Large)
    if t_cfg.get("auto_batch_size", False):
        model_probe = build_model(config).to(device)
        bs = auto_batch_size(model_probe, device,
                             image_size=config.get("model", {}).get("image_size", 384))
        del model_probe
        if device.type == "cuda":
            torch.cuda.empty_cache()
        config["training"]["batch_size"] = bs

    # Fold split
    labels = train_full_df["label"].values
    has_groups = "lesion_id" in train_full_df.columns and train_full_df["lesion_id"].nunique() > 1
    if has_groups:
        groups = train_full_df["lesion_id"].values
        kf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
        splits = list(kf.split(train_full_df, labels, groups))
        print(f"[Split] StratifiedGroupKFold (on lesion_id)")
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
        splits = list(kf.split(train_full_df, labels))
        print(f"[Split] StratifiedKFold")

    # Which folds to run
    fold_list = list(range(n_folds)) if args.fold < 0 else [args.fold]

    all_test_logits = []
    for fold_idx in fold_list:
        trn_idx, val_idx = splits[fold_idx]
        trn_df = train_full_df.iloc[trn_idx].reset_index(drop=True)
        val_df = train_full_df.iloc[val_idx].reset_index(drop=True)

        fold_log_dir = os.path.join(args.log, f"fold{fold_idx}")
        logits = train_fold(fold_idx, trn_df, val_df, test_df, config, device, fold_log_dir)
        all_test_logits.append(logits)

    # Fold-averaged ensemble prediction
    valid_logits = [lg for lg in all_test_logits if len(lg) > 0]
    if len(valid_logits) > 0 and len(test_df) > 0:
        avg_logits = np.mean(valid_logits, axis=0)
        preds = avg_logits.argmax(axis=1)
        test_labels = test_df["label"].values

        # Save ensemble logits regardless
        os.makedirs(args.log, exist_ok=True)
        np.save(os.path.join(args.log, "ensemble_logits.npy"), avg_logits)
        print(f"\n  Ensemble logits saved → {args.log}/ensemble_logits.npy")

        # Only print metrics if we have real ground truth
        has_gt = test_df["dx"].nunique() > 1 or test_df["dx"].iloc[0] != "MEL"
        if has_gt:
            acc  = accuracy_score(test_labels, preds)
            bal  = balanced_accuracy_score(test_labels, preds)
            f1m  = f1_score(test_labels, preds, average="macro", zero_division=0)
            cm   = confusion_matrix(test_labels, preds, labels=list(range(NUM_CLASSES)))
            print(f"\n{'='*60}")
            print(f"  {len(valid_logits)}-Fold ENSEMBLE (averaged logits)")
            print(f"{'='*60}")
            print(f"  Accuracy:          {acc:.4f}")
            print(f"  Balanced Accuracy: {bal:.4f}")
            print(f"  Macro F1:          {f1m:.4f}")
            print(f"\n{classification_report(test_labels, preds, target_names=VALID_CLASSES, digits=4)}")
            print(f"Confusion Matrix:\n{cm}")

    print("\nDone.")


if __name__ == "__main__":
    main()
