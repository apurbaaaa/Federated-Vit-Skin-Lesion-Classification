"""
Training utilities — EMA, MixUp/CutMix, scheduler, TTA eval, checkpoint, etc.
"""

from __future__ import annotations

import copy
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


# ============================================================================
# Seeding & Device
# ============================================================================

def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    """Maintains shadow (exponentially-averaged) copies of model parameters.

    Usage:
        ema = EMA(model, decay=0.9995)
        ...
        optimizer.step()
        ema.update()          # after each optimizer step
        ...
        ema.apply_shadow()    # swap in EMA weights for validation
        # validate
        ema.restore()         # swap back to training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._init()

    def _init(self) -> None:
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
                else:
                    self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def apply_shadow(self) -> None:
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self) -> None:
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        return {"shadow": {k: v.cpu() for k, v in self.shadow.items()}, "decay": self.decay}

    def load_state_dict(self, sd: dict) -> None:
        device = next(iter(self.shadow.values())).device if self.shadow else torch.device("cpu")
        self.shadow = {k: v.to(device) for k, v in sd["shadow"].items()}
        self.decay = sd.get("decay", self.decay)


# ============================================================================
# MixUp / CutMix
# ============================================================================

class MixUp:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        idx = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1 - lam) * images[idx]
        return mixed, labels, labels[idx], lam


class CutMix:
    def __init__(self, alpha: float = 1.0, prob: float = 0.7):
        self.alpha = alpha
        self.prob = prob

    @staticmethod
    def _rand_bbox(size, lam):
        W, H = size[2], size[3]
        cut = np.sqrt(1. - lam)
        cw, ch = int(W * cut), int(H * cut)
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1, y1 = np.clip(cx - cw // 2, 0, W), np.clip(cy - ch // 2, 0, H)
        x2, y2 = np.clip(cx + cw // 2, 0, W), np.clip(cy + ch // 2, 0, H)
        return x1, y1, x2, y2

    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        if np.random.rand() > self.prob:
            return images, labels, labels, 1.0
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(images.size(0), device=images.device)
        x1, y1, x2, y2 = self._rand_bbox(images.size(), lam)
        mixed = images.clone()
        mixed[:, :, x1:x2, y1:y2] = images[idx, :, x1:x2, y1:y2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        return mixed, labels, labels[idx], lam


class MixupCutmix:
    """Randomly choose MixUp or CutMix each batch."""
    def __init__(self, mixup_alpha=0.4, cutmix_alpha=1.0, cutmix_prob=0.7):
        self.mixup = MixUp(alpha=mixup_alpha)
        self.cutmix = CutMix(alpha=cutmix_alpha, prob=1.0)
        self.cutmix_prob = cutmix_prob

    def __call__(self, images, labels):
        if np.random.rand() < self.cutmix_prob:
            return self.cutmix(images, labels)
        return self.mixup(images, labels)


def mixup_criterion(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ============================================================================
# Warmup + Cosine Scheduler
# ============================================================================

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base * alpha for base in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base - self.min_lr) * cos for base in self.base_lrs]


# ============================================================================
# Gradient clipping
# ============================================================================

def clip_grad_norm(parameters, max_norm: float = 1.0) -> float:
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


# ============================================================================
# TTA evaluation
# ============================================================================

@torch.no_grad()
def evaluate_with_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_metadata: bool = True,
    use_amp: bool = True,
) -> Tuple[List[int], List[int], np.ndarray]:
    """Run TTA: average logits across augmented views."""
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    for batch in loader:
        images = batch["images"]                       # (B, T, C, H, W)
        labels = batch["label"]
        B, T = images.shape[:2]
        flat = images.view(-1, *images.shape[2:]).to(device, non_blocking=True)

        meta = None
        if use_metadata and "metadata" in batch:
            meta = batch["metadata"].to(device, non_blocking=True)  # (B, 13)
            meta = meta.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits_flat = model(flat, metadata=meta)["logits"]  # (B*T, C)

        logits = logits_flat.view(B, T, -1).mean(dim=1)        # (B, C)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
        all_logits.append(logits.cpu().numpy())
    return all_preds, all_labels, np.concatenate(all_logits, axis=0)


# ============================================================================
# Standard eval
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_metadata: bool = True,
    use_amp: bool = True,
) -> Dict:
    """Standard (no-TTA) evaluation — returns dict of metrics."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
    from data import NUM_CLASSES, VALID_CLASSES

    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        meta = batch.get("metadata")
        if meta is not None:
            meta = meta.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(images, metadata=meta if use_metadata else None)["logits"]
            loss = F.cross_entropy(logits, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    total = max(len(all_labels), 1)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    per_recall = [(cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0) for i in range(NUM_CLASSES)]
    return {
        "loss": running_loss / total,
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "confusion_matrix": cm,
        "per_class_recall": per_recall,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


# ============================================================================
# Checkpoint
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, ema, epoch, metric, path, config=None):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "ema_state_dict": ema.state_dict() if ema else None,
        "best_metric": metric,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, ema=None, device=None):
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ema and ckpt.get("ema_state_dict"):
        ema.load_state_dict(ckpt["ema_state_dict"])
    return ckpt


# ============================================================================
# Config
# ============================================================================

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Auto batch-size probe
# ============================================================================

def auto_batch_size(model: nn.Module, device: torch.device,
                    image_size: int = 384, in_channels: int = 4,
                    meta_dim: int = 13, start: int = 8, step: int = 2) -> int:
    """Search largest batch that fits in GPU memory (starts small for large models)."""
    if device.type != "cuda":
        return start
    model.eval()
    bs = start
    last_ok = max(step, 1)
    while bs <= 64:
        try:
            torch.cuda.empty_cache()
            x = torch.randn(bs, in_channels, image_size, image_size, device=device)
            m = torch.randn(bs, meta_dim, device=device)
            with torch.amp.autocast(device_type="cuda"):
                _ = model(x, metadata=m)
            del x, m, _
            torch.cuda.empty_cache()
            print(f"[AutoBS] batch_size {bs} OK")
            last_ok = bs
            bs += step
        except RuntimeError:
            torch.cuda.empty_cache()
            break
    print(f"[AutoBS] Using batch_size = {last_ok}")
    return last_ok
