"""
Training utilities for ISIC 2019 classification.

Includes:
    - Seed everything
    - Device selection
    - EMA (Exponential Moving Average)
    - MixUp augmentation
    - CutMix augmentation
    - Gradient clipping
    - Warmup scheduler
    - TTA evaluation
    - Checkpoint utilities
    - MPS stability helpers
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
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


# ============================================================================
# Seeding & Device
# ============================================================================

def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Get compute device.
    
    Parameters
    ----------
    device_str : str
        "auto", "mps", "cuda", or "cpu".
    
    Returns
    -------
    torch.device
        Selected device.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ============================================================================
# MPS Stability
# ============================================================================

def mps_sync() -> None:
    """Synchronize MPS device."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def mps_empty_cache() -> None:
    """Clear MPS cache."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    """Exponential Moving Average for model weights.
    
    Parameters
    ----------
    model : nn.Module
        Model to track.
    decay : float
        EMA decay rate.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._init_shadow()

    def _init_shadow(self) -> None:
        """Initialize shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay
                    )
                else:
                    self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def apply_shadow(self) -> None:
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self) -> None:
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        """Return state dict."""
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict."""
        self.shadow = state_dict["shadow"]
        self.decay = state_dict.get("decay", self.decay)


# ============================================================================
# MixUp
# ============================================================================

class MixUp:
    """MixUp augmentation.
    
    Parameters
    ----------
    alpha : float
        Beta distribution parameter.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp.
        
        Returns
        -------
        tuple
            (mixed_images, labels_a, labels_b, lam)
        """
        batch_size = images.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        indices = torch.randperm(batch_size, device=images.device)
        
        mixed_images = lam * images + (1 - lam) * images[indices]
        labels_a = labels
        labels_b = labels[indices]
        
        return mixed_images, labels_a, labels_b, lam


# ============================================================================
# CutMix
# ============================================================================

class CutMix:
    """CutMix augmentation.
    
    Parameters
    ----------
    alpha : float
        Beta distribution parameter.
    prob : float
        Probability of applying CutMix.
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5) -> None:
        self.alpha = alpha
        self.prob = prob

    def _rand_bbox(
        self,
        size: Tuple[int, ...],
        lam: float,
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix.
        
        Returns
        -------
        tuple
            (mixed_images, labels_a, labels_b, lam)
        """
        if np.random.rand() > self.prob:
            return images, labels, labels, 1.0
        
        batch_size = images.size(0)
        
        lam = np.random.beta(self.alpha, self.alpha)
        indices = torch.randperm(batch_size, device=images.device)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        images_mixed = images.clone()
        images_mixed[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        return images_mixed, labels, labels[indices], lam


# ============================================================================
# Combined MixUp/CutMix
# ============================================================================

class Mixup_Cutmix:
    """Combined MixUp and CutMix augmentation.
    
    Parameters
    ----------
    mixup_alpha : float
        MixUp alpha.
    cutmix_alpha : float
        CutMix alpha.
    cutmix_prob : float
        Probability of using CutMix vs MixUp.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        cutmix_prob: float = 0.5,
    ) -> None:
        self.mixup = MixUp(alpha=mixup_alpha)
        self.cutmix = CutMix(alpha=cutmix_alpha, prob=1.0)  # Always apply if selected
        self.cutmix_prob = cutmix_prob

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply either MixUp or CutMix."""
        if np.random.rand() < self.cutmix_prob:
            return self.cutmix(images, labels)
        else:
            return self.mixup(images, labels)


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute MixUp/CutMix loss."""
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ============================================================================
# Learning Rate Schedulers
# ============================================================================

class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing with linear warmup.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer.
    warmup_epochs : int
        Number of warmup epochs.
    total_epochs : int
        Total number of epochs.
    min_lr : float
        Minimum learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]


# ============================================================================
# Gradient Clipping
# ============================================================================

def clip_grad_norm(
    parameters,
    max_norm: float = 1.0,
) -> float:
    """Clip gradient norm."""
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


# ============================================================================
# TTA Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_with_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_metadata: bool = False,
) -> Tuple[List[int], List[int], np.ndarray]:
    """Evaluate model with TTA.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    loader : DataLoader
        TTA dataloader (returns stacked augmented images).
    device : torch.device
        Device.
    use_metadata : bool
        Whether data includes metadata.
    
    Returns
    -------
    tuple
        (predictions, labels, logits_array)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []
    
    for batch in loader:
        images = batch["images"]  # (B, num_tta, C, H, W)
        labels = batch["label"]
        
        batch_size, num_tta = images.shape[:2]
        
        # Flatten for batch processing
        images_flat = images.view(-1, *images.shape[2:]).to(device, non_blocking=False)
        
        # Prepare metadata if needed
        metadata = None
        if use_metadata and "metadata" in batch:
            # Repeat metadata for each TTA
            metadata = {
                "age": batch["metadata"]["age"].repeat_interleave(num_tta).to(device),
                "sex": batch["metadata"]["sex"].repeat_interleave(num_tta).to(device),
                "site": batch["metadata"]["site"].repeat_interleave(num_tta).to(device),
            }
        
        # Forward
        outputs = model(images_flat, metadata=metadata)
        logits_flat = outputs["logits"]  # (B * num_tta, num_classes)
        
        # Reshape and average
        logits = logits_flat.view(batch_size, num_tta, -1)  # (B, num_tta, C)
        avg_logits = logits.mean(dim=1)  # (B, C)
        
        mps_sync()
        
        preds = avg_logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
        all_logits.append(avg_logits.cpu().numpy())
    
    return all_preds, all_labels, np.concatenate(all_logits, axis=0)


# ============================================================================
# Checkpoint Utilities
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema: Optional[EMA],
    epoch: int,
    best_metric: float,
    filepath: str,
    config: Optional[dict] = None,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "ema_state_dict": ema.state_dict() if ema else None,
        "best_metric": best_metric,
        "config": config,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    ema: Optional[EMA] = None,
    device: torch.device = None,
) -> dict:
    """Load training checkpoint."""
    map_location = device if device else "cpu"
    checkpoint = torch.load(filepath, map_location=map_location)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if ema and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    
    return checkpoint


# ============================================================================
# Metrics helpers
# ============================================================================

def compute_accuracy(preds: List[int], labels: List[int]) -> float:
    """Compute accuracy."""
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)


# ============================================================================
# Config loading
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    # Test MixUp
    images = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 8, (8,))
    
    mixup = MixUp(alpha=0.2)
    mixed, la, lb, lam = mixup(images, labels)
    print(f"MixUp: lam={lam:.3f}")
    
    # Test CutMix
    cutmix = CutMix(alpha=1.0, prob=1.0)
    mixed, la, lb, lam = cutmix(images, labels)
    print(f"CutMix: lam={lam:.3f}")
    
    # Test combined
    mixup_cutmix = Mixup_Cutmix(mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_prob=0.5)
    mixed, la, lb, lam = mixup_cutmix(images, labels)
    print(f"MixUp/CutMix: lam={lam:.3f}")
    
    # Test scheduler
    import torch.optim as optim
    model = nn.Linear(10, 10)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=3, total_epochs=30)
    
    print("\nLR schedule:")
    for epoch in range(30):
        lr = scheduler.get_last_lr()[0]
        if epoch < 5 or epoch > 25:
            print(f"  Epoch {epoch}: lr={lr:.6f}")
        scheduler.step()
