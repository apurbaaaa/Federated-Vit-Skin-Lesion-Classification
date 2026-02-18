"""
Loss functions for ISIC 2019 skin lesion classification.

Includes:
    - Asymmetric Loss (for class imbalance)
    - Focal Loss
    - Dice Loss (for segmentation)
    - Combined BCE + Dice Loss
    - Joint Classification + Segmentation Loss
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Asymmetric Loss (ASL) - Recommended for class imbalance
# ============================================================================

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-class classification with imbalance.
    
    Reference: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    Adapted for multi-class (single-label) classification.
    
    Parameters
    ----------
    gamma_neg : float
        Focusing parameter for negative (easy) samples (default: 4).
    gamma_pos : float
        Focusing parameter for positive samples (default: 1).
    clip : float
        Probability clipping threshold (default: 0.05).
    weight : Optional[torch.Tensor]
        Class weights (default: None).
    eps : float
        Small constant for numerical stability.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        weight: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Raw model output (B, C).
        targets : torch.Tensor
            Class indices (B,).
        
        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)
        
        # One-hot encode targets
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Probability clipping for negative samples (asymmetric focusing)
        probs_pos = probs
        probs_neg = (probs + self.eps).clamp(max=1 - self.eps)
        
        if self.clip > 0:
            # Shift probability mass for negative samples
            probs_neg = (probs_neg - self.clip).clamp(min=self.eps)
        
        # Asymmetric focusing
        # For positive samples (target=1): use gamma_pos
        # For negative samples (target=0): use gamma_neg
        loss_pos = targets_one_hot * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets_one_hot) * torch.log(1 - probs_neg)
        
        # Apply focusing
        pt_pos = probs_pos
        pt_neg = 1 - probs_neg
        
        focal_weight_pos = (1 - pt_pos) ** self.gamma_pos
        focal_weight_neg = pt_neg ** self.gamma_neg
        
        loss = -(focal_weight_pos * loss_pos + focal_weight_neg * loss_neg)
        
        # Apply class weights
        if self.weight is not None:
            weight = self.weight.to(logits.device)
            loss = loss * weight.unsqueeze(0)
        
        # Sum over classes, mean over batch
        loss = loss.sum(dim=1).mean()
        
        return loss


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.
    
    Reference: "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    Parameters
    ----------
    gamma : float
        Focusing parameter (default: 2.0).
    alpha : Optional[torch.Tensor]
        Class weights (default: None).
    reduction : str
        Reduction method (default: "mean").
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Raw model output (B, C).
        targets : torch.Tensor
            Class indices (B,).
        
        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply class weights
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# Dice Loss (for segmentation)
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.
    
    Parameters
    ----------
    smooth : float
        Smoothing factor to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted mask logits (B, 1, H, W) or (B, H, W).
        target : torch.Tensor
            Ground truth mask (B, 1, H, W) or (B, H, W).
        
        Returns
        -------
        torch.Tensor
            Scalar Dice loss.
        """
        # Flatten to (B, -1)
        pred = torch.sigmoid(pred).view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


# ============================================================================
# BCE + Dice Combined Loss (for segmentation)
# ============================================================================

class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss for segmentation.
    
    Parameters
    ----------
    bce_weight : float
        Weight for BCE loss (default: 0.5).
    dice_weight : float
        Weight for Dice loss (default: 0.5).
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted mask logits (B, 1, H, W).
        target : torch.Tensor
            Ground truth mask (B, 1, H, W).
        
        Returns
        -------
        torch.Tensor
            Combined loss.
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ============================================================================
# Joint Loss (Classification + Segmentation)
# ============================================================================

class JointLoss(nn.Module):
    """Joint loss for classification and segmentation.
    
    Total Loss = Classification Loss + λ × Segmentation Loss
    
    Parameters
    ----------
    cls_loss : nn.Module
        Classification loss function.
    seg_loss : nn.Module
        Segmentation loss function.
    seg_weight : float
        Weight for segmentation loss (λ).
    """

    def __init__(
        self,
        cls_loss: nn.Module,
        seg_loss: nn.Module,
        seg_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.cls_loss = cls_loss
        self.seg_loss = seg_loss
        self.seg_weight = seg_weight

    def forward(
        self,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        seg_pred: Optional[torch.Tensor] = None,
        seg_target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        cls_logits : torch.Tensor
            Classification logits (B, num_classes).
        cls_targets : torch.Tensor
            Classification targets (B,).
        seg_pred : Optional[torch.Tensor]
            Segmentation predictions (B, 1, H, W).
        seg_target : Optional[torch.Tensor]
            Segmentation targets (B, 1, H, W).
        
        Returns
        -------
        tuple
            (total_loss, cls_loss, seg_loss)
        """
        cls_loss = self.cls_loss(cls_logits, cls_targets)
        
        if seg_pred is not None and seg_target is not None:
            seg_loss = self.seg_loss(seg_pred, seg_target)
            total_loss = cls_loss + self.seg_weight * seg_loss
        else:
            seg_loss = torch.tensor(0.0, device=cls_logits.device)
            total_loss = cls_loss
        
        return total_loss, cls_loss, seg_loss


# ============================================================================
# Factory function
# ============================================================================

def build_classification_loss(
    loss_type: str = "asymmetric",
    class_weights: Optional[torch.Tensor] = None,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Build classification loss function.
    
    Parameters
    ----------
    loss_type : str
        Type of loss: "asymmetric", "focal", "cross_entropy".
    class_weights : Optional[torch.Tensor]
        Class weights for imbalance handling.
    
    Returns
    -------
    nn.Module
        Loss function.
    """
    if loss_type == "asymmetric":
        return AsymmetricLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            weight=class_weights,
        )
    elif loss_type == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
        )
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    # Test Asymmetric Loss
    logits = torch.randn(8, 8)
    targets = torch.randint(0, 8, (8,))
    weights = torch.ones(8)
    
    asl = AsymmetricLoss(weight=weights)
    loss = asl(logits, targets)
    print(f"Asymmetric Loss: {loss.item():.4f}")
    
    # Test Focal Loss
    fl = FocalLoss(alpha=weights)
    loss = fl(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test Dice Loss
    pred_mask = torch.randn(4, 1, 64, 64)
    gt_mask = torch.randint(0, 2, (4, 1, 64, 64)).float()
    
    dice = DiceLoss()
    loss = dice(pred_mask, gt_mask)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # Test BCE + Dice
    bce_dice = BCEDiceLoss()
    loss = bce_dice(pred_mask, gt_mask)
    print(f"BCE + Dice Loss: {loss.item():.4f}")
