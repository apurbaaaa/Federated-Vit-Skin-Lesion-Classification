"""
Loss functions — Asymmetric Focal Loss (production, no class weights).

Reference: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
Adapted for single-label multi-class classification.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss — no class weights.

    Parameters
    ----------
    gamma_neg : float   Focusing for *wrong-class* log-probs (default 4).
    gamma_pos : float   Focusing for *correct-class* log-probs (default 1).
    clip      : float   Probability shift for negatives (default 0.05).
    eps       : float   Numerical stability.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C) raw scores
        targets : (B,)   class indices
        """
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)               # (B, C)
        one_hot = F.one_hot(targets, num_classes).float()   # (B, C)

        # probabilities for positive / negative positions
        p_pos = probs.clamp(min=self.eps)
        p_neg = probs.clamp(max=1.0 - self.eps)

        # asymmetric clipping on negatives
        if self.clip > 0:
            p_neg = (p_neg - self.clip).clamp(min=self.eps)

        # log terms
        loss_pos = one_hot * torch.log(p_pos)
        loss_neg = (1.0 - one_hot) * torch.log(1.0 - p_neg)

        # focal weighting
        w_pos = (1.0 - probs).clamp(min=0.0) ** self.gamma_pos
        w_neg = probs.clamp(min=0.0) ** self.gamma_neg

        loss = -(w_pos * loss_pos + w_neg * loss_neg)
        return loss.sum(dim=1).mean()


# ============================================================================
# Factory
# ============================================================================

def build_loss(config: dict) -> nn.Module:
    """Build loss from config dict (always asymmetric focal, no class weights)."""
    lcfg = config.get("loss", {})
    asl = lcfg.get("asymmetric", {})
    return AsymmetricFocalLoss(
        gamma_neg=float(asl.get("gamma_neg", 4)),
        gamma_pos=float(asl.get("gamma_pos", 1)),
        clip=float(asl.get("clip", 0.05)),
    )
