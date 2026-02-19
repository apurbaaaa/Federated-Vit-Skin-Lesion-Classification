"""
ISIC 2019 — SwinV2-Large-384 with Metadata Fusion.

Architecture
────────────
  1. Backbone : SwinV2-Large (timm), 384×384, drop_path 0.4
  2. Metadata : MLP branch  age+sex_onehot+site_onehot → 256 → 128
  3. Fusion   : concat(image_feat, meta_embed) → 512 → 8

All dimensions are detected dynamically from the backbone.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Metadata Branch  (age + sex-one-hot + site-one-hot)
# ============================================================================

class MetadataBranch(nn.Module):
    """
    Inputs (per sample):
        age_norm   : float  (1)
        sex_onehot : (3,)   male / female / unknown
        site_onehot: (9,)   9 anatomical sites

    Architecture:
        Linear(input_dim → 256) → BN → GELU → Dropout(0.4)
        Linear(256 → 128)       → BN → GELU
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, input_dim)  →  (B, output_dim)"""
        return self.net(x)


# ============================================================================
# Main Model
# ============================================================================

class ISICClassifier(nn.Module):
    """Production skin-lesion classifier.

    Parameters
    ----------
    backbone_name : str
        timm model identifier.
    num_classes : int
    image_size : int
    in_channels : int
        3 (RGB) or 4 (RGB+mask).
    pretrained, drop_path_rate : see timm docs.
    metadata_enabled : bool
        Turn on / off the metadata branch.
    meta_input_dim, meta_hidden_dim, meta_output_dim, meta_dropout :
        MetadataBranch hyper-parameters.
    cls_hidden_dim, cls_dropout :
        Final classifier MLP hyper-parameters.
    """

    def __init__(
        self,
        backbone_name: str = "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
        num_classes: int = 8,
        image_size: int = 384,
        in_channels: int = 4,
        pretrained: bool = True,
        drop_path_rate: float = 0.4,
        # metadata
        metadata_enabled: bool = True,
        meta_input_dim: int = 13,
        meta_hidden_dim: int = 256,
        meta_output_dim: int = 128,
        meta_dropout: float = 0.4,
        # classifier
        cls_hidden_dim: int = 512,
        cls_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.metadata_enabled = metadata_enabled
        self.num_classes = num_classes
        self.image_size = image_size
        self.in_channels = in_channels

        # ----- backbone (classification head removed) -----------------------
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,           # removes the final head
            drop_path_rate=drop_path_rate,
        )
        # dynamic feature dim
        self.backbone_dim = self.backbone.num_features
        print(f"[Model] Backbone feature dim: {self.backbone_dim}")

        # adapt first conv for 4-channel input
        if in_channels != 3:
            self._modify_input_channels(in_channels, pretrained)

        # ----- metadata branch (optional) -----------------------------------
        if metadata_enabled:
            self.metadata_branch = MetadataBranch(
                input_dim=meta_input_dim,
                hidden_dim=meta_hidden_dim,
                output_dim=meta_output_dim,
                dropout=meta_dropout,
            )
            classifier_in = self.backbone_dim + meta_output_dim
        else:
            classifier_in = self.backbone_dim

        # ----- classifier ---------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden_dim, num_classes),
        )
        self._init_classifier()

    # ---------------------------------------------------------------------- #
    # helpers
    # ---------------------------------------------------------------------- #
    def _modify_input_channels(self, in_channels: int, pretrained: bool) -> None:
        """Modify first projection to accept `in_channels` channels."""
        if hasattr(self.backbone, "patch_embed") and hasattr(self.backbone.patch_embed, "proj"):
            old = self.backbone.patch_embed.proj
            new = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=(old.bias is not None),
            )
            if pretrained:
                with torch.no_grad():
                    new.weight[:, :3] = old.weight
                    new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
                    if old.bias is not None:
                        new.bias.copy_(old.bias)
            self.backbone.patch_embed.proj = new
            print(f"[Model] patch_embed.proj → {in_channels} channels")

    def _init_classifier(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------- #
    # forward
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x        : (B, C, H, W)
        metadata : (B, meta_input_dim)  — flat vector of [age, sex_oh, site_oh]

        Returns
        -------
        {"logits": (B, num_classes)}
        """
        features = self.backbone(x)  # (B, backbone_dim)

        if self.metadata_enabled:
            if metadata is not None:
                meta_emb = self.metadata_branch(metadata)  # (B, meta_out)
            else:
                # Zero-fill metadata embedding so classifier dim stays consistent
                meta_emb = torch.zeros(
                    features.size(0), self.metadata_branch.output_dim,
                    device=features.device, dtype=features.dtype,
                )
            features = torch.cat([features, meta_emb], dim=1)

        logits = self.classifier(features)
        return {"logits": logits}

    # ---------------------------------------------------------------------- #
    # freeze / unfreeze
    # ---------------------------------------------------------------------- #
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[Model] backbone frozen")

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("[Model] backbone unfrozen")

    def get_head_parameters(self) -> List[Dict]:
        params = list(self.classifier.parameters())
        if self.metadata_enabled:
            params += list(self.metadata_branch.parameters())
        return [{"params": params}]

    def get_layerwise_lr_groups(
        self,
        base_lr: float = 1e-4,
        decay_rate: float = 0.75,
        weight_decay: float = 1e-5,
    ) -> List[Dict]:
        groups: List[Dict] = []

        # determine number of stages/layers
        if hasattr(self.backbone, "layers"):
            layers = list(self.backbone.layers)
        elif hasattr(self.backbone, "blocks"):
            layers = list(self.backbone.blocks)
        else:
            layers = []

        n_layers = len(layers)

        # patch embed (lowest LR)
        if hasattr(self.backbone, "patch_embed"):
            lr = base_lr * (decay_rate ** (n_layers + 1))
            groups.append({"params": list(self.backbone.patch_embed.parameters()),
                           "lr": lr, "weight_decay": weight_decay})

        # transformer stages
        for i, layer in enumerate(layers):
            lr = base_lr * (decay_rate ** (n_layers - i))
            groups.append({"params": list(layer.parameters()),
                           "lr": lr, "weight_decay": weight_decay})

        # norm
        if hasattr(self.backbone, "norm"):
            groups.append({"params": list(self.backbone.norm.parameters()),
                           "lr": base_lr, "weight_decay": weight_decay})

        # head (10× base LR)
        head_params = list(self.classifier.parameters())
        if self.metadata_enabled:
            head_params += list(self.metadata_branch.parameters())
        groups.append({"params": head_params, "lr": base_lr * 10,
                       "weight_decay": weight_decay})

        return groups

    def count_parameters(self) -> Dict[str, int]:
        d: Dict[str, int] = {
            "total": sum(p.numel() for p in self.parameters()),
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
        }
        if self.metadata_enabled:
            d["metadata"] = sum(p.numel() for p in self.metadata_branch.parameters())
        return d


# ============================================================================
# Factory
# ============================================================================

def get_layerwise_lr_groups(
    model: ISICClassifier,
    base_lr: float = 1e-4,
    decay_rate: float = 0.75,
    weight_decay: float = 1e-5,
) -> List[Dict]:
    """Standalone wrapper around ISICClassifier.get_layerwise_lr_groups."""
    return model.get_layerwise_lr_groups(base_lr, decay_rate, weight_decay)


def count_parameters(model: ISICClassifier) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters())


def build_model(config: dict) -> ISICClassifier:
    m = config.get("model", {})
    d = config.get("data", {})
    meta = m.get("metadata", {})
    cls  = m.get("classifier", {})

    in_ch = 4 if d.get("use_segmentation_mask", False) else 3

    return ISICClassifier(
        backbone_name=m.get("backbone", "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k"),
        num_classes=m.get("num_classes", 8),
        image_size=m.get("image_size", 384),
        in_channels=in_ch,
        pretrained=m.get("pretrained", True),
        drop_path_rate=float(m.get("drop_path_rate", 0.4)),
        metadata_enabled=meta.get("enabled", True),
        meta_input_dim=int(meta.get("input_dim", 13)),
        meta_hidden_dim=int(meta.get("hidden_dim", 256)),
        meta_output_dim=int(meta.get("output_dim", 128)),
        meta_dropout=float(meta.get("dropout", 0.4)),
        cls_hidden_dim=int(cls.get("hidden_dim", 512)),
        cls_dropout=float(cls.get("dropout", 0.5)),
    )
