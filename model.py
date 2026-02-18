"""
Vision Transformer for ISIC Classification with 4-channel input (RGB + Mask).

Architecture:
    1. Backbone: Swin / ViT with modified input layer for 4 channels
    2. Metadata Fusion: MLP embedding
    3. Classifier Head: MLP with dropout
    
No segmentation branch - masks are precomputed and used as spatial prior.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn


# ============================================================================
# Metadata Embedding
# ============================================================================

class MetadataEmbedding(nn.Module):
    """Embed categorical and numerical metadata features."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Embeddings
        self.sex_embed = nn.Embedding(3, embed_dim // 4)  # male, female, unknown
        self.site_embed = nn.Embedding(9, embed_dim // 2)  # 9 anatomical sites
        
        # Age projection (continuous)
        self.age_proj = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(inplace=True),
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        age: torch.Tensor,
        sex: torch.Tensor,
        site: torch.Tensor,
    ) -> torch.Tensor:
        if age.dim() == 1:
            age = age.unsqueeze(1)
        
        age_emb = self.age_proj(age)  # (B, embed_dim // 4)
        sex_emb = self.sex_embed(sex)  # (B, embed_dim // 4)
        site_emb = self.site_embed(site)  # (B, embed_dim // 2)
        
        concat = torch.cat([age_emb, sex_emb, site_emb], dim=1)  # (B, embed_dim)
        return self.fusion(concat)


# ============================================================================
# Main Model
# ============================================================================

class ISICClassifier(nn.Module):
    """Vision Transformer classifier with 4-channel input support.
    
    Parameters
    ----------
    backbone_name : str
        timm model name for backbone.
    num_classes : int
        Number of classification classes.
    image_size : int
        Input image size.
    in_channels : int
        Number of input channels (3 for RGB, 4 for RGB+mask).
    pretrained : bool
        Whether to use pretrained backbone weights.
    drop_path_rate : float
        DropPath rate for stochastic depth.
    metadata_enabled : bool
        Whether to use metadata fusion.
    metadata_embed_dim : int
        Metadata embedding dimension.
    classifier_hidden_dim : int
        Classifier hidden dimension.
    classifier_dropout : float
        Classifier dropout rate.
    """

    def __init__(
        self,
        backbone_name: str = "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
        num_classes: int = 8,
        image_size: int = 224,
        in_channels: int = 4,
        pretrained: bool = True,
        drop_path_rate: float = 0.2,
        metadata_enabled: bool = False,
        metadata_embed_dim: int = 64,
        classifier_hidden_dim: int = 512,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        
        self.metadata_enabled = metadata_enabled
        self.num_classes = num_classes
        self.image_size = image_size
        self.in_channels = in_channels
        
        # =====================================================================
        # Backbone
        # =====================================================================
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            drop_path_rate=drop_path_rate,
        )
        self.backbone_dim = self.backbone.num_features
        
        # Modify input layer if needed (4 channels)
        if in_channels != 3:
            self._modify_input_channels(in_channels, pretrained)
        
        # =====================================================================
        # Metadata Embedding (optional)
        # =====================================================================
        if metadata_enabled:
            self.metadata_embed = MetadataEmbedding(embed_dim=metadata_embed_dim)
            classifier_input_dim = self.backbone_dim + metadata_embed_dim
        else:
            classifier_input_dim = self.backbone_dim
        
        # =====================================================================
        # Classifier Head
        # =====================================================================
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(classifier_hidden_dim),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )
        
        self._init_classifier()

    def _modify_input_channels(self, in_channels: int, pretrained: bool) -> None:
        """Modify the first conv layer to accept different number of input channels."""
        # For Swin transformers
        if hasattr(self.backbone, "patch_embed"):
            patch_embed = self.backbone.patch_embed
            
            if hasattr(patch_embed, "proj"):
                old_conv = patch_embed.proj
                new_conv = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                )
                
                # Copy weights for first 3 channels if pretrained
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight[:, :3, :, :] = old_conv.weight
                        # Initialize new channel(s) with mean of existing channels
                        new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
                        if old_conv.bias is not None:
                            new_conv.bias = old_conv.bias
                
                patch_embed.proj = new_conv
                print(f"[Model] Modified patch_embed.proj to accept {in_channels} channels")
        
        # For ViT models
        elif hasattr(self.backbone, "patch_embed") and hasattr(self.backbone.patch_embed, "backbone"):
            # timm hybrid ViT
            backbone = self.backbone.patch_embed.backbone
            if hasattr(backbone, "stem"):
                old_conv = backbone.stem.conv
                new_conv = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                )
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight[:, :3, :, :] = old_conv.weight
                        new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
                        if old_conv.bias is not None:
                            new_conv.bias = old_conv.bias
                backbone.stem.conv = new_conv
                print(f"[Model] Modified stem.conv to accept {in_channels} channels")

    def _init_classifier(self) -> None:
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images (B, C, H, W) where C is 3 or 4.
        metadata : Optional[Dict]
            Metadata dict with "age", "sex", "site" tensors.
        
        Returns
        -------
        dict
            {"logits": (B, num_classes)}
        """
        # Backbone forward
        features = self.backbone(x)  # (B, backbone_dim)
        
        # Metadata fusion
        if self.metadata_enabled and metadata is not None:
            meta_emb = self.metadata_embed(
                metadata["age"],
                metadata["sex"],
                metadata["site"],
            )
            features = torch.cat([features, meta_emb], dim=1)
        
        # Classifier
        logits = self.classifier(features)
        
        return {"logits": logits}

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] Backbone unfrozen")

    def get_head_parameters(self) -> List[Dict]:
        """Get classifier and metadata embedding parameters."""
        params = list(self.classifier.parameters())
        if self.metadata_enabled:
            params.extend(list(self.metadata_embed.parameters()))
        return [{"params": params}]

    def get_layerwise_lr_groups(
        self,
        base_lr: float = 5e-5,
        decay_rate: float = 0.75,
        weight_decay: float = 0.05,
    ) -> List[Dict]:
        """Get parameter groups with layer-wise learning rate decay."""
        params = []
        
        # Get number of layers
        if hasattr(self.backbone, "layers"):
            num_layers = len(self.backbone.layers)
        elif hasattr(self.backbone, "blocks"):
            num_layers = len(self.backbone.blocks)
        else:
            num_layers = 12  # Default
        
        # Patch embedding (lowest LR)
        if hasattr(self.backbone, "patch_embed"):
            lr = base_lr * (decay_rate ** (num_layers + 1))
            params.append({
                "params": list(self.backbone.patch_embed.parameters()),
                "lr": lr,
                "weight_decay": weight_decay,
            })
        
        # Transformer layers
        if hasattr(self.backbone, "layers"):
            for i, layer in enumerate(self.backbone.layers):
                lr = base_lr * (decay_rate ** (num_layers - i))
                params.append({
                    "params": list(layer.parameters()),
                    "lr": lr,
                    "weight_decay": weight_decay,
                })
        elif hasattr(self.backbone, "blocks"):
            for i, block in enumerate(self.backbone.blocks):
                lr = base_lr * (decay_rate ** (num_layers - i))
                params.append({
                    "params": list(block.parameters()),
                    "lr": lr,
                    "weight_decay": weight_decay,
                })
        
        # Norm layer
        if hasattr(self.backbone, "norm"):
            params.append({
                "params": list(self.backbone.norm.parameters()),
                "lr": base_lr,
                "weight_decay": weight_decay,
            })
        
        # Classifier head (highest LR)
        head_params = list(self.classifier.parameters())
        if self.metadata_enabled:
            head_params.extend(list(self.metadata_embed.parameters()))
        
        params.append({
            "params": head_params,
            "lr": base_lr * 10,  # Higher LR for head
            "weight_decay": weight_decay,
        })
        
        return params

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        counts = {
            "total": sum(p.numel() for p in self.parameters()),
            "backbone": backbone_params,
            "classifier": classifier_params,
        }
        
        if self.metadata_enabled:
            counts["metadata"] = sum(p.numel() for p in self.metadata_embed.parameters())
        
        return counts


# ============================================================================
# Model Builder
# ============================================================================

def build_model(config: dict) -> ISICClassifier:
    """Build model from config.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    
    Returns
    -------
    ISICClassifier
        Model instance.
    """
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    
    # Determine input channels
    use_segmentation_mask = data_cfg.get("use_segmentation_mask", False)
    in_channels = 4 if use_segmentation_mask else 3
    
    return ISICClassifier(
        backbone_name=model_cfg.get("backbone", "swin_small_patch4_window7_224.ms_in22k_ft_in1k"),
        num_classes=model_cfg.get("num_classes", 8),
        image_size=model_cfg.get("image_size", 224),
        in_channels=in_channels,
        pretrained=model_cfg.get("pretrained", True),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.2)),
        metadata_enabled=model_cfg.get("metadata", {}).get("enabled", False),
        metadata_embed_dim=model_cfg.get("metadata", {}).get("embed_dim", 64),
        classifier_hidden_dim=model_cfg.get("classifier", {}).get("hidden_dim", 512),
        classifier_dropout=float(model_cfg.get("classifier", {}).get("dropout", 0.3)),
    )
