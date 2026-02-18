"""
Hybrid Segmentation-Guided Vision Transformer for ISIC Classification.

Architecture:
    1. Backbone: Swin-Small / EVA-Small / ViT-Base (384x384)
    2. Segmentation Branch: Lightweight Attention U-Net
    3. Feature Fusion: Attention / Concat / Cross-Attention
    4. Metadata Fusion: MLP embedding
    5. Classifier Head: MLP with dropout
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation import SegmentationBranch, build_fusion_module


# ============================================================================
# Metadata Embedding
# ============================================================================

class MetadataEmbedding(nn.Module):
    """Embed categorical and numerical metadata features.
    
    Parameters
    ----------
    feature_config : dict
        Configuration for each metadata feature.
        Expected keys: "age_approx", "sex", "anatom_site_general"
    embed_dim : int
        Output embedding dimension.
    """

    def __init__(
        self,
        feature_config: Optional[dict] = None,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Default categorical mappings
        self.sex_vocab = {"male": 0, "female": 1, "unknown": 2}
        self.site_vocab = {
            "anterior torso": 0,
            "upper extremity": 1,
            "lower extremity": 2,
            "posterior torso": 3,
            "lateral torso": 4,
            "head/neck": 5,
            "palms/soles": 6,
            "oral/genital": 7,
            "unknown": 8,
        }
        
        # Embeddings
        self.sex_embed = nn.Embedding(len(self.sex_vocab), embed_dim // 4)
        self.site_embed = nn.Embedding(len(self.site_vocab), embed_dim // 2)
        
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
        """
        Parameters
        ----------
        age : torch.Tensor
            Normalized age values (B,) or (B, 1).
        sex : torch.Tensor
            Sex indices (B,).
        site : torch.Tensor
            Anatomical site indices (B,).
        
        Returns
        -------
        torch.Tensor
            Metadata embedding (B, embed_dim).
        """
        if age.dim() == 1:
            age = age.unsqueeze(1)
        
        age_emb = self.age_proj(age)  # (B, embed_dim // 4)
        sex_emb = self.sex_embed(sex)  # (B, embed_dim // 4)
        site_emb = self.site_embed(site)  # (B, embed_dim // 2)
        
        # Concatenate
        concat = torch.cat([age_emb, sex_emb, site_emb], dim=1)  # (B, embed_dim)
        
        return self.fusion(concat)


# ============================================================================
# Main Hybrid Model
# ============================================================================

class HybridViT(nn.Module):
    """Hybrid Segmentation-Guided Vision Transformer.
    
    Parameters
    ----------
    backbone_name : str
        timm model name for backbone.
    num_classes : int
        Number of classification classes.
    image_size : int
        Input image size.
    pretrained : bool
        Whether to use pretrained backbone weights.
    drop_path_rate : float
        DropPath rate for stochastic depth.
    seg_enabled : bool
        Whether to enable segmentation branch.
    seg_encoder_channels : List[int]
        Segmentation encoder channels.
    seg_decoder_channels : List[int]
        Segmentation decoder channels.
    fusion_type : str
        Feature fusion type: "attention", "concat", "cross_attention".
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
        backbone_name: str = "swin_small_patch4_window12_384",
        num_classes: int = 8,
        image_size: int = 384,
        pretrained: bool = True,
        drop_path_rate: float = 0.2,
        seg_enabled: bool = False,
        seg_encoder_channels: List[int] = [64, 128, 256, 512],
        seg_decoder_channels: List[int] = [256, 128, 64, 32],
        fusion_type: str = "attention",
        metadata_enabled: bool = False,
        metadata_embed_dim: int = 64,
        classifier_hidden_dim: int = 512,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        
        self.seg_enabled = seg_enabled
        self.metadata_enabled = metadata_enabled
        self.num_classes = num_classes
        self.image_size = image_size
        
        # =====================================================================
        # Backbone: Swin / EVA / ViT
        # =====================================================================
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            drop_path_rate=drop_path_rate,
        )
        self.backbone_dim = self.backbone.num_features
        
        # =====================================================================
        # Segmentation Branch (optional)
        # =====================================================================
        if seg_enabled:
            self.seg_branch = SegmentationBranch(
                in_channels=3,
                encoder_channels=seg_encoder_channels,
                decoder_channels=seg_decoder_channels,
            )
            
            # Fusion module
            seg_feature_channels = seg_decoder_channels[-1]
            self.fusion = build_fusion_module(
                fusion_type=fusion_type,
                transformer_dim=self.backbone_dim,
                seg_channels=seg_feature_channels,
                output_dim=self.backbone_dim,
            )
        
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
        
        # Initialize classifier
        self._init_classifier()

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
        return_seg_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images (B, 3, H, W).
        metadata : Optional[Dict]
            Metadata dict with "age", "sex", "site" tensors.
        return_seg_mask : bool
            Whether to return segmentation mask.
        
        Returns
        -------
        Dict
            "logits": Classification logits (B, num_classes)
            "seg_mask": Segmentation mask (B, 1, H, W) if enabled
            "features": Backbone features (B, D)
        """
        outputs = {}
        
        # Backbone forward
        features = self.backbone(x)  # (B, D)
        outputs["features"] = features
        
        # Segmentation branch
        if self.seg_enabled:
            seg_mask, dec_features, bottleneck = self.seg_branch(x)
            outputs["seg_mask"] = seg_mask
            
            # Fuse with segmentation features
            seg_features = dec_features[-1]  # Use last decoder features
            features = self.fusion(features, seg_features)
        
        # Metadata fusion
        if self.metadata_enabled and metadata is not None:
            meta_embed = self.metadata_embed(
                age=metadata["age"],
                sex=metadata["sex"],
                site=metadata["site"],
            )
            features = torch.cat([features, meta_embed], dim=1)
        
        # Classification
        logits = self.classifier(features)
        outputs["logits"] = logits
        
        return outputs

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_head_parameters(self) -> List[nn.Parameter]:
        """Get head parameters (classifier + optional seg/fusion/metadata)."""
        params = list(self.classifier.parameters())
        
        if self.seg_enabled:
            params.extend(self.seg_branch.parameters())
            params.extend(self.fusion.parameters())
        
        if self.metadata_enabled:
            params.extend(self.metadata_embed.parameters())
        
        return params

    def get_layerwise_lr_groups(
        self,
        base_lr: float = 5e-5,
        decay_rate: float = 0.75,
        weight_decay: float = 0.05,
    ) -> List[Dict]:
        """Get parameter groups with layer-wise learning rate decay.
        
        For Swin Transformer, applies decay across stages.
        """
        param_groups = []
        
        # Head parameters (highest LR)
        head_params = []
        for param in self.classifier.parameters():
            if param.requires_grad:
                head_params.append(param)
        
        if self.seg_enabled:
            for param in self.seg_branch.parameters():
                if param.requires_grad:
                    head_params.append(param)
            for param in self.fusion.parameters():
                if param.requires_grad:
                    head_params.append(param)
        
        if self.metadata_enabled:
            for param in self.metadata_embed.parameters():
                if param.requires_grad:
                    head_params.append(param)
        
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "head",
            })
        
        # Backbone layers with decay
        # Swin has: patch_embed, layers.0-3, norm
        # We apply decay from early to late layers
        
        backbone_name = type(self.backbone).__name__.lower()
        
        if "swin" in backbone_name:
            # Swin Transformer layers
            layer_groups = self._get_swin_layer_groups(base_lr, decay_rate, weight_decay)
        elif "eva" in backbone_name:
            # EVA layers similar to ViT
            layer_groups = self._get_vit_layer_groups(base_lr, decay_rate, weight_decay)
        else:
            # Generic ViT
            layer_groups = self._get_vit_layer_groups(base_lr, decay_rate, weight_decay)
        
        param_groups.extend(layer_groups)
        
        return param_groups

    def _get_swin_layer_groups(
        self,
        base_lr: float,
        decay_rate: float,
        weight_decay: float,
    ) -> List[Dict]:
        """Get parameter groups for Swin Transformer."""
        groups = []
        
        # Patch embed (deepest, lowest LR)
        embed_params = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if "patch_embed" in name:
                embed_params.append(param)
        
        if embed_params:
            # Swin has 4 stages, embed is "stage -1"
            lr = base_lr * (decay_rate ** 5)
            groups.append({
                "params": embed_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "embed",
            })
        
        # Stages 0-3
        for stage_idx in range(4):
            stage_params = []
            for name, param in self.backbone.named_parameters():
                if not param.requires_grad:
                    continue
                if f"layers.{stage_idx}." in name:
                    stage_params.append(param)
            
            if stage_params:
                # Earlier stages get more decay
                lr = base_lr * (decay_rate ** (4 - stage_idx))
                groups.append({
                    "params": stage_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "name": f"stage_{stage_idx}",
                })
        
        # Final norm (full LR)
        norm_params = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name and "layers." not in name:
                norm_params.append(param)
        
        if norm_params:
            groups.append({
                "params": norm_params,
                "lr": base_lr,
                "weight_decay": 0.0,  # No decay for norms
                "name": "norm",
            })
        
        return groups

    def _get_vit_layer_groups(
        self,
        base_lr: float,
        decay_rate: float,
        weight_decay: float,
    ) -> List[Dict]:
        """Get parameter groups for ViT / EVA."""
        groups = []
        
        # Embed parameters
        embed_params = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ["patch_embed", "pos_embed", "cls_token"]):
                embed_params.append(param)
        
        # Count blocks
        num_blocks = 0
        for name, _ in self.backbone.named_parameters():
            if "blocks." in name:
                try:
                    block_idx = int(name.split("blocks.")[1].split(".")[0])
                    num_blocks = max(num_blocks, block_idx + 1)
                except:
                    pass
        
        if embed_params:
            lr = base_lr * (decay_rate ** (num_blocks + 1))
            groups.append({
                "params": embed_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "embed",
            })
        
        # Blocks with decay
        for block_idx in range(num_blocks):
            block_params = []
            for name, param in self.backbone.named_parameters():
                if not param.requires_grad:
                    continue
                if f"blocks.{block_idx}." in name:
                    block_params.append(param)
            
            if block_params:
                lr = base_lr * (decay_rate ** (num_blocks - block_idx))
                groups.append({
                    "params": block_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "name": f"block_{block_idx}",
                })
        
        # Final norm
        norm_params = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name and "blocks." not in name:
                norm_params.append(param)
        
        if norm_params:
            groups.append({
                "params": norm_params,
                "lr": base_lr,
                "weight_decay": 0.0,
                "name": "norm",
            })
        
        return groups

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        classifier = sum(p.numel() for p in self.classifier.parameters())
        
        counts = {
            "total": total,
            "trainable": trainable,
            "backbone": backbone,
            "classifier": classifier,
        }
        
        if self.seg_enabled:
            counts["segmentation"] = sum(p.numel() for p in self.seg_branch.parameters())
            counts["fusion"] = sum(p.numel() for p in self.fusion.parameters())
        
        if self.metadata_enabled:
            counts["metadata"] = sum(p.numel() for p in self.metadata_embed.parameters())
        
        return counts


# ============================================================================
# Factory function
# ============================================================================

def build_model(config: dict) -> HybridViT:
    """Build model from config dict.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary (from config.yaml).
    
    Returns
    -------
    HybridViT
        Initialized model.
    """
    model_cfg = config.get("model", {})
    
    return HybridViT(
        backbone_name=model_cfg.get("backbone", "swin_small_patch4_window12_384"),
        num_classes=config.get("classes", {}).get("num_classes", 8),
        image_size=model_cfg.get("image_size", 384),
        pretrained=model_cfg.get("pretrained", True),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.2),
        seg_enabled=model_cfg.get("segmentation", {}).get("enabled", False),
        seg_encoder_channels=model_cfg.get("segmentation", {}).get(
            "encoder_channels", [64, 128, 256, 512]
        ),
        seg_decoder_channels=model_cfg.get("segmentation", {}).get(
            "decoder_channels", [256, 128, 64, 32]
        ),
        fusion_type=model_cfg.get("segmentation", {}).get("fusion_type", "attention"),
        metadata_enabled=model_cfg.get("metadata", {}).get("enabled", False),
        metadata_embed_dim=model_cfg.get("metadata", {}).get("embed_dim", 64),
        classifier_hidden_dim=model_cfg.get("classifier", {}).get("hidden_dim", 512),
        classifier_dropout=model_cfg.get("classifier", {}).get("dropout", 0.3),
    )


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    
    # Test basic model
    print("\n=== Testing basic model (no seg, no metadata) ===")
    model = HybridViT(
        backbone_name="swin_small_patch4_window12_384",
        num_classes=8,
        pretrained=False,  # False for quick test
        seg_enabled=False,
        metadata_enabled=False,
    ).to(device)
    
    counts = model.count_parameters()
    print(f"Parameters: {counts}")
    
    x = torch.randn(2, 3, 384, 384, device=device)
    outputs = model(x)
    print(f"Input: {x.shape}")
    print(f"Logits: {outputs['logits'].shape}")
    
    # Test with metadata
    print("\n=== Testing with metadata ===")
    model_meta = HybridViT(
        backbone_name="swin_small_patch4_window12_384",
        num_classes=8,
        pretrained=False,
        seg_enabled=False,
        metadata_enabled=True,
    ).to(device)
    
    metadata = {
        "age": torch.randn(2, device=device),
        "sex": torch.randint(0, 3, (2,), device=device),
        "site": torch.randint(0, 9, (2,), device=device),
    }
    
    outputs = model_meta(x, metadata=metadata)
    print(f"Logits (with metadata): {outputs['logits'].shape}")
    
    # Test layerwise LR groups
    print("\n=== Layerwise LR groups ===")
    model.unfreeze_backbone()
    groups = model.get_layerwise_lr_groups(base_lr=5e-5, decay_rate=0.75)
    for g in groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:12s}: lr={g['lr']:.2e}, params={n_params:,}")
