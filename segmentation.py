"""
Segmentation branch for hybrid classification model.

Includes:
    - Lightweight U-Net encoder/decoder
    - Attention U-Net variant
    - Feature fusion modules (attention, concat, cross-attention)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Basic Blocks
# ============================================================================

class ConvBlock(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for Attention U-Net.
    
    Learns to focus on relevant spatial regions using gating signal.
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        inter_channels = inter_channels or skip_channels // 2
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        gate: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        gate : torch.Tensor
            Gating signal from decoder (lower resolution).
        skip : torch.Tensor
            Skip connection from encoder (higher resolution).
        
        Returns
        -------
        torch.Tensor
            Attention-weighted skip connection.
        """
        # Upsample gate to match skip resolution
        gate_up = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        
        g = self.W_g(gate_up)
        x = self.W_x(skip)
        
        psi = self.relu(g + x)
        psi = self.psi(psi)
        
        return skip * psi


# ============================================================================
# Lightweight U-Net Encoder
# ============================================================================

class UNetEncoder(nn.Module):
    """Lightweight U-Net encoder."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [64, 128, 256, 512],
    ) -> None:
        super().__init__()
        self.channels = channels
        
        # Encoder blocks
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        for ch in channels:
            self.enc_blocks.append(ConvBlock(prev_channels, ch))
            self.pools.append(nn.MaxPool2d(2))
            prev_channels = ch
        
        # Bottleneck
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns
        -------
        tuple
            (encoder_features, bottleneck_features)
        """
        enc_features = []
        
        for enc_block, pool in zip(self.enc_blocks, self.pools):
            x = enc_block(x)
            enc_features.append(x)
            x = pool(x)
        
        x = self.bottleneck(x)
        
        return enc_features, x


# ============================================================================
# Attention U-Net Decoder
# ============================================================================

class AttentionUNetDecoder(nn.Module):
    """Attention U-Net decoder with attention gates."""

    def __init__(
        self,
        encoder_channels: List[int] = [64, 128, 256, 512],
        decoder_channels: List[int] = [256, 128, 64, 32],
    ) -> None:
        super().__init__()
        
        # Reverse encoder channels for decoder
        enc_channels_rev = list(reversed(encoder_channels))
        bottleneck_channels = encoder_channels[-1] * 2
        
        self.ups = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        
        prev_channels = bottleneck_channels
        for i, (enc_ch, dec_ch) in enumerate(zip(enc_channels_rev, decoder_channels)):
            self.ups.append(
                nn.ConvTranspose2d(prev_channels, dec_ch, kernel_size=2, stride=2)
            )
            self.attention_gates.append(
                AttentionGate(gate_channels=dec_ch, skip_channels=enc_ch)
            )
            self.dec_blocks.append(
                ConvBlock(dec_ch + enc_ch, dec_ch)
            )
            prev_channels = dec_ch
        
        # Final output
        self.final_conv = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

    def forward(
        self,
        enc_features: List[torch.Tensor],
        bottleneck: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        enc_features : List[torch.Tensor]
            Encoder features [e1, e2, e3, e4].
        bottleneck : torch.Tensor
            Bottleneck features.
        
        Returns
        -------
        tuple
            (segmentation_mask, decoder_features)
        """
        enc_features_rev = list(reversed(enc_features))
        dec_features = []
        
        x = bottleneck
        for up, attn, dec_block, enc_feat in zip(
            self.ups, self.attention_gates, self.dec_blocks, enc_features_rev
        ):
            x = up(x)
            # Match spatial dimensions
            if x.shape[2:] != enc_feat.shape[2:]:
                x = F.interpolate(x, size=enc_feat.shape[2:], mode="bilinear", align_corners=False)
            
            # Attention gate
            enc_feat_attn = attn(x, enc_feat)
            
            # Concatenate and decode
            x = torch.cat([x, enc_feat_attn], dim=1)
            x = dec_block(x)
            dec_features.append(x)
        
        mask = self.final_conv(x)
        
        return mask, dec_features


# ============================================================================
# Full Segmentation Module
# ============================================================================

class SegmentationBranch(nn.Module):
    """Complete segmentation branch with encoder and decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: List[int] = [64, 128, 256, 512],
        decoder_channels: List[int] = [256, 128, 64, 32],
    ) -> None:
        super().__init__()
        
        self.encoder = UNetEncoder(in_channels, encoder_channels)
        self.decoder = AttentionUNetDecoder(encoder_channels, decoder_channels)
        
        # Store output channels for fusion
        self.bottleneck_channels = encoder_channels[-1] * 2
        self.decoder_channels = decoder_channels

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Returns
        -------
        tuple
            (segmentation_mask, decoder_features, bottleneck_features)
        """
        enc_features, bottleneck = self.encoder(x)
        mask, dec_features = self.decoder(enc_features, bottleneck)
        
        return mask, dec_features, bottleneck


# ============================================================================
# Feature Fusion Modules
# ============================================================================

class AttentionFusion(nn.Module):
    """Fuse transformer features with segmentation attention mask.
    
    Multiplies transformer feature map by segmentation attention mask.
    """

    def __init__(
        self,
        transformer_dim: int,
        seg_channels: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        
        # Project segmentation features to attention weights
        self.seg_proj = nn.Sequential(
            nn.Conv2d(seg_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )
        
        # Project transformer features
        self.trans_proj = nn.Linear(transformer_dim, transformer_dim)

    def forward(
        self,
        trans_features: torch.Tensor,
        seg_features: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        trans_features : torch.Tensor
            Transformer CLS token or pooled features (B, D).
        seg_features : torch.Tensor
            Segmentation decoder features (B, C, H, W).
        spatial_shape : Tuple[int, int]
            Spatial shape (H, W) for transformer patch features.
        
        Returns
        -------
        torch.Tensor
            Fused features (B, D).
        """
        # Generate attention weights from segmentation
        attn = self.seg_proj(seg_features)  # (B, 1, H, W)
        attn_pooled = F.adaptive_avg_pool2d(attn, 1).squeeze(-1).squeeze(-1)  # (B, 1)
        
        # Scale transformer features
        trans_proj = self.trans_proj(trans_features)
        fused = trans_proj * (1 + attn_pooled)
        
        return fused


class ConcatFusion(nn.Module):
    """Concatenate segmentation and transformer features."""

    def __init__(
        self,
        transformer_dim: int,
        seg_channels: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        
        # Pool segmentation features
        self.seg_pool = nn.AdaptiveAvgPool2d(1)
        self.seg_proj = nn.Linear(seg_channels, output_dim // 2)
        
        # Project transformer features
        self.trans_proj = nn.Linear(transformer_dim, output_dim // 2)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        trans_features: torch.Tensor,
        seg_features: torch.Tensor,
        spatial_shape: Tuple[int, int] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        trans_features : torch.Tensor
            Transformer features (B, D).
        seg_features : torch.Tensor
            Segmentation features (B, C, H, W).
        
        Returns
        -------
        torch.Tensor
            Fused features (B, output_dim).
        """
        # Pool and project segmentation features
        seg_pooled = self.seg_pool(seg_features).flatten(1)  # (B, C)
        seg_proj = self.seg_proj(seg_pooled)  # (B, output_dim // 2)
        
        # Project transformer features
        trans_proj = self.trans_proj(trans_features)  # (B, output_dim // 2)
        
        # Concatenate and fuse
        concat = torch.cat([trans_proj, seg_proj], dim=1)  # (B, output_dim)
        fused = self.fusion(concat)
        
        return fused


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between transformer and segmentation features."""

    def __init__(
        self,
        transformer_dim: int,
        seg_channels: int,
        num_heads: int = 4,
        output_dim: int = None,
    ) -> None:
        super().__init__()
        output_dim = output_dim or transformer_dim
        
        # Project segmentation features to key/value
        self.seg_proj = nn.Conv2d(seg_channels, transformer_dim, 1)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=transformer_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # Output projection
        self.out_proj = nn.Linear(transformer_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        trans_features: torch.Tensor,
        seg_features: torch.Tensor,
        spatial_shape: Tuple[int, int] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        trans_features : torch.Tensor
            Transformer features (B, D).
        seg_features : torch.Tensor
            Segmentation features (B, C, H, W).
        
        Returns
        -------
        torch.Tensor
            Fused features (B, output_dim).
        """
        B = trans_features.size(0)
        
        # Project and flatten segmentation features
        seg_proj = self.seg_proj(seg_features)  # (B, D, H, W)
        seg_flat = seg_proj.flatten(2).transpose(1, 2)  # (B, H*W, D)
        
        # Transformer features as query
        query = trans_features.unsqueeze(1)  # (B, 1, D)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(query, seg_flat, seg_flat)
        attn_out = attn_out.squeeze(1)  # (B, D)
        
        # Residual connection and projection
        fused = trans_features + attn_out
        fused = self.norm(self.out_proj(fused))
        
        return fused


def build_fusion_module(
    fusion_type: str,
    transformer_dim: int,
    seg_channels: int,
    output_dim: int = None,
) -> nn.Module:
    """Build fusion module based on type.
    
    Parameters
    ----------
    fusion_type : str
        Type: "attention", "concat", "cross_attention".
    transformer_dim : int
        Transformer feature dimension.
    seg_channels : int
        Segmentation feature channels.
    output_dim : int
        Output dimension (default: transformer_dim).
    
    Returns
    -------
    nn.Module
        Fusion module.
    """
    output_dim = output_dim or transformer_dim
    
    if fusion_type == "attention":
        return AttentionFusion(transformer_dim, seg_channels)
    elif fusion_type == "concat":
        return ConcatFusion(transformer_dim, seg_channels, output_dim)
    elif fusion_type == "cross_attention":
        return CrossAttentionFusion(transformer_dim, seg_channels, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    
    # Test segmentation branch
    seg_branch = SegmentationBranch(
        in_channels=3,
        encoder_channels=[32, 64, 128, 256],
        decoder_channels=[128, 64, 32, 16],
    ).to(device)
    
    x = torch.randn(2, 3, 384, 384, device=device)
    mask, dec_features, bottleneck = seg_branch(x)
    
    print(f"Input: {x.shape}")
    print(f"Mask: {mask.shape}")
    print(f"Bottleneck: {bottleneck.shape}")
    print(f"Decoder features: {[f.shape for f in dec_features]}")
    
    # Test fusion modules
    trans_features = torch.randn(2, 768, device=device)
    seg_features = dec_features[-1]  # Last decoder features
    
    for fusion_type in ["attention", "concat", "cross_attention"]:
        fusion = build_fusion_module(
            fusion_type=fusion_type,
            transformer_dim=768,
            seg_channels=seg_features.size(1),
            output_dim=512,
        ).to(device)
        
        fused = fusion(trans_features, seg_features)
        print(f"{fusion_type} fusion output: {fused.shape}")
