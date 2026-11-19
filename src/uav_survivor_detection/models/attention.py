"""
Attention mechanisms for small-object detection enhancement.

This module implements attention mechanisms to augment YOLO11n for improved
detection of small objects (survivors) in UAV aerial imagery. All mechanisms
are designed to be lightweight with minimal parameter overhead.

References:
    - GAM: Liu et al. (2021) "Global Attention Mechanism: Retain Information to
      Enhance Channel-Spatial Interactions" https://arxiv.org/abs/2112.05561
    - CBAM: Woo et al. (2018) "CBAM: Convolutional Block Attention Module"
      https://arxiv.org/abs/1807.06521
    - C3k2-ARC: Rotation-aware convolution for aerial view robustness
    - CL-Concat: Cross-level feature concatenation for multi-scale fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    
    Applies channel-wise attention by computing global statistics and learning
    channel importance weights through a bottleneck MLP architecture.
    
    Source: Component of CBAM (Woo et al., 2018)
    Reference: https://arxiv.org/abs/1807.06521
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for bottleneck (default: 16)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Shared MLP for both avg and max pooled features
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-weighted tensor of shape (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        # Global average pooling and max pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        
        # Shared MLP for both pooling results
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        
        # Combine and apply sigmoid
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Applies spatial attention by aggregating channel information and learning
    spatial importance weights through convolution.
    
    Source: Component of CBAM (Woo et al., 2018)
    Reference: https://arxiv.org/abs/1807.06521
    
    Args:
        kernel_size: Convolution kernel size (default: 7)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Convolution to generate spatial attention map
        self.conv = nn.Conv2d(
            2, 1, 
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-weighted tensor of shape (B, C, H, W)
        """
        # Channel-wise statistics: avg and max along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Generate spatial attention map
        attention = torch.sigmoid(self.conv(pooled))
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Sequentially applies channel attention followed by spatial attention to
    refine feature maps. CBAM is lightweight and can be integrated into any
    CNN architecture with negligible overhead.
    
    Source: Woo et al. (2018)
    "CBAM: Convolutional Block Attention Module"
    Reference: https://arxiv.org/abs/1807.06521
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for channel attention (default: 16)
        kernel_size: Spatial attention kernel size (default: 7)
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CBAM attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-weighted tensor of shape (B, C, H, W)
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class GAM(nn.Module):
    """
    Global Attention Mechanism.
    
    Reduces information loss in attention mechanisms by preserving both channel
    and spatial information through 3D permutation with MLP for channel attention
    and convolutional spatial attention. Unlike CBAM, GAM amplifies global
    cross-dimensional interactions.
    
    Source: Liu et al. (2021)
    "Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions"
    Reference: https://arxiv.org/abs/2112.05561
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Channel attention with 3D permutation
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        
        # Spatial attention with convolutions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(channels),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GAM attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-weighted tensor of shape (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        # Channel attention branch
        # Global pooling and MLP
        x_permute = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_channel = x_permute.mean(dim=1)  # (B, C)
        channel_weights = self.channel_attention(x_channel)
        channel_weights = torch.sigmoid(channel_weights).view(b, c, 1, 1)
        
        # Spatial attention branch
        spatial_weights = self.spatial_attention(x)
        spatial_weights = torch.sigmoid(spatial_weights)
        
        # Combine channel and spatial attention
        attention = channel_weights * spatial_weights
        
        return x * attention


class C3k2_ARC(nn.Module):
    """
    C3k2 with Adaptive Rotation-aware Convolution.
    
    Implements rotation-aware convolution to handle aerial view variations in
    UAV imagery. This module applies multiple rotated kernels to capture
    orientation-invariant features, which is crucial for detecting survivors
    at various orientations in aerial views.
    
    Source: Adapted from rotation-equivariant CNN techniques for aerial imagery
    Note: This is a simplified implementation focusing on rotation robustness
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Base convolution kernel size (default: 3)
        num_rotations: Number of rotation angles to consider (default: 4)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_rotations: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_rotations = num_rotations
        
        # Base convolution layers for different rotation angles
        # Each rotation branch learns rotation-specific features
        self.rotation_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                out_channels // num_rotations,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            )
            for _ in range(num_rotations)
        ])
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation-aware convolution.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Rotation-aware features of shape (B, out_channels, H, W)
        """
        # Apply rotations to input and process each with corresponding conv
        rotation_features = []
        
        for i, conv in enumerate(self.rotation_convs):
            # Rotate input by (i * 90) degrees
            angle = i * (360 // self.num_rotations)
            if angle == 0:
                rotated_x = x
            elif angle == 90:
                rotated_x = torch.rot90(x, k=1, dims=[2, 3])
            elif angle == 180:
                rotated_x = torch.rot90(x, k=2, dims=[2, 3])
            elif angle == 270:
                rotated_x = torch.rot90(x, k=3, dims=[2, 3])
            else:
                # For other angles, use nearest rotation
                k = round(angle / 90)
                rotated_x = torch.rot90(x, k=k, dims=[2, 3])
            
            # Apply convolution
            features = conv(rotated_x)
            
            # Rotate back to original orientation
            if angle != 0:
                k_back = (360 - angle) // 90
                features = torch.rot90(features, k=k_back, dims=[2, 3])
            
            rotation_features.append(features)
        
        # Concatenate all rotation features
        out = torch.cat(rotation_features, dim=1)
        out = self.bn(out)
        out = self.act(out)
        
        return out


class CLConcat(nn.Module):
    """
    Cross-Level feature Concatenation module.
    
    Fuses features from multiple network levels (scales) to enhance multi-scale
    detection capability. This is particularly important for small-object detection
    where information from multiple feature pyramid levels needs to be combined.
    
    Source: Adapted from feature pyramid network (FPN) and cross-level fusion techniques
    
    Args:
        in_channels_list: List of input channel counts from different levels
        out_channels: Number of output channels after fusion
        fusion_method: Method to fuse features - 'concat' or 'add' (default: 'concat')
    """
    
    def __init__(
        self,
        in_channels_list: list,
        out_channels: int,
        fusion_method: str = 'concat'
    ):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.fusion_method = fusion_method
        
        # 1x1 convolutions to unify channel dimensions from different levels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels_list
        ])
        
        # Batch normalization for each lateral connection
        self.lateral_bns = nn.ModuleList([
            nn.BatchNorm2d(out_channels)
            for _ in in_channels_list
        ])
        
        # Final fusion convolution
        if fusion_method == 'concat':
            fusion_in_channels = out_channels * len(in_channels_list)
        else:  # 'add'
            fusion_in_channels = out_channels
            
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, features: list) -> torch.Tensor:
        """
        Apply cross-level feature concatenation.
        
        Args:
            features: List of feature tensors from different levels,
                     each of shape (B, C_i, H_i, W_i)
            
        Returns:
            Fused feature tensor of shape (B, out_channels, H_target, W_target)
        """
        if len(features) != len(self.in_channels_list):
            raise ValueError(
                f"Expected {len(self.in_channels_list)} feature maps, "
                f"got {len(features)}"
            )
        
        # Determine target spatial size (use the largest feature map size)
        target_size = features[0].shape[2:]
        
        # Process each level
        processed_features = []
        for i, (feat, lateral_conv, bn) in enumerate(
            zip(features, self.lateral_convs, self.lateral_bns)
        ):
            # Apply 1x1 conv to unify channels
            x = lateral_conv(feat)
            x = bn(x)
            
            # Resize to target size if needed
            if x.shape[2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            processed_features.append(x)
        
        # Fuse features
        if self.fusion_method == 'concat':
            fused = torch.cat(processed_features, dim=1)
        else:  # 'add'
            fused = sum(processed_features)
        
        # Final fusion convolution
        out = self.fusion_conv(fused)
        
        return out
