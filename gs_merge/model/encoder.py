"""
Positional Encoding for Transformer

3D 좌표를 고차원 피처로 변환
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Learnable Positional Encoding for 3D coordinates"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.pos_proj = nn.Linear(3, d_model)
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [B, N, 3] 3D coordinates
        Returns:
            [B, N, d_model] positional features
        """
        return self.pos_proj(xyz)


class FourierPositionalEncoding(nn.Module):
    """Fixed Fourier Feature Positional Encoding"""
    
    def __init__(self, d_model: int, num_frequencies: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_frequencies = num_frequencies
        
        # Frequency bands
        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freqs', freqs)
        
        # Project from fourier features to d_model
        fourier_dim = 3 * num_frequencies * 2  # sin + cos for each dim
        self.proj = nn.Linear(fourier_dim + 3, d_model)  # +3 for raw xyz
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [B, N, 3] 3D coordinates
        Returns:
            [B, N, d_model] positional features
        """
        # Fourier features with NeRF-style pi scaling
        xyz_freq = xyz.unsqueeze(-1) * self.freqs * torch.pi  # [B, N, 3, F]
        sin_feat = torch.sin(xyz_freq)
        cos_feat = torch.cos(xyz_freq)
        
        # Flatten and concat
        fourier = torch.cat([sin_feat, cos_feat], dim=-1)  # [B, N, 3, 2F]
        fourier = fourier.view(*xyz.shape[:-1], -1)  # [B, N, 6F]
        
        # Concat with raw xyz
        features = torch.cat([xyz, fourier], dim=-1)  # [B, N, 3 + 6F]
        
        return self.proj(features)


class RatioEncoding(nn.Module):
    """Compression Ratio Encoding
    
    Supports both [B] and [B, 1] input shapes
    """
    
    def __init__(self, d_model: int, num_frequencies: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_frequencies = num_frequencies
        
        # Frequency bands
        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freqs', freqs)
        
        # Project from fourier features to d_model
        fourier_dim = num_frequencies * 2  # sin + cos for each dim
        self.mlp = nn.Linear(fourier_dim + 1, d_model)
    
    def forward(self, ratio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ratio: [B] or [B, 1] compression ratio
        Returns:
            [B, d_model] positional features
        """
        # Normalize to [B, 1] if needed
        if ratio.dim() == 1:
            ratio = ratio.unsqueeze(-1)  # [B] -> [B, 1]
        
        # Fourier features with NeRF-style pi scaling
        ratio_freq = ratio * self.freqs * torch.pi  # [B, 1, F]
        sin_feat = torch.sin(ratio_freq)
        cos_feat = torch.cos(ratio_freq)
        
        # Flatten and concat
        fourier = torch.cat([sin_feat, cos_feat], dim=-1)  # [B, 1, 2F]
        fourier = fourier.squeeze(1)  # [B, 2F]
        
        # Concat with raw ratio
        ratio_val = ratio.squeeze(-1)  # [B]
        features = torch.cat([ratio_val.unsqueeze(-1), fourier], dim=-1)  # [B, 1 + 2F]
        
        return self.mlp(features)  # [B, d_model]
    