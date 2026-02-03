"""
Output Heads for Gaussian Attributes

각 Gaussian 속성별 예측 헤드
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianHeads(nn.Module):
    """
    Gaussian 속성 예측 헤드들
    
    출력 형식 (log/logit space):
    - xyz: tanh [-1, 1] (voxel 내 정규화 좌표)
    - rotation: normalized quaternion
    - scale: log space
    - opacity: logit space
    - sh: raw values
    """
    
    def __init__(self, latent_dim: int, sh_dim: int = 48):
        super().__init__()
        
        self.head_pos = nn.Linear(latent_dim, 3)
        self.head_rot = nn.Linear(latent_dim, 4)
        self.head_scale = nn.Linear(latent_dim, 3)
        self.head_opacity = nn.Linear(latent_dim, 1)
        self.head_sh_dc = nn.Linear(latent_dim, 3)
        self.head_sh_rest = nn.Linear(latent_dim, sh_dim - 3)
        

        self.max_scale = 20.0  # scale의 최대값 제한 (실제 scale = exp(output))
    
    def forward(self, features: torch.Tensor) -> tuple:
        """
        Args:
            features: [B, M, latent_dim] decoder output
        
        Returns:
            xyz: [B, M, 3]
            rot: [B, M, 4]
            scale: [B, M, 3]
            opacity: [B, M, 1]
            sh: [B, M, 48]
        """
        # Position: Voxel 내부 좌표 (-1 ~ 1)
        xyz = torch.tanh(self.head_pos(features))
        
        # Rotation: Quaternion 정규화
        rot = F.normalize(self.head_rot(features), dim=-1)
        
        # Scale: log space (실제 scale = exp(출력)), 최대값 제한
        scale = self.head_scale(features).clamp(min=-20.0, max=3.0)
        
        # Opacity: logit space (실제 opacity = sigmoid(출력))
        opacity = self.head_opacity(features)
        
        # SH: 색상 계수
        sh_dc = self.head_sh_dc(features)
        sh_rest = self.head_sh_rest(features)
        sh = torch.cat([sh_dc, sh_rest], dim=-1)
        
        return xyz, rot, scale, opacity, sh
