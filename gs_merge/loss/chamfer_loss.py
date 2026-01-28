"""
Chamfer Loss: 간단한 Chamfer Distance 기반 손실 (Baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ChamferLoss(nn.Module):
    """단순 Chamfer Distance 기반 손실"""
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        rot_weight: float = 0.5,
        scale_weight: float = 0.5,
        opacity_weight: float = 0.3,
        sh_weight: float = 0.2
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.scale_weight = scale_weight
        self.opacity_weight = opacity_weight
        self.sh_weight = sh_weight
    
    def forward(
        self,
        input_g: Dict[str, torch.Tensor],
        output_g: Dict[str, torch.Tensor],
        input_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Chamfer distance + 매칭 기반 속성 손실"""
        B, N, _ = input_g['xyz'].shape
        M = output_g['xyz'].shape[1]
        device = input_g['xyz'].device
        
        # 거리 행렬
        dist = torch.cdist(output_g['xyz'], input_g['xyz'], p=2)
        if input_mask is not None:
            dist = dist.masked_fill(input_mask.unsqueeze(1), float('inf'))
        
        # 매칭
        min_out_to_in, match_idx = dist.min(dim=2)
        min_in_to_out, _ = dist.min(dim=1)
        
        # Position Loss
        if input_mask is not None:
            valid = ~input_mask
            pos_loss = min_out_to_in.mean() + (min_in_to_out * valid).sum() / valid.sum().clamp(min=1)
        else:
            pos_loss = min_out_to_in.mean() + min_in_to_out.mean()
        pos_loss = pos_loss / 2
        
        # 매칭된 속성
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, M)
        matched_rot = input_g['rotation'][batch_idx, match_idx]
        matched_scale = input_g['scale'][batch_idx, match_idx]
        matched_opacity = input_g['opacity'][batch_idx, match_idx]
        
        # Rotation Loss
        rot_dot = (output_g['rotation'] * matched_rot).sum(dim=-1).abs()
        rot_loss = (1 - rot_dot).mean()
        
        # Scale Loss
        scale_loss = F.l1_loss(output_g['scale'], matched_scale)
        
        # Opacity Loss
        opacity_loss = F.l1_loss(output_g['opacity'], matched_opacity)
        
        # SH Loss
        if 'sh_dc' in input_g and 'sh_dc' in output_g:
            matched_sh_dc = input_g['sh_dc'][batch_idx, match_idx]
            matched_sh_rest = input_g['sh_rest'][batch_idx, match_idx]
            sh_loss = F.l1_loss(output_g['sh_dc'], matched_sh_dc) + F.l1_loss(output_g['sh_rest'], matched_sh_rest)
        else:
            sh_loss = torch.tensor(0.0, device=device)
        
        # Total
        total_loss = (
            self.pos_weight * pos_loss +
            self.rot_weight * rot_loss +
            self.scale_weight * scale_loss +
            self.opacity_weight * opacity_loss +
            self.sh_weight * sh_loss
        )
        
        return total_loss, {
            'loss_pos': pos_loss.item(),
            'loss_rot': rot_loss.item(),
            'loss_scale': scale_loss.item(),
            'loss_opacity': opacity_loss.item(),
            'loss_sh': sh_loss.item() if isinstance(sh_loss, torch.Tensor) else sh_loss,
            'loss_total': total_loss.item()
        }
