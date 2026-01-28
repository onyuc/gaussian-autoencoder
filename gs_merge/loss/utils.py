"""
유틸리티 함수: Gaussian 텐서 파싱, 딕셔너리 변환
"""

import torch
from typing import Dict


def parse_gaussian_tensor(data: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    [B, N, 59] 또는 [N, 59] 텐서를 Gaussian 속성 딕셔너리로 분해
    
    순서: xyz(3) + rot(4) + scale(3) + opacity(1) + sh_dc(3) + sh_rest(45)
    """
    if data.dim() == 2:
        data = data.unsqueeze(0)
    
    return {
        'xyz': data[..., :3],
        'rotation': data[..., 3:7],
        'scale': data[..., 7:10],
        'opacity': data[..., 10:11],
        'sh_dc': data[..., 11:14],
        'sh_rest': data[..., 14:59]
    }


def model_output_to_dict(
    xyz: torch.Tensor, 
    rot: torch.Tensor, 
    scale: torch.Tensor, 
    opacity: torch.Tensor, 
    sh: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    모델 출력을 Gaussian 딕셔너리로 변환
    
    Args:
        xyz: [B, M, 3]
        rot: [B, M, 4]
        scale: [B, M, 3] (log space)
        opacity: [B, M, 1] (logit space)
        sh: [B, M, 48] (sh_dc + sh_rest)
    
    Returns:
        딕셔너리 형태의 Gaussian 속성
    """
    return {
        'xyz': xyz,
        'rotation': rot,
        'scale': scale,
        'opacity': opacity,
        'sh_dc': sh[..., :3],
        'sh_rest': sh[..., 3:48]
    }
