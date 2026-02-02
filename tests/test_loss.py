"""
Loss Function Tests

손실 함수 테스트
"""

import pytest
import torch

from gs_merge.loss import parse_gaussian_tensor, model_output_to_dict


class TestLossUtils:
    """손실 유틸리티 테스트"""
    
    def test_parse_gaussian_tensor(self, device):
        """Gaussian 텐서 파싱 테스트"""
        # [B, N, 59] 텐서
        data = torch.randn(2, 32, 59, device=device)
        
        parsed = parse_gaussian_tensor(data)
        
        assert 'xyz' in parsed
        assert 'rotation' in parsed
        assert 'scale' in parsed
        assert 'opacity' in parsed
        assert 'sh_dc' in parsed
        assert 'sh_rest' in parsed
        
        assert parsed['xyz'].shape == (2, 32, 3)
        assert parsed['rotation'].shape == (2, 32, 4)
        assert parsed['scale'].shape == (2, 32, 3)
        assert parsed['opacity'].shape == (2, 32, 1)
        assert parsed['sh_dc'].shape == (2, 32, 3)
        assert parsed['sh_rest'].shape == (2, 32, 45)
    
    def test_parse_gaussian_tensor_2d(self, device):
        """2D 텐서 파싱 테스트"""
        # [N, 59] 텐서
        data = torch.randn(32, 59, device=device)
        
        parsed = parse_gaussian_tensor(data)
        
        # 배치 차원 자동 추가
        assert parsed['xyz'].shape == (1, 32, 3)
    
    def test_model_output_to_dict(self, device):
        """모델 출력 변환 테스트"""
        B, M = 2, 16
        
        xyz = torch.randn(B, M, 3, device=device)
        rot = torch.randn(B, M, 4, device=device)
        scale = torch.randn(B, M, 3, device=device)
        opacity = torch.randn(B, M, 1, device=device)
        sh = torch.randn(B, M, 48, device=device)
        
        output = model_output_to_dict(xyz, rot, scale, opacity, sh)
        
        assert output['xyz'].shape == (B, M, 3)
        assert output['rotation'].shape == (B, M, 4)
        assert output['scale'].shape == (B, M, 3)
        assert output['opacity'].shape == (B, M, 1)
        assert output['sh_dc'].shape == (B, M, 3)
        assert output['sh_rest'].shape == (B, M, 45)
