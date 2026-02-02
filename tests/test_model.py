"""
Model Tests

GaussianMergingAE 모델 테스트
"""

import pytest
import torch

from gs_merge.model import GaussianMergingAE


class TestGaussianMergingAE:
    """GaussianMergingAE 테스트"""
    
    def test_model_creation(self, device):
        """모델 생성 테스트"""
        model = GaussianMergingAE(
            input_dim=59,
            latent_dim=128,
            num_inputs=64,
            num_queries=32,
            nhead=4,
            num_enc_layers=2,
            num_dec_layers=2,
            max_octree_level=8
        ).to(device)
        
        assert model is not None
        assert model.num_parameters > 0
    
    def test_forward_pass(self, sample_gaussian_data, sample_voxel_levels, sample_padding_mask, device):
        """Forward pass 테스트"""
        model = GaussianMergingAE(
            input_dim=59,
            latent_dim=128,
            num_inputs=64,
            num_queries=32,
            nhead=4,
            num_enc_layers=2,
            num_dec_layers=2
        ).to(device)
        
        outputs = model(
            sample_gaussian_data,
            sample_voxel_levels,
            sample_padding_mask
        )
        
        # 5개 출력 + tgt_padding_mask
        assert len(outputs) == 6
        
        xyz, rot, scale, opacity, sh, tgt_mask = outputs
        B = sample_gaussian_data.shape[0]
        M = 32  # num_queries
        
        assert xyz.shape == (B, M, 3)
        assert rot.shape == (B, M, 4)
        assert scale.shape == (B, M, 3)
        assert opacity.shape == (B, M, 1)
        assert sh.shape == (B, M, 48)
        assert tgt_mask.shape == (B, M)
    
    def test_gumbel_noise_scale(self, device):
        """Gumbel noise scale getter/setter 테스트"""
        model = GaussianMergingAE(
            input_dim=59,
            latent_dim=128,
            num_inputs=64,
            num_queries=32
        ).to(device)
        
        # 초기값 확인
        assert model.gumbel_noise_scale == 0.3
        
        # setter 테스트
        model.gumbel_noise_scale = 0.5
        assert model.gumbel_noise_scale == 0.5
        
        # 메서드 테스트
        model.set_gumbel_noise_scale(0.8)
        assert model.get_gumbel_noise_scale() == 0.8
        
        # 리셋 테스트
        model.reset_gumbel_noise_scale()
        assert model.gumbel_noise_scale == 0.3
    
    def test_training_mode(self, sample_gaussian_data, sample_voxel_levels, device):
        """Training/eval 모드 테스트"""
        model = GaussianMergingAE(
            input_dim=59,
            latent_dim=128,
            num_inputs=64,
            num_queries=32
        ).to(device)
        
        # Training mode
        model.train()
        outputs_train = model(sample_gaussian_data, sample_voxel_levels)
        
        # Eval mode
        model.eval()
        with torch.no_grad():
            outputs_eval = model(sample_gaussian_data, sample_voxel_levels)
        
        # 둘 다 valid output 생성
        assert len(outputs_train) == 6
        assert len(outputs_eval) == 6
