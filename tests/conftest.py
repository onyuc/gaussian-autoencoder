"""
Test fixtures for GS_Merge tests
"""

import pytest
import torch


@pytest.fixture
def device():
    """테스트용 디바이스"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def num_gaussians():
    return 64


@pytest.fixture
def latent_dim():
    return 128


@pytest.fixture
def sample_gaussian_data(batch_size, num_gaussians, device):
    """샘플 Gaussian 데이터 생성"""
    # 59 = xyz(3) + rot(4) + scale(3) + opacity(1) + sh(48)
    data = torch.randn(batch_size, num_gaussians, 59, device=device)
    
    # Normalize rotation (quaternion)
    data[..., 3:7] = torch.nn.functional.normalize(data[..., 3:7], dim=-1)
    
    return data


@pytest.fixture
def sample_voxel_levels(batch_size, device):
    """샘플 voxel levels"""
    return torch.randint(0, 8, (batch_size,), device=device)


@pytest.fixture
def sample_padding_mask(batch_size, num_gaussians, device):
    """샘플 패딩 마스크 (일부 위치만 패딩)"""
    mask = torch.zeros(batch_size, num_gaussians, dtype=torch.bool, device=device)
    # 마지막 10% 를 패딩으로 설정
    n_pad = num_gaussians // 10
    mask[:, -n_pad:] = True
    return mask
