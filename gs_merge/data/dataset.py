"""
VoxelDataset: Voxel 배치 생성을 위한 PyTorch Dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import random

from gs_merge.data.gaussian import GaussianData
from gs_merge.data.voxelizer import VoxelNode, OctreeVoxelizer


class VoxelDataset(Dataset):
    """
    Voxel 기반 Gaussian 배치 생성
    
    Args:
        gaussians: GaussianData 객체
        voxels: VoxelNode 리스트
        max_per_voxel: Voxel 당 최대 Gaussian 수
        shuffle_voxels: 매 epoch 마다 voxel 순서 섞기
    """
    
    def __init__(
        self,
        gaussians: GaussianData,
        voxels: List[VoxelNode],
        max_per_voxel: int = 128,
        shuffle_voxels: bool = True
    ):
        self.gaussians = gaussians
        self.voxels = voxels
        self.max_per_voxel = max_per_voxel
        self.shuffle_voxels = shuffle_voxels
        self.device = gaussians.xyz.device
    
    def __len__(self) -> int:
        return len(self.voxels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Returns:
            features: [max_per_voxel, 59] 정규화된 Gaussian features
            level: Voxel의 octree level
            mask: [max_per_voxel] 유효하지 않은 위치 마스크 (True = padding)
        """
        voxel = self.voxels[idx]
        
        # Gaussian features 정규화
        features = self._normalize_voxel(voxel)  # [N, 59]
        N = features.shape[0]
        
        # Padding
        if N < self.max_per_voxel:
            pad = torch.zeros(self.max_per_voxel - N, features.shape[1], device=features.device)
            features = torch.cat([features, pad], dim=0)
            mask = torch.zeros(self.max_per_voxel, dtype=torch.bool, device=features.device)
            mask[N:] = True
        elif N > self.max_per_voxel:
            # Random sample
            perm = torch.randperm(N, device=features.device)[:self.max_per_voxel]
            features = features[perm]
            mask = torch.zeros(self.max_per_voxel, dtype=torch.bool, device=features.device)
        else:
            mask = torch.zeros(self.max_per_voxel, dtype=torch.bool, device=features.device)
        
        return features, voxel.level, mask
    
    def _normalize_voxel(self, voxel: VoxelNode) -> torch.Tensor:
        """Voxel 내 Gaussian을 정규화 [N, 59]"""
        import math
        idx = voxel.indices
        half_size = voxel.size / 2
        
        local_xyz = (self.gaussians.xyz[idx] - voxel.center) / half_size
        local_scale = self.gaussians.scale[idx] - math.log(half_size)
        
        return torch.cat([
            local_xyz,                        # [N, 3]
            self.gaussians.rotation[idx],     # [N, 4]
            local_scale,                      # [N, 3]
            self.gaussians.opacity[idx],      # [N, 1]
            self.gaussians.sh_dc[idx],        # [N, 3]
            self.gaussians.sh_rest[idx]       # [N, 45]
        ], dim=-1)
    
    def collate_fn(self, batch: List[Tuple[torch.Tensor, int, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom collate for DataLoader"""
        features, levels, masks = zip(*batch)
        return (
            torch.stack(features),                                    # [B, max_per_voxel, 59]
            torch.tensor(levels, dtype=torch.long, device=self.device),  # [B]
            torch.stack(masks)                                        # [B, max_per_voxel]
        )
    
    def get_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """DataLoader 생성"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )


def prepare_batch(
    gaussians: GaussianData,
    voxels: List[VoxelNode],
    max_per_voxel: int = 128,
    batch_size: int = 16,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    배치 직접 생성 (DataLoader 없이)
    
    Args:
        gaussians: GaussianData 객체
        voxels: VoxelNode 리스트
        max_per_voxel: Voxel 당 최대 Gaussian 수
        batch_size: 배치 크기
        device: 연산 디바이스
    
    Returns:
        input_batch: [B, max_per_voxel, 59]
        levels: [B]
        masks: [B, max_per_voxel]
    """
    import math
    
    # 랜덤 샘플링
    batch_voxels = random.sample(voxels, min(batch_size, len(voxels)))
    
    batch_input = []
    batch_levels = []
    batch_masks = []
    
    for voxel in batch_voxels:
        idx = voxel.indices
        half_size = voxel.size / 2
        
        # Gaussian features 정규화
        local_xyz = (gaussians.xyz[idx] - voxel.center) / half_size
        local_scale = gaussians.scale[idx] - math.log(half_size)
        
        features = torch.cat([
            local_xyz,
            gaussians.rotation[idx],
            local_scale,
            gaussians.opacity[idx],
            gaussians.sh_dc[idx],
            gaussians.sh_rest[idx]
        ], dim=-1)  # [N, 59]
        
        N = features.shape[0]
        
        # Padding / Sampling
        if N < max_per_voxel:
            pad = torch.zeros(max_per_voxel - N, 59, device=device)
            padded = torch.cat([features, pad], dim=0)
            mask = torch.zeros(max_per_voxel, dtype=torch.bool, device=device)
            mask[N:] = True
        elif N > max_per_voxel:
            perm = torch.randperm(N, device=device)[:max_per_voxel]
            padded = features[perm]
            mask = torch.zeros(max_per_voxel, dtype=torch.bool, device=device)
        else:
            padded = features
            mask = torch.zeros(max_per_voxel, dtype=torch.bool, device=device)
        
        batch_input.append(padded)
        batch_levels.append(voxel.level)
        batch_masks.append(mask)
    
    return (
        torch.stack(batch_input),
        torch.tensor(batch_levels, device=device),
        torch.stack(batch_masks)
    )
