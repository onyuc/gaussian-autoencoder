"""
Voxel Utilities

Voxel ↔ Gaussian 변환, 병합, PLY 저장 유틸리티
"""

import torch
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from gs_merge.data.gaussian import GaussianData
from gs_merge.data.voxelizer import VoxelNode


def denormalize_gaussian(
    output: Dict[str, torch.Tensor],
    voxel: VoxelNode
) -> Dict[str, torch.Tensor]:
    """
    정규화된 모델 출력을 월드 좌표계로 변환
    
    Args:
        output: 모델 출력 딕셔너리 (log/logit space)
            - xyz: [N, 3] (정규화된 좌표 -1~1)
            - scale: [N, 3] (log space, voxel 기준)
            - rotation: [N, 4]
            - opacity: [N, 1] (logit space)
            - sh_dc: [N, 3]
            - sh_rest: [N, 45]
        voxel: VoxelNode (center, size)
    
    Returns:
        월드 좌표계의 Gaussian 딕셔너리 (실제 값)
    """
    half_size = voxel.size / 2
    
    # xyz: 정규화 좌표 → 월드 좌표
    world_xyz = output['xyz'] * half_size + voxel.center
    
    # scale: log space + voxel 오프셋 → 실제 값
    world_scale = torch.exp(output['scale'] + math.log(half_size))
    
    # opacity: logit → sigmoid
    world_opacity = torch.sigmoid(output['opacity'])
    
    return {
        'xyz': world_xyz,
        'rotation': output['rotation'],
        'scale': world_scale,
        'opacity': world_opacity,
        'sh_dc': output['sh_dc'],
        'sh_rest': output['sh_rest']
    }


def merge_gaussians_from_voxels(
    model_outputs: List[Dict[str, torch.Tensor]],
    voxels: List[VoxelNode],
    opacity_threshold: float = 0.005
) -> Dict[str, torch.Tensor]:
    """
    여러 Voxel의 모델 출력을 하나로 병합
    
    Args:
        model_outputs: Voxel별 모델 출력 리스트
        voxels: 대응하는 VoxelNode 리스트
        opacity_threshold: 이 값 미만의 opacity를 가진 Gaussian 제거
    
    Returns:
        병합된 Gaussian 딕셔너리 (월드 좌표)
    """
    all_xyz = []
    all_rot = []
    all_scale = []
    all_opacity = []
    all_sh_dc = []
    all_sh_rest = []
    
    filtered_count = 0
    total_count = 0
    
    for output, voxel in zip(model_outputs, voxels):
        # Denormalize
        world_g = denormalize_gaussian(output, voxel)
        
        # Opacity filtering
        opacity = world_g['opacity'].squeeze(-1)
        total_count += opacity.shape[0]
        
        if opacity_threshold > 0:
            valid_mask = opacity >= opacity_threshold
            filtered_count += (~valid_mask).sum().item()
            
            if valid_mask.sum() == 0:
                continue
            
            all_xyz.append(world_g['xyz'][valid_mask])
            all_rot.append(world_g['rotation'][valid_mask])
            all_scale.append(world_g['scale'][valid_mask])
            all_opacity.append(world_g['opacity'][valid_mask])
            all_sh_dc.append(world_g['sh_dc'][valid_mask])
            all_sh_rest.append(world_g['sh_rest'][valid_mask])
        else:
            all_xyz.append(world_g['xyz'])
            all_rot.append(world_g['rotation'])
            all_scale.append(world_g['scale'])
            all_opacity.append(world_g['opacity'])
            all_sh_dc.append(world_g['sh_dc'])
            all_sh_rest.append(world_g['sh_rest'])
    
    if len(all_xyz) == 0:
        raise ValueError("All Gaussians were filtered out!")
    
    if filtered_count > 0:
        print(f"  Filtered {filtered_count:,} / {total_count:,} Gaussians (opacity < {opacity_threshold})")
    
    return {
        'xyz': torch.cat(all_xyz, dim=0),
        'rotation': torch.cat(all_rot, dim=0),
        'scale': torch.cat(all_scale, dim=0),
        'opacity': torch.cat(all_opacity, dim=0),
        'sh_dc': torch.cat(all_sh_dc, dim=0),
        'sh_rest': torch.cat(all_sh_rest, dim=0)
    }


def save_merged_ply(
    merged_gaussians: Dict[str, torch.Tensor],
    output_path: str,
    verbose: bool = True
):
    """
    병합된 Gaussian을 PLY 파일로 저장
    
    Args:
        merged_gaussians: merge_gaussians_from_voxels의 출력
        output_path: 저장 경로
        verbose: 진행 상황 출력
    """
    from gs_merge.data.ply_io import save_ply
    from gs_merge.data.gaussian import GaussianData
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # GaussianData로 변환
    gaussians = GaussianData(
        xyz=merged_gaussians['xyz'],
        rotation=merged_gaussians['rotation'],
        scale=merged_gaussians['scale'],
        opacity=merged_gaussians['opacity'],
        sh_dc=merged_gaussians['sh_dc'],
        sh_rest=merged_gaussians['sh_rest']
    )
    
    save_ply(gaussians, str(output_path))
    
    if verbose:
        print(f"  ✅ Saved {gaussians.num_gaussians:,} Gaussians to {output_path}")


class VoxelGaussianConverter:
    """
    Voxel ↔ Gaussian 변환 유틸리티 클래스
    """
    
    def __init__(
        self,
        max_per_voxel: int = 128,
        output_queries: int = 32,
        opacity_threshold: float = 0.005,
        device: str = "cuda"
    ):
        self.max_per_voxel = max_per_voxel
        self.output_queries = output_queries
        self.opacity_threshold = opacity_threshold
        self.device = device
    
    def normalize_voxel(
        self,
        gaussians: GaussianData,
        voxel: VoxelNode
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Voxel 내 Gaussian을 정규화된 텐서로 변환
        
        Returns:
            features: [max_per_voxel, 59]
            mask: [max_per_voxel] (True = padding)
        """
        idx = voxel.indices
        half_size = voxel.size / 2
        
        # 정규화
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
        if N < self.max_per_voxel:
            pad = torch.zeros(self.max_per_voxel - N, 59, device=features.device)
            features = torch.cat([features, pad], dim=0)
            mask = torch.zeros(self.max_per_voxel, dtype=torch.bool, device=features.device)
            mask[N:] = True
        elif N > self.max_per_voxel:
            perm = torch.randperm(N, device=features.device)[:self.max_per_voxel]
            features = features[perm]
            mask = torch.zeros(self.max_per_voxel, dtype=torch.bool, device=features.device)
        else:
            mask = torch.zeros(self.max_per_voxel, dtype=torch.bool, device=features.device)
        
        return features, mask
    
    def denormalize_output(
        self,
        output: Dict[str, torch.Tensor],
        voxel: VoxelNode
    ) -> Dict[str, torch.Tensor]:
        """모델 출력을 월드 좌표로 변환"""
        return denormalize_gaussian(output, voxel)
    
    def merge_all(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        voxels: List[VoxelNode]
    ) -> Dict[str, torch.Tensor]:
        """여러 Voxel 출력 병합"""
        return merge_gaussians_from_voxels(
            outputs, voxels, 
            opacity_threshold=self.opacity_threshold
        )
