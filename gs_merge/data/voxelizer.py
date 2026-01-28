"""
OctreeVoxelizer: Gaussian Splatting 데이터를 Octree 기반 Voxel로 분할

전략:
1. Level 1: Scene을 균일한 Grid로 분할
2. Level 2~16: Octree 방식으로 max_gaussians 이하가 될 때까지 분할
3. Random Pruning & Sparse Voxel Compaction 적용
4. 캐시 지원으로 재계산 방지
"""

import torch
import math
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

from gs_merge.data.gaussian import GaussianData


@dataclass
class VoxelNode:
    """Octree의 각 노드(Voxel)"""
    level: int                      # Octree depth (0 = root)
    center: torch.Tensor            # [3] Voxel 중심 좌표
    size: float                     # Voxel 한 변의 길이
    indices: torch.Tensor           # 이 Voxel에 속한 Gaussian 인덱스들
    morton_code: int = 0            # Morton code (Z-order curve)
    
    @property
    def num_gaussians(self) -> int:
        return len(self.indices)
    
    @property
    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(min_corner, max_corner) 반환"""
        half = self.size / 2
        return (self.center - half, self.center + half)
    
    def to_dict(self) -> Dict:
        """Serializable dict로 변환"""
        return {
            'level': self.level,
            'center': self.center.cpu().numpy(),
            'size': self.size,
            'indices': self.indices.cpu().numpy(),
            'morton_code': self.morton_code
        }
    
    @classmethod
    def from_dict(cls, d: Dict, device: str = 'cpu') -> 'VoxelNode':
        """Dict에서 VoxelNode 복원"""
        import numpy as np
        return cls(
            level=d['level'],
            center=torch.from_numpy(d['center']).to(device),
            size=d['size'],
            indices=torch.from_numpy(d['indices']).to(device),
            morton_code=d['morton_code']
        )


class OctreeVoxelizer:
    """
    Gaussian Splatting 데이터를 Octree 기반 Voxel로 분할
    
    Args:
        voxel_size: Level 1 Grid의 voxel 크기
        max_level: 최대 Octree 깊이 (1~16)
        min_gaussians: 더 이상 분할하지 않을 최소 Gaussian 수
        max_gaussians: 분할을 트리거할 최대 Gaussian 수
        compact_threshold: 이 비율 미만으로 공간 사용 시 compact화
        device: 연산 디바이스
        cache_dir: Voxel 캐시 저장 디렉토리
    """
    
    def __init__(
        self,
        voxel_size: float = 100.0,
        max_level: int = 16,
        min_gaussians: int = 4,
        max_gaussians: int = 128,
        compact_threshold: float = 0.5,
        device: str = "cuda",
        cache_dir: Optional[str] = None
    ):
        self.voxel_size = voxel_size
        self.max_level = max_level
        self.min_gaussians = min_gaussians
        self.max_gaussians = max_gaussians
        self.compact_threshold = compact_threshold
        self.device = device
        self.cache_dir = cache_dir
        
        # 통계
        self.pruned_count = 0
        self.compact_count = 0
    
    def compute_scene_bounds(self, gaussians: GaussianData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scene의 AABB 계산"""
        xyz = gaussians.xyz
        return xyz.min(dim=0).values, xyz.max(dim=0).values
    
    def morton_encode(self, x: int, y: int, z: int) -> int:
        """3D 좌표를 Morton code로 인코딩"""
        def spread_bits(v):
            v = v & 0x3FF
            v = (v | (v << 16)) & 0x030000FF
            v = (v | (v << 8)) & 0x0300F00F
            v = (v | (v << 4)) & 0x030C30C3
            v = (v | (v << 2)) & 0x09249249
            return v
        return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)
    
    def _create_level1_grid(self, gaussians: GaussianData) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Level 1: Scene을 균일한 Grid로 분할"""
        min_bound, max_bound = self.compute_scene_bounds(gaussians)
        xyz = gaussians.xyz
        
        extent = max_bound - min_bound
        num_cells = torch.ceil(extent / self.voxel_size).int()
        nx, ny, nz = num_cells[0].item(), num_cells[1].item(), num_cells[2].item()
        
        print(f"  Level 1 Grid: {nx} x {ny} x {nz}")
        
        cell_indices = torch.floor((xyz - min_bound) / self.voxel_size).long()
        cell_indices[:, 0] = torch.clamp(cell_indices[:, 0], 0, nx - 1)
        cell_indices[:, 1] = torch.clamp(cell_indices[:, 1], 0, ny - 1)
        cell_indices[:, 2] = torch.clamp(cell_indices[:, 2], 0, nz - 1)
        
        grid_cells = []
        all_indices = torch.arange(gaussians.num_gaussians, device=self.device)
        
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    mask = (
                        (cell_indices[:, 0] == ix) &
                        (cell_indices[:, 1] == iy) &
                        (cell_indices[:, 2] == iz)
                    )
                    indices = all_indices[mask]
                    if len(indices) >= self.min_gaussians:
                        center = min_bound + torch.tensor(
                            [(ix + 0.5) * self.voxel_size,
                             (iy + 0.5) * self.voxel_size,
                             (iz + 0.5) * self.voxel_size],
                            device=self.device
                        )
                        grid_cells.append((center, indices))
        
        print(f"  Level 1: {len(grid_cells)} non-empty cells")
        return grid_cells
    
    def _subdivide_octree(
        self, gaussians: GaussianData, center: torch.Tensor,
        size: float, indices: torch.Tensor, level: int
    ) -> List[VoxelNode]:
        """Level 2+: Octree 방식으로 분할"""
        num_points = len(indices)
        
        if level >= self.max_level or num_points <= self.max_gaussians:
            return self._create_voxel_node(gaussians, center, size, indices, level)
        
        results = []
        half_size = size / 2
        quarter_size = size / 4
        xyz = gaussians.xyz[indices]
        
        for i in range(8):
            dx = quarter_size if (i & 1) else -quarter_size
            dy = quarter_size if (i & 2) else -quarter_size
            dz = quarter_size if (i & 4) else -quarter_size
            
            child_center = center + torch.tensor([dx, dy, dz], device=self.device)
            child_min = child_center - half_size / 2
            child_max = child_center + half_size / 2
            
            in_voxel = (
                (xyz[:, 0] >= child_min[0]) & (xyz[:, 0] < child_max[0]) &
                (xyz[:, 1] >= child_min[1]) & (xyz[:, 1] < child_max[1]) &
                (xyz[:, 2] >= child_min[2]) & (xyz[:, 2] < child_max[2])
            )
            
            child_indices = indices[in_voxel]
            if len(child_indices) >= self.min_gaussians:
                results.extend(self._subdivide_octree(
                    gaussians, child_center, half_size, child_indices, level + 1
                ))
        
        return results
    
    def _create_voxel_node(
        self, gaussians: GaussianData, center: torch.Tensor,
        size: float, indices: torch.Tensor, level: int
    ) -> List[VoxelNode]:
        """Voxel 노드 생성 (pruning & compact 적용)"""
        num_points = len(indices)
        
        if num_points < self.min_gaussians:
            return []
        
        # Random Pruning
        if num_points > self.max_gaussians:
            perm = torch.randperm(num_points, device=self.device)[:self.max_gaussians]
            self.pruned_count += num_points - self.max_gaussians
            indices = indices[perm]
        
        # Compact
        center, size, level = self._compact_voxel(gaussians, center, size, indices, level)
        
        # Morton code
        cx = int((center[0].item() / self.voxel_size + 100) * 5) & 0x3FF
        cy = int((center[1].item() / self.voxel_size + 100) * 5) & 0x3FF
        cz = int((center[2].item() / self.voxel_size + 100) * 5) & 0x3FF
        morton = self.morton_encode(cx, cy, cz)
        
        return [VoxelNode(level=level, center=center.clone(), size=size,
                          indices=indices, morton_code=morton)]
    
    def _compact_voxel(
        self, gaussians: GaussianData, center: torch.Tensor,
        size: float, indices: torch.Tensor, level: int
    ) -> Tuple[torch.Tensor, float, int]:
        """Sparse Voxel을 tight bounding box로 compact화"""
        xyz = gaussians.xyz[indices]
        
        min_xyz = xyz.min(dim=0).values
        max_xyz = xyz.max(dim=0).values
        tight_extent = (max_xyz - min_xyz).max().item()
        tight_size = tight_extent * 1.1
        
        min_size = self.voxel_size / (2 ** (self.max_level - 1))
        tight_size = max(tight_size, min_size)
        
        usage_ratio = tight_size / size
        
        if usage_ratio < self.compact_threshold:
            new_center = (min_xyz + max_xyz) / 2
            level_from_size = max(1, int(math.log2(self.voxel_size / tight_size) + 1))
            new_level = min(level_from_size, self.max_level)
            self.compact_count += 1
            return new_center, tight_size, new_level
        
        return center, size, level
    
    def voxelize(self, gaussians: GaussianData) -> List[VoxelNode]:
        """Gaussian 데이터를 Voxel로 분할"""
        self.pruned_count = 0
        self.compact_count = 0
        
        min_bound, max_bound = self.compute_scene_bounds(gaussians)
        extent = max_bound - min_bound
        
        print(f"  Scene bounds: {min_bound.cpu().numpy()} ~ {max_bound.cpu().numpy()}")
        print(f"  Total Gaussians: {gaussians.num_gaussians:,}")
        
        grid_cells = self._create_level1_grid(gaussians)
        
        all_voxels = []
        for center, indices in grid_cells:
            voxels = self._subdivide_octree(gaussians, center, self.voxel_size, indices, level=1)
            all_voxels.extend(voxels)
        
        all_voxels.sort(key=lambda v: v.morton_code)
        
        print(f"\n  === Voxelization Complete ===")
        print(f"  Generated {len(all_voxels)} voxels")
        if self.pruned_count > 0:
            print(f"  ⚠️  Random Pruning: {self.pruned_count:,} gaussians removed")
        if self.compact_count > 0:
            print(f"  📦 Compacted: {self.compact_count} sparse voxels")
        
        return all_voxels
    
    def _get_cache_path(self, ply_path: str) -> Path:
        """캐시 파일 경로 생성"""
        ply_path = Path(ply_path)
        config_str = f"vs{self.voxel_size}_ml{self.max_level}_min{self.min_gaussians}_max{self.max_gaussians}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_name = f"{ply_path.stem}_{config_hash}.voxel"
        
        cache_dir = Path(self.cache_dir) if self.cache_dir else ply_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / cache_name
    
    def save_voxels(self, voxels: List[VoxelNode], path: str) -> None:
        """Voxel 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'version': '2.0',
            'config': {
                'voxel_size': self.voxel_size,
                'max_level': self.max_level,
                'min_gaussians': self.min_gaussians,
                'max_gaussians': self.max_gaussians,
            },
            'voxels': [v.to_dict() for v in voxels],
            'stats': {
                'num_voxels': len(voxels),
                'pruned_count': self.pruned_count,
                'compact_count': self.compact_count,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"  💾 Saved {len(voxels)} voxels to {path}")
    
    def load_voxels(self, path: str) -> List[VoxelNode]:
        """Voxel 로드"""
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        voxels = [VoxelNode.from_dict(v, self.device) for v in data['voxels']]
        stats = data.get('stats', {})
        
        print(f"  📂 Loaded {len(voxels)} voxels from {path}")
        return voxels
    
    def voxelize_or_load(
        self, gaussians: GaussianData, ply_path: str, force_rebuild: bool = False
    ) -> List[VoxelNode]:
        """캐시가 있으면 로드, 없으면 새로 생성"""
        cache_path = self._get_cache_path(ply_path)
        
        if cache_path.exists() and not force_rebuild:
            try:
                print(f"\n📂 Found voxel cache: {cache_path}")
                return self.load_voxels(cache_path)
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}")
        
        print(f"\n🔨 Building voxels...")
        voxels = self.voxelize(gaussians)
        self.save_voxels(voxels, cache_path)
        return voxels
    
    def normalize_voxel_gaussians(self, gaussians: GaussianData, voxel: VoxelNode) -> torch.Tensor:
        """Voxel 내의 Gaussian들을 정규화 [N, 59]"""
        idx = voxel.indices
        half_size = voxel.size / 2
        
        local_xyz = (gaussians.xyz[idx] - voxel.center) / half_size
        local_scale = gaussians.scale[idx] - math.log(half_size)
        
        return torch.cat([
            local_xyz,
            gaussians.rotation[idx],
            local_scale,
            gaussians.opacity[idx],
            gaussians.sh_dc[idx],
            gaussians.sh_rest[idx]
        ], dim=-1)
