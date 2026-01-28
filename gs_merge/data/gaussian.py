"""
GaussianData: Gaussian Splatting 데이터 컨테이너

PLY에서 로드한 Gaussian 속성들을 담는 데이터 클래스
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class GaussianData:
    """
    Gaussian Splatting 데이터 컨테이너
    
    Attributes:
        xyz: [N, 3] Position (world coordinates)
        rotation: [N, 4] Quaternion (wxyz)
        scale: [N, 3] Scale (log space)
        opacity: [N, 1] Opacity (logit space)
        sh_dc: [N, 3] SH DC component
        sh_rest: [N, 45] SH higher order coefficients
    """
    xyz: torch.Tensor
    rotation: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor
    
    @property
    def num_gaussians(self) -> int:
        """Gaussian 개수"""
        return self.xyz.shape[0]
    
    @property
    def device(self) -> torch.device:
        """텐서가 있는 디바이스"""
        return self.xyz.device
    
    @property
    def sh_coeffs(self) -> torch.Tensor:
        """전체 SH coefficients [N, 48]"""
        return torch.cat([self.sh_dc, self.sh_rest], dim=-1)
    
    def to_flat(self) -> torch.Tensor:
        """
        모든 속성을 하나의 텐서로 합침 [N, 59]
        순서: xyz(3) + rot(4) + scale(3) + opacity(1) + sh(48)
        """
        return torch.cat([
            self.xyz,
            self.rotation,
            self.scale,
            self.opacity,
            self.sh_dc,
            self.sh_rest
        ], dim=-1)
    
    def to(self, device: torch.device) -> 'GaussianData':
        """모든 텐서를 지정된 디바이스로 이동"""
        return GaussianData(
            xyz=self.xyz.to(device),
            rotation=self.rotation.to(device),
            scale=self.scale.to(device),
            opacity=self.opacity.to(device),
            sh_dc=self.sh_dc.to(device),
            sh_rest=self.sh_rest.to(device)
        )
    
    def __getitem__(self, idx) -> 'GaussianData':
        """인덱싱 지원"""
        return GaussianData(
            xyz=self.xyz[idx],
            rotation=self.rotation[idx],
            scale=self.scale[idx],
            opacity=self.opacity[idx],
            sh_dc=self.sh_dc[idx],
            sh_rest=self.sh_rest[idx]
        )
    
    def clone(self) -> 'GaussianData':
        """깊은 복사"""
        return GaussianData(
            xyz=self.xyz.clone(),
            rotation=self.rotation.clone(),
            scale=self.scale.clone(),
            opacity=self.opacity.clone(),
            sh_dc=self.sh_dc.clone(),
            sh_rest=self.sh_rest.clone()
        )
    
    @staticmethod
    def from_flat(data: torch.Tensor) -> 'GaussianData':
        """
        [N, 59] 텐서에서 GaussianData 생성
        순서: xyz(3) + rot(4) + scale(3) + opacity(1) + sh_dc(3) + sh_rest(45)
        """
        return GaussianData(
            xyz=data[..., :3],
            rotation=data[..., 3:7],
            scale=data[..., 7:10],
            opacity=data[..., 10:11],
            sh_dc=data[..., 11:14],
            sh_rest=data[..., 14:59]
        )
    
    @staticmethod
    def empty(n: int, device: str = "cpu") -> 'GaussianData':
        """빈 GaussianData 생성"""
        return GaussianData(
            xyz=torch.zeros(n, 3, device=device),
            rotation=torch.zeros(n, 4, device=device),
            scale=torch.zeros(n, 3, device=device),
            opacity=torch.zeros(n, 1, device=device),
            sh_dc=torch.zeros(n, 3, device=device),
            sh_rest=torch.zeros(n, 45, device=device)
        )
    
    @staticmethod
    def concat(gaussians: list) -> 'GaussianData':
        """여러 GaussianData를 하나로 합침"""
        return GaussianData(
            xyz=torch.cat([g.xyz for g in gaussians], dim=0),
            rotation=torch.cat([g.rotation for g in gaussians], dim=0),
            scale=torch.cat([g.scale for g in gaussians], dim=0),
            opacity=torch.cat([g.opacity for g in gaussians], dim=0),
            sh_dc=torch.cat([g.sh_dc for g in gaussians], dim=0),
            sh_rest=torch.cat([g.sh_rest for g in gaussians], dim=0)
        )
