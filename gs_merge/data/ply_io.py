"""
PLY I/O: Gaussian Splatting PLY 파일 읽기/쓰기

3DGS 표준 PLY 포맷 지원
"""

import torch
import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path

from gs_merge.data.gaussian import GaussianData


def load_ply(ply_path: str, device: str = "cpu") -> GaussianData:
    """
    3DGS PLY 파일을 로드하여 GaussianData로 반환
    
    Args:
        ply_path: PLY 파일 경로
        device: 'cpu' or 'cuda'
    
    Returns:
        GaussianData 객체
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # Position
    xyz = np.stack([
        vertex['x'], 
        vertex['y'], 
        vertex['z']
    ], axis=-1).astype(np.float32)
    
    # Opacity (logit space)
    opacity = vertex['opacity'].reshape(-1, 1).astype(np.float32)
    
    # Scale (log space)
    scale = np.stack([
        vertex['scale_0'],
        vertex['scale_1'],
        vertex['scale_2']
    ], axis=-1).astype(np.float32)
    
    # Rotation (quaternion: wxyz)
    rotation = np.stack([
        vertex['rot_0'],
        vertex['rot_1'],
        vertex['rot_2'],
        vertex['rot_3']
    ], axis=-1).astype(np.float32)
    
    # SH Coefficients - DC component
    sh_dc = np.stack([
        vertex['f_dc_0'],
        vertex['f_dc_1'],
        vertex['f_dc_2']
    ], axis=-1).astype(np.float32)
    
    # SH Coefficients - Higher order (f_rest_0 ~ f_rest_44)
    sh_rest_list = []
    for i in range(45):
        try:
            sh_rest_list.append(vertex[f'f_rest_{i}'])
        except ValueError:
            # SH degree가 낮은 경우 없는 coefficient는 0으로 채움
            sh_rest_list.append(np.zeros(len(vertex), dtype=np.float32))
    sh_rest = np.stack(sh_rest_list, axis=-1).astype(np.float32)
    
    return GaussianData(
        xyz=torch.from_numpy(xyz).to(device),
        rotation=torch.from_numpy(rotation).to(device),
        scale=torch.from_numpy(scale).to(device),
        opacity=torch.from_numpy(opacity).to(device),
        sh_dc=torch.from_numpy(sh_dc).to(device),
        sh_rest=torch.from_numpy(sh_rest).to(device)
    )


def save_ply(
    path: str,
    gaussians: GaussianData = None,
    *,
    xyz: np.ndarray = None,
    rotation: np.ndarray = None,
    scale: np.ndarray = None,
    opacity: np.ndarray = None,
    sh_dc: np.ndarray = None,
    sh_rest: np.ndarray = None
):
    """
    Gaussian 데이터를 3DGS 호환 PLY 파일로 저장
    
    Args:
        path: 저장 경로
        gaussians: GaussianData 객체 (또는 개별 속성들)
        xyz, rotation, scale, opacity, sh_dc, sh_rest: 개별 numpy 배열
    """
    # GaussianData 객체가 주어진 경우 분해
    if gaussians is not None:
        xyz = gaussians.xyz.cpu().numpy()
        rotation = gaussians.rotation.cpu().numpy()
        scale = gaussians.scale.cpu().numpy()
        opacity = gaussians.opacity.cpu().numpy()
        sh_dc = gaussians.sh_dc.cpu().numpy()
        sh_rest = gaussians.sh_rest.cpu().numpy()
    
    N = xyz.shape[0]
    
    # opacity가 2D인 경우 flatten
    if opacity.ndim == 2:
        opacity = opacity.squeeze(-1)
    
    # 디렉토리 생성
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Structured array 생성
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),  # normals (unused)
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    
    # SH rest coefficients
    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))
    
    dtype.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])
    
    elements = np.empty(N, dtype=dtype)
    
    # Position
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    
    # Normals (unused)
    elements['nx'] = 0
    elements['ny'] = 0
    elements['nz'] = 0
    
    # SH DC
    elements['f_dc_0'] = sh_dc[:, 0]
    elements['f_dc_1'] = sh_dc[:, 1]
    elements['f_dc_2'] = sh_dc[:, 2]
    
    # SH Rest
    for i in range(45):
        if i < sh_rest.shape[1]:
            elements[f'f_rest_{i}'] = sh_rest[:, i]
        else:
            elements[f'f_rest_{i}'] = 0
    
    # Opacity
    elements['opacity'] = opacity
    
    # Scale
    elements['scale_0'] = scale[:, 0]
    elements['scale_1'] = scale[:, 1]
    elements['scale_2'] = scale[:, 2]
    
    # Rotation
    elements['rot_0'] = rotation[:, 0]
    elements['rot_1'] = rotation[:, 1]
    elements['rot_2'] = rotation[:, 2]
    elements['rot_3'] = rotation[:, 3]
    
    # Create PLY
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
