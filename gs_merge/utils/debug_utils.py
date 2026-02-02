"""
Debug Utilities for GMAE Training

디버그 통계 출력 및 시각화 유틸리티
"""

import torch
from typing import Dict, Optional


def print_gaussian_stats(
    input_g: Dict[str, torch.Tensor], 
    output_g: Dict[str, torch.Tensor],
    loss_density: torch.Tensor,
    loss_render: torch.Tensor,
    iteration: int,
    input_mask: Optional[torch.Tensor] = None,
    output_mask: Optional[torch.Tensor] = None
):
    """
    Gaussian 입출력 통계 출력 (opacity > 0.005 필터링 + 마스크 고려)
    
    Args:
        input_g: 입력 가우시안 파라미터
        output_g: 출력 가우시안 파라미터
        loss_density: Density loss 값
        loss_render: Render loss 값
        iteration: 현재 iteration 번호
        input_mask: 입력 패딩 마스크 (True=패딩)
        output_mask: 출력 패딩 마스크 (True=패딩)
    """
    with torch.no_grad():
        # Raw data
        in_xyz = input_g['xyz']
        in_scale_real = torch.exp(input_g['scale'])
        in_opacity_real = torch.sigmoid(input_g['opacity'])
        
        out_xyz = output_g['xyz']
        out_scale_real = torch.exp(output_g['scale'])
        out_opacity_real = torch.sigmoid(output_g['opacity'])
        
        # 마스크 고려 (패딩된 위치 제거)
        if input_mask is not None:
            in_valid_mask = ~input_mask
        else:
            in_valid_mask = torch.ones_like(in_opacity_real.squeeze(-1), dtype=torch.bool)
            
        if output_mask is not None:
            out_valid_mask = ~output_mask
        else:
            out_valid_mask = torch.ones_like(out_opacity_real.squeeze(-1), dtype=torch.bool)

        # Count filtered
        in_total = in_xyz.shape[0] * in_xyz.shape[1]  # B * N
        out_total = out_xyz.shape[0] * out_xyz.shape[1]  # B * M
        in_removed = in_total - in_valid_mask.sum().item()
        out_removed = out_total - out_valid_mask.sum().item()
        
        # Apply filter
        in_xyz_filtered = in_xyz[in_valid_mask]
        in_scale_filtered = in_scale_real[in_valid_mask]
        in_opacity_filtered = in_opacity_real.squeeze(-1)[in_valid_mask]
        
        out_xyz_filtered = out_xyz[out_valid_mask]
        out_scale_filtered = out_scale_real[out_valid_mask]
        out_opacity_filtered = out_opacity_real.squeeze(-1)[out_valid_mask]
        
        print(f"\n{'='*60}")
        print(f"[DEBUG iter={iteration}] Input vs Output 분포 (opacity > 0.005)")
        print(f"{'='*60}")
        print(f"  Filtered Gaussians (마스크 포함):")
        in_valid_count = in_total - in_removed
        out_valid_count = out_total - out_removed
        actual_ratio = out_valid_count / max(in_valid_count, 1)
        print(f"    Total Input(B, N):  {in_total} → {in_valid_count} (removed: {in_removed}, {in_removed/max(in_total, 1)*100:.1f}%)")
        print(f"    Total Output(B, M): {out_total} → {out_valid_count} (removed: {out_removed}, {out_removed/max(out_total, 1)*100:.1f}%)")
        print(f"    Compression Ratio: {actual_ratio:.4f} ({actual_ratio*100:.1f}%)")
        
        if in_xyz_filtered.numel() > 0 and out_xyz_filtered.numel() > 0:
            print(f"  XYZ:")
            print(f"    Input  - min: {in_xyz_filtered.min():.3f}, max: {in_xyz_filtered.max():.3f}, mean: {in_xyz_filtered.mean():.3f}")
            print(f"    Output - min: {out_xyz_filtered.min():.3f}, max: {out_xyz_filtered.max():.3f}, mean: {out_xyz_filtered.mean():.3f}")
            print(f"  Scale (실제값):")
            print(f"    Input  - min: {in_scale_filtered.min():.4f}, max: {in_scale_filtered.max():.4f}, mean: {in_scale_filtered.mean():.4f}")
            print(f"    Output - min: {out_scale_filtered.min():.4f}, max: {out_scale_filtered.max():.4f}, mean: {out_scale_filtered.mean():.4f}")
            print(f"  Opacity (0~1):")
            print(f"    Input  - min: {in_opacity_filtered.min():.3f}, max: {in_opacity_filtered.max():.3f}, mean: {in_opacity_filtered.mean():.3f}")
            print(f"    Output - min: {out_opacity_filtered.min():.3f}, max: {out_opacity_filtered.max():.3f}, mean: {out_opacity_filtered.mean():.3f}")
            print(f"  Opacity > 0.5 비율:")
            print(f"    Input:  {(in_opacity_filtered > 0.5).float().mean()*100:.1f}%")
            print(f"    Output: {(out_opacity_filtered > 0.5).float().mean()*100:.1f}%")
        else:
            print(f"  ⚠ No valid Gaussians after filtering!")
            
        print(f"  Losses: density={loss_density.item():.4f}, render={loss_render.item():.4f}")
        print(f"{'='*60}\n")


def print_loss_breakdown(
    loss_dict: Dict[str, torch.Tensor],
    iteration: int
):
    """
    Loss 항목별 상세 출력
    
    Args:
        loss_dict: Loss 딕셔너리 (key: loss name, value: loss tensor)
        iteration: 현재 iteration 번호
    """
    print(f"\n[Loss Breakdown - iter={iteration}]")
    total = sum(v.item() for v in loss_dict.values())
    for name, value in loss_dict.items():
        percentage = (value.item() / total * 100) if total > 0 else 0
        print(f"  {name:20s}: {value.item():8.4f} ({percentage:5.1f}%)")
    print(f"  {'Total':20s}: {total:8.4f}\n")
