#!/usr/bin/env python3
"""
Gaussian PLY Compressor

학습된 모델로 PLY 파일을 축소하여 저장

Usage:
    python compress.py --ply input.ply --checkpoint best.pt --output compressed.ply
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs_merge import (
    GaussianMergingAE,
    GaussianData, load_ply, save_ply, OctreeVoxelizer, VoxelNode,
    denormalize_gaussian, merge_gaussians_from_voxels, save_merged_ply
)
from gs_merge.loss import model_output_to_dict


def compress_ply(
    ply_path: str,
    checkpoint_path: str,
    output_path: str,
    voxel_size: float = 100,
    max_level: int = 16,
    max_gaussians_per_voxel: int = 128,
    num_queries: int = 128,
    opacity_threshold: float = 0.005,
    device: str = "cuda"
):
    """
    PLY 파일을 로드하고 머지 모델로 축소하여 저장
    """
    # =========================================
    # 1. Load PLY & Voxelize
    # =========================================
    print(f"Loading PLY: {ply_path}")
    gaussians = load_ply(ply_path, device=device)
    print(f"  Loaded {gaussians.num_gaussians:,} gaussians")
    
    print(f"\nVoxelizing...")
    voxelizer = OctreeVoxelizer(
        voxel_size=voxel_size,
        max_level=max_level,
        min_gaussians=1,
        max_gaussians=max_gaussians_per_voxel,
        compact_threshold=0.5,
        device=device,
        cache_dir=None
    )
    voxels = voxelizer.voxelize_or_load(gaussians, ply_path, force_rebuild=False)
    print(f"  Created {len(voxels)} voxels")
    
    # =========================================
    # 2. Load Model
    # =========================================
    print(f"\nLoading model: {checkpoint_path}")
    
    model = GaussianMergingAE(
        input_dim=59,
        latent_dim=256,
        num_inputs = 128,
        num_queries=num_queries,
        nhead=8,
        num_enc_layers=4,
        num_dec_layers=4,
        max_octree_level=max_level
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    print("  Model loaded successfully")
    
    # =========================================
    # 3. Process Each Voxel
    # =========================================
    print(f"\nCompressing voxels...")
    
    all_outputs = []
    batch_size = 64  # 배치 크기 설정
    all_outputs = []
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(voxels), batch_size), desc="Processing batches"):
            batch_voxels = voxels[batch_start:batch_start + batch_size]
            B = len(batch_voxels)
            
            # 배치 입력 준비
            batch_inputs = []
            batch_levels = []
            batch_padding_masks = []
            
            for voxel in batch_voxels:
                import math
                # Voxel 내 Gaussian 추출 및 정규화
                idx = voxel.indices
                half_size = voxel.size / 2
                
                xyz_norm = (gaussians.xyz[idx] - voxel.center) / half_size
                xyz_norm = torch.clamp(xyz_norm, -1.0, 1.0)
                scale_norm = gaussians.scale[idx] - math.log(half_size)
                
                # Input tensor [N, 59]
                input_tensor = torch.cat([
                    xyz_norm,
                    gaussians.rotation[idx],
                    scale_norm,
                    gaussians.opacity[idx],
                    gaussians.sh_dc[idx],
                    gaussians.sh_rest[idx]
                ], dim=-1)
                
                # Padding
                N = input_tensor.shape[0]
                if N < max_gaussians_per_voxel:
                    padding = torch.zeros(max_gaussians_per_voxel - N, 59, device=device)
                    input_tensor = torch.cat([input_tensor, padding], dim=0)
                    padding_mask = torch.zeros(max_gaussians_per_voxel, dtype=torch.bool, device=device)
                    padding_mask[N:] = True
                else:
                    input_tensor = input_tensor[:max_gaussians_per_voxel]
                    padding_mask = torch.zeros(max_gaussians_per_voxel, dtype=torch.bool, device=device)
                
                batch_inputs.append(input_tensor)
                batch_levels.append(voxel.level)
                batch_padding_masks.append(padding_mask)
            
            # Stack to batch tensors
            batch_inputs = torch.stack(batch_inputs, dim=0)  # [B, N, 59]
            batch_levels = torch.tensor(batch_levels, device=device)  # [B]
            batch_padding_masks = torch.stack(batch_padding_masks, dim=0)  # [B, N]
            
            # Model forward
            xyz_out, rot_out, scale_out, opacity_out, sh_out, output_mask = model(
                batch_inputs, batch_levels, batch_padding_masks
            )
            
            # 배치별로 결과 분리 및 필터링
            for i in range(B):
                valid_mask = ~output_mask[i]  # [M], True=유효
                
                output = {
                    'xyz': xyz_out[i][valid_mask],
                    'rotation': rot_out[i][valid_mask],
                    'scale': scale_out[i][valid_mask],
                    'opacity': opacity_out[i][valid_mask],
                    'sh_dc': sh_out[i][valid_mask][..., :3],
                    'sh_rest': sh_out[i][valid_mask][..., 3:]
                }
                
                all_outputs.append(output)
    
    # =========================================
    # 4. Merge All Voxels
    # =========================================
    print(f"\nMerging results...")
    
    merged = merge_gaussians_from_voxels(
        all_outputs, voxels, 
        opacity_threshold=opacity_threshold
    )
    
    print(f"  Original Gaussians: {gaussians.num_gaussians:,}")
    print(f"  Compressed Gaussians: {merged['xyz'].shape[0]:,}")
    print(f"  Compression Ratio: {gaussians.num_gaussians / merged['xyz'].shape[0]:.2f}x")
    
    # =========================================
    # 5. Save as PLY
    # =========================================
    save_merged_ply(merged, output_path)
    print(f"\n✅ Done!")


def main():
    parser = argparse.ArgumentParser(description="Compress Gaussian PLY using trained merge model")
    parser.add_argument("--ply", type=str, required=True, help="Input PLY file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output PLY file path")
    parser.add_argument("--voxel_size", type=float, default=100, help="Initial voxel size")
    parser.add_argument("--max_level", type=int, default=16, help="Max octree level")
    parser.add_argument("--max_gaussians", type=int, default=128, help="Max gaussians per voxel")
    parser.add_argument("--num_queries", type=int, default=128, help="Output gaussians per voxel")
    parser.add_argument("--opacity_threshold", type=float, default=0.005, help="Min opacity to keep")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    compress_ply(
        ply_path=args.ply,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        voxel_size=args.voxel_size,
        max_level=args.max_level,
        max_gaussians_per_voxel=args.max_gaussians,
        num_queries=args.num_queries,
        opacity_threshold=args.opacity_threshold,
        device=args.device
    )


if __name__ == "__main__":
    main()
