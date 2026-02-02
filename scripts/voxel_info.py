#!/usr/bin/env python3
"""
Voxel Info Printer

PLY 파일을 voxelize 한 뒤 voxel 통계를 출력

Usage:
    python scripts/voxel_info.py --ply path/to/point_cloud.ply
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs_merge import load_ply, OctreeVoxelizer


def _percentile(arr: np.ndarray, p: float) -> float:
    """배열의 백분위수 계산"""
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def _print_header(title: str) -> None:
    """섹션 제목 출력"""
    print("\n" + title)
    print("=" * len(title))


def print_voxel_stats(voxels, gaussians, voxelizer: OctreeVoxelizer) -> None:
    """Voxel 통계 출력"""
    if len(voxels) == 0:
        print("No voxels generated.")
        return

    # 통계 계산
    counts = np.array([v.num_gaussians for v in voxels], dtype=np.int64)
    sizes = np.array([v.size for v in voxels], dtype=np.float32)
    levels = np.array([v.level for v in voxels], dtype=np.int64)

    total_voxels = len(voxels)
    total_gaussians_in_voxels = int(counts.sum())
    original_gaussians = gaussians.num_gaussians if gaussians is not None else None

    # Voxel 전체 bounds 계산
    min_corners = []
    max_corners = []
    for v in voxels:
        half = v.size / 2
        center = v.center.detach().cpu().numpy()
        min_corners.append(center - half)
        max_corners.append(center + half)
    min_corner = np.min(np.stack(min_corners, axis=0), axis=0)
    max_corner = np.max(np.stack(max_corners, axis=0), axis=0)

    # === Voxel Summary ===
    _print_header("Voxel Summary")
    print(f"Total Voxels: {total_voxels:,}")
    if original_gaussians is not None:
        print(f"Original Gaussians: {original_gaussians:,}")
        print(f"Gaussians in Voxels: {total_gaussians_in_voxels:,}")
        if voxelizer.pruned_count > 0:
            print(f"⚠️  Pruned Gaussians: {voxelizer.pruned_count:,}")
        kept_ratio = total_gaussians_in_voxels / max(1, original_gaussians)
        print(f"Kept Ratio: {kept_ratio * 100:.2f}%")

    print(f"\nVoxel Size Range: min={sizes.min():.6f}, mean={sizes.mean():.6f}, max={sizes.max():.6f}")
    print(f"Gaussians/Voxel Statistics:")
    print(f"  - Min: {counts.min()}")
    print(f"  - Mean: {counts.mean():.2f}")
    print(f"  - Max: {counts.max()}")
    print(f"  - p50 (Median): {int(_percentile(counts, 50))}")
    print(f"  - p90: {int(_percentile(counts, 90))}")
    print(f"  - p99: {int(_percentile(counts, 99))}")

    # === Level Distribution ===
    _print_header("Octree Level Distribution")
    uniq_levels, level_counts = np.unique(levels, return_counts=True)
    for lvl, cnt in zip(uniq_levels.tolist(), level_counts.tolist()):
        ratio = cnt / total_voxels * 100
        bar_length = int(ratio / 2)
        bar = "█" * bar_length
        print(f"Level {lvl:02d}: {cnt:6,} voxels ({ratio:5.2f}%) {bar}")

    # === Spatial AABB ===
    _print_header("Voxel Grid AABB")
    print(f"Min Corner: [{min_corner[0]:8.3f}, {min_corner[1]:8.3f}, {min_corner[2]:8.3f}]")
    print(f"Max Corner: [{max_corner[0]:8.3f}, {max_corner[1]:8.3f}, {max_corner[2]:8.3f}]")
    extent = max_corner - min_corner
    print(f"Extent:     [{extent[0]:8.3f}, {extent[1]:8.3f}, {extent[2]:8.3f}]")

    # === Voxelizer Stats ===
    _print_header("Voxelizer Processing Stats")
    print(f"Random Pruning: {voxelizer.pruned_count:,} gaussians removed")
    print(f"Sparse Compaction: {voxelizer.compact_count:,} voxels compacted")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print voxel statistics from a PLY file")
    parser.add_argument("--ply", type=str, required=True, help="Input PLY file path")
    parser.add_argument("--voxel_size", type=float, default=100.0, help="Initial voxel size (default: 100.0)")
    parser.add_argument("--max_level", type=int, default=16, help="Max octree level (default: 16)")
    parser.add_argument("--min_gaussians", type=int, default=4, help="Min gaussians per voxel (default: 4)")
    parser.add_argument("--max_gaussians", type=int, default=128, help="Max gaussians per voxel (default: 128)")
    parser.add_argument("--compact_threshold", type=float, default=0.5, help="Compact threshold (default: 0.5)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Voxel cache directory (default: same as PLY)")
    parser.add_argument("--force_rebuild", action="store_true", help="Ignore cache and rebuild voxels")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")

    args = parser.parse_args()

    # Device 설정
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available. Falling back to CPU.")
        device = "cpu"

    # PLY 파일 확인
    ply_path = Path(args.ply)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    # PLY 로드
    print(f"📂 Loading PLY: {ply_path}")
    gaussians = load_ply(str(ply_path), device=device)
    print(f"  ✅ Loaded {gaussians.num_gaussians:,} gaussians")

    # Voxelizer 생성
    print(f"\n🔨 Voxelizing...")
    voxelizer = OctreeVoxelizer(
        voxel_size=args.voxel_size,
        max_level=args.max_level,
        min_gaussians=args.min_gaussians,
        max_gaussians=args.max_gaussians,
        compact_threshold=args.compact_threshold,
        device=device,
        cache_dir=args.cache_dir,
    )

    # Voxelize
    voxels = voxelizer.voxelize_or_load(gaussians, str(ply_path), force_rebuild=args.force_rebuild)

    # 통계 출력
    print("\n" + "=" * 70)
    print_voxel_stats(voxels, gaussians, voxelizer)
    print("\n" + "=" * 70)
    print("✅ Done!\n")


if __name__ == "__main__":
    main()
