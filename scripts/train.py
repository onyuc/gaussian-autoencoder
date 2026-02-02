#!/usr/bin/env python3
"""
Gaussian Merging AE Training Script

Voxel 단위로 Gaussian을 압축하는 모델 학습

Usage:
    python train.py --ply path/to/point_cloud.ply --epochs 100
    python train.py --config configs/default.yaml --ply model.ply
"""

import os
import sys
import random

import torch

# 패키지 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs_merge import (
    GaussianMergingAE,
    load_ply, OctreeVoxelizer,
    GMAELoss, ChamferLoss,
)
from gs_merge.data import VoxelDataset
from gs_merge.config import parse_args_with_config
from gs_merge.training import (
    Trainer,
    GumbelNoiseScheduler,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)


def setup_model(args, device: str) -> GaussianMergingAE:
    """모델 생성"""
    model = GaussianMergingAE(
        input_dim=getattr(args, 'input_dim', 59),
        latent_dim=getattr(args, 'latent_dim', 256),
        num_inputs=getattr(args, 'max_gaussians', 128),
        num_queries=args.num_queries,
        nhead=getattr(args, 'nhead', 8),
        num_enc_layers=getattr(args, 'num_enc_layers', 4),
        num_dec_layers=getattr(args, 'num_dec_layers', 4),
        max_octree_level=args.max_level
    )
    return model


def setup_criterion(args, save_dir: str):
    """손실 함수 생성"""
    if args.loss_type == "gmae":
        return GMAELoss(
            lambda_density=getattr(args, 'lambda_density', 1.0),
            lambda_render=getattr(args, 'lambda_render', 2.0),
            lambda_sparsity=getattr(args, 'lambda_sparsity', 0.01),
            n_density_samples=getattr(args, 'n_density_samples', 1024),
            render_resolution=getattr(args, 'render_resolution', 64),
            warmup_iterations=getattr(args, 'warmup_iterations', 1000),
            debug_save_dir=os.path.join(save_dir, "debug_renders"),
            debug_save_interval=getattr(args, 'debug_save_interval', 100)
        )
    else:
        return ChamferLoss()


def setup_data(args, device: str):
    """데이터 로더 생성"""
    # Load PLY
    print(f"\nLoading PLY: {args.ply}")
    gaussians = load_ply(args.ply, device=device)
    print(f"  Loaded {gaussians.num_gaussians:,} gaussians")
    
    # Voxelize
    print(f"\nVoxelizing...")
    voxelizer = OctreeVoxelizer(
        voxel_size=args.voxel_size,
        max_level=args.max_level,
        min_gaussians=getattr(args, 'min_gaussians', 8),
        max_gaussians=args.max_gaussians,
        compact_threshold=0.5,
        device=device,
        cache_dir=args.cache_dir
    )
    voxels = voxelizer.voxelize_or_load(gaussians, args.ply, force_rebuild=args.force_rebuild)
    print(f"  Created {len(voxels)} voxels")
    
    # Train/Val split
    random.shuffle(voxels)
    n_val = int(len(voxels) * args.val_split)
    train_voxels = voxels[n_val:]
    val_voxels = voxels[:n_val]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_voxels)} voxels")
    print(f"  Val: {len(val_voxels)} voxels")
    
    # Create datasets
    train_dataset = VoxelDataset(gaussians, train_voxels, args.max_gaussians)
    val_dataset = VoxelDataset(gaussians, val_voxels, args.max_gaussians)
    
    train_loader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    val_loader = val_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)
    
    return train_loader, val_loader


def setup_callbacks(args) -> list:
    """콜백 설정"""
    callbacks = [
        CheckpointCallback(
            save_dir=args.save_dir,
            save_interval=getattr(args, 'save_interval', 10),
            save_best=True,
            save_last=True,
            monitor="val_loss_total",
            mode="min"
        ),
        LoggingCallback(
            log_interval=10,
            log_to_file=True,
            log_path=os.path.join(args.save_dir, "training.log")
        ),
    ]
    
    # Early stopping (optional)
    if getattr(args, 'early_stopping_patience', 0) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                patience=args.early_stopping_patience,
                monitor="val_loss_total",
                mode="min"
            )
        )
    
    return callbacks


def main():
    # Parse arguments with config
    args = parse_args_with_config()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup
    train_loader, val_loader = setup_data(args, device)
    model = setup_model(args, device)
    criterion = setup_criterion(args, args.save_dir)
    callbacks = setup_callbacks(args)
    
    print(f"\nModel parameters: {model.num_parameters:,}")
    
    # Gumbel Scheduler
    gumbel_scheduler = GumbelNoiseScheduler(
        model=model,
        initial_scale=getattr(args, 'gumbel_initial_scale', 0.5),
        final_scale=getattr(args, 'gumbel_final_scale', 0.0),
        warmup_epochs=getattr(args, 'gumbel_warmup_epochs', 10),
        total_epochs=args.epochs,
        schedule_type=getattr(args, 'gumbel_schedule_type', 'warmup_cosine')
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        max_epochs=args.epochs,
        device=device,
        callbacks=callbacks,
        gumbel_scheduler=gumbel_scheduler,
        lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 1e-5),
        grad_clip=getattr(args, 'grad_clip', 1.0),
    )
    
    # Set compression ratio range (배치마다 랜덤 샘플링)
    trainer.compression_ratio_min = getattr(args, 'compression_ratio_min', 0.5)
    trainer.compression_ratio_max = getattr(args, 'compression_ratio_max', 0.5)
    print(f"\nCompression ratio range: [{trainer.compression_ratio_min}, {trainer.compression_ratio_max}]")
    
    # Resume if specified
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"  Resumed at epoch {start_epoch}")
    
    # Train
    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
