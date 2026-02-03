#!/usr/bin/env python3
"""
Gaussian Merging AE Training Script

Voxel 단위로 Gaussian을 압축하는 모델 학습
Multi-GPU 학습 지원 (Accelerate 라이브러리)

Usage:
    # Single GPU
    python train.py --ply path/to/point_cloud.ply --epochs 100
    
    # Multi-GPU with Accelerate
    accelerate launch --multi_gpu --num_processes=2 train.py --ply model.ply --use_accelerate
    
    # Multi-GPU with config file
    accelerate launch --config_file accelerate_config.yaml train.py --config configs/default.yaml --ply model.ply
"""

import os
import sys
import random

import torch

# Accelerate for Multi-GPU
try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None

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
    TensorBoardCallback,
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
            warmup_iterations=getattr(args, 'warmup_iterations', 1000)
        )
    else:
        return ChamferLoss()


def setup_data(args, device: str, accelerator=None):
    """데이터 로더 생성"""
    is_main = accelerator is None or accelerator.is_main_process
    
    # PLY 파일 목록 처리 (단일 또는 여러 개)
    ply_files = args.ply if isinstance(args.ply, list) else [args.ply]
    
    all_voxel_data = []  # (voxel, gaussians) 튜플 리스트
    
    for ply_path in ply_files:
        # Load PLY
        if is_main:
            print(f"\nLoading PLY: {ply_path}")
        gaussians = load_ply(ply_path, device=device)
        if is_main:
            print(f"  Loaded {gaussians.num_gaussians:,} gaussians")
        
        # Voxelize
        if is_main:
            print(f"  Voxelizing...")
        voxelizer = OctreeVoxelizer(
            voxel_size=args.voxel_size,
            max_level=args.max_level,
            min_gaussians=getattr(args, 'min_gaussians', 8),
            max_gaussians=args.max_gaussians,
            compact_threshold=0.5,
            device=device,
            cache_dir=args.cache_dir
        )
        voxels = voxelizer.voxelize_or_load(gaussians, ply_path, force_rebuild=args.force_rebuild)
        if is_main:
            print(f"  Created {len(voxels)} voxels")
        
        # 각 voxel에 해당 gaussians 객체 매핑
        for voxel in voxels:
            all_voxel_data.append((voxel, gaussians))
    
    if is_main:
        print(f"\nTotal voxels from {len(ply_files)} file(s): {len(all_voxel_data)}")
    
    # Train/Val split
    random.shuffle(all_voxel_data)
    n_val = int(len(all_voxel_data) * args.val_split)
    train_voxel_data = all_voxel_data[n_val:]
    val_voxel_data = all_voxel_data[:n_val]
    
    if is_main:
        print(f"\nDataset split:")
        print(f"  Train: {len(train_voxel_data)} voxels")
        print(f"  Val: {len(val_voxel_data)} voxels")
    
    # Create datasets (이제 (voxel, gaussians) 튜플 리스트를 받음)
    train_dataset = VoxelDataset(train_voxel_data, args.max_gaussians)
    val_dataset = VoxelDataset(val_voxel_data, args.max_gaussians)
    
    # DataLoader 설정
    num_workers = getattr(args, 'num_workers', 0)
    pin_memory = getattr(args, 'pin_memory', False)
    
    # Multi-GPU 학습 시 DistributedSampler 사용
    train_loader = train_dataset.get_dataloader(
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = val_dataset.get_dataloader(
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader


def setup_callbacks(args, is_main_process: bool = True) -> list:
    """콜백 설정"""
    # 메인 프로세스에서만 체크포인트 저장
    if not is_main_process:
        return []
    
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
    
    # TensorBoard logging (선택적)
    if getattr(args, 'use_tensorboard', False):
        tensorboard_dir = getattr(args, 'tensorboard_dir', '')
        if not tensorboard_dir:
            # save_dir 이름을 run name으로 사용
            run_name = os.path.basename(args.save_dir.rstrip('/'))
            tensorboard_dir = os.path.join(args.save_dir, 'tensorboard', run_name)
        callbacks.append(
            TensorBoardCallback(
                log_dir=tensorboard_dir,
                log_histograms=getattr(args, 'tensorboard_log_histograms', True),
                log_interval=getattr(args, 'tensorboard_log_interval', 0)
            )
        )
        print(f"TensorBoard enabled: {tensorboard_dir}")
    
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
    
    # Accelerate 초기화
    use_accelerate = getattr(args, 'use_accelerate', False) and ACCELERATE_AVAILABLE
    accelerator = None
    
    if use_accelerate:
        # DDP 설정
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=getattr(args, 'find_unused_parameters', False)
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
            kwargs_handlers=[ddp_kwargs],
        )
        device = accelerator.device
        is_main = accelerator.is_main_process
        
        if is_main:
            print(f"\n{'='*60}")
            print(f"Multi-GPU Training with Accelerate")
            print(f"  Number of GPUs: {accelerator.num_processes}")
            print(f"  Mixed Precision: {accelerator.mixed_precision}")
            print(f"  Device: {device}")
            print(f"{'='*60}")
    else:
        device = args.device if torch.cuda.is_available() else "cpu"
        is_main = True
        print(f"Using device: {device}")
    
    # Setup
    train_loader, val_loader = setup_data(args, device, accelerator)
    model = setup_model(args, device)
    criterion = setup_criterion(args, args.save_dir)
    callbacks = setup_callbacks(args, is_main)
    
    if is_main:
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
    
    # Create Trainer with Accelerate support
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
        # Multi-GPU support
        accelerator=accelerator,
        use_accelerate=use_accelerate,
        # Debug settings
        debug_save_dir=getattr(args, 'debug_save_dir', './debug_renders'),
        debug_save_epochs=getattr(args, 'debug_save_epochs', 5),
    )
    
    # Set compression ratio range (배치마다 랜덤 샘플링)
    trainer.compression_ratio_min = getattr(args, 'compression_ratio_min', 0.5)
    trainer.compression_ratio_max = getattr(args, 'compression_ratio_max', 0.5)
    if is_main:
        print(f"\nCompression ratio range: [{trainer.compression_ratio_min}, {trainer.compression_ratio_max}]")
    
    # Resume if specified
    start_epoch = 1
    if args.resume:
        if is_main:
            print(f"\nResuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        if is_main:
            print(f"  Resumed at epoch {start_epoch}")
    
    # Train
    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
