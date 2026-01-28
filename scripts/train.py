#!/usr/bin/env python3
"""
Gaussian Merging AE Training Script

Voxel 단위로 Gaussian을 압축하는 모델 학습
- GMAELoss: Density KL + Rendering + Sparsity 손실

Usage:
    python train.py --ply path/to/point_cloud.ply --epochs 100
"""

import os
import sys
import random

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple

# 패키지 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs_merge import (
    GaussianMergingAE,
    GaussianData, load_ply, OctreeVoxelizer, VoxelNode,
    GMAELoss, ChamferLoss,
    parse_gaussian_tensor, model_output_to_dict,
    get_training_parser
)
from gs_merge.data.dataset import VoxelDataset
from gs_merge.config.args import parse_args_with_config


class Trainer:
    """Gaussian Merging AE Trainer"""
    
    def __init__(
        self,
        model: GaussianMergingAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_type: str = "gmae",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        log_interval: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        self.log_interval = log_interval
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer & Scheduler
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=lr * 0.01)
        
        # Loss Function
        if loss_type == "gmae":
            self.criterion = GMAELoss(
                lambda_density=getattr(self, 'lambda_density', 1.0),
                lambda_render=getattr(self, 'lambda_render', 2.0),
                lambda_sparsity=getattr(self, 'lambda_sparsity', 0.01),
                n_density_samples=getattr(self, 'n_density_samples', 1024),
                render_resolution=getattr(self, 'render_resolution', 64),
                warmup_iterations=getattr(self, 'warmup_iterations', 200),
                debug_save_dir=os.path.join(save_dir, "debug_renders"),
                debug_save_interval=getattr(self, 'debug_save_interval', 100)
            )
            self.loss_keys = ['loss_density', 'loss_render', 'loss_sparsity', 'loss_total']
        else:
            self.criterion = ChamferLoss()
            self.loss_keys = ['loss_pos', 'loss_rot', 'loss_scale', 'loss_opacity', 'loss_sh', 'loss_total']
        
        # Logging
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        
        loss_accum = {k: 0.0 for k in self.loss_keys}
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, levels, masks) in enumerate(pbar):
            data = data.to(self.device)
            levels = levels.to(self.device)
            masks = masks.to(self.device)
            
            # Forward
            pred_xyz, pred_rot, pred_scale, pred_opacity, pred_sh = self.model(data, levels, masks)
            
            # Prepare dicts for loss
            input_dict = parse_gaussian_tensor(data)
            output_dict = model_output_to_dict(pred_xyz, pred_rot, pred_scale, pred_opacity, pred_sh)
            
            # Loss
            loss, loss_dict = self.criterion(input_dict, output_dict, input_mask=masks, output_mask=None)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate
            for k in self.loss_keys:
                if k in loss_dict:
                    loss_accum[k] += loss_dict[k]
            n_batches += 1
            
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'den': f"{loss_dict.get('loss_density', 0):.4f}"
                })
        
        return {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        loss_accum = {k: 0.0 for k in self.loss_keys}
        n_batches = 0
        
        for data, levels, masks in self.val_loader:
            data = data.to(self.device)
            levels = levels.to(self.device)
            masks = masks.to(self.device)
            
            pred_xyz, pred_rot, pred_scale, pred_opacity, pred_sh = self.model(data, levels, masks)
            
            input_dict = parse_gaussian_tensor(data)
            output_dict = model_output_to_dict(pred_xyz, pred_rot, pred_scale, pred_opacity, pred_sh)
            
            loss, loss_dict = self.criterion(input_dict, output_dict, input_mask=masks, output_mask=None)
            
            for k in self.loss_keys:
                if k in loss_dict:
                    loss_accum[k] += loss_dict[k]
            n_batches += 1
        
        return {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, os.path.join(self.save_dir, 'last.pt'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best.pt'))
        
        if epoch % 10 == 0:
            torch.save(checkpoint, os.path.join(self.save_dir, f'epoch_{epoch}.pt'))
    
    def load_checkpoint(self, path: str) -> int:
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint['epoch']
    
    def train(self):
        """전체 학습 루프"""
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"  Epochs: {self.max_epochs}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader) if self.val_loader else 0}")
        print(f"  Device: {self.device}")
        print(f"  Loss type: {type(self.criterion).__name__}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            self.scheduler.step()
            
            is_best = False
            current_val = val_loss.get('loss_total', float('inf'))
            if current_val < self.best_val_loss:
                self.best_val_loss = current_val
                is_best = True
            
            self.save_checkpoint(epoch, is_best)
            
            print(f"\nEpoch {epoch}/{self.max_epochs}")
            print(f"  Train: total={train_loss.get('loss_total', 0):.4f} "
                  f"density={train_loss.get('loss_density', 0):.4f} "
                  f"render={train_loss.get('loss_render', 0):.4f}")
            if val_loss:
                print(f"  Val:   total={val_loss.get('loss_total', 0):.4f} "
                      f"density={val_loss.get('loss_density', 0):.4f} "
                      f"render={val_loss.get('loss_render', 0):.4f}")
                if is_best:
                    print(f"  ✓ New best model!")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")


def main():
    # Parse arguments with config
    args = parse_args_with_config()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using device: {device}")
    
    # 1. Load PLY
    print(f"\nLoading PLY: {args.ply}")
    gaussians = load_ply(args.ply, device=device)
    print(f"Loaded {gaussians.num_gaussians:,} gaussians")
    
    # 2. Voxelize
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
    
    # 3. Train/Val split
    random.shuffle(voxels)
    n_val = int(len(voxels) * args.val_split)
    train_voxels = voxels[n_val:]
    val_voxels = voxels[:n_val]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_voxels)} voxels")
    print(f"  Val: {len(val_voxels)} voxels")
    
    # 4. Create datasets
    train_dataset = VoxelDataset(gaussians, train_voxels, args.max_gaussians)
    val_dataset = VoxelDataset(gaussians, val_voxels, args.max_gaussians)
    
    train_loader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    val_loader = val_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)
    
    # 5. Create model
    model = GaussianMergingAE(
        input_dim=getattr(args, 'input_dim', 59),
        latent_dim=getattr(args, 'latent_dim', 256),
        num_queries=args.num_queries,
        nhead=getattr(args, 'nhead', 8),
        num_enc_layers=getattr(args, 'num_enc_layers', 4),
        num_dec_layers=getattr(args, 'num_dec_layers', 4),
        max_octree_level=args.max_level
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_type=args.loss_type,
        lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 1e-5),
        max_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        log_interval=10
    )
    
    # Pass loss config to trainer BEFORE GMAELoss initialization
    if args.loss_type == "gmae":
        trainer.lambda_density = getattr(args, 'lambda_density', 1.0)
        trainer.lambda_render = getattr(args, 'lambda_render', 2.0)
        trainer.lambda_sparsity = getattr(args, 'lambda_sparsity', 0.01)
        trainer.n_density_samples = getattr(args, 'n_density_samples', 1024)
        trainer.render_resolution = getattr(args, 'render_resolution', 64)
        trainer.debug_save_interval = getattr(args, 'debug_save_interval', 10)
        trainer.warmup_iterations = getattr(args, 'warmup_iterations', 200)
        
        # Re-initialize criterion with correct warmup_iterations
        trainer.criterion = GMAELoss(
            lambda_density=trainer.lambda_density,
            lambda_render=trainer.lambda_render,
            lambda_sparsity=trainer.lambda_sparsity,
            n_density_samples=trainer.n_density_samples,
            render_resolution=trainer.render_resolution,
            warmup_iterations=trainer.warmup_iterations,
            debug_save_dir=os.path.join(args.save_dir, "debug_renders"),
            debug_save_interval=trainer.debug_save_interval
        )
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed at epoch {start_epoch}")
    
    # 7. Train
    trainer.train()


if __name__ == "__main__":
    main()
