"""
Trainer Class

Gaussian Merging AE 모델 학습을 담당하는 클래스
"""

import os
from typing import Dict, Optional, List, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from tqdm import tqdm

from gs_merge.training.schedulers import GumbelNoiseScheduler
from gs_merge.training.callbacks import Callback, CallbackList, CheckpointCallback


class Trainer:
    """
    Gaussian Merging AE Trainer
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더 (선택)
        criterion: 손실 함수
        optimizer: 옵티마이저 (None이면 AdamW 사용)
        scheduler: LR 스케줄러 (None이면 CosineAnnealingLR 사용)
        max_epochs: 최대 에폭 수
        device: 학습 디바이스
        callbacks: 콜백 리스트
        gumbel_scheduler: Gumbel noise 스케줄러 (선택)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        max_epochs: int = 100,
        device: str = "cuda",
        callbacks: Optional[List[Callback]] = None,
        gumbel_scheduler: Optional[GumbelNoiseScheduler] = None,
        compression_ratio_scheduler: Optional[Any] = None,
        # Default optimizer settings
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.device = device
        self.grad_clip = grad_clip
        
        # Optimizer
        if optimizer is None:
            self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer
        
        # LR Scheduler
        if scheduler is None:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=lr * 0.01)
        else:
            self.scheduler = scheduler
        
        # Gumbel Scheduler
        self.gumbel_scheduler = gumbel_scheduler
        
        # Compression Ratio Scheduler
        self.compression_ratio_scheduler = compression_ratio_scheduler
        
        # Compression ratio range for random sampling (배치마다 랜덤)
        self.compression_ratio_min = 0.1
        self.compression_ratio_max = 1.0
        
        # Callbacks
        self.callbacks = CallbackList(callbacks)
        
        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        self.callbacks.on_epoch_begin(self, epoch)
        
        loss_accum: Dict[str, float] = {}
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            self.callbacks.on_batch_begin(self, batch_idx)
            
            # Move to device
            batch = self._move_batch_to_device(batch)
            
            # Forward
            loss, loss_dict = self._forward_step(batch)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            
            self.optimizer.step()
            self.global_step += 1
            
            # Accumulate losses
            for k, v in loss_dict.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v
            n_batches += 1
            
            # Progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            self.callbacks.on_batch_end(self, batch_idx, loss.item())
        
        # Average losses
        avg_losses = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
        return avg_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """검증"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        loss_accum: Dict[str, float] = {}
        n_batches = 0
        
        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)
            loss, loss_dict = self._forward_step(batch)
            
            for k, v in loss_dict.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v
            n_batches += 1
        
        return {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
    
    def _move_batch_to_device(self, batch):
        """배치를 디바이스로 이동"""
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
        elif isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch
    
    def _forward_step(self, batch) -> tuple:
        """Forward 스텝 - 서브클래스에서 오버라이드 가능"""
        # Default: (data, levels, masks) 형태
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            data, levels, masks = batch
            
            # 배치마다 랜덤한 compression ratio 샘플링
            compression_ratio = torch.rand(1).item() * (self.compression_ratio_max - self.compression_ratio_min) + self.compression_ratio_min
            
            outputs = self.model(data, levels, masks, compression_ratio=compression_ratio)
            
            if self.criterion:
                from gs_merge.loss import parse_gaussian_tensor, model_output_to_dict
                
                # outputs이 tuple인 경우 처리
                if isinstance(outputs, tuple):
                    pred_xyz, pred_rot, pred_scale, pred_opacity, pred_sh = outputs[:5]
                    tgt_mask = outputs[5] if len(outputs) > 5 else None
                else:
                    raise ValueError("Unexpected model output format")
                
                input_dict = parse_gaussian_tensor(data)
                output_dict = model_output_to_dict(pred_xyz, pred_rot, pred_scale, pred_opacity, pred_sh)
                
                loss, loss_dict = self.criterion(input_dict, output_dict, input_mask=masks, output_mask=tgt_mask)
                return loss, loss_dict
        
        raise NotImplementedError("Override _forward_step for custom batch handling")
    
    def train(self, start_epoch: int = 1):
        """전체 학습 루프"""
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"  Epochs: {self.max_epochs}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader) if self.val_loader else 0}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")
        
        self.callbacks.on_train_begin(self)
        
        for epoch in range(start_epoch, self.max_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
            self.val_history.append(val_metrics)
            
            # Schedulers
            if self.scheduler:
                self.scheduler.step()
            
            gumbel_scale = 0.0
            if self.gumbel_scheduler:
                gumbel_scale = self.gumbel_scheduler.step(epoch)
            
            compress_ratio = 0.5
            if self.compression_ratio_scheduler:
                compress_ratio = self.compression_ratio_scheduler.step(epoch)
            
            # Metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Callbacks
            self.callbacks.on_epoch_end(self, epoch, all_metrics)
            
            # Logging
            self._log_epoch(epoch, train_metrics, val_metrics, gumbel_scale)
            
            # Early stopping check
            for cb in self.callbacks.callbacks:
                if hasattr(cb, 'should_stop') and cb.should_stop:
                    print(f"\nTraining stopped early at epoch {epoch}")
                    self.callbacks.on_train_end(self)
                    return
        
        self.callbacks.on_train_end(self)
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
    
    def _log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, gumbel_scale: float):
        """에폭 로깅"""
        train_loss = train_metrics.get('train_loss_total', 0)
        val_loss = val_metrics.get('val_loss_total', 0)
        lr = self.scheduler.get_last_lr()[0] if self.scheduler else 0
        
        print(f"\nEpoch {epoch}/{self.max_epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        if val_metrics:
            print(f"  Val loss:   {val_loss:.4f}")
        print(f"  LR: {lr:.6f}  Gumbel: {gumbel_scale:.4f}")
    
    def save_checkpoint(self, path: str):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        if self.gumbel_scheduler:
            checkpoint['gumbel_scheduler_state_dict'] = self.gumbel_scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.gumbel_scheduler and checkpoint.get('gumbel_scheduler_state_dict'):
            self.gumbel_scheduler.load_state_dict(checkpoint['gumbel_scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        return self.current_epoch
