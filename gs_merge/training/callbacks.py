"""
Training Callbacks

학습 중 특정 시점에 실행되는 콜백 함수들
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path

import torch

if TYPE_CHECKING:
    from gs_merge.training.trainer import Trainer


class Callback(ABC):
    """콜백 기본 클래스"""
    
    def on_train_begin(self, trainer: Trainer, **kwargs):
        """학습 시작 시"""
        pass
    
    def on_train_end(self, trainer: Trainer, **kwargs):
        """학습 종료 시"""
        pass
    
    def on_epoch_begin(self, trainer: Trainer, epoch: int, **kwargs):
        """에폭 시작 시"""
        pass
    
    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """에폭 종료 시"""
        pass
    
    def on_batch_begin(self, trainer: Trainer, batch_idx: int, **kwargs):
        """배치 시작 시"""
        pass
    
    def on_batch_end(self, trainer: Trainer, batch_idx: int, loss: float, **kwargs):
        """배치 종료 시"""
        pass


class CallbackList:
    """콜백 리스트 관리"""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback):
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer, **kwargs):
        for cb in self.callbacks:
            cb.on_train_begin(trainer, **kwargs)
    
    def on_train_end(self, trainer, **kwargs):
        for cb in self.callbacks:
            cb.on_train_end(trainer, **kwargs)
    
    def on_epoch_begin(self, trainer, epoch, **kwargs):
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch, **kwargs)
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics, **kwargs)
    
    def on_batch_begin(self, trainer, batch_idx, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_begin(trainer, batch_idx, **kwargs)
    
    def on_batch_end(self, trainer, batch_idx, loss, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch_idx, loss, **kwargs)


class CheckpointCallback(Callback):
    """
    체크포인트 저장 콜백
    
    Args:
        save_dir: 저장 디렉토리
        save_interval: 저장 간격 (에폭)
        save_best: 베스트 모델 저장 여부
        save_last: 마지막 모델 저장 여부
        monitor: 모니터링할 메트릭
        mode: "min" 또는 "max"
    """
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_interval: int = 10,
        save_best: bool = True,
        save_last: bool = True,
        monitor: str = "val_loss_total",
        mode: str = "min"
    ):
        self.save_dir = Path(save_dir)
        self.save_interval = save_interval
        self.save_best = save_best
        self.save_last = save_last
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
    
    def on_train_begin(self, trainer, **kwargs):
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        # 마지막 모델 저장
        if self.save_last:
            self._save_checkpoint(trainer, epoch, "last.pt")
        
        # 주기적 저장
        if self.save_interval > 0 and epoch % self.save_interval == 0:
            self._save_checkpoint(trainer, epoch, f"epoch_{epoch}.pt")
        
        # 베스트 모델 저장
        if self.save_best and self.monitor in metrics:
            current = metrics[self.monitor]
            is_best = (self.mode == "min" and current < self.best_value) or \
                      (self.mode == "max" and current > self.best_value)
            
            if is_best:
                self.best_value = current
                self._save_checkpoint(trainer, epoch, "best.pt")
                print(f"  ✓ New best model! ({self.monitor}: {current:.4f})")
    
    def _save_checkpoint(self, trainer, epoch: int, filename: str):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
            'best_value': self.best_value,
        }
        
        # 추가 스케줄러 상태
        if hasattr(trainer, 'gumbel_scheduler') and trainer.gumbel_scheduler:
            checkpoint['gumbel_scheduler_state_dict'] = trainer.gumbel_scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)


class EarlyStoppingCallback(Callback):
    """
    Early Stopping 콜백
    
    Args:
        patience: 개선 없이 대기할 에폭 수
        monitor: 모니터링할 메트릭
        mode: "min" 또는 "max"
        min_delta: 개선으로 인정할 최소 변화량
    """
    
    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_loss_total",
        mode: str = "min",
        min_delta: float = 1e-4
    ):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if self.monitor not in metrics:
            return
        
        current = metrics[self.monitor]
        
        if self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n  Early stopping triggered at epoch {epoch}")
                print(f"  No improvement in {self.patience} epochs")


class LoggingCallback(Callback):
    """
    로깅 콜백
    
    Args:
        log_interval: 로그 출력 간격 (배치)
        log_to_file: 파일 로깅 여부
        log_path: 로그 파일 경로
    """
    
    def __init__(
        self,
        log_interval: int = 10,
        log_to_file: bool = False,
        log_path: Optional[str] = None
    ):
        self.log_interval = log_interval
        self.log_to_file = log_to_file
        self.log_path = Path(log_path) if log_path else None
        self.epoch_losses = []
    
    def on_train_begin(self, trainer, **kwargs):
        if self.log_to_file and self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'w') as f:
                f.write("epoch,train_loss,val_loss,lr,gumbel_scale\n")
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        train_loss = metrics.get('train_loss_total', 0)
        val_loss = metrics.get('val_loss_total', 0)
        lr = trainer.scheduler.get_last_lr()[0] if trainer.scheduler else 0
        gumbel = trainer.gumbel_scheduler.get_scale() if hasattr(trainer, 'gumbel_scheduler') else 0
        
        if self.log_to_file and self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{lr:.8f},{gumbel:.4f}\n")
