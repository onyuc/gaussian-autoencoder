"""
Learning Schedulers

학습 중 하이퍼파라미터를 동적으로 조절하는 스케줄러들
- GumbelNoiseScheduler: 쿼리 선택 노이즈 스케줄링
- CompressionRatioScheduler: 압축 비율 스케줄링
"""

import math
from typing import Optional, Literal

import torch.nn as nn


class BaseScheduler:
    """스케줄러 기본 클래스"""
    
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        warmup_epochs: int = 0,
        total_epochs: int = 100,
        schedule_type: str = "cosine"
    ):
        self.initial_value = initial_value
        self.final_value = final_value
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        self.current_epoch = 0
        self._current_value = initial_value
    
    def step(self, epoch: Optional[int] = None) -> float:
        """에폭 기반 스케줄링 스텝"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        self._current_value = self._compute_value(self.current_epoch)
        return self._current_value
    
    def _compute_value(self, epoch: int) -> float:
        """현재 에폭에 맞는 값 계산"""
        if self.schedule_type == "constant":
            return self.initial_value
        
        elif self.schedule_type == "linear":
            progress = epoch / max(self.total_epochs, 1)
            return self.initial_value + (self.final_value - self.initial_value) * progress
        
        elif self.schedule_type == "cosine":
            progress = epoch / max(self.total_epochs, 1)
            cos_value = (1 + math.cos(math.pi * progress)) / 2
            return self.final_value + (self.initial_value - self.final_value) * cos_value
        
        elif self.schedule_type == "exponential":
            decay_rate = 0.95
            return max(
                self.final_value,
                self.initial_value * (decay_rate ** epoch)
            )
        
        elif self.schedule_type == "warmup_cosine":
            if epoch < self.warmup_epochs:
                return self.initial_value
            else:
                post_warmup = epoch - self.warmup_epochs
                remaining = self.total_epochs - self.warmup_epochs
                if remaining <= 0:
                    return self.final_value
                progress = post_warmup / remaining
                cos_value = (1 + math.cos(math.pi * progress)) / 2
                return self.final_value + (self.initial_value - self.final_value) * cos_value
        
        elif self.schedule_type == "warmup_linear":
            if epoch < self.warmup_epochs:
                # Warmup: 0 → initial_value
                return self.initial_value * (epoch / max(self.warmup_epochs, 1))
            else:
                # Linear decay
                post_warmup = epoch - self.warmup_epochs
                remaining = self.total_epochs - self.warmup_epochs
                if remaining <= 0:
                    return self.final_value
                progress = post_warmup / remaining
                return self.initial_value + (self.final_value - self.initial_value) * progress
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def get_value(self) -> float:
        """현재 값 반환"""
        return self._current_value
    
    def state_dict(self) -> dict:
        """스케줄러 상태 저장"""
        return {
            'current_epoch': self.current_epoch,
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'schedule_type': self.schedule_type,
            'current_value': self._current_value
        }
    
    def load_state_dict(self, state_dict: dict):
        """스케줄러 상태 로드"""
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.initial_value = state_dict.get('initial_value', self.initial_value)
        self.final_value = state_dict.get('final_value', self.final_value)
        self.warmup_epochs = state_dict.get('warmup_epochs', self.warmup_epochs)
        self.total_epochs = state_dict.get('total_epochs', self.total_epochs)
        self.schedule_type = state_dict.get('schedule_type', self.schedule_type)
        self._current_value = state_dict.get('current_value', self._compute_value(self.current_epoch))


class GumbelNoiseScheduler(BaseScheduler):
    """
    Gumbel Noise Scale Scheduler
    
    학습 초기에는 높은 noise로 탐색을 촉진하고,
    학습이 진행될수록 noise를 줄여 안정적인 수렴 유도
    
    Args:
        model: GaussianMergingAE 모델 (gumbel_noise_scale 속성 필요)
        initial_scale: 초기 noise scale (높을수록 탐색)
        final_scale: 최종 noise scale (0 = 결정적 선택)
        warmup_epochs: warmup 에폭 수
        total_epochs: 전체 에폭 수
        schedule_type: 스케줄 타입
    
    Supported schedule types:
        - "constant": 고정값
        - "linear": 선형 감소
        - "cosine": 코사인 어닐링
        - "exponential": 지수 감소
        - "warmup_cosine": warmup 후 코사인 감소 (권장)
        - "warmup_linear": warmup 후 선형 감소
    """
    
    def __init__(
        self,
        model: nn.Module,
        initial_scale: float = 0.5,
        final_scale: float = 0.0,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        schedule_type: str = "warmup_cosine"
    ):
        super().__init__(
            initial_value=initial_scale,
            final_value=final_scale,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            schedule_type=schedule_type
        )
        self.model = model
        
        # 모델에 초기값 설정
        if hasattr(model, 'gumbel_noise_scale'):
            model.gumbel_noise_scale = initial_scale
    
    def step(self, epoch: Optional[int] = None) -> float:
        """스케줄러 스텝 및 모델 업데이트"""
        new_scale = super().step(epoch)
        
        if hasattr(self.model, 'gumbel_noise_scale'):
            self.model.gumbel_noise_scale = new_scale
        
        return new_scale
    
    def get_scale(self) -> float:
        """현재 noise scale 반환 (alias)"""
        return self.get_value()


class CompressionRatioScheduler(BaseScheduler):
    """
    Compression Ratio Scheduler
    
    학습 초기에는 높은 압축률(많은 출력)로 쉬운 학습,
    점진적으로 압축률을 낮춰 어려운 태스크로 전환
    
    또는 반대로:
    학습 초기에는 낮은 압축률로 핵심만 학습,
    점진적으로 더 많은 Gaussian을 사용하도록 확장
    
    Args:
        initial_ratio: 초기 압축 비율 (0.0 ~ 1.0)
        final_ratio: 최종 압축 비율
        warmup_epochs: warmup 에폭 수
        total_epochs: 전체 에폭 수
        schedule_type: 스케줄 타입
        mode: "easy_to_hard" (높음→낮음) 또는 "hard_to_easy" (낮음→높음)
    """
    
    def __init__(
        self,
        initial_ratio: float = 0.8,
        final_ratio: float = 0.3,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        schedule_type: str = "warmup_linear",
        mode: Literal["easy_to_hard", "hard_to_easy"] = "easy_to_hard"
    ):
        # mode에 따라 initial/final 설정
        if mode == "hard_to_easy":
            initial_ratio, final_ratio = final_ratio, initial_ratio
        
        super().__init__(
            initial_value=initial_ratio,
            final_value=final_ratio,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            schedule_type=schedule_type
        )
        self.mode = mode
    
    def get_ratio(self) -> float:
        """현재 압축 비율 반환 (alias)"""
        return max(0.0, min(1.0, self.get_value()))
