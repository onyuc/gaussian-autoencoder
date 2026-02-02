"""
Scheduler Tests

학습 스케줄러 테스트
"""

import pytest
import torch
import torch.nn as nn

from gs_merge.training.schedulers import (
    BaseScheduler,
    GumbelNoiseScheduler,
    CompressionRatioScheduler
)


class MockModel(nn.Module):
    """테스트용 모의 모델"""
    def __init__(self):
        super().__init__()
        self._gumbel_noise_scale = 0.0
    
    @property
    def gumbel_noise_scale(self):
        return self._gumbel_noise_scale
    
    @gumbel_noise_scale.setter
    def gumbel_noise_scale(self, value):
        self._gumbel_noise_scale = value


class TestBaseScheduler:
    """BaseScheduler 테스트"""
    
    def test_linear_schedule(self):
        """선형 스케줄 테스트"""
        scheduler = BaseScheduler(
            initial_value=1.0,
            final_value=0.0,
            total_epochs=10,
            schedule_type="linear"
        )
        
        values = [scheduler.step(e) for e in range(11)]
        
        assert values[0] == pytest.approx(1.0, abs=0.01)
        assert values[5] == pytest.approx(0.5, abs=0.01)
        assert values[10] == pytest.approx(0.0, abs=0.01)
    
    def test_cosine_schedule(self):
        """코사인 스케줄 테스트"""
        scheduler = BaseScheduler(
            initial_value=1.0,
            final_value=0.0,
            total_epochs=10,
            schedule_type="cosine"
        )
        
        values = [scheduler.step(e) for e in range(11)]
        
        assert values[0] == pytest.approx(1.0, abs=0.01)
        assert values[10] == pytest.approx(0.0, abs=0.01)
        # 코사인은 중간에서 더 빠르게 감소
        assert values[5] < 0.6
    
    def test_warmup_cosine_schedule(self):
        """Warmup 코사인 스케줄 테스트"""
        scheduler = BaseScheduler(
            initial_value=1.0,
            final_value=0.0,
            warmup_epochs=3,
            total_epochs=10,
            schedule_type="warmup_cosine"
        )
        
        # Warmup 동안은 초기값 유지
        assert scheduler.step(0) == pytest.approx(1.0, abs=0.01)
        assert scheduler.step(1) == pytest.approx(1.0, abs=0.01)
        assert scheduler.step(2) == pytest.approx(1.0, abs=0.01)
        
        # Warmup 이후 감소
        assert scheduler.step(3) < 1.0
        assert scheduler.step(10) == pytest.approx(0.0, abs=0.01)
    
    def test_state_dict(self):
        """상태 저장/복원 테스트"""
        scheduler = BaseScheduler(
            initial_value=1.0,
            final_value=0.0,
            total_epochs=10,
            schedule_type="linear"
        )
        
        # 몇 스텝 진행
        for _ in range(5):
            scheduler.step()
        
        # 상태 저장
        state = scheduler.state_dict()
        
        # 새 스케줄러에 복원
        new_scheduler = BaseScheduler(0, 0, 0, 0)
        new_scheduler.load_state_dict(state)
        
        assert new_scheduler.current_epoch == scheduler.current_epoch
        assert new_scheduler.get_value() == pytest.approx(scheduler.get_value(), abs=0.01)


class TestGumbelNoiseScheduler:
    """GumbelNoiseScheduler 테스트"""
    
    def test_model_update(self):
        """모델 업데이트 테스트"""
        model = MockModel()
        
        scheduler = GumbelNoiseScheduler(
            model=model,
            initial_scale=0.5,
            final_scale=0.0,
            total_epochs=10,
            schedule_type="linear"
        )
        
        # 초기값 확인
        assert model.gumbel_noise_scale == pytest.approx(0.5, abs=0.01)
        
        # 스텝 진행
        scheduler.step(5)
        assert model.gumbel_noise_scale == pytest.approx(0.25, abs=0.01)
        
        scheduler.step(10)
        assert model.gumbel_noise_scale == pytest.approx(0.0, abs=0.01)


class TestCompressionRatioScheduler:
    """CompressionRatioScheduler 테스트"""
    
    def test_easy_to_hard(self):
        """Easy to hard 모드 테스트"""
        scheduler = CompressionRatioScheduler(
            initial_ratio=0.8,
            final_ratio=0.3,
            total_epochs=10,
            schedule_type="linear",
            mode="easy_to_hard"
        )
        
        assert scheduler.get_ratio() == pytest.approx(0.8, abs=0.01)
        scheduler.step(10)
        assert scheduler.get_ratio() == pytest.approx(0.3, abs=0.01)
    
    def test_hard_to_easy(self):
        """Hard to easy 모드 테스트"""
        scheduler = CompressionRatioScheduler(
            initial_ratio=0.8,
            final_ratio=0.3,
            total_epochs=10,
            schedule_type="linear",
            mode="hard_to_easy"
        )
        
        # 모드가 반대이므로 initial=0.3, final=0.8
        assert scheduler.get_ratio() == pytest.approx(0.3, abs=0.01)
        scheduler.step(10)
        assert scheduler.get_ratio() == pytest.approx(0.8, abs=0.01)
