# 변경 사항: 입력 가우시안 통계 추가

## 개요
TensorBoard에서 입력(input) 가우시안과 출력(output/reconstructed) 가우시안의 통계를 모두 볼 수 있도록 개선했습니다.

## 변경 내용

### 1. Loss 모듈 (`gs_merge/loss/gmae_loss.py`)

#### `_compute_gaussian_stats()` 메서드
- **추가**: `prefix` 파라미터로 통계 키에 접두어 추가 가능
- **동작**: `prefix='input'` → `input_opacity_mean`, `input_scale_mean` 등
- **동작**: `prefix='output'` → `output_opacity_mean`, `output_scale_mean` 등

#### `forward()` 메서드
```python
# 이전: 출력 가우시안만
gaussian_stats = self._compute_gaussian_stats(output_g, output_mask)

# 변경 후: 입력 + 출력 모두
output_stats = self._compute_gaussian_stats(output_g, output_mask, prefix='output')
input_stats = self._compute_gaussian_stats(input_g, input_mask, prefix='input')

loss_dict = {
    ...,
    **output_stats,  # output_opacity_mean, output_scale_mean 등
    **input_stats    # input_opacity_mean, input_scale_mean 등
}
```

#### 히스토그램 데이터
```python
# 이전
debug_info['histograms'] = self._get_histograms(output_g, output_mask)

# 변경 후: 입력/출력 구분
debug_info['histograms'] = {
    **{f'output_{k}': v for k, v in self._get_histograms(output_g, output_mask).items()},
    **{f'input_{k}': v for k, v in self._get_histograms(input_g, input_mask).items()}
}
```

### 2. TensorBoard Callback (`gs_merge/training/callbacks.py`)

#### 스칼라 로깅 개선
```python
# 입력 가우시안 통계
if key.startswith('train_input_'):
    metric_name = key.replace('train_input_', '')
    self.writer.add_scalar(f'Gaussian/Input/{metric_name}', value, epoch)

# 출력 가우시안 통계
elif key.startswith('train_output_'):
    metric_name = key.replace('train_output_', '')
    self.writer.add_scalar(f'Gaussian/Output/{metric_name}', value, epoch)
```

#### 히스토그램 로깅
- `Distribution/input_opacity` - 입력 Opacity 분포
- `Distribution/input_scale` - 입력 Scale 분포
- `Distribution/output_opacity` - 출력 Opacity 분포
- `Distribution/output_scale` - 출력 Scale 분포

### 3. 문서 업데이트 (`TENSORBOARD.md`)

#### 추가된 섹션
- 입력 가우시안 통계 목록
- 출력 가우시안 통계 목록
- 입력 vs 출력 비교 분석 예시

## TensorBoard에서 확인 가능한 통계

### 입력 가우시안 (Gaussian/Input/*)
- `opacity_mean`, `opacity_std`, `opacity_min`, `opacity_max`, `opacity_high_ratio`
- `scale_mean`, `scale_std`, `scale_min`, `scale_max`
- `scale_x/y/z_mean`, `scale_x/y/z_max`, `scale_x/y/z_min`
- `count_valid`

### 출력 가우시안 (Gaussian/Output/*)
- 동일한 통계 항목

### 히스토그램 (Distribution/*)
- `input_opacity`, `input_scale`
- `output_opacity`, `output_scale`

## 사용 예시

### TensorBoard에서 비교
1. **Opacity 보존 확인**
   - `Gaussian/Input/opacity_mean`
   - `Gaussian/Output/opacity_mean`
   - 두 그래프를 overlay하여 reconstruction 품질 확인

2. **Scale 변화 추이**
   - `Gaussian/Input/scale_mean`
   - `Gaussian/Output/scale_mean`
   - 모델이 가우시안 크기를 어떻게 변환하는지 분석

3. **분포 시각화**
   - `Distribution/input_opacity` vs `Distribution/output_opacity`
   - 히스토그램으로 전체 분포 변화 확인

## 테스트 결과

```bash
$ python -c "from gs_merge.loss import GMAELoss; ..."

Loss dict keys:
  ✓ input_count_valid
  ✓ input_opacity_high_ratio
  ✓ input_opacity_max
  ✓ input_opacity_mean
  ✓ input_opacity_min
  ✓ input_opacity_std
  ✓ input_scale_max
  ✓ input_scale_mean
  ... (19개)
  
  ✓ output_count_valid
  ✓ output_opacity_high_ratio
  ✓ output_opacity_max
  ✓ output_opacity_mean
  ... (19개)

Histogram keys:
  ✓ output_opacity
  ✓ output_scale
  ✓ input_opacity
  ✓ input_scale

✅ 입력/출력 가우시안 통계가 모두 계산됩니다!
```

## 호환성

- 기존 코드와 100% 호환
- 레거시 지원: prefix 없는 통계도 여전히 작동
- 추가 설정 불필요 (자동으로 적용됨)
