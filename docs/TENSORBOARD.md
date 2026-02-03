# TensorBoard 로깅 가이드

## 개요

TensorBoard를 통해 학습 과정을 실시간으로 모니터링하고 분석할 수 있습니다.

## 기능

### 1. Loss 추적
- **Training/Validation Loss**: 전체 손실 및 개별 손실 컴포넌트
  - `Loss/train/loss_total` - 전체 학습 손실
  - `Loss/train/loss_density` - 밀도 필드 매칭 손실
  - `Loss/train/loss_render` - 렌더링 손실
  - `Loss/train/loss_sparsity` - Sparsity 정규화 손실
  - `Loss/val/*` - 검증 손실들

### 2. 가우시안 통계

#### 입력(Input) 가우시안 통계
- **Opacity 통계**
  - `Gaussian/Input/opacity_mean` - 평균 opacity
  - `Gaussian/Input/opacity_std` - 표준편차
  - `Gaussian/Input/opacity_min/max` - 최소/최대값
  - `Gaussian/Input/opacity_high_ratio` - 높은 opacity 비율 (>0.5)

- **Scale 통계**
  - `Gaussian/Input/scale_mean` - 평균 scale (전체)
  - `Gaussian/Input/scale_std` - 표준편차
  - `Gaussian/Input/scale_min/max` - 최소/최대값
  - `Gaussian/Input/scale_x/y/z_mean` - 축별 평균 scale
  - `Gaussian/Input/scale_x/y/z_max` - 축별 최대 scale
  - `Gaussian/Input/scale_x/y/z_min` - 축별 최소 scale

- **유효 가우시안 개수**
  - `Gaussian/Input/count_valid` - 배치당 평균 유효 가우시안 개수

#### 출력(Output/Reconstructed) 가우시안 통계
- **Opacity 통계**
  - `Gaussian/Output/opacity_mean` - 평균 opacity
  - `Gaussian/Output/opacity_std` - 표준편차
  - `Gaussian/Output/opacity_min/max` - 최소/최대값
  - `Gaussian/Output/opacity_high_ratio` - 높은 opacity 비율 (>0.5)

- **Scale 통계**
  - `Gaussian/Output/scale_mean` - 평균 scale (전체)
  - `Gaussian/Output/scale_std` - 표준편차
  - `Gaussian/Output/scale_min/max` - 최소/최대값
  - `Gaussian/Output/scale_x/y/z_mean` - 축별 평균 scale
  - `Gaussian/Output/scale_x/y/z_max` - 축별 최대 scale
  - `Gaussian/Output/scale_x/y/z_min` - 축별 최소 scale
  - *(주: 2 이상의 scale은 자동으로 클램프되어 보기 쉽게 표시)*

- **유효 가우시안 개수**
  - `Gaussian/Output/count_valid` - 배치당 평균 유효 가우시안 개수

### 3. 히스토그램 (분포)
- **입력 가우시안**
  - `Distribution/input_opacity` - 입력 Opacity 분포
  - `Distribution/input_scale` - 입력 Scale 분포 (평균, 2 이상 클램프)

- **출력 가우시안**
  - `Distribution/output_opacity` - 출력 Opacity 분포
  - `Distribution/output_scale` - 출력 Scale 분포 (평균, 2 이상 클램프)

### 4. 학습 파라미터
- `Training/learning_rate` - 학습률
- `Training/gumbel_scale` - Gumbel noise scale

## 설정

### config 파일에서 설정 (권장)

`configs/default.yaml`:
```yaml
tensorboard:
  enabled: true
  log_dir: ""  # 비어있으면 save_dir/tensorboard/run_name 자동 사용
  log_histograms: true
  log_interval: 0  # 0=epoch only (권장)
```

**자동 경로 설정:**
- `log_dir`이 비어있으면 `{save_dir}/tensorboard/{run_name}` 사용
- `run_name`은 `save_dir`의 마지막 디렉토리 이름
- 예: `save_dir="/scratch/.../checkpoints/ssim_512"` → `tensorboard/ssim_512/`

### 커맨드라인에서 설정

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --use_tensorboard true \
  --tensorboard_dir ./runs \
  --tensorboard_log_histograms true
```

## TensorBoard 실행

### 1. TensorBoard 서버 시작

기본 경로 사용 (save_dir 내부):
```bash
# 예: save_dir이 /scratch/.../checkpoints/ssim_512인 경우
tensorboard --logdir=/scratch/rchkl2380/Workspace/gaussian-autoencoder/checkpoints/ssim_512/tensorboard --port=6006
```

또는 상위 디렉토리에서 여러 실험 동시 비교:
```bash
tensorboard --logdir=/scratch/rchkl2380/Workspace/gaussian-autoencoder/checkpoints --port=6006
```

### 2. 브라우저에서 접속

로컬:
```
http://localhost:6006
```

원격 서버 (SSH 포트 포워딩):
```bash
ssh -L 6006:localhost:6006 user@server
```
그 다음 로컬 브라우저에서 `http://localhost:6006` 접속

### 3. 실시간 모니터링

학습 중에도 TensorBoard가 자동으로 업데이트됩니다. 새로고침 간격을 조정하거나 자동 새로고침을 활성화할 수 있습니다.

## 주요 뷰

### SCALARS (권장)
- Loss curves (학습/검증)
- 가우시안 통계 시계열 (숫자 기반, epoch별)
- Learning rate, Gumbel scale 추이

### DISTRIBUTIONS / HISTOGRAMS
- **입력 가우시안 분포**
  - `Distribution/input_opacity` - 입력 Opacity 분포 변화
  - `Distribution/input_scale` - 입력 Scale 분포 변화
- **출력 가우시안 분포**
  - `Distribution/output_opacity` - 출력 Opacity 분포 변화
  - `Distribution/output_scale` - 출력 Scale 분포 변화 (2 이상 클램프)
- **참고:** Batch-level logging은 비활성화됨 (epoch만 사용)

### 분석 예시

#### 입력 vs 출력 비교
TensorBoard에서 `Gaussian/Input/opacity_mean`과 `Gaussian/Output/opacity_mean`을 동시에 선택하면:
- 모델이 입력의 opacity를 얼마나 잘 보존하는지 확인 가능
- Reconstruction 품질 모니터링

Scale도 마찬가지로:
- `Gaussian/Input/scale_mean` vs `Gaussian/Output/scale_mean`
- 입력 대비 출력의 크기 변화 추이 확인

## 팁

### 1. 여러 실험 비교
실험마다 다른 `save_dir`을 사용하면 자동으로 구분됨:

```yaml
# Experiment 1
output:
  save_dir: "/scratch/.../checkpoints/exp1_ssim"

# Experiment 2  
output:
  save_dir: "/scratch/.../checkpoints/exp2_l1"
```

TensorBoard를 상위 디렉토리로 실행:
```bash
tensorboard --logdir=/scratch/.../checkpoints
```

TensorBoard가 각 `exp1_ssim`, `exp2_l1`을 자동으로 인식하여 비교 가능!

### 2. 메모리 절약
히스토그램 로깅은 메모리를 사용할 수 있습니다. 필요 없으면 비활성화:
```yaml
tensorboard:
  log_histograms: false
```

### 3. Epoch-only Logging (기본 설정)
Batch-level logging은 데이터가 너무 많아 비활성화됨.
Epoch별로만 로깅되므로 깔끔하고 보기 쉬움.
- Step = Epoch 번호
- 모든 스칼라와 히스토그램은 epoch 단위로 기록

## 문제 해결

### TensorBoard가 설치되지 않음
```bash
pip install tensorboard
# 또는
conda install tensorboard
```

### 로그가 보이지 않음
1. `log_dir` 경로 확인
2. 학습이 최소 1 epoch 완료되었는지 확인
3. TensorBoard 서버 재시작

### 원격 서버 접속 문제
SSH 포트 포워딩 확인:
```bash
ssh -L 6006:localhost:6006 -L 6007:localhost:6007 user@server
```

## 추가 정보

TensorBoard 공식 문서: https://www.tensorflow.org/tensorboard
