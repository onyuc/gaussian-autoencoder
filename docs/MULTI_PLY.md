# 여러 PLY 파일 학습 가이드

## 사용 방법

### 1. 단일 PLY 파일 (기존 방식)
```bash
python scripts/train.py --ply path/to/model.ply --config configs/default.yaml
```

### 2. 여러 PLY 파일 (새로운 방식)
```bash
python scripts/train.py \
  --ply path/to/model1.ply path/to/model2.ply path/to/model3.ply \
  --config configs/default.yaml
```

또는:
```bash
python scripts/train.py \
  --ply scene/*.ply \
  --config configs/default.yaml
```

## 동작 방식

1. **각 PLY 파일을 독립적으로 로드**
   - 각 PLY 파일마다 별도의 GaussianData 객체 생성
   - 각각 Voxelize 수행

2. **모든 Voxel을 하나로 합침**
   - 모든 PLY 파일의 voxel들이 하나의 데이터셋으로 통합
   - Train/Val split은 전체 voxel 기준으로 수행

3. **호환성 유지**
   - 기존 코드와 100% 호환
   - 단일 PLY 파일도 동일하게 작동

## 예시

### 여러 장면 동시 학습
```bash
python scripts/train.py \
  --ply \
    /data/scenes/scene1/point_cloud.ply \
    /data/scenes/scene2/point_cloud.ply \
    /data/scenes/scene3/point_cloud.ply \
  --config configs/default.yaml \
  --epochs 1000 \
  --batch_size 32
```

### Multi-GPU 학습
```bash
accelerate launch --multi_gpu --num_processes=4 scripts/train.py \
  --ply scene1.ply scene2.ply scene3.ply \
  --config configs/default.yaml \
  --use_accelerate
```

## 장점

1. **데이터 다양성**: 여러 장면/객체를 동시에 학습하여 일반화 성능 향상
2. **데이터 증강**: 더 많은 voxel 샘플로 robust한 모델 학습
3. **효율성**: 한 번의 학습으로 여러 장면 처리
4. **캐시 활용**: 각 PLY 파일의 voxelization 결과가 개별적으로 캐시됨

## 주의사항

1. **메모리**: 여러 PLY 파일을 동시에 메모리에 로드하므로 메모리 사용량 증가
2. **Voxel 크기**: 모든 PLY 파일이 동일한 `voxel_size`와 `max_level` 설정 사용
3. **캐시**: 각 PLY 파일마다 별도의 캐시 파일이 생성됨

## 출력 예시

```
Loading PLY: scene1.ply
  Loaded 1,234,567 gaussians
  Voxelizing...
  Created 456 voxels

Loading PLY: scene2.ply
  Loaded 2,345,678 gaussians
  Voxelizing...
  Created 789 voxels

Total voxels from 2 file(s): 1245

Dataset split:
  Train: 1120 voxels
  Val: 125 voxels
```
