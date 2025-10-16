# 모델 요약 및 성능 비교 (Conv-TasNet 변형들)

이 문서는 본 저장소에서 사용한 Conv-TasNet 기반 모델 변형과 학습/평가 설정, 그리고 테스트 데이터 기준 성능을 정리합니다. 모델별 Git 태그와 재현(체크아웃/평가) 방법도 함께 안내합니다.

## 모델 변형과 태그

- baseline — 태그: `baseline`
  - 기본 스케줄러(코드 기본값: ReduceLROnPlateau)
  - 기본 증강(랜덤 볼륨·시간 쉬프트·랜덤 노이즈) 사용
- CosineAnnealingLR — 태그: `CosineAnnealingLR`
  - 스케줄러를 CosineAnnealingLR로 교체한 변형
- Augmented (with NoiseGenerator) — 태그: `NoiseGenAugmented`
  - 노이즈 생성(모사) 모델로 사전학습(pretrain) 후, 실제 데이터에 파인튜닝하기 전 단계 결과
- AugmentedFinetune — 태그: `AugmentedFinetune`
  - 위 Augmented 가중치를 실제 데이터로 파인튜닝한 결과

> 참고: 태그 이름은 Git에 미리 달아두신 기준입니다. 태그 체크아웃으로 각 변형을 재현할 수 있습니다.

## 공통 아키텍처와 학습 요약

- Generator: Conv-TasNet
  - Encoder(Conv1d) → gLN(Global Layer Norm) → Bottleneck(1x1) → TCN 블록 반복(Residual/Skip) → Mask 예측 → Decoder(ConvTranspose1d)
  - Depthwise separable conv, gLN, skip 합산(skip_sum), Dropout(0.1) 적용
- Discriminator(GAN): 1D Conv 스택 + InstanceNorm + SpectralNorm + AdaptiveAvgPool1d
- 손실 조합(train.py CONFIG 기준)
  - SI-SDR(주 손실, torchmetrics 기반 함수 래핑) — `lambda_sdr = 2.0`
  - MR-STFT 손실 — `lambda_stft = 1.0`
  - Log-Mel Perceptual 손실 — `lambda_perc = 0.8`
  - Adversarial(MSE) — `lambda_adv = 0.005`, warmup `adv_warmup_epochs = 8`
  - 라벨 스무딩(smooth_labels) 유틸 제공
- 최적화/스케줄링(기본값)
  - Optim: Adam (G는 weight_decay=1e-4)
  - Scheduler: ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
  - Early Stopping: patience=10
  - AMP 옵션, torch.compile 옵션 제공(환경 호환 시)
- 데이터/증강(dataset.py)
  - 학습 시: 랜덤 볼륨(0.8~1.2), 시간 쉬프트(±0.2s), 추가 랜덤 노이즈(SNR 15~30 dB)
  - 평가 시: 전체 구간 사용(크롭/증강 없음)

## 테스트 성능 비교 (test 데이터)

표의 “denoised / noisy”는 각각 “복원 신호 vs 정답(향상 점수) / 입력 노이즈 신호 vs 정답(원 점수)”을 의미합니다.

| 모델 | Git 태그 | PESQ | STOI | SI-SDR (dB) |
|---|---|---:|---:|---:|
| noisy | . | 1.4936 | 0.6958 | -9.1918 |
| baseline | baseline | 1.9658 | 0.7945 | 6.9085 |
| CosineAnnealingLR | CosineAnnealingLR | 2.0348 | 0.7990 | 7.0838 |
| Augmented (with NoiseGenerator) | NoiseGenAugmented | 1.4678 | 0.7139 | -9.1296 |
| AugmentedFinetune | AugmentedFinetune | 0.6659 | 0.7632 | 5.4110 |

### 간단 해석

- 종합 1위: CosineAnnealingLR — PESQ 2.0348, STOI 0.7990, SI-SDR 7.0838 dB로 전반적 최고.
- baseline: PESQ 1.9658, STOI 0.7945, SI-SDR 6.9085 dB로 근소하게 2위.
- NoiseGenAugmented: 사전학습만으로는 PESQ와 SI-SDR이 낮고(1.4678, -9.1296 dB), STOI는 0.7139로 소폭 향상.
- NoiseGenAugmented_finetune: 파인튜닝 후 PESQ/STOI/SI-SDR이 회복(1.6659 / 0.7632 / 5.4110 dB)되지만, baseline 및 CosineAnnealingLR에는 소폭 미치지 못함.

지표 간 트레이드오프가 존재하며, Augmented는 명료도(STOI)에 특히 강점, CosineAnnealingLR은 균형적으로 개선, Finetune은 PESQ/명료도는 회복하되 SI-SDR은 덜 개선되는 양상을 보였습니다.

## 재현(체크아웃 및 평가) 방법

다음은 PowerShell 기준 예시입니다. 태그를 체크아웃한 뒤, 요구 패키지를 설치하고 평가 스크립트를 실행하세요. 실행 옵션은 `test.py`/`train.py`에서 확인하세요.

```powershell
# 1) 태그 가져오기 및 체크아웃
git fetch --all --tags

# baseline
git checkout tags/baseline -b exp-baseline
# 또는 다른 변형
# git checkout tags/CosineAnnealingLR -b exp-cosine
# git checkout tags/NoiseGenAugmented -b exp-aug
# git checkout tags/AugmentedFinetune -b exp-aug-ft

# 2) 의존성 설치 (가상환경 권장)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) 평가 실행 (예시)
# 사전 학습된 체크포인트 경로는 변형/출력 디렉터리(CONFIG.output_root)별로 상이할 수 있습니다.
# 기본 출력 디렉터리 예: convtasnet_baseline/checkpoints/best_model.pth
python test.py
```

- 학습/출력 경로는 `train.py`의 `CONFIG["output_root"]`에 의해 결정됩니다.
- 체크포인트/로그는 `<output_root>/checkpoints`, `<output_root>/training_log.csv` 등에 저장됩니다.

## 파일 안내

- `model.py`: Generator(Conv-TasNet)와 Discriminator 정의
- `train.py`: 학습 루프, 손실 조합, 로깅/체크포인트, 스케줄러/얼리 스토핑 설정
- `dataset.py`: 데이터 로딩 및 증강
- `utils.py`: 메트릭, 손실 유틸, 시각화/오디오 저장, 체크포인트 로드/저장
- `test.py`: 테스트 데이터 평가(지표 계산 및 샘플 저장)

필요 시 추가 실험(예: 드롭아웃 비율 조절, 스케줄러 파라미터 튜닝, 증강 강도 변화)을 통해 지표 간 균형을 맞출 수 있습니다.
