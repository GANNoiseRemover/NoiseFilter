# SimpleModel: Speech Denoising with Conv-TasNet (+ GAN, MR-STFT, Perceptual)

이 저장소는 Conv-TasNet 기반의 음성 잡음 제거(speech denoising) 모델 구현과 학습/평가/데이터 진단 파이프라인을 제공합니다.

- 모델: Conv-TasNet(Temporal ConvNet) + Global Layer Norm, 잔차/스킵, Depthwise Separable Conv
- 추가 손실: SI-SDR, 다중 해상도 STFT(MR-STFT), Log-Mel Perceptual
- 선택적 판별기(Discriminator)로 GAN 손실 지원
- 데이터 진단/정렬/정규화 유틸리티 포함

지원 메트릭: PESQ, STOI, SI-SDR

## 빠른 시작 (Windows PowerShell)

1. (선택) 가상환경 생성 및 활성화

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 의존성 설치

```powershell
pip install -r .\requirements.txt
```

3. 데이터 준비

- 이미 `dataset/`에 `train/validation/test` 폴더 구조와 `clean/noisy` 짝이 있다면 4로 이동
- 없다면 `preprocess.py`가 시연용 더미 데이터를 생성하고 CSV(filelist)를 만들어 줍니다.

```powershell
python .\preprocess.py
```

실행 후 `dataset/train.csv, validation.csv, test.csv`가 생성됩니다.

4. (권장) 데이터 진단/정렬/정규화

- CSV를 기반으로 진단 리포트 및 정규화/정렬된 복사본을 생성할 수 있습니다.
- 예시(훈련/검증/테스트 각각 실행):

```powershell
# 훈련 세트 진단 + RMS 정규화 + 정렬된 페어 생성 (요구되는 옵션에 맞게 조정)
python .\diagnose_dataset.py --csv dataset\train.csv --sr 16000 `
  --write-report diagnostics_train `
  --volume-normalize --target-rms-db -23 --write-adjusted --normalized-outdir normalized `
  --write-aligned --aligned-outdir aligned

# 검증 세트
python .\diagnose_dataset.py --csv dataset\validation.csv --sr 16000 `
  --write-report diagnostics_val `
  --volume-normalize --target-rms-db -23 --write-adjusted --normalized-outdir normalized `
  --write-aligned --aligned-outdir aligned

# 테스트 세트
python .\diagnose_dataset.py --csv dataset\test.csv --sr 16000 `
  --write-report diagnostics_test `
  --volume-normalize --target-rms-db -23 --write-adjusted --normalized-outdir normalized `
  --write-aligned --aligned-outdir aligned
```

결과물: `diagnostics_*/diagnostics_report.csv`, `summary.json`, 히스토그램 이미지, 선택적으로 `normalized/`와 `aligned/` 폴더 및 해당 CSV

5. 학습

```powershell
python .\train.py
```

- 기본 설정은 `train.py`의 `CONFIG`에 정의되어 있으며, 기본적으로 아래 CSV를 사용하도록 되어 있습니다:
  - train: `diagnostics_train/normalized.csv`
  - val:   `diagnostics_val/normalized.csv`
- 위 진단 단계를 생략했다면 `CONFIG`의 `train_csv`, `val_csv`를 `dataset/train.csv`, `dataset/validation.csv`로 바꾸세요.

6. 테스트(베스트 모델)

```powershell
python .\test.py
```

- `test.py`는 기본적으로 `convtasnet_realdata_v3/checkpoints/best_model.pth`를 사용하고, 테스트 CSV를 `diagnostics_test/normalized.csv` (없으면 `dataset/test.csv`)에서 읽습니다.
- 결과물은 `convtasnet_realdata_v3/test_results` 폴더에 저장됩니다(요약 JSON, 샘플 WAV, 스펙트로그램 이미지).

## 프로젝트 구조 개요

```text
Conv-TasNet/
├─ dataset.py                 # DenoisingDataset: CSV 기반 clean/noisy 로더(+학습용 증강)
├─ preprocess.py              # 더미 데이터/CSV 생성 유틸리티
├─ diagnose_dataset.py        # 데이터 진단/정렬/정규화(리포트, 히스토그램, CSV, 보정본)
├─ dataseteval.py             # 폴더 단위(PESQ/STOI) 빠른 평가 유틸리티
├─ model.py                   # ConvTasNet + Discriminator 구현
├─ utils.py                   # 메트릭/체크포인트/STFT/Perceptual 등 유틸
├─ train.py                   # 학습 파이프라인
├─ test.py                    # 단일 체크포인트 평가
├─ requirements.txt
└─ dataset/                   # 데이터 루트 (train/validation/test + clean/noisy)
```

데이터셋 폴더 구조 예시:

```text
dataset/
  train/
    clean/*.wav
    noisy/*.wav
  validation/
    clean/*.wav
    noisy/*.wav
  test/
    clean/*.wav
    noisy/*.wav
```

각 split의 `clean`/`noisy`는 동일 파일명으로 1:1 매칭되어야 하며, `preprocess.py`/진단 스크립트가 이를 기반으로 CSV(`*.csv`)를 생성합니다.

## 데이터 로더(DenoisingDataset)

- 입력: `clean_path`, `noisy_path` 컬럼을 가진 CSV
- 학습 모드: 길이 `segment_len`(샘플)로 랜덤 크롭/패딩 후 증강 적용
  - 증강: 동일 gain/shift를 clean/noisy에 동시 적용 + noisy에만 추가 랜덤노이즈(SNR 15~30dB)
- 평가 모드: 전체 길이를 반환(크롭/패딩/증강 없음)

## 모델(ConvTasNet) 및 손실/메트릭

- ConvTasNet 구성:
  - Encoder(Conv1d stride=win_len/2) → gLN → Bottleneck(1x1)
  - TCN 블록(잔차/스킵 분리, Depthwise Separable Conv, gLN, PReLU, Dropout)
  - Skip 합산 → 1x1 마스크 추정(sigmoid) → Decoder(ConvTranspose1d)
- 출력 길이는 입력과 동일하도록 pad/crop 보정
- 판별기(Discriminator): spectral norm + adaptive avg pool 1D, 최종 1x1
- 기본 손실
  - SI-SDR(shift-invariant 변형) + 수치안전성 보강
  - MR-STFT Loss(여러 FFT/Hop/Win 조합)
  - Log-Mel Perceptual Loss(L1 on log-mel)
  - (선택) Adversarial Loss(MSE)
- 평가 메트릭: PESQ, STOI, SI-SDR

## 학습 설정(CONFIG) 하이라이트 (`train.py`)

- 경로/CSV
  - `output_root`: 로그/체크포인트/평가 결과 저장 루트 (기본: `convtasnet_realdata_v3`)
  - `train_csv`, `val_csv`: 기본은 `diagnostics_*/normalized.csv`를 가정. 일반 CSV를 쓰려면 해당 경로로 변경
- 학습 하이퍼파라미터
  - `epochs`, `batch_size`, `learning_rate_g/d` (+ 파인튜닝용 별도 LR)
  - `segment_duration`(초) → `segment_len = sample_rate * segment_duration`
- 가속/안정화
  - `use_amp`(CUDA에서 자동 혼합정밀), `use_torch_compile`(PyTorch 2.x),
  - `steps_per_epoch`(작은 데이터셋에서 업데이트 수 제어)
- 손실 가중치/워밍업
  - `use_adv/use_stft/use_perc`, `lambda_*`, `adv_warmup_epochs`
- 체크포인트
  - `resume_checkpoint`(학습 재개), `fine_tuning_checkpoint`(사전학습 파인튜닝)
  - 저장: `output_root/checkpoints/`

## 체크포인트와 평가(`test.py`)

- 기본적으로 `output_root/checkpoints/best_model.pth` 로드
- 키 불일치 시 진단을 출력하고, `allow_partial_load=True`로 일부 가중치만 로드도 가능(권장X)
- 테스트는 배치 1, 전체 길이 단위로 실행하며, 처음 몇 개 샘플의 WAV/스펙트럼 이미지를 저장합니다.

출력 예)

```plain
==================================================
최종 평가 결과
  - 평균 PESQ  : 1.9016 / 1.4936
  - 평균 STOI  : 0.6951 / 0.6958
  - 평균 SI-SDR: -30.8265 / -9.1918 dB
    (향상점수 / 원점수: denoised vs clean / noisy vs clean)
==================================================
평가 결과 및 샘플 오디오가 'convtasnet_realdata_v3/test_results'에 저장되었습니다.
```

## 자주 묻는 질문(FAQ) 및 문제 해결

- PESQ/STOI 에러: 샘플레이트가 16kHz인지 확인하세요. `pesq()`는 모드에 따라 SR 제약이 있습니다.
- CUDA 메모리 부족: `batch_size` 축소, `segment_duration` 축소, 또는 `use_amp: True` 확인
- torchaudio 로드 에러: 파일 경로나 권한 확인, stereo→mono 변환은 자동 처리됩니다.
- 체크포인트 키 불일치: 학습/평가 시점의 모델 구조가 달라졌을 수 있습니다.
  - 동일 실험 폴더의 `best_model.pth`를 사용하거나, `allow_partial_load=True`로 우회(성능 저하 가능)
- Windows 빌드 이슈: 특정 패키지(특히 PESQ)에서 빌드 툴 요구 시 Microsoft C++ Build Tools 설치가 필요할 수 있습니다.

## 라이선스/주의

- PESQ는 ITU 표준 관련 라이선스가 있을 수 있으며, 연구/개발 외 사용 시 각 라이선스 정책을 확인하세요.
- 데이터 저작권 및 개인정보를 준수하세요.

## 참고: 주요 파일별 역할

- `preprocess.py`: 더미 데이터 생성(없을 때) + `train/validation/test` CSV(filelist) 생성
- `dataset.py`: CSV 기반 clean/noisy 로딩, 학습 시 볼륨/쉬프트/노이즈 증강
- `model.py`: ConvTasNet + Discriminator
- `utils.py`: SI-SDR 변형 손실, MR-STFT/Perceptual Loss, 메트릭 계산, 체크포인트 입출력 등
- `diagnose_dataset.py`: 파일 유효성/샘플레이트/채널/지연/정규화 등 진단 및 보정/정렬본 생성
- `train.py`: 학습 루프, AMP/compile, 체크포인트/스케줄러, 주기적 저장
- `test.py`: 최고성능 체크포인트 로드 후 전체 길이 평가 및 샘플 저장
- `dataseteval.py`: 폴더 기반 빠른 PESQ/STOI 평균
