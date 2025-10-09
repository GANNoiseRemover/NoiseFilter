"""
한국어 WAV 데이터셋을 학습용 폴더 구조로 정리
korean_dataset_wav/ → dataset_korean_pretrain/
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ============================================
# 설정
# ============================================

INPUT_DIR = './korean_dataset_wav'  # 변환된 WAV 파일
OUTPUT_DIR = './dataset_korean_pretrain'  # 학습용 폴더

VALIDATION_SPLIT = 0.1  # 10%를 validation으로
TEST_SPLIT = 0.1       # 10%를 test로

# ============================================
# 메인 함수
# ============================================

def organize_korean_dataset():
    """
    한국어 데이터셋 정리
    - train/clean 90%
    - validation/clean 10%
    - test/clean 10%
    - noisy는 학습 시 자동 생성
    """
    print("=" * 60)
    print("한국어 데이터셋 정리")
    print("=" * 60)
    print()
    
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    if not input_path.exists():
        print(f"❌ 입력 폴더가 없습니다: {INPUT_DIR}")
        return
    
    # 모든 WAV 파일 수집
    print("📂 WAV 파일 수집 중...")
    wav_files = []
    
    # 하위 폴더 탐색
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            wav_files.extend(list(subdir.glob('*.wav')))
    
    # 직접 파일도 확인
    wav_files.extend(list(input_path.glob('*.wav')))
    
    if not wav_files:
        print(f"❌ WAV 파일을 찾을 수 없습니다: {INPUT_DIR}")
        return
    
    print(f"✅ 총 {len(wav_files)}개 파일 발견")
    print()
    
    # 랜덤 섞기
    random.shuffle(wav_files)

    # Train / Validation / Test 분할
    val_count = int(len(wav_files) * VALIDATION_SPLIT)
    test_count = int(len(wav_files) * TEST_SPLIT)
    train_count = len(wav_files) - val_count - test_count

    val_files = wav_files[:val_count]
    test_files = wav_files[val_count:val_count+test_count]
    train_files = wav_files[val_count+test_count:]

    print(f"📊 데이터 분할:")
    print(f"   Train: {len(train_files)}개")
    print(f"   Validation: {len(val_files)}개")
    print(f"   Test: {len(test_files)}개")
    print()

    # 출력 폴더 생성
    train_clean_dir = output_path / 'train' / 'clean'
    val_clean_dir = output_path / 'validation' / 'clean'
    test_clean_dir = output_path / 'test' / 'clean'

    train_clean_dir.mkdir(parents=True, exist_ok=True)
    val_clean_dir.mkdir(parents=True, exist_ok=True)
    test_clean_dir.mkdir(parents=True, exist_ok=True)

    # Train 복사
    print("📦 Train 데이터 복사 중...")
    for i, file in enumerate(tqdm(train_files, desc="Train")):
        new_name = f"korean_{i:06d}.wav"
        shutil.copy2(file, train_clean_dir / new_name)

    # Validation 복사
    print("📦 Validation 데이터 복사 중...")
    for i, file in enumerate(tqdm(val_files, desc="Validation")):
        new_name = f"korean_{i:06d}.wav"
        shutil.copy2(file, val_clean_dir / new_name)

    # Test 복사
    print("📦 Test 데이터 복사 중...")
    for i, file in enumerate(tqdm(test_files, desc="Test")):
        new_name = f"korean_{i:06d}.wav"
        shutil.copy2(file, test_clean_dir / new_name)

    print()
    print("=" * 60)
    print("✅ 완료!")
    print("=" * 60)
    print(f"📁 출력 폴더: {OUTPUT_DIR}")
    print()
    print("📊 최종 구조:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── train/")
    print(f"   │   └── clean/  ({len(train_files)}개)")
    print(f"   ├── validation/")
    print(f"   │   └── clean/  ({len(val_files)}개)")
    print(f"   └── test/")
    print(f"       └── clean/  ({len(test_files)}개)")
    print()

if __name__ == "__main__":
    organize_korean_dataset()
