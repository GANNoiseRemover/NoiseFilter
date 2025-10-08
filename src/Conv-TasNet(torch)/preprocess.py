import os
import glob
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
import numpy as np

# --- 설정 ---
# 실제 데이터셋이 위치한 루트 디렉토리를 지정하세요.
# 예: './speech_data'
DATASET_ROOT = 'dataset' 
# 생성된 CSV 파일이 저장될 위치를 지정하세요.
OUTPUT_DIR = DATASET_ROOT 
SAMPLE_RATE = 16000
NOISE_LEVEL = 0.5


def create_dummy_data_structured(root_dir, num_files_per_split):
    """시연을 위해 train/validation/test 구조를 가진 더미 데이터셋을 생성합니다."""
    print("="*50)
    print(f"시연용 더미 데이터셋 생성을 시작합니다...")
    print(f"생성 위치: {os.path.abspath(root_dir)}")
    print("="*50)

    for split in ['train', 'validation', 'test']:
        clean_dir = os.path.join(root_dir, split, 'clean')
        noisy_dir = os.path.join(root_dir, split, 'noisy')
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(noisy_dir, exist_ok=True)

        num_files = num_files_per_split[split]

        for i in tqdm(range(num_files), desc=f"'{split}' 데이터 생성 중"):
            # 2~5초 사이의 다양한 길이로 오디오 생성
            duration = np.random.uniform(2.0, 5.0)
            num_samples = int(SAMPLE_RATE * duration)

            time = torch.linspace(0., duration, num_samples)
            clean_wav = (torch.sin(2 * torch.pi * 220 * time) + 0.5 * torch.sin(2 * torch.pi * 880 * time))
            clean_wav = clean_wav.unsqueeze(0) * 0.5

            noise = torch.randn_like(clean_wav) * NOISE_LEVEL
            noisy_wav = clean_wav + noise
            noisy_wav = torch.clamp(noisy_wav, -1, 1)

            filename = f'{i:04d}.wav'
            torchaudio.save(os.path.join(clean_dir, filename), clean_wav, SAMPLE_RATE)
            torchaudio.save(os.path.join(noisy_dir, filename), noisy_wav, SAMPLE_RATE)

    print("\n더미 데이터셋 생성이 완료되었습니다.")

def generate_filelists(root_dir, output_dir):
    """
    데이터셋 구조를 스캔하여 train, validation, test용 CSV 파일 목록을 생성합니다.
    """
    print("\n" + "="*50)
    print("파일 목록 CSV 생성을 시작합니다...")
    print(f"데이터셋 루트: {root_dir}")
    print(f"CSV 저장 위치: {output_dir}")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'validation', 'test']:
        clean_glob_path = os.path.join(root_dir, split, 'clean', '*.wav')
        # glob.glob은 파일 목록을 반환합니다.
        clean_files = sorted(glob.glob(clean_glob_path))

        if not clean_files:
            print(f"경고: '{split}' 스플릿에서 .wav 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue

        filelist = []
        for clean_path in tqdm(clean_files, desc=f"'{split}' 목록 생성 중"):
            filename = os.path.basename(clean_path)
            noisy_path = os.path.join(root_dir, split, 'noisy', filename)

            if os.path.exists(noisy_path):
                # 어떤 위치에서든 스크립트를 실행할 수 있도록 절대 경로로 저장합니다.
                filelist.append({
                    'clean_path': os.path.abspath(clean_path),
                    'noisy_path': os.path.abspath(noisy_path)
                })
            else:
                print(f"경고: '{clean_path}'에 해당하는 noisy 파일을 찾을 수 없습니다. ({noisy_path})")

        if not filelist:
            print(f"'{split}' 스플릿에 대한 파일 쌍을 찾지 못했습니다.")
            continue
            
        df = pd.DataFrame(filelist)
        output_csv_path = os.path.join(output_dir, f'{split}.csv')
        df.to_csv(output_csv_path, index=False)
        print(f"-> '{output_csv_path}' 생성 완료 ({len(df)}개 파일).")

if __name__ == '__main__':
    # 만약 'speech_data' 디렉토리가 없다면, 시연용 더미 데이터를 생성합니다.
    # 실제 데이터셋이 있다면 이 부분은 실행되지 않습니다.
    if not os.path.exists(DATASET_ROOT):
        num_files = {'train': 100, 'validation': 20, 'test': 20}
        create_dummy_data_structured(DATASET_ROOT, num_files)
    
    # 데이터셋 디렉토리를 스캔하여 파일 목록(CSV)을 생성합니다.
    generate_filelists(DATASET_ROOT, OUTPUT_DIR)

