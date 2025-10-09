import os
import glob
import torch
import torchaudio
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal # create_noise_profile.py에서 사용되지는 않지만, 혹시 몰라 포함
import random
from tqdm import tqdm

# ==============================================================================
# --- 설정 (CONFIG) ---
# 사용자의 환경에 맞게 이 부분을 수정하세요.
# ==============================================================================
CONFIG = {
    # --- 기본 경로 ---
    "base_dataset_dir": "dataset_korean_pretrain", # 정리된 데이터셋 폴더 경로
    "output_csv_dir": "dataset_csvs", # 생성될 CSV 파일들이 저장될 폴더
    
    # --- 처리할 데이터 분할 ---
    # 필요한 것만 남기고 주석 처리하거나 지울 수 있습니다.
    "splits": ["train", "validation", "test"],
    
    # --- 노이즈 프로파일 ---
    # create_noise_profile.py로 생성한 '지문' 파일 경로
    "noise_profile_path": "noise_psd_profile.npy",
    
    # --- 오디오 및 노이즈 생성 파라미터 ---
    "target_sample_rate": 16000,
    "snr_range_db": (5, 15), # 믹싱 시 사용할 SNR(dB) 범위 (낮을수록 노이즈가 큼)
    "audio_format": ".wav", # clean 파일의 확장자
}
# ==============================================================================

def generate_custom_noise(shape, noise_profile, sample_rate, device=None):
    """
    주어진 노이즈 프로파일(PSD)을 기반으로 커스텀 노이즈를 생성합니다.
    """
    length = shape[-1]
    profile_freqs, profile_psd = noise_profile[0], noise_profile[1]

    psd_interpolator = interp1d(profile_freqs, profile_psd, bounds_error=False, fill_value="extrapolate")
    noise_fft_freqs = np.fft.rfftfreq(length, d=1./sample_rate)
    target_psd = psd_interpolator(noise_fft_freqs)
    target_psd[target_psd < 0] = 0

    scaling_filter = torch.from_numpy(np.sqrt(target_psd)).to(torch.float32)
    white_noise = torch.randn(*shape[:-1], length // 2 + 1, dtype=torch.complex64)
    custom_noise_freq = white_noise * scaling_filter
    custom_noise = torch.fft.irfft(custom_noise_freq, n=length)
    
    # 파워 정규화
    if torch.std(custom_noise) > 1e-6:
        custom_noise *= (torch.std(torch.randn(shape)) / torch.std(custom_noise))
    
    return custom_noise.to(device)

def process_split(config, split, noise_profile):
    """
    하나의 데이터 분할(train, validation, test)에 대한 노이즈 합성을 처리합니다.
    """
    base_dir = config["base_dataset_dir"]
    clean_dir = os.path.join(base_dir, split, "clean")
    noisy_dir = os.path.join(base_dir, split, "noisy")
    
    print(f"\n[{split.upper()}] 분할 처리 시작...")
    print(f"Clean 오디오 경로: {clean_dir}")
    
    if not os.path.isdir(clean_dir):
        print(f"경고: '{clean_dir}' 폴더를 찾을 수 없습니다. 이 분할은 건너뜁니다.")
        return

    # 출력 폴더 생성
    os.makedirs(noisy_dir, exist_ok=True)
    print(f"Noisy 오디오 저장 경로: {noisy_dir}")

    # clean 파일 목록 가져오기
    clean_files = glob.glob(os.path.join(clean_dir, f"*{config['audio_format']}"))
    if not clean_files:
        print(f"경고: '{clean_dir}' 폴더에서 오디오 파일을 찾을 수 없습니다.")
        return
        
    print(f"총 {len(clean_files)}개의 파일을 처리합니다.")
    
    file_mappings = []

    for clean_path in tqdm(clean_files, desc=f"Processing {split}"):
        try:
            # 1. Clean 오디오 로드 및 전처리
            clean_wav, sr = torchaudio.load(clean_path)
            if sr != config["target_sample_rate"]:
                resampler = torchaudio.transforms.Resample(sr, config["target_sample_rate"])
                clean_wav = resampler(clean_wav)
            if clean_wav.shape[0] > 1:
                clean_wav = torch.mean(clean_wav, dim=0, keepdim=True)

            # 2. 커스텀 노이즈 생성
            noise = generate_custom_noise(clean_wav.shape, noise_profile, config["target_sample_rate"])

            # 3. 랜덤 SNR로 믹싱
            snr_db = random.uniform(*config["snr_range_db"])
            clean_power = torch.mean(clean_wav.pow(2))
            noise_power = torch.mean(noise.pow(2))

            if clean_power < 1e-8 or noise_power < 1e-8:
                # 소리가 거의 없는 파일은 노이즈를 섞지 않고 원본을 복사
                noisy_wav = clean_wav
            else:
                snr_linear = 10**(snr_db / 10)
                scale = (clean_power / (snr_linear * noise_power)).sqrt()
                noisy_wav = clean_wav + noise * scale
            
            noisy_wav = torch.clamp(noisy_wav, -1.0, 1.0)

            # 4. Noisy 오디오 파일 저장
            filename = os.path.basename(clean_path)
            noisy_path = os.path.join(noisy_dir, filename)
            torchaudio.save(noisy_path, noisy_wav, config["target_sample_rate"])

            # 5. CSV 매핑 정보 추가
            file_mappings.append({
                "clean_path": os.path.abspath(clean_path),
                "noisy_path": os.path.abspath(noisy_path)
            })

        except Exception as e:
            print(f"\n오류 발생: '{clean_path}' 처리 중 문제 발생. 건너뜁니다. ({e})")
    
    # # 6. CSV 파일 저장
    # if file_mappings:
    #     df = pd.DataFrame(file_mappings)
    #     csv_output_dir = config["output_csv_dir"]
    #     os.makedirs(csv_output_dir, exist_ok=True)
    #     csv_path = os.path.join(csv_output_dir, f"{split}_pairs.csv")
    #     df.to_csv(csv_path, index=False)
    #     print(f"'{split}' 분할에 대한 CSV 파일이 '{csv_path}'에 저장되었습니다.")

def main():
    # 노이즈 프로파일 로드
    try:
        noise_profile = np.load(CONFIG["noise_profile_path"])
        print(f"'{CONFIG['noise_profile_path']}'에서 노이즈 프로파일을 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"치명적 오류: 노이즈 프로파일 파일 '{CONFIG['noise_profile_path']}'을 찾을 수 없습니다.")
        print("먼저 'create_noise_profile.py'를 실행하여 프로파일을 생성해주세요.")
        return

    # 설정된 각 분할에 대해 처리 실행
    for split in CONFIG["splits"]:
        process_split(CONFIG, split, noise_profile)
    
    print("\n모든 작업이 완료되었습니다.")

if __name__ == '__main__':
    main()