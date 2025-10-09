import torch
import torchaudio
import pandas as pd
import numpy as np
from scipy import signal
from tqdm import tqdm

# --- 설정 ---
# 실제 (clean, noisy) 쌍이 있는 평가용/검증용 CSV 파일
EVAL_CSV_PATH = "diagnostics_train/normalized.csv" # <--- 실제 파일 경로로 수정하세요
# 생성될 노이즈 프로파일(지문) 파일 이름
OUTPUT_PROFILE_PATH = "noise_psd_profile.npy" 
SAMPLE_RATE = 16000
N_FFT = 4096 # PSD 계산을 위한 FFT 사이즈 (값이 클수록 주파수 해상도가 높아짐)

def main():
    print("실제 노이즈 샘플로부터 노이즈 프로파일 생성을 시작합니다...")
    
    filelist = pd.read_csv(EVAL_CSV_PATH)
    
    all_psds = []

    for idx in tqdm(range(len(filelist))):
        row = filelist.iloc[idx]
        clean_path = row['clean_path']
        noisy_path = row['noisy_path']

        try:
            clean_wav, sr = torchaudio.load(clean_path)
            noisy_wav, _ = torchaudio.load(noisy_path)
        except Exception as e:
            print(f"파일 로드 에러 (인덱스 {idx}): {e}")
            continue

        # 전처리
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            clean_wav = resampler(clean_wav)
            noisy_wav = resampler(noisy_wav)
        
        if clean_wav.shape[0] > 1: clean_wav = torch.mean(clean_wav, dim=0, keepdim=True)
        if noisy_wav.shape[0] > 1: noisy_wav = torch.mean(noisy_wav, dim=0, keepdim=True)
            
        min_len = min(clean_wav.shape[1], noisy_wav.shape[1])
        clean_wav = clean_wav[:, :min_len]
        noisy_wav = noisy_wav[:, :min_len]

        # 노이즈 추출
        residual_noise = (noisy_wav - clean_wav).squeeze().numpy()
        
        # PSD 계산
        freqs, psd = signal.welch(residual_noise, fs=SAMPLE_RATE, nperseg=N_FFT)
        all_psds.append(psd)

    if not all_psds:
        print("오류: PSD를 계산할 수 있는 파일이 없습니다.")
        return

    # 모든 노이즈 샘플의 PSD를 평균내어 최종 프로파일 생성
    mean_psd = np.mean(all_psds, axis=0)

    # (선택적이지만 권장) PSD를 약간 스무딩하여 일반화 성능 높이기
    # Savitzky-Golay 필터를 사용하여 부드럽게 만듭니다.
    # window_length는 홀수, polyorder는 window_length보다 작게 설정
    smoothed_psd = signal.savgol_filter(mean_psd, window_length=51, polyorder=3)
    # 음수 값이 나오지 않도록 클리핑
    smoothed_psd[smoothed_psd < 0] = 0

    # 주파수 축과 스무딩된 PSD 값을 함께 저장
    noise_profile = np.array([freqs, smoothed_psd])
    
    np.save(OUTPUT_PROFILE_PATH, noise_profile)
    
    print(f"\n노이즈 프로파일 생성이 완료되었습니다. '{OUTPUT_PROFILE_PATH}' 파일에 저장되었습니다.")
    
    # 결과 시각화
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, mean_psd, label='Mean PSD (Raw)')
    plt.semilogy(freqs, smoothed_psd, label='Smoothed PSD (Final Profile)', linewidth=2)
    plt.title("Generated Noise Profile")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()