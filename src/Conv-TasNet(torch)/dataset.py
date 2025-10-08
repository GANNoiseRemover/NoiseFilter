import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class DenoisingDataset(Dataset):
    """
    잡음 제거 모델 학습을 위한 데이터셋 클래스.
    CSV 파일 목록을 읽어 clean/noisy 오디오 쌍을 로드합니다.
    오디오 파일의 길이가 다양할 경우, 지정된 길이로 랜덤하게 잘라내거나 패딩합니다.
    """
    def __init__(self, filelist_path, segment_len, sample_rate, for_evaluation=False):
        """
        Args:
            filelist_path (str): 'clean_path', 'noisy_path' 컬럼을 포함한 CSV 파일 경로
            segment_len (int): 오디오를 잘라낼 고정 길이 (샘플 수 기준)
            sample_rate (int): 오디오 샘플링 레이트
            for_evaluation (bool): 평가 모드일 경우 True. True이면 오디오를 자르지 않고 전체를 반환.
        """
        self.filelist = pd.read_csv(filelist_path)
        self.segment_len = segment_len
        self.sample_rate = sample_rate
        self.for_evaluation = for_evaluation
        # 증강 파이프라인 설정 (학습 시에만 적용)
        self.enable_augmentation = not for_evaluation
        # 볼륨/쉬프트는 noisy/clean에 동일하게, 노이즈는 noisy에만 적용
        # 증강 함수는 내부에서 파라미터를 받아 처리하도록 변경


    def random_volume(self, wav, gain=None):
        # 0.8~1.2배 랜덤 볼륨
        if gain is None:
            gain = torch.empty(1).uniform_(0.8, 1.2).item()
        return wav * gain

    def random_noise(self, wav, snr_db=None):
        # SNR 15~30dB 랜덤 노이즈 추가
        if snr_db is None:
            snr_db = torch.empty(1).uniform_(15, 30).item()
        rms = wav.pow(2).mean().sqrt()
        noise_std = rms / (10 ** (snr_db / 20))
        noise = torch.randn_like(wav) * noise_std
        return wav + noise

    def random_shift(self, wav, shift=None):
        # 최대 0.2초(3200샘플) 내에서 랜덤 시간 쉬프트
        max_shift = int(0.2 * self.sample_rate)
        if shift is None:
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift == 0:
            return wav
        elif shift > 0:
            return torch.nn.functional.pad(wav, (shift, 0))[: wav.shape[-1]]
        else:
            return torch.nn.functional.pad(wav, (0, -shift))[-shift:]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        row = self.filelist.iloc[idx]
        clean_path = row['clean_path']
        noisy_path = row['noisy_path']

        # 오디오 파일 로드, (채널, 시간) 형태
        try:
            clean_wav, sr_clean = torchaudio.load(clean_path)
            noisy_wav, sr_noisy = torchaudio.load(noisy_path)
        except Exception as e:
            print(f"오디오 파일 로드 에러 (인덱스 {idx}): {clean_path}")
            # 문제가 발생하면 0으로 채워진 더미 데이터를 반환
            return torch.zeros(self.segment_len), torch.zeros(self.segment_len)

        # 샘플링 레이트 확인 (필요 시 재조정)
        if sr_clean != self.sample_rate:
            clean_wav = torchaudio.transforms.Resample(sr_clean, self.sample_rate)(clean_wav)
        if sr_noisy != self.sample_rate:
            noisy_wav = torchaudio.transforms.Resample(sr_noisy, self.sample_rate)(noisy_wav)

        # 단일 채널(모노)로 변환
        if clean_wav.shape[0] > 1:
            clean_wav = torch.mean(clean_wav, dim=0, keepdim=True)
        if noisy_wav.shape[0] > 1:
            noisy_wav = torch.mean(noisy_wav, dim=0, keepdim=True)

        # 평가 모드에서는 전체 오디오 반환 (길이 맞추기 X)
        if self.for_evaluation:
            return noisy_wav.squeeze(0), clean_wav.squeeze(0)

        # 학습/검증 모드에서는 고정된 길이로 자르거나 패딩
        current_len = noisy_wav.shape[1]
        if current_len > self.segment_len:
            # 길이가 길면 랜덤하게 자르기
            start = torch.randint(0, current_len - self.segment_len + 1, (1,)).item()
            end = start + self.segment_len
            noisy_wav = noisy_wav[:, start:end]
            clean_wav = clean_wav[:, start:end]
        elif current_len < self.segment_len:
            # 길이가 짧으면 0으로 패딩
            padding_len = self.segment_len - current_len
            noisy_wav = torch.nn.functional.pad(noisy_wav, (0, padding_len))
            clean_wav = torch.nn.functional.pad(clean_wav, (0, padding_len))

        # --- 증강 파이프라인 적용 (학습 시에만) ---
        if self.enable_augmentation:
            noisy_wav = noisy_wav.squeeze(0)
            clean_wav = clean_wav.squeeze(0)
            # 동일한 랜덤 볼륨/쉬프트 파라미터 생성
            gain = torch.empty(1).uniform_(0.8, 1.2).item()
            max_shift = int(0.2 * self.sample_rate)
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            # 볼륨/쉬프트 증강 동시 적용
            noisy_wav = self.random_volume(noisy_wav, gain)
            clean_wav = self.random_volume(clean_wav, gain)
            noisy_wav = self.random_shift(noisy_wav, shift)
            clean_wav = self.random_shift(clean_wav, shift)
            # 노이즈 증강은 noisy에만 적용
            noisy_wav = self.random_noise(noisy_wav)
            return noisy_wav, clean_wav
        else:
            return noisy_wav.squeeze(0), clean_wav.squeeze(0)

