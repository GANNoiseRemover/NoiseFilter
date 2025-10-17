import torch
import torch.nn as nn
import torchaudio
import os
import numpy as np
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import Tuple

# --- TorchMetrics (SI-SDR) ---
_HAS_TORCHMETRICS = True
try:
    # Prefer functional API for loss (avoids metric state accumulation)
    from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as _tm_si_sdr
except Exception:
    try:
        # Fallback older alias
        from torchmetrics.functional.audio import si_sdr as _tm_si_sdr
    except Exception:
        print("정보: torchmetrics의 SI-SDR 함수를 불러올 수 없습니다. 로컬 구현으로 대체합니다.")
        _HAS_TORCHMETRICS = False

def set_seed(seed):
    """시드 고정으로 재현성 확보"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def si_sdr_loss(preds, target, epsilon=1e-8):
    """수치 안전성을 강화한 SI-SDR 손실 함수
    - 0 에너지 구간에서 무한대가 발생하지 않도록 분자/분모에 epsilon 추가
    - log10 입력값을 합리적인 범위로 클램프하여 inf 방지
    """
    # (B, T) 가정
    preds = preds.float()
    target = target.float()

    dot = torch.sum(preds * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target * target, dim=-1, keepdim=True)
    alpha = dot / (target_energy + epsilon)
    target_scaled = alpha * target
    noise = target_scaled - preds

    s_target = torch.sum(target_scaled * target_scaled, dim=-1)
    s_noise = torch.sum(noise * noise, dim=-1)
    ratio = (s_target + epsilon) / (s_noise + epsilon)
    # 매우 큰 값은 학습에 악영향을 주므로 상한
    ratio = torch.clamp(ratio, min=epsilon, max=1e7)
    loss = -10.0 * torch.log10(ratio)
    # NaN/Inf 방지 – 비정상값을 0으로 대체
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.mean(loss)

def si_sdr_loss_shift_invariant(preds: torch.Tensor, target: torch.Tensor, max_shift: int = 160, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Shift-invariant SI-SDR: small integer shifts around 0 are searched, best SI-SDR is used.
    Args:
        preds: (B, T)
        target: (B, T)
        max_shift: +/- samples to search (160 ~= 10 ms @ 16kHz)
    """
    if preds.dim() == 3:
        preds = preds.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    B, T = preds.shape
    best_loss = None
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            p = preds
            t = target
        elif shift > 0:
            p = preds[:, :-shift]
            t = target[:, shift:]
        else:  # shift < 0
            p = preds[:, -shift:]
            t = target[:, : T + shift]

        if p.numel() == 0:
            continue

        alpha = (torch.sum(p * t, dim=-1, keepdim=True)) / (torch.sum(t * t, dim=-1, keepdim=True) + epsilon)
        target_scaled = alpha * t
        noise = target_scaled - p
        val = (torch.sum(target_scaled * target_scaled, dim=-1) + epsilon) / (torch.sum(noise * noise, dim=-1) + epsilon)
        loss = -10 * torch.log10(val + epsilon)
        loss = torch.mean(loss)
        if best_loss is None:
            best_loss = loss
        else:
            best_loss = torch.minimum(best_loss, loss)

    return best_loss if best_loss is not None else si_sdr_loss(preds, target, epsilon)


def si_sdr_loss_torchmetrics(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    SI-SDR loss using TorchMetrics (differentiable functional).
    Returns negative SI-SDR (mean over batch) to be minimized.
    - Supports inputs of shape (B, T) or (B, 1, T).
    - Aligns lengths and clamps to [-1, 1] for numerical stability.
    If torchmetrics is unavailable, falls back to the local implementation.
    """
    # Squeeze channel dim if provided
    if preds.dim() == 3 and preds.size(1) == 1:
        preds = preds.squeeze(1)
    if target.dim() == 3 and target.size(1) == 1:
        target = target.squeeze(1)

    # Ensure 2D (B, T)
    if preds.dim() == 1:
        preds = preds[None, :]
    if target.dim() == 1:
        target = target[None, :]

    # Length align and clamp
    min_len = min(preds.shape[-1], target.shape[-1])
    preds = torch.nan_to_num(preds[..., :min_len].float().clamp(-1.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target[..., :min_len].float().clamp(-1.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)

    if _HAS_TORCHMETRICS:
        try:
            si_sdr_vals = _tm_si_sdr(preds, target)  # shape: (B,) or scalar depending on version
            loss = -si_sdr_vals.mean()
            # guard
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            return loss
        except Exception:
            pass
    # Fallback to local stable SI-SDR if torchmetrics is not available or failed
    return si_sdr_loss(preds, target)

def _compute_single_metric(clean, denoised, sr):
    """단일 샘플에 대한 PESQ, STOI, SI-SDR 계산 (병렬 처리를 위함)"""
    try:
        # PESQ 계산 시 발생하는 UserWarning을 무시
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pesq_val = pesq(sr, clean, denoised, 'wb')
    except Exception:
        # PESQ 계산 실패 시 -1.0으로 처리 (로그에서 확인 가능)
        pesq_val = -1.0
    
    try:
        stoi_val = stoi(clean, denoised, sr, extended=False)
    except Exception:
        stoi_val = 0.0

    # SI-SDR 계산
    alpha = np.sum(denoised * clean) / (np.sum(clean**2) + 1e-8)
    target_scaled = alpha * clean
    noise = target_scaled - denoised
    si_sdr_val = 10 * np.log10((np.sum(target_scaled**2) + 1e-8) / (np.sum(noise**2) + 1e-8))
    
    return pesq_val, stoi_val, si_sdr_val

def calculate_metrics(clean_batch, denoised_batch, sr):
    """배치 단위로 메트릭을 병렬 계산
    안전성 강화를 위해 float32 변환 및 [-1,1] 클램프를 적용한 뒤 계산합니다.
    """
    # dtype/범위 보정: AMP/모델 출력의 half, 범위 초과 및 NaN/Inf에 대비
    clean_batch = clean_batch.detach().float().clamp(-1.0, 1.0)
    denoised_batch = denoised_batch.detach().float().clamp(-1.0, 1.0)
    clean_batch = torch.nan_to_num(clean_batch, nan=0.0, posinf=0.0, neginf=0.0)
    denoised_batch = torch.nan_to_num(denoised_batch, nan=0.0, posinf=0.0, neginf=0.0)

    clean_batch_np = clean_batch.cpu().numpy().astype(np.float32, copy=False)
    denoised_batch_np = denoised_batch.cpu().numpy().astype(np.float32, copy=False)
    
    total_pesq, total_stoi, total_si_sdr = 0, 0, 0
    batch_size = clean_batch_np.shape[0]

    with ThreadPoolExecutor() as executor:
        # 각 샘플을 별도의 스레드에서 계산
        futures = [executor.submit(_compute_single_metric, c.squeeze(), d.squeeze(), sr) for c, d in zip(clean_batch_np, denoised_batch_np)]
        results = [f.result() for f in futures]

    for pesq_val, stoi_val, si_sdr_val in results:
        total_pesq += pesq_val
        total_stoi += stoi_val
        total_si_sdr += si_sdr_val
        
    return total_pesq / batch_size, total_stoi / batch_size, total_si_sdr / batch_size

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """체크포인트 저장"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filepath)

def load_checkpoint(checkpoint_path, model_g, model_d=None, optimizer_g=None, optimizer_d=None):
    """체크포인트 로드
    - torch.compile로 저장되어 '_orig_mod.' 접두사가 붙은 state_dict도 자동 대응
    - 로드 실패 시 strict=False로 부분 로드를 시도
    반환: (start_epoch, best_val_loss)
    """
    if not os.path.isfile(checkpoint_path):
        print(f"=> 체크포인트가 없습니다: '{checkpoint_path}'")
        return 0, float('inf')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    def _load_with_fallback(module: torch.nn.Module, state_dict: dict, name: str):
        try:
            module.load_state_dict(state_dict, strict=True)
            return True
        except Exception:
            # torch.compile로 저장된 경우 키에 '_orig_mod.' 접두사가 붙을 수 있음
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                remapped = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                try:
                    module.load_state_dict(remapped, strict=True)
                    print(f"[정보] {name}: '_orig_mod.' 접두사를 제거하여 로드했습니다.")
                    return True
                except Exception:
                    module.load_state_dict(remapped, strict=False)
                    print(f"[경고] {name}: strict=False로 부분 로드했습니다 (성능 저하 가능).")
                    return False
            else:
                module.load_state_dict(state_dict, strict=False)
                print(f"[경고] {name}: strict=False로 부분 로드했습니다 (성능 저하 가능).")
                return False

    # Generator
    g_ok = False
    if 'model_g_state_dict' in checkpoint:
        g_ok = _load_with_fallback(model_g, checkpoint['model_g_state_dict'], 'Generator')
    else:
        print("[경고] 체크포인트에 'model_g_state_dict'가 없습니다.")

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Discriminator (옵션)
    if model_d and 'model_d_state_dict' in checkpoint:
        _load_with_fallback(model_d, checkpoint['model_d_state_dict'], 'Discriminator')
    if optimizer_g and 'optimizer_g_state_dict' in checkpoint:
        try:
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        except Exception:
            print("[경고] Generator 옵티마이저 상태 로드 실패, 새로 초기화합니다.")
    if optimizer_d and 'optimizer_d_state_dict' in checkpoint:
        try:
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        except Exception:
            print("[경고] Discriminator 옵티마이저 상태 로드 실패, 새로 초기화합니다.")
    
    print(f"'{checkpoint_path}'에서 체크포인트 로드 완료. Epoch {start_epoch}.")
    return start_epoch, best_val_loss

def save_sample_audios(output_dir, filename_base, clean, noisy, denoised, sr):
    """평가 중 샘플 오디오 3종(clean, noisy, denoised)을 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    denoised_fp32 = denoised.to(torch.float32)

    tensors_to_save = {
        "01_clean.wav": clean.cpu(),
        "02_noisy.wav": noisy.cpu(),
        "03_denoised.wav": denoised_fp32.cpu()
    }
    
    for filename, tensor in tensors_to_save.items():
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(os.path.join(output_dir, f"{filename_base}_{filename}"), tensor, sr)


def save_spectrogram_images(output_dir, filename_base, clean, noisy, denoised, sr):
    """스펙트로그램 비교 이미지(clean, noisy, denoised)를 저장"""
    os.makedirs(output_dir, exist_ok=True)

    denoised_fp32 = denoised.to(torch.float32)
    
    if clean.ndim == 1: clean = clean.unsqueeze(0)
    if noisy.ndim == 1: noisy = noisy.unsqueeze(0)
    if denoised_fp32.ndim == 1: denoised_fp32 = denoised_fp32.unsqueeze(0)
    
    # [수정] n_fft 값을 명시적으로 설정하여 경고를 해결하고 더 상세한 스펙트로그램을 생성합니다.
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024)
    
    spec_clean = transform(clean.cpu())
    spec_noisy = transform(noisy.cpu())
    spec_denoised = transform(denoised_fp32.cpu())

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f'Spectrograms for {filename_base}', fontsize=16)

    for ax, spec, title in zip(axes, [spec_clean, spec_noisy, spec_denoised], ['Clean', 'Noisy', 'Denoised']):
        im = ax.imshow(spec.log2().squeeze(0).numpy(), aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel("Time Frame")
        ax.set_ylabel("Mel Frequency Bin")
        fig.colorbar(im, ax=ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{filename_base}_spectrogram_comparison.png"))
    plt.close(fig)



# --- 멀티해상도 STFT 손실 ---
class MRSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=(512, 1024, 2048),
        hop_sizes=(128, 256, 512),
        win_lengths=(512, 1024, 2048),
        eps: float = 1e-7,
        sr: int = 16000,
        freq_weighting: str | None = None,  # None | 'hf'
        hf_alpha: float = 1.5,             # ramp exponent for high-freq emphasis
        hf_cutoff_hz: float = 3500.0,      # start boosting above this frequency
        hf_boost: float = 1.5,             # overall scale of high-freq weight
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.eps = eps
        self.sr = sr
        self.freq_weighting = freq_weighting
        self.hf_alpha = hf_alpha
        self.hf_cutoff_hz = hf_cutoff_hz
        self.hf_boost = hf_boost

    def stft_mag(self, x, n_fft, hop_length, win_length):
        # x: (B, T)
        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
        X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                       window=window, center=True, pad_mode='reflect', return_complex=True)
        mag = torch.abs(X)
        return mag

    def forward(self, pred, target):
        # pred/target: (B, T), [-1,1], float32 권장
        total_sc = 0.0
        total_mag = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            P = self.stft_mag(pred, n_fft, hop, win)
            T = self.stft_mag(target, n_fft, hop, win)
            # Spectral convergence
            sc = torch.linalg.norm(T - P) / (torch.linalg.norm(T) + self.eps)
            # Log-magnitude loss
            log_diff = torch.abs(torch.log(T + self.eps) - torch.log(P + self.eps))
            if self.freq_weighting == 'hf':
                # emphasize high-frequency bins using a simple ramp above cutoff
                F = T.size(1)
                # cutoff bin based on Nyquist
                nyq = self.sr / 2.0
                cutoff_bin = int(min(F - 1, max(0, round(self.hf_cutoff_hz / nyq * (F - 1)))))
                # ramp from 0..1 across bins
                ramp = torch.linspace(0.0, 1.0, F, device=T.device, dtype=T.dtype).pow(self.hf_alpha)
                # boost only above cutoff
                mask = torch.zeros_like(ramp)
                if cutoff_bin < F:
                    mask[cutoff_bin:] = 1.0
                weight = 1.0 + self.hf_boost * (ramp * mask)
                # normalize average weight ~1 to not change global scale much
                weight = weight * (F / weight.sum())
                # apply per-frequency weight
                log_diff = log_diff * weight.view(1, F, 1)
            mag = torch.mean(log_diff)
            total_sc = total_sc + sc
            total_mag = total_mag + mag
        return total_sc + total_mag


class PreEmphasisLoss(nn.Module):
    """
    간단한 프리엠퍼시스(고주파 강조) 필터 도메인에서 L1 손실을 계산해
    치찰/파열음과 같은 고주파 트랜지언트 보존을 돕습니다.
    """
    def __init__(self, coeff: float = 0.97):
        super().__init__()
        self.coeff = coeff

    def preemph(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        x = x.float()
        # y[n] = x[n] - a * x[n-1]
        y = x.clone()
        y[:, 1:] = x[:, 1:] - self.coeff * x[:, :-1]
        y[:, :1] = x[:, :1]
        return y

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len].float()
        target = target[..., :min_len].float()
        yp = self.preemph(pred)
        yt = self.preemph(target)
        return torch.mean(torch.abs(yp - yt))


# --- Perceptual Loss (Log-Mel L1) ---
class LogMelPerceptualLoss(nn.Module):
    """
    로그-멜 스펙트로그램 공간에서 L1 손실을 계산하는 간단한 퍼셉추얼 로스.
    가벼우면서도 청감 관련 특성을 어느 정도 반영합니다.
    """
    def __init__(self, sr: int = 16000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 80, eps: float = 1e-6):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (B, T)
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        pred = pred.float().clamp(-1, 1)
        target = target.float().clamp(-1, 1)

        # 길이 정렬
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        # 모듈 초기화 시점의 장치와 다를 수 있어, 여기서 to(device)를 호출하되 상태를 덮어쓰지 않습니다.
        mel = self.mel.to(pred.device)

        M_pred = mel(pred)
        M_tgt = mel(target)
        logM_pred = torch.log(M_pred + self.eps)
        logM_tgt = torch.log(M_tgt + self.eps)
        return torch.mean(torch.abs(logM_pred - logM_tgt))


def smooth_labels(tensor: torch.Tensor, low: float = 0.9, high: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GAN 라벨 스무딩 유틸.
    - real_target: 1 대신 low (예: 0.9)
    - fake_target: 0 대신 high (예: 0.0)
    출력 텐서는 (B, 1, 1) 형태로 맞춰 반환합니다.
    """
    b = tensor.size(0)
    device = tensor.device
    real = torch.full((b, 1, 1), low, device=device)
    fake = torch.full((b, 1, 1), high, device=device)
    return real, fake
