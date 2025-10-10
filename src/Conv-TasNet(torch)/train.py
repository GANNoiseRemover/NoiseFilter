import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

# --- 프로젝트 파일 임포트 ---
from model import ConvTasNet, Discriminator
from dataset import DenoisingDataset
from utils import (set_seed, calculate_metrics, save_checkpoint, 
                   load_checkpoint, save_sample_audios, save_spectrogram_images, MRSTFTLoss,
                   LogMelPerceptualLoss, smooth_labels, si_sdr_loss_torchmetrics)
# preprocess.py는 직접 임포트하지 않고, 스크립트로 별도 실행합니다.

# ==============================================================================
# --- 중앙 설정 (CONFIG) ---
# 모든 하이퍼파라미터와 경로를 여기서 관리합니다.
# ==============================================================================
CONFIG = {
    # --- 경로 설정 ---
    "output_root": "convtasnet_baseline",
    # preprocess.py가 생성한 CSV 파일들이 있는 디렉토리
    "dataset_dir": "dataset", 
    "train_csv": "diagnostics_train/normalized.csv",         # 직접 지정 시 사용. 예: "diagnostics_train/normalized.csv"
    "val_csv": "diagnostics_val/normalized.csv",           # 직접 지정 시 사용. 예: "diagnostics_val/normalized.csv"
    "resume_checkpoint": "",  # 예: "training_output/checkpoints/checkpoint_epoch_10.pth"
    "fine_tuning_checkpoint": "", # 예: "path/to/pretrained_model.pth"
    # --- 파인튜닝 전용 옵션 ---
    "is_finetune": False,             # 파인튜닝 모드 활성화
    "freeze_encoder_epochs": 1,       # 초반 N 에포크 encoder/bottleneck 동결
    "disable_d_epochs": 4,            # 초반 N 에포크 Discriminator/Adv 비활성화
    "disable_perc_epochs": 2,         # 초반 N 에포크 Perceptual loss 비활성화(작은 데이터 안정화)

    # --- 학습 하이퍼파라미터 ---
    "seed": 42,
    "epochs": 100,
    "batch_size": 8,
    "learning_rate_g": 5e-5,
    "learning_rate_d": 2e-4,
    # 파인튜닝 권장 학습률 (is_finetune=True일 때 아래 값을 사용)
    "learning_rate_g_ft": 1e-5,
    "learning_rate_d_ft": 1e-4,

    # --- 성능 가속화 설정 ---
    "use_amp": True,  # Automatic Mixed Precision (AMP) 사용 여부. GPU 사용 시 속도 향상.
    "use_torch_compile": True, # PyTorch 2.0+의 torch.compile() 사용 여부. 호환성에 따라 속도 향상 또는 저하 가능.
    "detect_anomaly": False,  # autograd anomaly detection (디버깅 시 True 권장)

    # --- 데이터로더 설정 ---
    "num_workers": 4, # 사용 가능 CPU 코어에 맞춰 조정
    "pin_memory": True,
    # 작은 데이터셋 파인튜닝 시, 에포크당 업데이트 수를 늘리고 싶다면 설정
    # 0이면 비활성화, N>0이면 에포크마다 N 스텝이 되도록 샘플러가 중복 추출합니다.
    "steps_per_epoch": 400,

    # --- 오디오 속성 ---
    "sample_rate": 16000,
    "segment_duration": 2, # 초 단위, 학습 시 사용할 오디오 조각 길이

    # --- 모델 구조 ---
    "model_params": {
        "enc_dim": 512,
        "win_len": 16,
        "num_spk": 1,
        "num_layers": 3,
        "num_blocks": 8,
        "conv_channels": 512,
        "kernel_size": 3,
    },

    # --- 손실 함수 가중치 ---
    "use_adv": True,     # GAN 손실 사용 여부
    "use_stft": True,    # MR-STFT 손실 사용 여부
    "use_perc": True,    # Perceptual(Log-Mel) 손실 사용 여부

    "lambda_adv": 0.005,  # GAN 손실 가중치
    "adv_warmup_epochs": 8, # N 에포크까지 0.0, 이후 자동으로 켜짐
    "lambda_sdr": 2.0,   # SI-SDR loss 가중치
    "lambda_stft": 1.0,  # MR-STFT loss 가중치
    "lambda_perc": 0.8,  # Perceptual (Log-Mel) loss 가중치

    # --- 스케줄러 ---
    "lr_scheduler": {
        "type": "plateau",  # 'plateau'만 지원
        "factor": 0.5,
        "patience": 3,
        "min_lr": 1e-6
    },

    # --- 로깅 및 저장 ---
    "save_interval": 5, # N 에포크마다 체크포인트 저장
    "early_stopping_patience": 10,
}
# ==============================================================================

def main(config):
    # --- 1. 초기 설정 ---
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    if config.get("detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)

    output_dir = config['output_root']
    chkpt_dir = os.path.join(output_dir, 'checkpoints')
    eval_dir = os.path.join(output_dir, 'eval_results')
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training_log.csv')

    # --- 2. 데이터 준비 ---
    # CSV 경로 직접 지정 우선
    train_filelist = config.get('train_csv') or os.path.join(config['dataset_dir'], 'train.csv')
    val_filelist = config.get('val_csv') or os.path.join(config['dataset_dir'], 'validation.csv')

    if not os.path.exists(train_filelist) or not os.path.exists(val_filelist):
        print(f"오류: '{train_filelist}' 또는 '{val_filelist}' 파일을 찾을 수 없습니다.")
        print("먼저 'preprocess.py' 또는 진단 스크립트로 CSV를 생성해주세요.")
        return

    segment_len = config['sample_rate'] * config['segment_duration']
    
    train_dataset = DenoisingDataset(train_filelist, segment_len, config['sample_rate'])
    # 검증 데이터는 랜덤 크롭 없이 전체 신호를 사용해 일관성 있게 평가
    val_dataset = DenoisingDataset(val_filelist, segment_len, config['sample_rate'], for_evaluation=True)
    
    print(f"데이터셋 크기 - 학습: {len(train_dataset)}, 검증: {len(val_dataset)}")

    # 작은 데이터셋에서 업데이트 수를 늘리기 위해 replacement 샘플러를 옵션으로 지원
    if config.get('steps_per_epoch', 0) and config['steps_per_epoch'] > 0:
        from torch.utils.data import RandomSampler
        num_samples = int(config['steps_per_epoch']) * int(config['batch_size'])
        sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples)
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=False,
            sampler=sampler, num_workers=config['num_workers'], pin_memory=config['pin_memory']
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True,
            num_workers=config['num_workers'], pin_memory=config['pin_memory']
        )
    val_loader = DataLoader(
        # 검증 시에는 다양한 길이를 처리하기 위해 batch_size를 1로 설정합니다.
        val_dataset, batch_size=1, shuffle=False,
        num_workers=config['num_workers'], pin_memory=config['pin_memory']
    )

    # --- 3. 모델, 옵티마이저, 손실 함수 정의 ---
    model_g = ConvTasNet(**config['model_params']).to(device)
    model_d = Discriminator().to(device)

    # --- [성능 가속화] torch.compile 적용 ---
    if config.get("use_torch_compile", False) and hasattr(torch, 'compile'):
        print("torch.compile()을 사용하여 Generator를 최적화합니다... (첫 에포크는 컴파일 시간으로 인해 약간 느릴 수 있습니다)")
        try:
            model_g = torch.compile(model_g)
        except Exception as e:
            print(f"torch.compile(model_g) 적용 중 오류 발생: {e}. 일반 모드로 계속합니다.")


    lr_g = config['learning_rate_g_ft'] if config.get('is_finetune') else config['learning_rate_g']
    lr_d = config['learning_rate_d_ft'] if config.get('is_finetune') else config['learning_rate_d']
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=lr_g, weight_decay=1e-4)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=lr_d)
    scheduler_g = None
    if config.get("lr_scheduler", {}).get("type") == "plateau":
        s = config["lr_scheduler"]
        scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_g, mode='min', factor=s.get("factor", 0.5), patience=s.get("patience", 3),
            min_lr=s.get("min_lr", 1e-6), verbose=True
        )
    
    adversarial_loss = nn.MSELoss().to(device)
    # Prefer torchmetrics SI-SDR for stable gradients; keep shift-invariant available if needed
    reconstruction_loss = si_sdr_loss_torchmetrics
    mrstft_loss = MRSTFTLoss().to(device)
    perc_loss_fn = LogMelPerceptualLoss(sr=config['sample_rate']).to(device)

    # --- [성능 가속화 & 경고 수정] AMP GradScaler 초기화 ---
    use_amp = config.get("use_amp", False) and device.type == 'cuda'
    # 최신 PyTorch API에 맞게 수정
    scaler_g = torch.amp.GradScaler('cuda', enabled=use_amp)
    scaler_d = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("Automatic Mixed Precision (AMP)을 활성화합니다.")


    # --- 4. 체크포인트 로드 (재개 또는 미세조정) ---
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config['resume_checkpoint']:
        start_epoch, best_val_loss = load_checkpoint(
            config['resume_checkpoint'], model_g, model_d, optimizer_g, optimizer_d
        )
    elif config['fine_tuning_checkpoint']:
        _, _ = load_checkpoint(config['fine_tuning_checkpoint'], model_g)
        print(f"'{config['fine_tuning_checkpoint']}'에서 Generator 가중치를 로드하여 미세조정을 시작합니다.")

    # --- 5. 로깅 및 베이스라인 측정 준비 ---
    if os.path.exists(log_file) and start_epoch > 0:
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame(columns=['epoch', 'train_g_loss', 'train_d_loss', 'val_loss', 'pesq', 'stoi', 'si_sdr'])

    if start_epoch == 0:
        print("\n학습 시작 전 베이스라인 성능을 측정합니다...")
        baseline_pesq, baseline_stoi, baseline_si_sdr = 0, 0, 0
        total_batches = 0
        model_g.eval()
        with torch.no_grad():
            for noisy, clean in tqdm(val_loader, desc="Baseline 측정 중"):
                # 메트릭 계산 전 보정: 길이 정렬 + float32 + [-1,1] 클램프
                min_len = min(clean.shape[-1], noisy.shape[-1])
                clean_b = clean[..., :min_len].float().clamp(-1.0, 1.0)
                noisy_b = noisy[..., :min_len].float().clamp(-1.0, 1.0)
                pesq_val, stoi_val, si_sdr_val = calculate_metrics(clean_b, noisy_b, config['sample_rate'])
                baseline_pesq += pesq_val; baseline_stoi += stoi_val; baseline_si_sdr += si_sdr_val
                total_batches += 1
        
        baseline_log = {
            'epoch': -1, 'pesq': baseline_pesq / total_batches,
            'stoi': baseline_stoi / total_batches, 'si_sdr': baseline_si_sdr / total_batches
        }
        # [경고 수정] concat 방식을 약간 변경하여 경고 발생 가능성 줄임
        log_df = pd.concat([log_df, pd.DataFrame([baseline_log])], ignore_index=True)
        log_df.to_csv(log_file, index=False)
        print(f"베이스라인 성능: PESQ={baseline_log['pesq']:.3f}, STOI={baseline_log['stoi']:.3f}, SI-SDR={baseline_log['si_sdr']:.3f} dB")
    
    # --- 7. 학습 및 검증 루프 ---
    patience_counter = 0
    for epoch in range(start_epoch, config['epochs']):
        # Adversarial loss warmup: adv_warmup_epochs까지 0.0, 이후 자동 변경
        if epoch < config.get("adv_warmup_epochs", 5):
            w_adv = 0.0
        else:
            w_adv = config["lambda_adv"]

        start_time = time.time()
        
        # --- 학습 단계 ---
        model_g.train(); model_d.train()
        # 파인튜닝: 초반 특정 에포크 동안 encoder/bottleneck 동결
        if config.get('is_finetune', False) and epoch < config.get('freeze_encoder_epochs', 0):
            for p in model_g.encoder.parameters():
                p.requires_grad = False
            for p in model_g.bottleneck.parameters():
                p.requires_grad = False
        else:
            for p in model_g.encoder.parameters():
                p.requires_grad = True
            for p in model_g.bottleneck.parameters():
                p.requires_grad = True
        total_train_g_loss, total_train_d_loss = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} [학습]")
        effective_steps = 0  # NaN/Inf 배치 스킵 반영
        for noisy, clean in progress_bar:
            noisy, clean = noisy.to(device), clean.to(device)
            # 입력 안정화
            noisy = torch.nan_to_num(noisy.float().clamp(-1, 1), nan=0.0, posinf=0.0, neginf=0.0)
            clean = torch.nan_to_num(clean.float().clamp(-1, 1), nan=0.0, posinf=0.0, neginf=0.0)
            valid = torch.ones(noisy.size(0), 1, 1, device=device)
            fake = torch.zeros(noisy.size(0), 1, 1, device=device)
            
            # --- [성능 가속화 & 경고 수정] AMP Autocast 적용 ---
            # --- Generator 학습 ---
            optimizer_g.zero_grad(set_to_none=True)
            # forward만 autocast로 감싸고, 일부 손실(특히 로그/분수 포함)은 FP32로 계산
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                denoised = model_g(noisy)
            denoised_wav = denoised.squeeze(1)
            loss_sdr = reconstruction_loss(denoised_wav, clean)
            # STFT/Perceptual은 FP32에서 계산해 underflow/NaN 방지
            loss_stft = mrstft_loss(denoised_wav.float(), clean.float()) if config.get('use_stft', True) else torch.tensor(0.0, device=device, dtype=loss_sdr.dtype)
            # Perceptual warmup: 초반 disable_perc_epochs 동안 0으로 처리
            if config.get('use_perc', True) and not (config.get('is_finetune', False) and epoch < config.get('disable_perc_epochs', 0)):
                loss_perc = perc_loss_fn(denoised_wav.float(), clean.float())
            else:
                loss_perc = torch.tensor(0.0, device=device, dtype=loss_sdr.dtype)
            if config.get('is_finetune', False) and epoch < config.get('disable_d_epochs', 0):
                loss_adv = torch.tensor(0.0, device=device, dtype=loss_sdr.dtype)
            else:
                if config.get('use_adv', True):
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        pred_fake = model_d(denoised)
                        valid_smooth, fake_smooth = smooth_labels(valid, low=0.9, high=0.0)
                        valid_smooth = valid_smooth.to(dtype=pred_fake.dtype)
                        loss_adv = adversarial_loss(pred_fake, valid_smooth)
                else:
                    loss_adv = torch.tensor(0.0, device=device, dtype=loss_sdr.dtype)

            # 에포크별 유효 가중치(Perceptual warmup, Adv warmup/토글 반영)
            w_sdr = config['lambda_sdr']
            w_stft = (config['lambda_stft'] if config.get('use_stft', True) else 0.0)
            w_perc = (config['lambda_perc'] if (config.get('use_perc', True) and not (config.get('is_finetune', False) and epoch < config.get('disable_perc_epochs', 0))) else 0.0)
            w_adv = (0.0 if (config.get('is_finetune', False) and epoch < config.get('disable_d_epochs', 0)) else (config['lambda_adv'] if config.get('use_adv', True) else 0.0))

            g_loss = (w_sdr * loss_sdr + w_stft * loss_stft + w_perc * loss_perc + w_adv * loss_adv)

            # 비정상 손실 방지: NaN/Inf면 배치 스킵
            g_loss = g_loss.to(torch.float32)
            if not torch.isfinite(g_loss):
                progress_bar.set_postfix_str("g_loss non-finite, skip batch")
                continue

            scaler_g.scale(g_loss).backward()
            # gradient clipping (AMP 사용 시 unscale 후 clip)
            if use_amp:
                scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # --- [성능 가속화 & 경고 수정] AMP Autocast 적용 ---
            # --- Discriminator 학습 ---
            if (config.get('is_finetune', False) and epoch < config.get('disable_d_epochs', 0)) or not config.get('use_adv', True):
                d_loss = torch.tensor(0.0, device=device)
            else:
                optimizer_d.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred_real = model_d(clean)
                    valid_smooth, fake_smooth = smooth_labels(valid, low=0.9, high=0.0)
                    valid_smooth = valid_smooth.to(dtype=pred_real.dtype)
                    loss_real = adversarial_loss(pred_real, valid_smooth)
                    pred_fake = model_d(denoised.detach())
                    fake_smooth = fake_smooth.to(dtype=pred_fake.dtype)
                    loss_fake = adversarial_loss(pred_fake, fake_smooth)
                    d_loss = 0.5 * (loss_real + loss_fake)
                if torch.isfinite(d_loss):
                    scaler_d.scale(d_loss).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    progress_bar.set_postfix_str("d_loss non-finite, skip D step")

            total_train_g_loss += float(g_loss.detach().cpu())
            total_train_d_loss += float(d_loss.detach().cpu())
            effective_steps += 1
            progress_bar.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}")

        denom = max(effective_steps, 1)
        avg_train_g_loss = total_train_g_loss / denom
        avg_train_d_loss = total_train_d_loss / denom

        # --- 검증 단계 ---
        model_g.eval()
        total_val_loss, total_pesq, total_stoi, total_si_sdr = 0, 0, 0, 0
        with torch.no_grad():
            for i, (noisy, clean) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch}/{config['epochs']} [검증]")):
                noisy, clean = noisy.to(device), clean.to(device)
                
                # --- [성능 가속화 & 경고 수정] 검증 시에도 Autocast 적용 ---
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    denoised = model_g(noisy)
                
                min_len = min(denoised.shape[-1], clean.shape[-1])
                denoised = torch.nan_to_num(denoised[..., :min_len].float().clamp(-1.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)
                clean = torch.nan_to_num(clean[..., :min_len].float().clamp(-1.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)

                # 검증 손실은 SI-SDR + MR-STFT로 구성 (Adv 제외)
                val_sdr = reconstruction_loss(denoised.squeeze(1), clean)
                val_stft = mrstft_loss(denoised.squeeze(1), clean)
                val_perc = perc_loss_fn(denoised.squeeze(1), clean)
                total_val_loss += (config['lambda_sdr'] * val_sdr + config['lambda_stft'] * val_stft + config['lambda_perc'] * val_perc).item()
                pesq_val, stoi_val, si_sdr_val = calculate_metrics(clean, denoised, config['sample_rate'])
                total_pesq += pesq_val; total_stoi += stoi_val; total_si_sdr += si_sdr_val
                if i < 5 and epoch % config['save_interval'] == 0:
                    save_sample_audios(eval_dir, f"epoch_{epoch}_sample_{i}", clean.squeeze(0), noisy.squeeze(0), denoised.squeeze(0), config['sample_rate'])
                    save_spectrogram_images(eval_dir, f"epoch_{epoch}_sample_{i}", clean.squeeze(0), noisy.squeeze(0), denoised.squeeze(0), config['sample_rate'])

        avg_val_loss = total_val_loss / len(val_loader); avg_pesq = total_pesq / len(val_loader)
        avg_stoi = total_stoi / len(val_loader); avg_si_sdr = total_si_sdr / len(val_loader)
        
        # 스케줄러 스텝 (Plateau)
        if scheduler_g is not None:
            scheduler_g.step(avg_val_loss)

        # 현재 LR 출력
        current_lr = optimizer_g.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} 완료 ({epoch_time:.2f}초) - Val Loss: {avg_val_loss:.4f}, PESQ: {avg_pesq:.3f}, STOI: {avg_stoi:.3f}, SI-SDR: {avg_si_sdr:.3f} dB, LR: {current_lr:.2e}")
        
        new_log = {'epoch': epoch, 'train_g_loss': avg_train_g_loss, 'train_d_loss': avg_train_d_loss, 'val_loss': avg_val_loss, 'pesq': avg_pesq, 'stoi': avg_stoi, 'si_sdr': avg_si_sdr}
        log_df = pd.concat([log_df, pd.DataFrame([new_log])], ignore_index=True); log_df.to_csv(log_file, index=False)
        
        is_best = avg_val_loss < best_val_loss
        if is_best: best_val_loss = avg_val_loss; patience_counter = 0
        else: patience_counter += 1

        checkpoint_state = {'epoch': epoch, 'model_g_state_dict': model_g.state_dict(), 'model_d_state_dict': model_d.state_dict(), 'optimizer_g_state_dict': optimizer_g.state_dict(), 'optimizer_d_state_dict': optimizer_d.state_dict(), 'best_val_loss': best_val_loss, 'config': config}
        if epoch % config['save_interval'] == 0: save_checkpoint(checkpoint_state, False, chkpt_dir, f'checkpoint_epoch_{epoch}.pth')
        if is_best: save_checkpoint(checkpoint_state, True, chkpt_dir)

        if patience_counter >= config['early_stopping_patience']:
            print(f"{config['early_stopping_patience']} 에포크 동안 성능 개선이 없어 조기 종료합니다."); break

    print(f"\n학습 완료. 최고 검증 손실: {best_val_loss:.4f}");

if __name__ == '__main__':
    main(CONFIG)

