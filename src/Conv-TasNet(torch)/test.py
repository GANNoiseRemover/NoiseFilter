import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 프로젝트 파일 임포트 ---
from model import ConvTasNet
from dataset import DenoisingDataset
from utils import calculate_metrics, save_sample_audios, save_spectrogram_images, load_checkpoint

def test(config):
    """학습된 모델을 테스트 데이터셋으로 평가하는 함수"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # --- 1. 경로 설정 ---
    output_dir = config['output_root']
    test_results_dir = os.path.join(output_dir, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    # 명시적 체크포인트 경로가 있으면 우선 사용
    best_model_path = config.get('checkpoint_path') or os.path.join(output_dir, 'checkpoints', 'best_model.pth')
    if not os.path.exists(best_model_path):
        print(f"오류: '{best_model_path}'에서 모델 파일을 찾을 수 없습니다.")
        print("먼저 train.py를 실행하여 모델을 학습시켜주세요.")
        return

    # --- 2. 데이터 준비 ---
    test_filelist = config.get('test_csv') or os.path.join(config['dataset_dir'], 'test.csv')
    if not os.path.exists(test_filelist):
        print(f"오류: '{test_filelist}' 파일을 찾을 수 없습니다.")
        print("preprocess.py 또는 진단 스크립트로 'test.csv'를 생성했는지 확인해주세요.")
        return

    # 테스트 시에는 전체 오디오를 평가해야 하므로, for_evaluation=True로 설정합니다.
    # segment_len은 사용되지 않지만 DenoisingDataset의 인자로 필요합니다.
    test_dataset = DenoisingDataset(test_filelist, 0, config['sample_rate'], for_evaluation=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 다양한 길이를 처리하기 위해 batch_size는 1로 고정
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=config.get('pin_memory', True)
    )
    
    print(f"테스트 데이터셋 크기: {len(test_dataset)}")

    # --- 3. 모델 로드 ---
    checkpoint = torch.load(best_model_path, map_location=device)
    model_config = checkpoint['config']['model_params']
    model_g = ConvTasNet(**model_config).to(device)
    # train.py와 동일하게 load_checkpoint 함수 사용 (경고 및 호환성 대응)
    try:
        load_checkpoint(best_model_path, model_g)
    except Exception as e:
        print("\n[체크포인트 로드 실패: state_dict 키 불일치]")
        print(f"경로: {best_model_path}")
        # 키 차이 진단
        try:
            ckpt_sd = checkpoint['model_g_state_dict']
            model_keys = set(model_g.state_dict().keys())
            ckpt_keys = set(ckpt_sd.keys())
            missing = sorted(list(model_keys - ckpt_keys))
            unexpected = sorted(list(ckpt_keys - model_keys))
            print(f"- 누락 키 개수: {len(missing)} (예: {missing[:5]})")
            print(f"- 예상치 못한 키 개수: {len(unexpected)} (예: {unexpected[:5]})")
        except Exception:
            pass
        print("힌트:")
        print("- train.py에서 사용한 output_root/checkpoints/best_model.pth와 동일한 파일인지 확인")
        print("- 최근 구조 변경(gLN/SpectralNorm 등) 이후 학습한 체크포인트인지 확인")
        print("- test.py의 output_root 또는 checkpoint_path를 최신 실험 폴더로 지정")
        # 선택적으로 부분 로드 허용
        if config.get('allow_partial_load', False):
            print("[경고] allow_partial_load=True: strict=False로 가능한 가중치만 로드합니다. 성능이 저하될 수 있습니다.")
            model_g.load_state_dict(checkpoint['model_g_state_dict'], strict=False)
        else:
            # 실패한 예외를 다시 올려 호출자에게 알림
            raise e
    model_g.eval()
    print(f"'{best_model_path}'에서 최고 성능 모델을 로드했습니다.")

    # --- 4. 평가 루프 ---
    total_pesq, total_stoi, total_si_sdr = 0, 0, 0
    total_pesq_orig, total_stoi_orig, total_si_sdr_orig = 0, 0, 0
    
    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="🧪 테스트 데이터로 모델 평가 중")
        for i, (noisy, clean) in enumerate(progress_bar):
            noisy, clean = noisy.to(device), clean.to(device)
            
            # (batch, time) -> (batch, 1, time)
            if noisy.dim() == 2:
                noisy = noisy.unsqueeze(1)
            # AMP 추론 적용 (CUDA에서만)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                denoised = model_g(noisy).squeeze(1)

            # [오류 해결 로직] 메트릭 계산 전 길이 맞추기
            min_len = min(denoised.shape[-1], clean.shape[-1])
            denoised = denoised[..., :min_len].float().clamp(-1.0, 1.0)
            clean = clean[..., :min_len].float().clamp(-1.0, 1.0)

            # 향상점수: denoised vs clean
            pesq_val, stoi_val, si_sdr_val = calculate_metrics(
                clean, denoised, config['sample_rate']
            )
            total_pesq += pesq_val
            total_stoi += stoi_val
            total_si_sdr += si_sdr_val
            # 원점수: noisy vs clean
            pesq_orig, stoi_orig, si_sdr_orig = calculate_metrics(
                clean, noisy.squeeze(1), config['sample_rate']
            )
            total_pesq_orig += pesq_orig
            total_stoi_orig += stoi_orig
            total_si_sdr_orig += si_sdr_orig
            
            # 처음 5개의 샘플에 대해 결과 저장
            if i < 5:
                filename_base = f"test_sample_{i}"
                save_sample_audios(test_results_dir, filename_base, clean.squeeze(0), noisy.squeeze(1), denoised.squeeze(0), config['sample_rate'])
                save_spectrogram_images(test_results_dir, filename_base, clean.squeeze(0), noisy.squeeze(1), denoised.squeeze(0), config['sample_rate'])

    # --- 5. 최종 결과 출력 및 저장 ---
    avg_pesq = total_pesq / len(test_loader)
    avg_stoi = total_stoi / len(test_loader)
    avg_si_sdr = total_si_sdr / len(test_loader)
    avg_pesq_orig = total_pesq_orig / len(test_loader)
    avg_stoi_orig = total_stoi_orig / len(test_loader)
    avg_si_sdr_orig = total_si_sdr_orig / len(test_loader)

    print("\n" + "="*50)
    print("📋 최종 평가 결과 📋")
    print(f"  - 평균 PESQ  : {avg_pesq:.4f} / {avg_pesq_orig:.4f}")
    print(f"  - 평균 STOI  : {avg_stoi:.4f} / {avg_stoi_orig:.4f}")
    print(f"  - 평균 SI-SDR: {avg_si_sdr:.4f} / {avg_si_sdr_orig:.4f} dB")
    print("    (향상점수 / 원점수: denoised vs clean / noisy vs clean)")
    print("="*50)

    # 결과 요약 파일 저장
    results = {
        'model_path': str(best_model_path),
        'avg_pesq': float(avg_pesq),
        'avg_pesq_orig': float(avg_pesq_orig),
        'avg_stoi': float(avg_stoi),
        'avg_stoi_orig': float(avg_stoi_orig),
        'avg_si_sdr': float(avg_si_sdr),
        'avg_si_sdr_orig': float(avg_si_sdr_orig)
    }
    with open(os.path.join(test_results_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"평가 결과 및 샘플 오디오가 '{test_results_dir}'에 저장되었습니다.")


if __name__ == '__main__':
    # train.py의 CONFIG와 동일한 설정을 사용합니다.
    # 필요한 부분만 가져오거나 train.py에서 CONFIG를 import 할 수도 있습니다.
    TEST_CONFIG = {
        "output_root": "convtasnet_additionalAugmented_finetune",
        "checkpoint_path": "convtasnet_additionalAugmented_finetune/checkpoints/best_model.pth",  # 명시적 경로 지정
        "dataset_dir": "dataset", 
        "test_csv": "diagnostics_test/normalized.csv",  # 직접 지정 시 사용. 예: "diagnostics_test/normalized.csv"
        "sample_rate": 16000,
        "num_workers": 4,
        "pin_memory": True,
        "use_amp": True,
        # 부분 로드 허용(안전하지 않음): 키 불일치 시 strict=False로 로딩
        "allow_partial_load": False,
    }
    test(TEST_CONFIG)
