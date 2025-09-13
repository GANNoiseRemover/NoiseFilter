import numpy as np
import soundfile as sf

# --- 설정 (사용자 수정 필요) ---
# 💾 손상된 원본 파일 이름을 정확하게 입력하세요.
CORRUPTED_FILENAME = 'MIC_B_silent_only_voice.wav' 

# 💾 새로 저장할 파일 이름을 지정하세요.
RECOVERED_FILENAME = 'recovered_audioB.wav'
# --- 설정 끝 ---

# 이전 녹음 스크립트에서 사용했던 오디오 정보
SAMPLE_RATE = 44100
CHANNELS = 1
# sounddevice의 기본 데이터 타입은 32비트 float입니다.
DATA_TYPE = np.float32 

try:
    print(f"'{CORRUPTED_FILENAME}' 파일에서 Raw 데이터를 읽는 중...")
    
    # 파일에서 순수 바이너리 데이터를 numpy 배열로 읽어옵니다.
    raw_audio_data = np.fromfile(CORRUPTED_FILENAME, dtype=DATA_TYPE)
    
    print(f"✅ 데이터 읽기 완료! (총 {raw_audio_data.size}개의 샘플)")

    stereo_data = raw_audio_data.reshape(-1, CHANNELS)
    
    # soundfile 라이브러리를 사용해 올바른 헤더와 함께 새 파일로 저장합니다.
    print(f"'{RECOVERED_FILENAME}' 파일로 저장하는 중...")
    sf.write(RECOVERED_FILENAME, raw_audio_data, SAMPLE_RATE)
    
    print("\n🎉 파일 복구가 완료되었습니다!")

except FileNotFoundError:
    print(f"❌ 오류: '{CORRUPTED_FILENAME}' 파일을 찾을 수 없습니다. 파일 이름을 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")