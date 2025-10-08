import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time

# --- 사용자 설정 ---
# 'list_audio_devices.py'를 실행하여 확인한 실제 마이크 인덱스 번호로 변경해주세요.

# === 사용자 설정 ===
MIC_A_INDEX = 1  # 🎤 첫 번째 마이크의 인덱스 번호
MIC_B_INDEX = 4  # 🎤 두 번째 마이크의 인덱스 번호

DURATION = 60         # 한 번에 녹음할 시간 (초)
REPEAT = 150           # 반복 횟수
SAMPLE_RATE = 44100   # 샘플링 속도 (Hz)
CHANNELS = 1          # 채널 수 (1 = 모노, 2 = 스테레오)
SAVE_DIR = r'soundfromAIHub1'  # 저장할 폴더명 (예: recordings)
FILENAME_FORMAT_A = 'micA_{:04d}.wav'  # 첫 번째 마이크 파일명 포맷
FILENAME_FORMAT_B = 'micB_{:04d}.wav'  # 두 번째 마이크 파일명 포맷
# --------------------

# 각 스레드의 녹음 결과를 저장할 딕셔너리
recordings = {}

def record_mic(device_index, duration, sample_rate, channels, result_dict):
    """지정된 장치에서 오디오를 녹음하고 결과를 딕셔너리에 저장하는 함수"""
    print(f"[{threading.current_thread().name}] 장치 {device_index}에서 {duration}초 동안 녹음을 시작합니다...")
    
    try:
        # 주어진 시간 동안 오디오 녹음
        recording_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, device=device_index, dtype='float32')
        sd.wait()  # 녹음이 끝날 때까지 대기
        
        # 결과를 딕셔너리에 저장
        result_dict[device_index] = recording_data
        print(f"[{threading.current_thread().name}] 장치 {device_index} 녹음 완료.")
        
    except Exception as e:
        print(f"오류 발생 (장치 {device_index}): {e}")
        result_dict[device_index] = None



import os

def ensure_save_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    """메인 실행 함수"""
    print(f"=== 🎤 두 채널 동시 녹음 반복 시작 (총 {REPEAT}회, 각 {DURATION}초) ===")
    ensure_save_dir(SAVE_DIR)

    for i in range(1, REPEAT + 1):
        print(f"\n--- 반복 {i}/{REPEAT} ---")
        # 각 반복마다 결과 딕셔너리 초기화
        recordings.clear()

        # 각 마이크 녹음을 위한 스레드 생성
        thread_a = threading.Thread(target=record_mic, name="Mic-A", args=(MIC_A_INDEX, DURATION, SAMPLE_RATE, CHANNELS, recordings))
        thread_b = threading.Thread(target=record_mic, name="Mic-B", args=(MIC_B_INDEX, DURATION, SAMPLE_RATE, CHANNELS, recordings))

        # 스레드 시작
        start_time = time.time()
        thread_a.start()
        thread_b.start()

        # 모든 스레드가 작업을 마칠 때까지 대기
        thread_a.join()
        thread_b.join()
        end_time = time.time()
        print(f"녹음 완료 (소요: {end_time - start_time:.2f}초). 파일 저장 중...")

        # 파일명 생성
        filename_a = os.path.join(SAVE_DIR, FILENAME_FORMAT_A.format(i))
        filename_b = os.path.join(SAVE_DIR, FILENAME_FORMAT_B.format(i))

        # 녹음된 데이터를 WAV 파일로 저장
        if recordings.get(MIC_A_INDEX) is not None:
            sf.write(filename_a, recordings[MIC_A_INDEX], SAMPLE_RATE)
            print(f"✅ '{filename_a}' 파일 저장 완료!")
        else:
            print(f"❌ '{filename_a}' 파일 저장 실패. (장치 {MIC_A_INDEX} 녹음 데이터 없음)")
        
        if recordings.get(MIC_B_INDEX) is not None:
            sf.write(filename_b, recordings[MIC_B_INDEX], SAMPLE_RATE)
            print(f"✅ '{filename_b}' 파일 저장 완료!")
        else:
            print(f"❌ '{filename_b}' 파일 저장 실패. (장치 {MIC_B_INDEX} 녹음 데이터 없음)")

    print(f"\n=== 모든 반복 녹음 및 저장 완료! ===")

if __name__ == "__main__":
    main()
