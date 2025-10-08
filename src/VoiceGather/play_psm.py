import sounddevice as sd
import numpy as np
import struct
import sys
import os

"""
PSM 파일 재생 스크립트
- 16비트 signed PCM, little-endian, 모노 또는 스테레오, 샘플레이트 44100Hz 기준
- 파일 확장자: .psm (raw PCM 데이터)
- 사용법: python play_psm.py <파일경로> [샘플레이트] [채널수]

100분 녹음했는데, 1088까지 재생했어요.
"""

def play_psm(filename, samplerate=44100, channels=1):
    # 파일 크기 확인
    filesize = os.path.getsize(filename)
    sample_width = 2  # 16비트 = 2바이트
    total_samples = filesize // (sample_width * channels)

    with open(filename, 'rb') as f:
        raw = f.read()
        # int16로 변환
        audio = np.frombuffer(raw, dtype='<i2')  # little-endian 16bit signed
        if channels > 1:
            audio = audio.reshape(-1, channels)
    print(f"재생: {filename} | 샘플레이트: {samplerate} | 채널: {channels} | 샘플수: {total_samples}")
    sd.play(audio, samplerate=samplerate)
    sd.wait()
    print("재생 완료.")

def play_pcm_recursive(target_dir, samplerate, channels):
    """폴더 내 모든 하위 폴더까지 pcm 파일 재생"""
    for root, dirs, files in os.walk(target_dir):
        pcm_files = [f for f in files if f.lower().endswith('.pcm')]
        pcm_files.sort()
        for fname in pcm_files:
            fullpath = os.path.join(root, fname)
            relpath = os.path.relpath(fullpath, target_dir)
            print(f"\n===== {relpath} 재생 =====")
            play_psm(fullpath, samplerate, channels)

if __name__ == "__main__":
    # === 아래 값만 수정해서 사용하세요 ===
    TARGET = r"KsponSpeech_01"  # 폴더명 또는 파일명 (상대/절대경로 모두 가능)
    SAMPLERATE = 16000           # 샘플레이트
    CHANNELS = 1                 # 채널 수
    # =============================

    if os.path.isdir(TARGET):
        play_pcm_recursive(TARGET, SAMPLERATE, CHANNELS)
    else:
        play_psm(TARGET, SAMPLERATE, CHANNELS)
