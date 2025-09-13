# find_mic.py
import sounddevice as sd

print("사용 가능한 오디오 입력 장치 목록:")
print(sd.query_devices())
