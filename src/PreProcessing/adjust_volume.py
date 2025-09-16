import os
import glob
from pydub import AudioSegment

def match_target_amplitude(sound_a_path, sound_b_path, output_path):
    """
    sound_a의 볼륨을 sound_b의 볼륨에 맞춰 조절하고 새 파일로 저장하는 함수.
    (이전 코드와 동일)
    """
    try:
        sound_a = AudioSegment.from_file(sound_a_path)
        sound_b = AudioSegment.from_file(sound_b_path)

        change_in_dbfs = sound_b.dBFS - sound_a.dBFS
        adjusted_sound_a = sound_a.apply_gain(change_in_dbfs)

        adjusted_sound_a.export(output_path, format="wav")
        
        print(f"✅ 성공: '{os.path.basename(sound_a_path)}'의 볼륨을 {change_in_dbfs:.2f}dB 조절하여 저장했습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {os.path.basename(sound_a_path)} 처리 중 오류. ({e})")


def process_folder(source_folder, output_folder):
    """
    지정된 폴더 내의 모든 micA, micB 파일 쌍을 찾아 볼륨을 조절합니다.
    """
    # 결과물을 저장할 폴더가 없으면 새로 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' 폴더를 생성했습니다.")

    # 소스 폴더에서 'micA_*.wav' 패턴을 가진 모든 파일을 찾음
    # 예: ['source_audio/micA_01.wav', 'source_audio/micA_02.wav', ...]
    search_pattern = os.path.join(source_folder, 'micA_*.wav')
    mic_a_files = glob.glob(search_pattern)

    if not mic_a_files:
        print("처리할 'micA_*.wav' 파일을 찾을 수 없습니다. 파일 이름과 경로를 확인해주세요.")
        return

    print(f"총 {len(mic_a_files)}개의 micA 파일을 찾았습니다. 처리를 시작합니다.")
    
    # 찾은 micA 파일을 하나씩 순회
    for a_file_path in mic_a_files:
        # micA 파일 경로를 기반으로 짝이 되는 micB 파일 경로 생성
        # 예: 'source_audio/micA_01.wav' -> 'source_audio/micB_01.wav'
        b_file_path = a_file_path.replace("micA_", "micB_")

        # 결과 파일 경로 생성
        # 예: 'source_audio/micA_01.wav' -> 'output_audio/micA_01_matched.wav'
        file_name = os.path.basename(a_file_path)
        output_name = file_name.replace(".wav", "_matched.wav")
        output_file_path = os.path.join(output_folder, output_name)
        
        # 짝이 되는 micB 파일이 실제로 존재하는지 확인
        if os.path.exists(b_file_path):
            # 파일이 존재하면 볼륨 조절 함수 호출
            match_target_amplitude(a_file_path, b_file_path, output_file_path)
        else:
            # 짝이 없으면 경고 메시지 출력
            print(f"⚠️ 경고: '{os.path.basename(a_file_path)}'의 짝인 '{os.path.basename(b_file_path)}'을 찾을 수 없습니다.")

# --- 코드 실행 부분 ---
if __name__ == "__main__":
    # 원본 파일들이 있는 폴더와 결과물을 저장할 폴더를 지정
    SOURCE_DIRECTORY = "C:\\Users\\kesa0\\Downloads\\Lecture\\202502\\종설프\\SoundData\\BeforeProcess\\house_with_fan"
    OUTPUT_DIRECTORY = "C:\\Users\\kesa0\\Downloads\\Lecture\\202502\\종설프\\SoundData\\AfterProcess\\house_with_fan"

    process_folder(SOURCE_DIRECTORY, OUTPUT_DIRECTORY)
    print("\n모든 작업이 완료되었습니다.")