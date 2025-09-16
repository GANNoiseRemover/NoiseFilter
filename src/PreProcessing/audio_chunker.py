import os
import glob
from pydub import AudioSegment
from pydub.utils import make_chunks

def split_audio_in_folders(source_base_dir, matched_base_dir, chunk_output_base_dir, chunk_length_sec=3):
    """
    여러 시나리오 폴더를 순회하며 오디오 파일을 3초 단위로 자릅니다.

    :param source_base_dir: micB 원본 파일들이 있는 최상위 폴더 (예: '1_source_audio')
    :param matched_base_dir: 볼륨 조절된 micA 파일들이 있는 최상위 폴더 (예: '2_matched_audio')
    :param chunk_output_base_dir: 잘린 파일들을 저장할 최상위 폴더 (예: '3_chunked_audio')
    :param chunk_length_sec: 자를 시간 단위 (초)
    """
    chunk_length_ms = int(chunk_length_sec * 1000)
    global_chunk_counter = 1  # 모든 파일에 걸쳐 고유 번호를 매기기 위한 카운터

    print("오디오 분할 작업을 시작합니다...")

    # 소스 폴더 하위의 모든 시나리오 폴더를 찾음 (예: 'house_ambient')
    try:
        scenario_folders = [f for f in os.listdir(source_base_dir) if os.path.isdir(os.path.join(source_base_dir, f))]
    except FileNotFoundError:
        print(f"❌ 오류: '{source_base_dir}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 각 시나리오 폴더에 대해 작업 반복
    for scenario in scenario_folders:
        print(f"\n📁 시나리오 '{scenario}' 처리 중...")
        
        # 현재 시나리오에 맞는 폴더 경로 설정
        source_scenario_dir = os.path.join(source_base_dir, scenario)
        matched_scenario_dir = os.path.join(matched_base_dir, scenario)
        output_scenario_dir = os.path.join(chunk_output_base_dir, scenario)

        # 결과물 저장 폴더 생성
        if not os.path.exists(output_scenario_dir):
            os.makedirs(output_scenario_dir)

        # 현재 시나리오 폴더에서 micB 파일 목록 가져오기
        search_pattern = os.path.join(source_scenario_dir, 'micB_*.wav')
        mic_b_files = glob.glob(search_pattern)

        if not mic_b_files:
            print(f"  - '{scenario}' 폴더에서 처리할 'micB_*.wav' 파일을 찾지 못했습니다.")
            continue

        # micB 파일을 기준으로 페어링 및 분할 작업
        for noise_path in mic_b_files:
            base_name = os.path.basename(noise_path) # 예: 'micB_01.wav'
            
            # 짝이 되는 clean 파일 경로 추정
            clean_name = base_name.replace('micB_', 'micA_').replace('.wav', '_matched.wav')
            clean_path = os.path.join(matched_scenario_dir, clean_name)

            # 두 파일이 모두 존재하는지 확인
            if not os.path.exists(clean_path):
                print(f"  ⚠️ 경고: '{noise_path}'의 짝인 '{clean_path}'를 찾을 수 없어 건너뜁니다.")
                continue

            try:
                # 오디오 파일 로드
                noise_audio = AudioSegment.from_file(noise_path)
                clean_audio = AudioSegment.from_file(clean_path)
                
                # 두 오디오 중 짧은 것을 기준으로 길이를 맞춤 (녹음 시작/종료 오차 보정)
                min_len = min(len(noise_audio), len(clean_audio))
                noise_audio = noise_audio[:min_len]
                clean_audio = clean_audio[:min_len]

                # pydub의 make_chunks 함수로 오디오를 일정 길이로 자름
                noise_chunks = make_chunks(noise_audio, chunk_length_ms)
                clean_chunks = make_chunks(clean_audio, chunk_length_ms)

                # 잘린 조각들을 파일로 저장
                for i, (clean_chunk, noise_chunk) in enumerate(zip(clean_chunks, noise_chunks)):
                    # 마지막 조각이 너무 짧으면 버릴 수 있음 (예: 1초 미만)
                    if len(clean_chunk) < 1000:
                        continue
                    
                    # 파일 이름 생성 및 저장
                    clean_output_path = os.path.join(output_scenario_dir, f"clean_{global_chunk_counter}.wav")
                    noise_output_path = os.path.join(output_scenario_dir, f"noise_{global_chunk_counter}.wav")
                    
                    clean_chunk.export(clean_output_path, format="wav")
                    noise_chunk.export(noise_output_path, format="wav")
                    
                    global_chunk_counter += 1
                
                print(f"  - ✅ '{base_name}'와 짝 파일 분할 완료.")

            except Exception as e:
                print(f"  - ❌ 오류: '{base_name}' 처리 중 오류 발생: {e}")

    print(f"\n✨ 모든 작업 완료! 총 {global_chunk_counter - 1}개의 clean/noise 쌍을 생성했습니다.")


# --- 코드 실행 부분 ---
if __name__ == "__main__":
    # 1. 원본 micB 파일이 있는 기본 폴더
    SOURCE_BASE_DIR = "C:\\Users\\kesa0\\Downloads\\Lecture\\202502\\종설프\\SoundData\\Source"
    
    # 2. 볼륨 조절된 파일(micA_*_matched.wav)이 있는 기본 폴더
    MATCHED_BASE_DIR = "C:\\Users\\kesa0\\Downloads\\Lecture\\202502\\종설프\\SoundData\\VolumeAdjusted"

    # 3. 최종적으로 잘린 파일들을 저장할 기본 폴더
    CHUNK_OUTPUT_DIR = "C:\\Users\\kesa0\\Downloads\\Lecture\\202502\\종설프\\SoundData\\Chunked"
    
    # 자를 단위 시간 (초)
    CHUNK_SECONDS = 3

    split_audio_in_folders(SOURCE_BASE_DIR, MATCHED_BASE_DIR, CHUNK_OUTPUT_DIR, CHUNK_SECONDS)