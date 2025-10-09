FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 시스템 패키지 업데이트 및 빌드 도구 설치
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

CMD ["python", "create_noise_profile.py"]