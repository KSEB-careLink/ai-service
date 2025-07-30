# 1. Python 3.11 slim 버전 사용
FROM python:3.11-slim

# 2. 시스템 패키지 설치 (ffmpeg + librosa용 의존성 + git)
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 전체 프로젝트 복사
COPY . .

# 6. FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
