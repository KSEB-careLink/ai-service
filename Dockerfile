FROM python:3.11-slim

# ffmpeg 및 필요한 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt 및 voicefixer wheel 먼저 복사
COPY requirements.txt .
COPY voicefixer-*.whl ./

# 의존성 설치 (캐시 없이)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install voicefixer-*.whl

# 전체 소스 복사 (가장 마지막 단계에 — 캐시 이점 최대화)
COPY . .

EXPOSE 8000

# uvicorn 실행 명령 (FastAPI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
