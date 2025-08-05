FROM python:3.11-slim

# ffmpeg 등 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ requirements 및 wheel 먼저 복사
COPY requirements.txt ./
COPY voicefixer-*.whl ./  # ← 이거 맞습니다! (와일드카드 OK)

# ✅ 의존성 설치 (캐시 타게)
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 전체 소스 복사는 마지막에 (캐시 보호)
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
