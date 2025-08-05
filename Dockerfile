# 베이스 이미지
FROM python:3.11-slim

# ffmpeg 및 기타 패키지
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# ✅ 1. requirements만 복사 → pip 캐시 유지
COPY requirements.txt ./

# ✅ 2. pip install — 캐시 보장됨
RUN pip install --no-cache-dir -r requirements.txt \
    -f https://download.pytorch.org/whl/torch_stable.html

# ✅ 3. voicefixer wheel 복사 및 설치 (이건 별도로)
COPY voicefixer-*.whl ./
RUN pip install voicefixer-*.whl

# ✅ 4. 나머지 전체 코드 복사
COPY . .

# FastAPI 포트 노출
EXPOSE 8000

# 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
