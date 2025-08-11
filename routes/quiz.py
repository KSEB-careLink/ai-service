from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from firebase_admin import firestore, storage
from firebase.firebase_init import bucket
from uuid import uuid4
from llm.gpt_client import generate_quiz_only
from tts.elevenlabs_client import text_to_speech, process_audio_speed
from enums import ToneEnum
import os
import re
import tempfile
from enum import Enum

router = APIRouter()
db = firestore.client()

# 🔹 카테고리 Enum 추가 (선택사항 - 더 엄격한 검증을 원한다면)
class CategoryEnum(str, Enum):
    FAMILY = "가족"
    NEIGHBORHOOD = "동네"
    SCHOOL_DAYS = "학창시절"
    TRAVEL = "여행"
    PATIENT_FAVORITES = "환자가 좋아하는 것"

@router.post("/generate-quiz")
async def generate_quiz_endpoint(
    guardian_uid: str = Form(...),
    patient_uid: str = Form(...),
    patient_name: str = Form(...),
    photo_description: str = Form(...),
    relationship: str = Form(...),
    category: str = Form(...),  # 🔹 카테고리 파라미터 추가
    tone: ToneEnum = Form(...),
    voice_id: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # 🔹 카테고리 유효성 검증 (선택사항)
        valid_categories = ["가족", "동네", "학창시절", "여행", "환자가 좋아하는 것"]
        if category not in valid_categories:
            raise HTTPException(
                status_code=400, 
                detail=f"유효하지 않은 카테고리입니다. 허용된 카테고리: {', '.join(valid_categories)}"
            )

        # ✅ 이미지 임시 저장
        ext = os.path.splitext(image.filename)[1]
        safe_filename = f"{uuid4().hex}{ext}"
        image_path = os.path.join(tempfile.gettempdir(), safe_filename)

        with open(image_path, "wb") as f:
            f.write(await image.read())

        # ✅ 퀴즈 생성 (카테고리 정보도 함께 전달)
        result = generate_quiz_only(patient_name, photo_description, relationship, tone, image_path, category)  # 🔹 category 추가
        print("🧪 generate_quiz_only 결과:\n", result)

        # ✅ 퀴즈 파싱
        quiz_list = []
        quiz_block = {"question": "", "options": [], "answer": ""}
        capture_options = False

        for line in result.strip().splitlines():
            line = line.strip()
            if line.startswith("질문:") or line.startswith("퀴즈 문제:"):
                if quiz_block["question"] and quiz_block["options"] and quiz_block["answer"]:
                    quiz_list.append(quiz_block)
                    quiz_block = {"question": "", "options": [], "answer": ""}
                quiz_block["question"] = line.split(":", 1)[1].strip()
            elif line.startswith("선택지:"):
                capture_options = True
            elif line.startswith("정답:"):
                capture_options = False
                raw = line.split("정답:")[1].strip()
                match = re.match(r"\d+번[.,]?\s*(.+)", raw)
                quiz_block["answer"] = match.group(1).strip() if match else raw
            elif capture_options:
                match = re.match(r"\d+번[.,]?\s*(.+)", line)
                if match:
                    quiz_block["options"].append(match.group(1).strip())

        # 마지막 블록 추가
        if quiz_block["question"] and quiz_block["options"] and quiz_block["answer"]:
            quiz_list.append(quiz_block)

        if not quiz_list:
            raise HTTPException(status_code=400, detail="퀴즈 정보가 부족합니다.")

        # ✅ 각 퀴즈별 TTS 생성 및 업로드
        response_data = []
        for quiz in quiz_list:
            quiz_text = f"{quiz['question']}\n" + "\n".join(
                [f"{i+1}번, {opt}" for i, opt in enumerate(quiz["options"])]
            )

            quiz_mp3 = f"quiz_{uuid4().hex}.mp3"
            text_to_speech(quiz_text, voice_id, quiz_mp3)
            process_audio_speed(quiz_mp3, quiz_mp3, speed=0.83)

            blob = bucket.blob(f"tts/quiz/{guardian_uid}/{patient_uid}/{quiz_mp3}")
            blob.upload_from_filename(quiz_mp3)
            blob.make_public()
            quiz_url = f"https://storage.googleapis.com/{bucket.name}/tts/quiz/{guardian_uid}/{patient_uid}/{quiz_mp3}"

            os.remove(quiz_mp3)

            response_data.append({
                "question": quiz["question"],
                "options": quiz["options"],
                "answer": quiz["answer"],
                "quiz_tts_url": quiz_url
            })

        # ✅ 이미지 파일 정리
        os.remove(image_path)

        return {
            "message": f"퀴즈 {len(response_data)}개 생성 완료",
            "category": category,  # 🔹 응답에 카테고리 정보도 포함
            "quizzes": response_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))