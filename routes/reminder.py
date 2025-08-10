from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from firebase_admin import firestore, storage
from firebase.firebase_init import bucket
from uuid import uuid4
from llm.gpt_client import generate_reminder_sentence
from tts.elevenlabs_client import text_to_speech, process_audio_speed
from enums import ToneEnum
import os
import tempfile  # ✅ 추가

router = APIRouter()
db = firestore.client()

@router.post("/generate-reminder")
async def generate_reminder_endpoint(
    guardian_uid: str = Form(...),
    patient_uid: str = Form(...),
    patient_name: str = Form(...),
    photo_description: str = Form(...),
    relationship: str = Form(...),
    tone: ToneEnum = Form(...),
    voice_id: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # ✅ 이미지 임시 저장
        ext = os.path.splitext(image.filename)[1]  # 예: ".jpg"
        safe_filename = f"{uuid4().hex}{ext}"
        image_path = os.path.join(tempfile.gettempdir(), safe_filename)
        
        with open(image_path, "wb") as f:
            f.write(await image.read())

        # ✅ 디버깅: 파일 존재 여부 확인
        print("임시 이미지 저장 경로:", image_path)
        print("파일 존재 여부:", os.path.exists(image_path))

        result = generate_reminder_sentence(patient_name, photo_description, relationship, tone, image_path)
        print("📝 photo_description:", photo_description)
        print("🔧 GPT 응답:", result)

        # ✅ generate_reminder_sentence에서 사용하는 동안 삭제 지연
        try:
            result = generate_reminder_sentence(
                patient_name, photo_description, relationship, tone, image_path
            )
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

        # 회상 문장 추출
        reminder_text = result.strip()

        if not reminder_text or len(reminder_text) < 5:
            raise HTTPException(status_code=400, detail="회상 문장이 없습니다.")

        # TTS 생성
        reminder_mp3 = f"reminder_{uuid4().hex}.mp3"
        text_to_speech(reminder_text, voice_id, reminder_mp3)
        process_audio_speed(reminder_mp3, reminder_mp3, speed=0.83)

        # 업로드
        blob = bucket.blob(f"tts/reminder/{guardian_uid}/{patient_uid}/{reminder_mp3}")
        blob.upload_from_filename(reminder_mp3)
        blob.make_public()
        reminder_url = f"https://storage.googleapis.com/{bucket.name}/tts/reminder/{guardian_uid}/{patient_uid}/{reminder_mp3}"

        # TTS 파일 삭제
        if os.path.exists(reminder_mp3):
            os.remove(reminder_mp3)

        return {
            "message": "회상 문장 생성 완료",
            "reminder_text": reminder_text,
            "tts_url": reminder_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
