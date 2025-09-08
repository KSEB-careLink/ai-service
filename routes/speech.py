from fastapi import APIRouter, Form, HTTPException
from firebase_admin import firestore, storage
from firebase.firebase_init import bucket
from tts.elevenlabs_client import text_to_speech, process_audio_speed
from uuid import uuid4
import os

router = APIRouter()
db = firestore.client()

@router.post("/speech")
async def speech_endpoint(
    guardian_uid: str = Form(...),
    patient_uid: str = Form(...),
    text: str = Form(...),
    voice_id: str = Form(...)
):
    try:
        # ✅ 텍스트 → mp3 생성
        mp3_filename = f"speech_{uuid4().hex}.mp3"
        result_path = text_to_speech(text, voice_id, mp3_filename)

        if not result_path or not os.path.exists(result_path):
            raise HTTPException(status_code=500, detail="TTS 변환 실패")

        # ✅ 속도 조절
        process_audio_speed(mp3_filename, mp3_filename, speed=0.83)

        # ✅ Firebase Storage 업로드 (경로 수정됨)
        blob_path = f"tts/sample/{guardian_uid}/{patient_uid}/{mp3_filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(mp3_filename)
        blob.make_public()                # 퍼블릭 공개
        tts_url = blob.public_url         # 공개 URL

        # ✅ 로컬 파일 삭제
        os.remove(mp3_filename)

        return {
            "message": "TTS 변환 및 업로드 완료",
            "tts_url": tts_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
