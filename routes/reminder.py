# routes/reminder.py
from fastapi import APIRouter, Form, HTTPException
from firebase_admin import firestore, storage
from firebase.firebase_init import bucket
from uuid import uuid4
from llm.gpt_client import generate_reminder
from tts.elevenlabs_client import text_to_speech, process_audio_speed
from enums import ToneEnum
import os

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
    voice_id: str = Form(...)
):
    try:
        result = generate_reminder(patient_name, photo_description, relationship, tone)

        reminder_text = ""
        for line in result.splitlines():
            if line.strip().startswith("회상 문장:"):
                reminder_text = line.split("회상 문장:")[1].strip()
                break

        if not reminder_text:
            raise HTTPException(status_code=400, detail="회상 문장이 없습니다.")

        reminder_mp3 = f"reminder_{uuid4().hex}.mp3"
        text_to_speech(reminder_text, voice_id, reminder_mp3)
        process_audio_speed(reminder_mp3, reminder_mp3, speed=0.83)

        blob = bucket.blob(f"tts/{guardian_uid}/{patient_uid}/{reminder_mp3}")
        blob.upload_from_filename(reminder_mp3)
        reminder_url = f"https://storage.googleapis.com/{bucket.name}/tts/reminder/{guardian_uid}/{patient_uid}/{reminder_mp3}"

        os.remove(reminder_mp3)

        return {
            "message": "회상 문장 생성 완료",
            "reminder_text": reminder_text,
            "tts_url": reminder_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
