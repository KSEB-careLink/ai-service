# routes/quiz.py
from fastapi import APIRouter, Form, HTTPException
from firebase_admin import firestore, storage
from firebase.firebase_init import bucket
from uuid import uuid4
from llm.gpt_client import generate_reminder
from tts.elevenlabs_client import text_to_speech, process_audio_speed
from enums import ToneEnum
import os
import re

router = APIRouter()

db = firestore.client()

@router.post("/generate-quiz")
async def generate_quiz_endpoint(
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

        quiz_question = ""
        quiz_options = []
        quiz_answer = ""
        capture_options = False

        for line in result.strip().splitlines():
            line = line.strip()
            if line.startswith("퀴즈 문제:"):
                quiz_question = line.split("퀴즈 문제:")[1].strip()
            elif line.startswith("선택지:"):
                capture_options = True
            elif line.startswith("정답:"):
                capture_options = False
                raw = line.split("정답:")[1].strip()
                match = re.match(r"\d+번[.,]?\s*(.+)", raw)
                quiz_answer = match.group(1).strip() if match else raw
            elif capture_options:
                match = re.match(r"\d+번[.,]?\s*(.+)", line)
                if match:
                    quiz_options.append(match.group(1).strip())

        if not quiz_question or not quiz_answer or not quiz_options:
            raise HTTPException(status_code=400, detail="퀴즈 정보가 부족합니다.")

        quiz_text = f"{quiz_question}\n" + "\n".join(
            [f"{i+1}번, {opt}" for i, opt in enumerate(quiz_options)]
        )

        quiz_mp3 = f"quiz_{uuid4().hex}.mp3"
        text_to_speech(quiz_text, voice_id, quiz_mp3)
        process_audio_speed(quiz_mp3, quiz_mp3, speed=0.83)

        blob = bucket.blob(f"tts/{guardian_uid}/{patient_uid}/{quiz_mp3}")
        blob.upload_from_filename(quiz_mp3)
        quiz_url = f"https://storage.googleapis.com/{bucket.name}/tts/quiz/{guardian_uid}/{patient_uid}/{quiz_mp3}"

        os.remove(quiz_mp3)

        return {
            "message": "퀴즈 생성 완료",
            "question": quiz_question,
            "options": quiz_options,
            "answer": quiz_answer,
            "quiz_tts_url": quiz_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
