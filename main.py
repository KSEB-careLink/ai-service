from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from firebase.firebase_init import bucket
from firebase_admin import firestore, storage
from uuid import uuid4
import os
import traceback
from enums import ToneEnum
from llm.gpt_client import generate_reminder
from scripts.register_voice import register_voice
from scripts.register_voice import router as voice_router 
from voicefixer.voicefixer import VoiceFixer
import subprocess
import torchaudio
import torchaudio.transforms as T
import re
from tts.elevenlabs_client import text_to_speech, create_voice, process_audio_speed

app = FastAPI()

db = firestore.client()
bucket = storage.bucket()

app.include_router(voice_router)

class ReminderInput(BaseModel):
    patient_name: str
    photo_description: str
    tone: ToneEnum

class TTSRequest(BaseModel):
    text: str
    voice_id: str

@app.post("/generate-and-read")
async def generate_and_read(
    guardian_uid: str = Form(...),
    patient_uid: str = Form(...),
    voice_id: str = Form(...),  # ✅ Node.js에서 가져와 전달
    patient_name: str = Form(...),
    photo_description: str = Form(...),
    relationship: str = Form(...),
    tone: ToneEnum = Form(...)
):
    try:
        # ✅ Form으로 받은 relationship 우선 사용
        if not relationship:
            profile_doc = db.collection("users").document(guardian_uid).collection("profile").document("info").get()
            if profile_doc.exists:
                relationship = profile_doc.to_dict().get("relationship", "보호자")
            else:
                relationship = "어르신"

        # 1. 회상 문장 및 퀴즈 생성
        result = generate_reminder(
            patient_name=patient_name,
            photo_description=photo_description,
            relation=relationship,
            tone=tone
        )

        print("🧠 GPT 응답 결과:\n", result)

        # 파싱
        reminder_text = ""
        quiz_question = ""
        quiz_options = []
        quiz_answer = ""
        capture_options = False

        for line in result.strip().splitlines():
            line = line.strip()
            if line.startswith("회상 문장:"):
                reminder_text = line.split("회상 문장:")[1].strip()
            elif line.startswith("퀴즈 문제:"):
                quiz_question = line.split("퀴즈 문제:")[1].strip()
            elif line.startswith("선택지:"):
                capture_options = True
            elif line.startswith("정답:"):
                capture_options = False
                raw = line.split("정답:")[1].strip()
                match = re.match(r"\d+번[.,]?\s*(.+)", raw)
                if match:
                    quiz_answer = match.group(1).strip()
                else:
                    quiz_answer = raw
            elif capture_options:
                match = re.match(r"\d+번[.,]?\s*(.+)", line)
                if match:
                    quiz_options.append(match.group(1).strip())

        print("🎯 파싱된 선택지 목록:", quiz_options)

        # 2. mp3 생성
        reminder_mp3 = f"reminder_{uuid4().hex}.mp3"
        text_to_speech(reminder_text, voice_id, reminder_mp3)
        process_audio_speed(reminder_mp3, reminder_mp3, speed=0.83)

        readable_nums = {1: "첫 번째", 2: "두 번째", 3: "세 번째", 4: "네 번째"}
        options_text = "\n".join([
            f"{readable_nums[i+1]}, {opt}" for i, opt in enumerate(quiz_options)
        ])
        quiz_text = f"{quiz_question}\n{options_text}"

        quiz_mp3 = f"quiz_{uuid4().hex}.mp3"
        text_to_speech(quiz_text, voice_id, quiz_mp3)
        process_audio_speed(quiz_mp3, quiz_mp3, speed=0.83)

        # 3. Firebase Storage 업로드
        reminder_blob = bucket.blob(f"tts/{guardian_uid}/{patient_uid}/{reminder_mp3}")
        reminder_blob.upload_from_filename(reminder_mp3)
        reminder_url = f"https://storage.googleapis.com/{bucket.name}/tts/{guardian_uid}/{patient_uid}/{reminder_mp3}"

        quiz_blob = bucket.blob(f"tts/{guardian_uid}/{patient_uid}/{quiz_mp3}")
        quiz_blob.upload_from_filename(quiz_mp3)
        quiz_url = f"https://storage.googleapis.com/{bucket.name}/tts/{guardian_uid}/{patient_uid}/{quiz_mp3}"


        # 4. Firestore 저장
        print("📝 정답 내용 확인:", quiz_answer)

        if not quiz_answer:
            raise HTTPException(status_code=400, detail="GPT 응답에 퀴즈 정답이 없습니다.")

        doc_data = {
            "reminder_text": reminder_text,
            "quiz_question": quiz_question,
            "quiz_options": quiz_options,
            "quiz_answer": quiz_answer,
            "tts_url": reminder_url,
            "quiz_tts_url": quiz_url,
            "voice_id": voice_id,
            "guardian_id": guardian_uid,
            "patient_id": patient_uid,
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        res = db.collection("users").document(patient_uid).collection("reminders").add(doc_data)
        print("✅ Firestore 저장 완료:", res)

        os.remove(reminder_mp3)
        os.remove(quiz_mp3)

        return {
            "message": "회상 문장 + 퀴즈 + mp3 + 저장 완료",
            "reminder": reminder_text,
            "question": quiz_question,
            "tts_url": reminder_url,
            "quiz_tts_url": quiz_url,
            "voice_id": voice_id
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
