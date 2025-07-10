from firebase.firebase_init import bucket
import os
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from llm.gpt_client import generate_reminder
from tts.elevenlabs_client import text_to_speech, create_voice, process_audio_speed
from scripts.register_voice import register_voice

# 🔥 Firebase 연결
import firebase.firebase_init  # firebase_init.py에서 초기화
from firebase_admin import firestore, storage

from fastapi import HTTPException
import traceback
from enums import ToneEnum

app = FastAPI()

db = firestore.client()
bucket = storage.bucket()

# ✅ 회상 문장 입력용 모델
class ReminderInput(BaseModel):
    patient_name: str
    photo_description: str
    tone: ToneEnum

# ✅ TTS 요청용 모델
class TTSRequest(BaseModel):
    text: str
    voice_id: str

# 🔹 전체 통합 API: 음성 등록 + 회상 문장 + 퀴즈 + TTS + Firebase 저장
@app.post("/generate-and-read")
async def generate_and_read(
    name: str = Form(...),  # 🔹 보호자 이름 추가
    file: UploadFile = File(...),  # 🔹 보호자 음성 파일
    patient_name: str = Form(...),
    photo_description: str = Form(...),
    relationship: str = Form(...),
    tone: ToneEnum = Form(...)  # 🔹 말투를 Enum으로 제한
):
    try:
        user_id = "test_user"  # Firebase Auth 연동 전까지는 임시

        # 🔎 Firestore에서 user profile 문서 가져오기
        profile_doc = db.collection("users").document(user_id).collection("profile").document("info").get()
        if profile_doc.exists:
            relationship = profile_doc.to_dict().get("relationship", "보호자")
        else:
            relationship = "보호자"  # 기본값

        # 1. 보호자 음성 등록
        temp_filename = f"temp_{uuid4().hex}.mp3"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        voice_id = register_voice(temp_filename, name)
        os.remove(temp_filename)

        # 2. GPT로 회상 문장 + 퀴즈 생성
        prompt = f"""
        당신은 치매 환자의 회상을 도와주는 회상 도우미입니다.

        다음 정보를 바탕으로 회상 문장과 적절한 유형의 객관식 퀴즈 문제를 생성해주세요.

        - 환자 이름: {patient_name}
        - 사진 설명: {photo_description}
        - 보호자와의 관계: {relationship}
        - 보호자의 말투: {tone}

        퀴즈 유형은 다음 중 하나를 자동으로 선택해서 생성해주세요:
        1. 이름 맞추기 – 사진 속 사람, 장소, 물건의 이름을 맞추는 문제
        2. 시각 회상 – 사진 배경이나 상황을 설명하고 기억을 유도하는 문제
        3. 자유 회상 – 사진을 보고 떠오를 수 있는 기억을 기반으로 보기 4개를 주고, 가장 관련 있는 것을 고르는 **객관식 퀴즈**로 만들어주세요

        환자가 이해하기 쉽게 다정하고 천천히 말하는 어조로 구성해주세요.

        출력 형식:
        회상 문장: ...
        퀴즈 유형: ...
        퀴즈 문제: ...
        선택지: 보기1, 보기2, 보기3, 보기4
        정답: ...
        """
        # 이 줄을 수정 👇
        result = generate_reminder(prompt, relation=relationship, tone=tone)

        print("🧠 GPT 응답 결과:\n", result)

        # 🔧 파싱: 줄 순서 상관없이 안전하게 분리
        reminder_text = ""
        quiz_question = ""
        quiz_options = []
        quiz_answer = ""

        for line in result.strip().splitlines():
            if line.startswith("회상 문장:"):
                reminder_text = line.replace("회상 문장:", "").strip()
            elif line.startswith("퀴즈 문제:"):
                quiz_question = line.replace("퀴즈 문제:", "").strip()
            elif line.startswith("선택지:"):
                quiz_options = line.replace("선택지:", "").strip().split(", ")
            elif line.startswith("정답:"):
                quiz_answer = line.replace("정답:", "").strip()

        # 3. 회상 문장 mp3 생성
        reminder_mp3 = f"reminder_{uuid4().hex}.mp3"
        text_to_speech(reminder_text, voice_id, reminder_mp3)
        process_audio_speed(reminder_mp3, reminder_mp3, speed=0.9)

        # 4. 퀴즈 문제 mp3 생성
        quiz_text = f"{quiz_question} " + " ".join([f"{i+1}번 {opt}" for i, opt in enumerate(quiz_options)])
        quiz_mp3 = f"quiz_{uuid4().hex}.mp3"
        text_to_speech(quiz_text, voice_id, quiz_mp3)
        process_audio_speed(quiz_mp3, quiz_mp3, speed=0.9)

        # 5. Firebase Storage에 mp3 업로드
        reminder_blob = bucket.blob(f"tts/{user_id}/{reminder_mp3}")
        reminder_blob.upload_from_filename(reminder_mp3)
        reminder_url = f"https://storage.googleapis.com/{bucket.name}/tts/{user_id}/{reminder_mp3}"

        quiz_blob = bucket.blob(f"tts/{user_id}/{quiz_mp3}")
        quiz_blob.upload_from_filename(quiz_mp3)
        quiz_url = f"https://storage.googleapis.com/{bucket.name}/tts/{user_id}/{quiz_mp3}"

        # 6. Firestore에 문장 + 문제 + mp3 정보 저장
        doc_data = {
            "reminder_text": reminder_text,
            "quiz_question": quiz_question,
            "quiz_options": quiz_options,
            "quiz_answer": quiz_answer,
            "tts_url": reminder_url,
            "quiz_tts_url": quiz_url,
            "voice_id": voice_id,
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        print("📝 저장할 데이터:", doc_data)
        db.collection("users").document(user_id).collection("reminders").add(doc_data)

        # 7. 임시 파일 정리
        os.remove(reminder_mp3)
        os.remove(quiz_mp3)

        return {
            "message": "회상 문장 + 퀴즈 + mp3 + 저장 완료",
            "reminder": reminder_text,
            "question": quiz_question,
            "tts_url": reminder_url,
            "quiz_tts_url": quiz_url,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
