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
from voicefixer.voicefixer import VoiceFixer
import subprocess
import torchaudio
import torchaudio.transforms as T
import re
from tts.elevenlabs_client import text_to_speech, create_voice, process_audio_speed

app = FastAPI()

db = firestore.client()
bucket = storage.bucket()

# ✅ 고급 전처리 함수: VoiceFixer + VAD + mp3 변환

def preprocess_for_elevenlabs(input_mp3: str) -> str:
    def mp3_to_wav(mp3_path: str) -> str:
        wav_path = mp3_path.replace(".mp3", ".wav")
        subprocess.run(["ffmpeg", "-y", "-i", mp3_path, wav_path], check=True)
        return wav_path

    def apply_voicefixer(wav_path: str) -> str:
        vf = VoiceFixer()
        cleaned_wav = wav_path.replace(".wav", "_vf.wav")
        vf.restore(input=wav_path, output=cleaned_wav, cuda=False, mode=1)
        return cleaned_wav

    def apply_vad(wav_path: str) -> str:
        waveform, sample_rate = torchaudio.load(wav_path)
        vad = T.Vad(sample_rate=sample_rate)
        voiced = vad(waveform)
        voiced_path = wav_path.replace(".wav", "_vad.wav")
        torchaudio.save(voiced_path, voiced, sample_rate)
        return voiced_path

    def to_final_mp3(wav_path: str) -> str:
        mp3_path = wav_path.replace(".wav", "_final.mp3")
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-af", "highpass=f=300, lowpass=f=3000",
            "-ar", "22050", "-ac", "1", "-b:a", "64k",
            mp3_path
        ], check=True)
        return mp3_path

    wav = mp3_to_wav(input_mp3)
    cleaned = apply_voicefixer(wav)
    voiced = apply_vad(cleaned)
    final_mp3 = to_final_mp3(voiced)

    # cleanup
    for path in [wav, cleaned, voiced]:
        try: os.remove(path)
        except: pass

    return final_mp3

class ReminderInput(BaseModel):
    patient_name: str
    photo_description: str
    tone: ToneEnum

class TTSRequest(BaseModel):
    text: str
    voice_id: str

@app.post("/generate-and-read")
async def generate_and_read(
    name: str = Form(...),
    file: UploadFile = File(...),
    patient_name: str = Form(...),
    photo_description: str = Form(...),
    relationship: str = Form(...),
    tone: ToneEnum = Form(...)
):
    try:
        user_id = "test_user"

        # ✅ Form으로 받은 relationship을 우선 사용하고, 비어 있으면 Firestore fallback
        if not relationship:
            profile_doc = db.collection("users").document(user_id).collection("profile").document("info").get()
            if profile_doc.exists:
                relationship = profile_doc.to_dict().get("relationship", "보호자")
            else:
                relationship = "어르신"

        # 1. 보호자 음성 등록
        temp_filename = f"temp_{uuid4().hex}.mp3"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        cleaned_path = preprocess_for_elevenlabs(temp_filename)

        cleaned_blob = bucket.blob(f"cleaned_voice/{user_id}/{os.path.basename(cleaned_path)}")
        cleaned_blob.upload_from_filename(cleaned_path)

        voice_id = register_voice(cleaned_path, name, guardian_uid=user_id)

        os.remove(temp_filename)
        try:
            os.remove(cleaned_path)
        except FileNotFoundError:
            print(f"파일 삭제 실패: {cleaned_path}")

        # 2. 회상 문장 및 퀴즈 생성
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
                    quiz_answer = raw  # 혹시 포맷이 달라도 대비
            elif capture_options:
                match = re.match(r"\d+번[.,]?\s*(.+)", line)
                if match:
                    quiz_options.append(match.group(1).strip())

        print("🎯 파싱된 선택지 목록:", quiz_options)

        # 3. mp3 생성
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

        # 4. Firebase 업로드
        reminder_blob = bucket.blob(f"tts/{user_id}/{reminder_mp3}")
        reminder_blob.upload_from_filename(reminder_mp3)
        reminder_url = f"https://storage.googleapis.com/{bucket.name}/tts/{user_id}/{reminder_mp3}"

        quiz_blob = bucket.blob(f"tts/{user_id}/{quiz_mp3}")
        quiz_blob.upload_from_filename(quiz_mp3)
        quiz_url = f"https://storage.googleapis.com/{bucket.name}/tts/{user_id}/{quiz_mp3}"

        # 5. Firestore 저장
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
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        res= db.collection("users").document(user_id).collection("reminders").add(doc_data)
        print("✅ Firestore 저장 완료:", res)

        # 6. 정리
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

# @app.post("/generate-only")
# async def generate_only(
#     patient_name: str = Form(...),
#     photo_description: str = Form(...),
#     relationship: str = Form(...),
#     tone: ToneEnum = Form(...)
# ):
#     try:
#         user_id = "test_user"

#         # 회상 문장 및 퀴즈 생성
#         result = generate_reminder(
#             patient_name=patient_name,
#             photo_description=photo_description,
#             relation=relationship,
#             tone=tone
#         )

#         print("🧠 GPT 응답 결과:\n", result)

#         # 파싱
        
#         reminder_text = ""
#         quiz_question = ""
#         quiz_options = []
#         quiz_answer = ""
#         capture_options = False

#         for line in result.strip().splitlines():
#             line = line.strip()
#             if line.startswith("회상 문장:"):
#                 reminder_text = line.split("회상 문장:")[1].strip()
#             elif line.startswith("퀴즈 문제:"):
#                 quiz_question = line.split("퀴즈 문제:")[1].strip()
#             elif line.startswith("선택지:"):
#                 capture_options = True
#             elif line.startswith("정답:"):
#                 capture_options = False
#                 raw = line.split("정답:")[1].strip()
#                 match = re.match(r"\d+번[.,]?\s*(.+)", raw)
#                 if match:
#                     quiz_answer = match.group(1).strip()
#                 else:
#                     quiz_answer = raw  # 혹시 포맷이 달라도 대비
#             elif capture_options:
#                 match = re.match(r"\d+번[.,]?\s*(.+)", line)
#                 if match:
#                     quiz_options.append(match.group(1).strip())

#         # for line in result.strip().splitlines():
#         #     line = line.strip()
#         #     if line.startswith("회상 문장:"):
#         #         reminder_text = line.split("회상 문장:")[1].strip()
#         #     elif line.startswith("퀴즈 문제:"):
#         #         quiz_question = line.split("퀴즈 문제:")[1].strip()
#         #     elif line.startswith("선택지:"):
#         #         # 👇 쉼표 기준으로 split (GPT가 1번, 계곡, 2번, 바다 식으로 응답할 경우 대응)
#         #         raw = line.replace("선택지:", "").strip()
#         #         items = re.findall(r"\d+번[.,]?\s*[^,\n]+", raw)
#         #         for item in items:
#         #             opt = re.sub(r"^\d+번[.,]?\s*", "", item).strip()
#         #             quiz_options.append(opt)
#         #     elif line.startswith("정답:"):
#         #         # 정답: 1번, 계곡 → "계곡"만 뽑기
#         #         ans_raw = line.split("정답:")[1].strip()
#         #         answer_match = re.match(r"\d+번[.,]?\s*(.+)", ans_raw)
#         #         if answer_match:
#         #             quiz_answer = answer_match.group(1).strip()
#         #     elif capture_options:
#         #         match = re.match(r"^\d+번[.\s]+(.+)", line)
#         #         if match:
#         #             quiz_options.append(match.group(1).strip())


#         print("🎯 파싱된 선택지 목록:", quiz_options)

#         if not quiz_answer:
#             raise HTTPException(status_code=400, detail="GPT 응답에 퀴즈 정답이 없습니다.")

#         # Firestore 저장
#         doc_data = {
#             "reminder_text": reminder_text,
#             "quiz_question": quiz_question,
#             "quiz_options": quiz_options,
#             "quiz_answer": quiz_answer,
#             "created_at": firestore.SERVER_TIMESTAMP,
#         }
#         res = db.collection("users").document(user_id).collection("reminders").add(doc_data)
#         print("✅ Firestore 저장 완료:", res)

#         return {
#             "message": "문장 및 퀴즈 Firestore 저장 완료",
#             "reminder": reminder_text,
#             "question": quiz_question,
#             "options": quiz_options,
#             "answer": quiz_answer,
#         }

#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))
