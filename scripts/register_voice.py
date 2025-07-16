import os

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from io import BytesIO
from firebase.firebase_init import db, bucket
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from uuid import uuid4
import traceback

from voicefixer.voicefixer import VoiceFixer
import subprocess
import torchaudio
import torchaudio.transforms as T

router = APIRouter()

load_dotenv()
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# ✅ ffmpeg 절대 경로
FFMPEG_PATH = "C:/Program Files/ffmpeg-7.0.2-essentials_build/ffmpeg-7.0.2-essentials_build/bin/ffmpeg.exe"  # ← 본인 PC의 경로로 수정
print("✅ FFMPEG_PATH:", FFMPEG_PATH)
print("✅ 존재 여부:", os.path.exists(FFMPEG_PATH))

def update_firestore_voice_id(guardian_uid: str, new_voice_id: str):
    """ Firestore 보호자 Document에 voiceId 저장 """
    try:
        guardian_ref = db.collection("users").document(guardian_uid)
        guardian_ref.update({"voiceId": new_voice_id})
        print(f"✅ Firestore에 voiceId 저장 완료! (guardian_uid: {guardian_uid})")
    except Exception as e:
        print("❌ Firestore 업데이트 실패:", e)

def register_voice(file_path: str, voice_name: str, guardian_uid: str):
    """ 음성 파일을 ElevenLabs에 등록 """
    with open(file_path, "rb") as f:
        audio_bytes = BytesIO(f.read())

    voice = elevenlabs.voices.ivc.create(
        name=voice_name,
        files=[audio_bytes]
    )

    new_voice_id = voice.voice_id
    print("✅ Voice 등록 완료! 새 Voice ID:", new_voice_id)
    return new_voice_id

def preprocess_for_elevenlabs(input_mp3: str) -> str:
    """ VoiceFixer 및 VAD 등 전처리 후 mp3 생성 """

    def mp3_to_wav(mp3_path: str) -> str:
        wav_path = mp3_path.replace(".mp3", ".wav")
        subprocess.run([FFMPEG_PATH, "-y", "-i", mp3_path, wav_path], check=True)
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
            FFMPEG_PATH, "-y", "-i", wav_path,
            "-af", "highpass=f=300, lowpass=f=3000",
            "-ar", "22050", "-ac", "1", "-b:a", "64k",
            mp3_path
        ], check=True)
        return mp3_path

    wav = mp3_to_wav(input_mp3)
    cleaned = apply_voicefixer(wav)
    voiced = apply_vad(cleaned)
    final_mp3 = to_final_mp3(voiced)

    for path in [wav, cleaned, voiced]:
        try: os.remove(path)
        except: pass

    return final_mp3

@router.post("/register-voice")
async def register_voice_endpoint(
    guardian_uid: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...)
):
    temp_filename = f"temp_{uuid4().hex}.mp3"
    cleaned_path = None

    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # ✅ 전처리
        cleaned_path = preprocess_for_elevenlabs(temp_filename)

        # ✅ Firebase Storage 업로드
        cleaned_blob = bucket.blob(f"cleaned_voice/{guardian_uid}/{os.path.basename(cleaned_path)}")
        cleaned_blob.upload_from_filename(cleaned_path)

        # ✅ ElevenLabs 등록
        new_voice_id = register_voice(cleaned_path, name, guardian_uid)

        # ✅ Firestore에 voiceId 저장
        update_firestore_voice_id(guardian_uid, new_voice_id)

        return {
            "message": "보호자 목소리 등록 완료!",
            "voice_id": new_voice_id
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if cleaned_path and os.path.exists(cleaned_path):
            os.remove(cleaned_path)
