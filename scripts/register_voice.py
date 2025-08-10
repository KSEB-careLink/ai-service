import os
import shutil
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from io import BytesIO
from firebase.firebase_init import db, bucket
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from uuid import uuid4
import traceback
import subprocess
import torchaudio
import torchaudio.transforms as T
from voicefixer import VoiceFixer
import torch
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

router = APIRouter()

# 🔑 환경 변수 로딩
load_dotenv()
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# ✅ 시스템에 설치된 ffmpeg 경로 사용
FFMPEG_PATH = shutil.which("ffmpeg")
if not FFMPEG_PATH:
    raise RuntimeError("❌ ffmpeg not found in system PATH")

print("✅ Using ffmpeg at:", FFMPEG_PATH)

def update_firestore_voice_id(guardian_uid: str, new_voice_id: str):
    """ Firestore 보호자 Document에 voiceId 저장 """
    try:
        guardian_ref = db.collection("guardians").document(guardian_uid)
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
        print(f"🔄 변환: {mp3_path} → {wav_path} | 크기: {os.path.getsize(wav_path)/1024:.2f} KB")
        return wav_path

    def apply_voicefixer(wav_path: str) -> str:
        vf = VoiceFixer()
        cleaned_wav = wav_path.replace(".wav", "_vf.wav")
        vf.restore(input=wav_path, output=cleaned_wav, cuda=True, mode=1)
        print(f"✨ VoiceFixer 적용 완료: {cleaned_wav} | 크기: {os.path.getsize(cleaned_wav)/1024:.2f} KB")
        return cleaned_wav

    def apply_vad(wav_path: str) -> str:
        waveform, sample_rate = torchaudio.load(wav_path)
        vad = T.Vad(sample_rate=sample_rate)
        voiced = vad(waveform)

        if voiced.abs().sum() < 1e-3:
            raise ValueError("🛑 VAD 결과가 무음입니다. 음성 입력 확인 요망.")

        voiced_path = wav_path.replace(".wav", "_vad.wav")
        torchaudio.save(voiced_path, voiced, sample_rate)
        print(f"🎙️ VAD 적용 완료: {voiced_path} | 크기: {os.path.getsize(voiced_path)/1024:.2f} KB")
        return voiced_path

    def to_final_mp3(wav_path: str) -> str:
        mp3_path = wav_path.replace(".wav", "_final.mp3")
        subprocess.run([
            FFMPEG_PATH, "-y", "-i", wav_path,
            "-af", "highpass=f=300, lowpass=f=3000",
            "-ar", "22050", "-ac", "1", "-b:a", "64k",
            mp3_path
        ], check=True)

        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) < 1024:
            raise RuntimeError(f"❌ 최종 mp3 파일이 비정상입니다: {mp3_path}")

        print(f"✅ 최종 MP3 생성: {mp3_path} | 크기: {os.path.getsize(mp3_path)/1024:.2f} KB")
        return mp3_path

    # 🧪 순차적 처리
    wav = mp3_to_wav(input_mp3)
    cleaned = apply_voicefixer(wav)
    voiced = apply_vad(cleaned)
    final_mp3 = to_final_mp3(voiced)

    # ✅ 임시 파일 삭제
    for path in [wav, cleaned, voiced]:
        try:
            os.remove(path)
        except:
            print(f"⚠️ 임시 파일 삭제 실패: {path}")

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
        # ✅ 업로드된 파일 저장
        with open(temp_filename, "wb") as buffer:
            data = await file.read()
            buffer.write(data)
            print(f"📥 업로드 받은 파일 저장: {temp_filename} | 크기: {len(data)/1024:.2f} KB")

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
        # ✅ 임시 파일 정리
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if cleaned_path and os.path.exists(cleaned_path):
            os.remove(cleaned_path)
