from fastapi import APIRouter, Form, HTTPException
from firebase_admin import firestore
from firebase.firebase_init import bucket
from tts.elevenlabs_client import text_to_speech, process_audio_speed
from uuid import uuid4
import os
import math

router = APIRouter()
db = firestore.client() 

@router.post("/speech")
async def speech_endpoint(
    guardian_uid: str = Form(...),
    patient_uid: str = Form(...),
    text: str = Form(...),
    voice_id: str = Form(...),
    speed: float = Form(0.83),  
    file_prefix: str = Form("tts/sample")  
):
    text = (text or "").strip()
    guardian_uid = (guardian_uid or "").strip()
    patient_uid = (patient_uid or "").strip()
    voice_id = (voice_id or "").strip()
    if not (text and guardian_uid and patient_uid and voice_id):
        raise HTTPException(status_code=400, detail="필수 입력이 누락되었습니다.")

    if not math.isfinite(speed):
        speed = 0.83
    speed = max(0.5, min(2.0, float(speed)))

    uid = uuid4().hex
    tmp_dir = "/tmp"
    raw_name = f"speech_{uid}.mp3"
    raw_path = os.path.join(tmp_dir, raw_name)
    final_name = f"speech_{uid}_final.mp3"
    final_path = os.path.join(tmp_dir, final_name)

    blob_path = f"{file_prefix}/{guardian_uid}/{patient_uid}/{final_name}"

    try:
        # 1) ElevenLabs 합성 → 로컬 mp3 저장
        result_path = text_to_speech(text=text, voice_id=voice_id, file_name=raw_path)
        if not result_path or not os.path.exists(result_path):
            raise HTTPException(status_code=502, detail="TTS 변환 실패(파일 미생성)")

        # 2) 후처리(atempo=속도조절)
        process_audio_speed(input_path=raw_path, output_path=final_path, speed=speed)
        if not os.path.exists(final_path):
            raise HTTPException(status_code=500, detail="후처리 실패(파일 미생성)")

        # 3) Firebase Storage 업로드(+ 공개 URL)
        blob = bucket.blob(blob_path)
        blob.cache_control = "public, max-age=31536000"
        blob.upload_from_filename(final_path, content_type="audio/mpeg")
        blob.make_public()
        tts_url = blob.public_url

        return {
            "message": "TTS 변환 및 업로드 완료",
            "tts_url": tts_url,
            "blob_path": blob_path,
            "speed_applied": speed
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/speech 처리 중 오류: {e}")
    finally:
        # 4) 임시 파일 정리
        for p in (raw_path, final_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass

# from fastapi import APIRouter, Form, HTTPException
# from firebase_admin import firestore, storage
# from firebase.firebase_init import bucket
# from tts.elevenlabs_client import text_to_speech, process_audio_speed
# from uuid import uuid4
# import os

# router = APIRouter()
# db = firestore.client()

# @router.post("/speech")
# async def speech_endpoint(
#     guardian_uid: str = Form(...),
#     patient_uid: str = Form(...),
#     text: str = Form(...),
#     voice_id: str = Form(...)
# ):
#     try:
#         # ✅ 텍스트 → mp3 생성
#         mp3_filename = f"speech_{uuid4().hex}.mp3"
#         result_path = text_to_speech(text, voice_id, mp3_filename)

#         if not result_path or not os.path.exists(result_path):
#             raise HTTPException(status_code=500, detail="TTS 변환 실패")

#         # ✅ 속도 조절
#         process_audio_speed(mp3_filename, mp3_filename, speed=0.83)

#         # ✅ Firebase Storage 업로드 (경로 수정됨)
#         blob_path = f"tts/sample/{guardian_uid}/{patient_uid}/{mp3_filename}"
#         blob = bucket.blob(blob_path)
#         blob.upload_from_filename(mp3_filename)
#         blob.make_public()                # 퍼블릭 공개
#         tts_url = blob.public_url         # 공개 URL

#         # ✅ 로컬 파일 삭제
#         os.remove(mp3_filename)

#         return {
#             "message": "TTS 변환 및 업로드 완료",
#             "tts_url": tts_url
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
