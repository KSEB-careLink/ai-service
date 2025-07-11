import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from io import BytesIO
from firebase.firebase_init import db  # ✅ Firestore 가져오기

load_dotenv()
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def update_firestore_voice_id(guardian_uid: str, new_voice_id: str):
    """ Firestore 보호자 Document에 voiceId 저장 """
    try:
        guardian_ref = db.collection("users").document(guardian_uid)
        guardian_ref.update({"voiceId": new_voice_id})
        print(f"✅ Firestore에 voiceId 저장 완료! (guardian_uid: {guardian_uid})")
    except Exception as e:
        print("❌ Firestore 업데이트 실패:", e)

def register_voice(file_path: str, voice_name: str, guardian_uid: str, save_to_firestore: bool = True):
    """ 
    음성 파일을 ElevenLabs에 등록하고, Firestore에 voiceId 저장 
    """
    with open(file_path, "rb") as f:
        audio_bytes = BytesIO(f.read())

    voice = elevenlabs.voices.ivc.create(
        name=voice_name,
        files=[audio_bytes]
    )

    new_voice_id = voice.voice_id
    print("✅ Voice 등록 완료!")
    print("새 Voice ID:", new_voice_id)

    if save_to_firestore:
        update_firestore_voice_id(guardian_uid, new_voice_id)

    return new_voice_id
