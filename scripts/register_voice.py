import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from io import BytesIO

load_dotenv()

# ElevenLabs 클라이언트 초기화
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def update_env_voice_id(new_voice_id: str, env_path=".env"):
    """ .env 파일에 ELEVENLABS_VOICE_ID를 자동 갱신하는 함수 """
    with open(env_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    found = False
    for i, line in enumerate(lines):
        if line.startswith("ELEVENLABS_VOICE_ID="):
            lines[i] = f"ELEVENLABS_VOICE_ID={new_voice_id}\n"
            found = True
            break

    if not found:
        lines.append(f"\nELEVENLABS_VOICE_ID={new_voice_id}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("✅ .env 파일에 Voice ID가 자동으로 갱신되었습니다!")

def register_voice(file_path: str, voice_name: str, env_update: bool = True):
    """ 
    음성 파일을 ElevenLabs에 등록하고, 필요하면 .env에 자동 저장하는 함수 
    """
    with open(file_path, "rb") as f:
        audio_bytes = BytesIO(f.read())

    # Voice 등록 요청
    voice = elevenlabs.voices.ivc.create(
        name=voice_name,
        files=[audio_bytes]
    )

    new_voice_id = voice.voice_id
    print("✅ Voice 등록 완료!")
    print("새 Voice ID:", new_voice_id)

    if env_update:
        update_env_voice_id(new_voice_id)

    return new_voice_id
