import os
import requests
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def text_to_speech(text: str, voice_id: str, file_name="output.mp3"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "model_id": "eleven_multilingual_v2",
        "text": text,
        "voice_settings": {
            "stability": 0.35,
            "similarity_boost": 0.75,
            "style" : 0.35,
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        with open(file_name, "wb") as f:
            f.write(response.content)
        return file_name
    else:
        print("❌ TTS 실패:", response.status_code, response.text)
        return None
    
def create_voice(voice_name: str, file_path: str):
    url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    files = {"files": open(file_path, "rb")}
    data = {"name": voice_name, "description": "보호자 음성 자동 등록"}

    response = requests.post(url, headers=headers, files=files, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        print("Voice 등록 실패:", response.status_code, response.text)
        return None
    
import subprocess


def process_audio_speed(input_path: str, output_path: str, speed=0.8):
    # 임시 출력 파일
    tmp_output = f"tmp_{output_path}"

    # ffmpeg 명령어
    cmd = ["ffmpeg", "-i", input_path, "-filter:a", f"atempo={speed}", tmp_output, "-y"]
    subprocess.run(cmd, check=True)

    # 기존 파일 덮어쓰기
    os.replace(tmp_output, output_path)

    print(f"✅ 속도 조절된 파일 저장 완료 (ffmpeg atempo): {output_path}")
    return output_path


