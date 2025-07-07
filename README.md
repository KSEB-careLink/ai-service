# 🧠 AI 기반 치매 회상 유도 백엔드 시스템

이 프로젝트는 치매 환자의 기억 회복을 돕기 위해 보호자의 음성을 기반으로 맞춤 회상 문장과 문제를 생성하고, 음성으로 전달하는 FastAPI 기반의 백엔드 시스템입니다.

---

## 🚀 주요 기능

- ✅ 보호자 음성 업로드 및 Voice ID 생성
- ✅ GPT 기반 회상 문장 & 퀴즈 생성
- ✅ ElevenLabs TTS를 통한 음성 합성 (mp3)
- ✅ Firebase Storage에 음성 파일 저장
- ✅ Firestore에 문장/퀴즈/URL/메타데이터 저장
- ✅ 모든 작업을 `/generate-and-read` 하나의 API에서 처리

---

## 🛠️ 기술 스택

| 분야 | 기술 |
|------|------|
| 언어 | Python 3.10+ |
| 웹 프레임워크 | FastAPI |
| AI 모델 | OpenAI GPT-4 (via `openai` 라이브러리) |
| 음성 합성 | ElevenLabs Text-to-Speech API |
| 음성 클로닝 | ElevenLabs Voice Lab / Voice Create API |
| 데이터베이스 | Firebase Firestore |
| 파일 저장소 | Firebase Storage |
| 인증 키 관리 | `dotenv` 환경변수 로딩 (.env) |
| 기타 | uuid, requests, os, traceback 등 |

---

## 📁 프로젝트 구조
ai_service/
├── firebase/
│ └── firebase_init.py # Firebase 초기화
├── llm/
│ └── gpt_client.py # GPT 회상 문장 생성
├── scripts/
│ └── register_voice.py # 보호자 음성 등록 → voice_id 생성
├── tts/
│ ├── init.py
│ └── elevenlabs_client.py # ElevenLabs TTS 기능
├── main.py # FastAPI 진입점
├── .gitignore
└── README.md

