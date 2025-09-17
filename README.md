# 🧠 CareLink AI Backend

치매 환자의 기억 회상을 돕기 위해 **회상 문장/퀴즈 생성 (LLM)**, **음성 합성(TTS)**, **보호자 음성 등록**, **월 단위 정답률 예측(LSTM)** 을 제공하는 FastAPI 백엔드입니다.

---

## 🚀 핵심 기능

### 1) LLM 기반 콘텐츠 생성
- **회상 문장 생성**: `generate_reminder_sentence(patient_name, photo_description, relation, tone, image_path)` 
- **퀴즈 생성(3문항)**: `generate_quiz_only(patient_name, photo_description, relation, tone, image_path)` 
- **키워드 추출**  
  - 설명 기반: `extract_terms(photo_description)`  
  - 이미지 기반: `extract_visible_terms_from_image(image_path)`  

#### 엔드포인트
- `POST /generate-reminder` → 회상 문장 + TTS(mp3) + Firebase 업로드(URL 반환) 
- `POST /generate-quiz` → 3개 퀴즈 + 각 문항 TTS(mp3) + 업로드(URL 반환) 

---

### 2) 음성 합성(TTS)
- **텍스트→음성**: `text_to_speech(text, voice_id, file_name="output.mp3")` 
- **속도 보정**: `process_audio_speed(input_path, output_path, speed=0.8)` 
- **보이스 등록(ElevenLabs)**: `create_voice(voice_name, file_path)` 

#### 엔드포인트
- `POST /speech` → 일반 텍스트를 TTS 변환 → 속도 보정 후 Firebase 업로드(URL 반환)

---

### 3) 보호자 음성 등록 & 전처리
보호자의 음성을 ElevenLabs에 등록하기 전에, **잡음 제거·VAD(Voice Activity Detection)·음질 보정** 과정을 수행합니다.

- **Firestore 업데이트**: `update_firestore_voice_id(guardian_uid, new_voice_id)` → 보호자 Document에 voiceId 저장 
- **ElevenLabs 등록**: `register_voice(file_path, voice_name, guardian_uid)` → 음성 파일을 ElevenLabs Voice로 등록하고 `voice_id` 발급 
- **전처리 파이프라인**: `preprocess_for_elevenlabs(input_mp3)` 
  1. **MP3 → WAV 변환**: `ffmpeg` 사용  
  2. **VoiceFixer 복원**: `vf.restore()` 로 음성 잡음 제거 및 품질 향상  
  3. **VAD(Voice Activity Detection)**: `torchaudio.transforms.Vad` 로 무음 제거 및 발화 구간 추출  
  4. **최종 MP3 변환**: 고역/저역 필터링, 샘플링 레이트 다운샘플링, 모노 채널 변환  

#### 엔드포인트
- `POST /register-voice`
  - 보호자 음성을 업로드  
  - VoiceFixer + VAD 기반 전처리 후 Firebase Storage 업로드  
  - ElevenLabs Voice 등록 → `voice_id` 발급  
  - Firestore에 `voiceId` 저장

---

### 4) 향후 10일 및 월 단위 정답률 예측(LSTM) 
- model_final 안에 있는 README 참고

## 📡 API 엔드포인트 요약

| Method | Path | 설명 |
|--------|------|------|
| POST   | `/generate-reminder` | 회상 문장 생성 → TTS → Firebase 업로드 (URL 반환) |
| POST   | `/generate-quiz` | 퀴즈 3개 생성 → 각 TTS → Firebase 업로드 (URL 반환) |
| POST   | `/speech` | 자유 텍스트 TTS 변환 → Firebase 업로드 (URL 반환) |
| POST   | `/register-voice` | 보호자 음성 **전처리(VoiceFixer+VAD+ffmpeg)** → ElevenLabs 등록 → Firestore 저장 |
| GET    | `/predict-accuracy-live` | 환자 향후 10일, 월 정답률 예측(LSTM) |

---

## 🔧 환경 변수

- `OPENAI_API_KEY` : LLM 호출용  
- `ELEVENLABS_API_KEY` : TTS/보이스 등록 키  
- `NODE_API_BASE_URL`, `NODE_API_TIMEOUT` : Node API 설정  
- `MODEL_DIR`, `MODEL_CKPT` : LSTM 체크포인트 경로  
- Cold-start 관련: `EXPECTED_DAILY_SOLVES`, `BASELINE_ACC`, `COLD_DAYS_OK`, `COLD_ATTEMPTS_OK`  
- `FFMPEG_PATH` : ffmpeg 실행 파일 절대 경로 (VoiceFixer 전처리용)  

---

## 🧪 파이프라인 요약

1. **보호자 음성 등록** (`/register-voice`)  
   → 업로드된 mp3를 **VoiceFixer**로 잡음 제거 및 음질 복원  
   → **VAD(Voice Activity Detection)** 적용해 발화 구간만 추출  
   → ffmpeg 필터링(고역/저역, 다운샘플링) 후 최종 mp3 생성  
   → Firebase Storage 저장 → ElevenLabs Voice 등록(`voice_id` 발급) → Firestore에 `voiceId` 업데이트  

2. **회상 문장/퀴즈 생성** (`/generate-reminder`, `/generate-quiz`)  
   → GPT-4 기반 텍스트 생성 (허용/금칙 단어 규칙 적용)  
   → 보호자 음성(`voice_id`)으로 TTS 변환 → Firebase 업로드(URL 반환)  

3. **자유 TTS 변환** (`/speech`)  
   → 임의 텍스트 입력 → TTS 변환 + 속도 보정 → Firebase 업로드(URL 반환)  

4. **정답률 예측** (`/predict-accuracy-live`)  
   → Node API 로그 기반으로 최근 W일 데이터 집계  
   → LSTM 추론 + Cold-start 보정 → 다음 10일, 이번 달 예상 정답률 반환  

---

