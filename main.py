from fastapi import FastAPI
from firebase_admin import firestore
from firebase.firebase_init import bucket

# 라우터 모듈들
from scripts.register_voice import router as voice_router
from routes.reminder import router as reminder_router
from routes.quiz import router as quiz_router
from routes.predict_accuracy import router as predict_router 

app = FastAPI()

@app.get("/")
def root():
    return {"message": "CareLink Server is Live!"}

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

# Firebase 초기화
db = firestore.client()
bucket = bucket

# 라우터 등록
app.include_router(voice_router)
app.include_router(reminder_router)
app.include_router(quiz_router)
app.include_router(predict_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

