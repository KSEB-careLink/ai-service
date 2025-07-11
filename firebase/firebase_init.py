# firebase/firebase_init.py
import os
import firebase_admin
from firebase_admin import credentials, storage, firestore

if not firebase_admin._apps:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "serviceAccountKey.json")
    cred = credentials.Certificate(json_path)

    # ✅ 먼저 초기화
    firebase_admin.initialize_app(cred, {
        "storageBucket": "carelink-a228a.firebasestorage.app"
    })

# ✅ 초기화 이후에 bucket 선언
bucket = storage.bucket()

# ✅ Firestore 클라이언트 선언
db = firestore.client()
