# LSTM 기반 퀴즈 정답률 예측 파이프라인

이 프로젝트는 **환자의 퀴즈 정답률**을 시계열 데이터로 학습하여, 향후 일정 기간(`HORIZON`) 동안의 정답률을 예측하는 LSTM 모델을 구현한 코드입니다.  

---

## 주요 기능

- `quiz_logs.csv` 데이터를 불러와 **일별 정답률/응답시간**을 집계  
- 결측 날짜 자동 보간(`ffill`/`bfill`) 처리  
- **시퀀스 데이터셋** 생성 (입력 구간 `WINDOW`, 예측 구간 `HORIZON`)  
- **LSTM 모델 학습 및 검증** (Early Stopping 적용)  
- **학습 곡선 시각화** (`training_curves.png`)  
- 전체 데이터에 대해 **향후 10일 평균 정답률 예측** (`predictions.csv`)  

---

## 파일 구조

```bash
project/
│── quiz_logs.csv          # 입력 데이터 (필수)
│── main.py                # 전체 학습/예측 스크립트
│── best_model_10d.pt      # 학습된 모델 가중치 (저장됨)
│── predictions.csv        # 최종 예측 결과
│── training_curves.png    # 학습 곡선 시각화
│── README.md              # 프로젝트 설명
```

---

## 설치 및 실행 방법

### 1. 환경 설정
Python 3.8 이상 권장.

```bash
pip install -r requirements.txt
```

`requirements.txt` 예시:
```txt
pandas
numpy
torch
scikit-learn
matplotlib
```

---

### 2. 데이터 준비
`quiz_logs.csv` 파일이 필요. 

### 3. 실행

## 주요 파라미터

코드 상단에서 설정 가능:
```python
WINDOW      = 45   # 입력 시퀀스 길이 (일 단위)
HORIZON     = 10   # 예측 기간 (일 단위)
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
TRAIN_RATIO = 0.7  # train/val/test 분할 비율
EARLY_STOP  = 5    # val 손실이 개선되지 않으면 학습 조기 종료
```

---

## 📊 출력 결과

1. **학습 곡선**
   - `training_curves.png`  
   - `Train Loss (MSE)` / `Val MSE` / `Val MAE` 변화 확인 가능  

2. **예측 결과**
   - `predictions.csv`  
   예시:
   ```csv
   patient_id,input_start,input_end,pred_start,pred_end,pred_10d
   U001,2023-01-01,2023-02-14,2023-02-15,2023-02-24,0.7421
   U002,2023-01-03,2023-02-16,2023-02-17,2023-02-26,0.5347
   ```
   - `pred_10d` → 예측된 향후 10일 평균 정답률 (0~1 사이 값)

---

## 참고

- 모델 구조: **2-layer LSTM + FC layer**
- 손실 함수: `MSELoss`
- 최적화 알고리즘: `Adam`
- 예측 시에는 `torch.sigmoid()`를 적용하여 확률(정답률) 형태로 변환
