import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint

df = pd.read_csv("monthly_summary.csv")
X = df[["avg_acc_rate_90d", "avg_time_90d"]]
y = df["target_acc_rate"]

X = X[~y.isna()]
y = y[~y.isna()]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_dist = {
    # 'n_estimators': randint(100, 500),
    # 'max_depth': randint(5, 30),
    # 'min_samples_split': randint(2, 10),
    # 'min_samples_leaf': randint(1, 5) 첫번째 레슨 
    # 'n_estimators': randint(100, 300),
    # 'max_depth': randint(3,15),
    # 'min_samples_split': randint(2, 10),
    # 'min_samples_leaf': randint(2, 10),
    # 'max_features': ['sqrt', 'log2', None] #두번째 레슨, 바뀐 모델 첫번째 레슨
    # 'n_estimators': randint(100, 400),           
    # 'max_depth': randint(10, 20),               
    # 'min_samples_split': randint(2, 15),       
    # 'min_samples_leaf': randint(1, 5),           
    # 'max_features': ['sqrt', 'log2', None] #바뀐 데이터 두번째 레슨 
    'n_estimators': randint(250, 320),           
    'max_depth': randint(16, 22),                
    'min_samples_split': randint(4, 7),          
    'min_samples_leaf': randint(1, 3),          
    'max_features': [None, 'sqrt']               
}

random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    # n_iter=50,
    # cv=3,
    # scoring='neg_mean_absolute_error', # 정답률 예측용 
    # verbose=1,
    # n_jobs=-1,
    # random_state=42 #두번째 레슨, 바뀐 모델 첫번째 레슨
    # n_iter=100,
    # cv=5,
    # scoring='neg_mean_absolute_error', # 정답률 예측용 
    # verbose=1,
    # n_jobs=-1,
    # random_state=42 세번째 레슨 ---------
    # n_iter=100,        
    # cv=3,
    # scoring='neg_mean_absolute_error',
    # verbose=1,
    # n_jobs=-1,
    # random_state=42 #바뀐 데이터 두번째 레슨 
    n_iter=60,           
    cv=5,                
    scoring='neg_mean_absolute_error',  
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("✅ Randomized Search 완료")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
print("Best Params:", random_search.best_params_)

# MSE: 0.002161612935788271 첫번째
# R²: 0.6778432476464396
# Best Params: {'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 330}

# MSE: 0.0021420583705024847 두번째 
# R²: 0.6807575692355898
# Best Params: {'max_depth': 11, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 202}

# MSE: 0.002380148381516181 세번째 
# R²: 0.6452737398015174
# Best Params: {}

#시각화
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

import seaborn as sns 

importances = best_model.feature_importances_
features = X.columns

plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

#모델 저장 pk1
import pickle

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Pickle 형식(.pkl)으로 모델 저장 완료: best_model.pkl")

#모델 저장 onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

onnx_model = convert_sklearn(best_model, initial_types=initial_type)

with open("best_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ ONNX 형식(.onnx)으로 모델 저장 완료: best_model.onnx")

#================================== 치매 특화 데이터
# MSE: 0.0006822980717637931
# R²: 0.9459276589945914
# Best Params: {'max_depth': 14, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 9, 'n_estimators': 111}

# MSE: 0.0005278646255044753 파인튜닝
# R²: 0.9581665591386112
# Best Params: {'max_depth': 19, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 272}

# MSE: 0.0005242653650111136
# R²: 0.9584518016870199
# Best Params: {'max_depth': 21, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 296}