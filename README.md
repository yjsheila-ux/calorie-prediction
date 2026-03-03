# 🔥 Calorie Prediction Hybrid Model

## 📌 Project Overview

운동 데이터를 활용하여 소모 칼로리를 예측하는 회귀 모델을 개발하였다.  
단순 머신러닝 모델이 아닌, **도메인 공식 기반 역공학 + ML 잔차 보정 + 앙상블 최적화** 구조를 설계하였다.

---

## 🧠 Core Idea

이 프로젝트의 핵심은 단순히 타겟 값을 직접 예측하는 것이 아닌

1. 기존 칼로리 계산 공식을 역공학(reverse engineering)하여 최적의 계수를 찾고
2. 해당 공식 예측값과 실제값의 잔차(residual)를 머신러닝 모델이 학습
3. 최종 예측값 = 공식 예측값 + ML 보정값

이라는 **Hybrid Modeling 구조**를 적용한 것이다.

---

## 🏗 Modeling Architecture

### 1️⃣ 공식 역공학 (Formula Optimization)

- Keytel 칼로리 공식 형태를 가정
- L-BFGS-B 최적화를 사용하여 파라미터 학습
- 남성 / 여성 파라미터 분리 학습

→ 도메인 기반 초기 예측값 생성

---

### 2️⃣ Residual Learning (잔차 보정)

```python
residual = y - formula_prediction```


ML 모델은 직접 y를 예측하지 않고  
공식의 오차(residual)만 학습하도록 설계하였다.

---

### 3️⃣ 핵심 Feature Engineering

- BMI 생성
- Duration × BPM
- Duration × Temperature
- 공식 예측값(fp)
- 공식 예측값의 소수점 부분(fp_frac)

특히 `fp_frac` (공식 예측값의 소수점 부분)이  
반올림 패턴과 연관되어 성능 향상에 크게 기여하였다.

---

### 4️⃣ OOF 기반 앙상블

- XGBoost
- LightGBM
- CatBoost
- ElasticNet

K-Fold OOF 예측을 활용하여
Nelder-Mead 방법으로 가중치 최적화 수행

가중치 조건:
- 가중치 합 = 1
- 음수 방지 (abs 후 정규화)

---

## 📊 Results

- Hybrid 설계 적용 후 성능 개선
- 공식 단독 모델 대비 오차 감소
- 잔차 보정 모델에서 추가 개선
- 최종 앙상블에서 최적 성능 달성

최종 RMSE = 0.061825
Score = 0.0616359607
---

## 🔎 Key Insights

1. 기존 칼로리 공식은 이미 상당히 정확했다.
2. 오차는 특정 패턴을 가졌으며, 소수점 정보가 중요 신호였다.
3. 직접 예측보다 잔차 보정 방식이 더 안정적이었다.
4. 도메인 지식 + ML 결합이 단일 모델보다 효과적이었다.

---

## 🛠 Tech Stack

- Python
- NumPy
- Pandas
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- Optuna
- SciPy (L-BFGS-B)

---

## 📂 Project Structure

calorie-prediction/
│
├── notebooks/calorie_prediction.ipynb
│
├── submission/
│ └── submission.csv
│
├── requirements.txt
└── README.md

## 🚀 How to Run

```bash
pip install -r requirements.txt```

notebooks/calorie_prediction.ipynb 실행 후 전체 셀 실행
