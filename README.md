# 🧠 Credit Risk Probability Model for Alternative Data  

---

## 📌 Overview

This project develops a robust, interpretable credit risk model for **Bati Bank’s Buy-Now-Pay-Later (BNPL)** product using behavioral data from an eCommerce partner. The absence of traditional credit history or default labels is addressed through **proxy target engineering**, supported by a full **ML pipeline**, **model tracking**, and **API deployment**.

---

## 🎯 Objectives

- Predict customer credit risk probabilities in a BNPL context.
- Engineer features from raw transaction data.
- Build a proxy for default using behavioral signals (RFM).
- Train, evaluate, and serve models with traceability.
- Ensure interpretability and regulatory readiness (Basel II).
- Package and deploy the model via FastAPI and Docker.

---

## 🧠 Business Understanding

### Basel II & Model Interpretability
The **Basel II Accord** requires banks to measure and manage risk rigorously. This compels us to build **interpretable models** with **clear audit trails** for regulatory compliance.

### Why a Proxy Target?
True default labels were unavailable. We constructed a **proxy variable** using **K-means clustering on RFM metrics**, labeling low-engagement customers as high risk. This enabled supervised modeling, though it introduces labeling noise.

### Model Trade-offs
- **Logistic Regression + WoE** offers transparency for regulators.
- **Random Forest/XGBoost** yield better accuracy but need explainability tools (e.g., SHAP).

---

## 📊 Methodology

### 1. Data Understanding & EDA
- Source: ~95k transaction records, 16 columns.
- Tasks: Skew detection, missing values, outlier detection, fraud flag analysis.
- Key Fields: `TransactionId`, `Amount`, `Value`, `ProductCategory`, `TransactionStartTime`.

### 2. Feature Engineering & Target Creation
- Created **Recency, Frequency, Monetary (RFM)** features.
- Encoded categorical variables; extracted time features.
- Generated **proxy target (`is_high_risk`)** using **KMeans clustering** on RFM.

### 3. Model Training & Evaluation
- Models: `Logistic Regression`, `Random Forest`, `XGBoost`.
- Used **GridSearchCV** for tuning, **MLflow** for tracking.
- Metrics: `ROC-AUC`, `F1`, `Precision`, `Recall`.
- Feature importance visualized and interpreted.

### 4. Deployment
- Created a `/predict` **FastAPI endpoint**.
- Loads best model from **MLflow Registry**.
- Entire pipeline **Dockerized** using `Dockerfile` and `docker-compose.yml`.

---

## 🧪 Usage

### Setup

```bash
pip install -r requirements.txt

python scripts/train.py

uvicorn src.api.main:app --reload

docker-compose up --build

```

---

## 📦 Tech Stack

| Layer            | Tools & Libraries                          |
|------------------|---------------------------------------------|
| Data Processing  | pandas, numpy, sklearn.pipeline             |
| Modeling         | scikit-learn, xgboost, mlflow               |
| Deployment       | FastAPI, Uvicorn, Docker, docker-compose    |
| CI/CD            | GitHub Actions, pytest, flake8              |
| Experimentation  | MLflow                                      |
| Version Control  | Git, DVC                                    |

