<div align="center">

<img src="https://img.shields.io/badge/LoanSpark-AI%20Loan%20System-63B3ED?style=for-the-badge&logo=lightning&logoColor=white" alt="LoanSpark"/>

# ⚡ LoanSpark — AI Loan Underwriting System

**A full-stack AI-powered loan approval and fraud detection platform.**  
Built with FastAPI, scikit-learn, Mistral (Ollama), and vanilla JS.

<br/>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Mistral-black?style=flat-square&logo=ollama&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

<br/>

</div>

---

## 📌 What is LoanSpark?

LoanSpark is an end-to-end AI loan underwriting system that simulates how a real fintech company processes loan applications. It has **two separate interfaces** — one for the applicant and one for the bank — connected by a FastAPI backend that runs two machine learning models in real time.

When an applicant fills out the chat-based form, the system simultaneously:
- Evaluates their **financial profile** using a Gradient Boosting model
- Detects **fraud signals** using a Random Forest model trained on behavioral biometrics
- Generates a **personalized AI explanation** using a locally-run Mistral LLM
- Produces a complete **bank risk report** with fraud probability, behavioral flags, and recommended actions

> 💡 Everything runs **100% locally** — no paid APIs, no cloud required.

---

## 🖥️ Screenshots

| Applicant Chat Interface | Bank Risk Dashboard |
|:---:|:---:|
| Chat-based loan application with progress tracking | Real-time fraud & risk analysis for bank officers |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FRONTEND                            │
│  ┌──────────────────┐      ┌──────────────────────────┐ │
│  │   index.html     │      │      bank.html           │ │
│  │  Applicant Chat  │      │   Bank Risk Dashboard    │ │
│  │  - Progress bar  │      │   - Fraud score          │ │
│  │  - Choice inputs │      │   - Behavioral flags     │ │
│  │  - Result card   │      │   - KYC summary          │ │
│  └────────┬─────────┘      └────────────┬─────────────┘ │
└───────────┼────────────────────────────┼───────────────┘
            │ HTTP / REST                │
┌───────────▼────────────────────────────▼───────────────┐
│                  FastAPI Backend (main.py)              │
│                   http://localhost:8000                 │
├─────────────┬──────────────┬──────────────┬────────────┤
│  database   │  ml_models   │  behavior_   │  ollama_   │
│  .py        │  .py         │  tracker.py  │  client.py │
│  SQLite     │  sklearn     │  Biometrics  │  Mistral   │
│  sessions   │  pipelines   │  scoring     │  LLM       │
└─────────────┴──────────────┴──────────────┴────────────┘
```

---

## ✨ Key Features

### 🧠 Dual ML Models
- **Loan Approval Model** — Gradient Boosting Classifier trained on 12,000 synthetic applicant records with realistic Indian fintech distributions
- **Fraud Detection Model** — Random Forest Classifier with `class_weight="balanced"` combining financial signals with live behavioral biometrics

### 🕵️ Behavioral Biometrics (Silent Fraud Detection)
While the applicant fills the form, the system secretly tracks:
- Time taken to answer each question
- Number of edits per field
- Tab/window switches
- Total session duration

These signals feed directly into the fraud model. Fraudsters hesitate on income fields, edit excessively, and answer inconsistently — the model learns this.

### 🤖 Local AI with Mistral
Uses [Ollama](https://ollama.com) to run **Mistral** locally — generates personalized approval/rejection messages and improvement suggestions without any API cost or internet dependency.

### 🏦 Bank Risk Dashboard
A separate secured dashboard (password protected) showing:
- Fraud probability gauge
- Loan approval confidence score
- KYC summary
- Behavioral anomaly flags
- Financial health indicators
- Full action plan for the bank officer

### 🔧 sklearn Pipelines
Both models are wrapped in `Pipeline([StandardScaler(), model])` — ensures identical normalization at training and prediction time, eliminating one of the most common ML production bugs.

---

## 🗂️ Project Structure

```
LoanSpark/
│
├── 🚀 START.bat              # One-click launcher (Windows)
├── ⚙️  SETUP.bat              # First-time setup (Windows)
├── 🛑 STOP.bat               # Force stop all processes
├── 📖 README.md              # This file
│
├── backend/
│   ├── main.py               # FastAPI server & all API routes
│   ├── ml_models.py          # Model loading & prediction functions
│   ├── train_data.py         # Data generation & model training
│   ├── decision_engine.py    # Final APPROVE / REVIEW / REJECT logic
│   ├── behavior_tracker.py   # Behavioral biometrics aggregation
│   ├── database.py           # SQLite session management
│   ├── ollama_client.py      # Mistral LLM integration
│   ├── requirements.txt      # Python dependencies
│   ├── models/
│   │   ├── loan_model.pkl    # Trained loan approval pipeline
│   │   └── fraud_model.pkl   # Trained fraud detection pipeline
│   └── data/
│       └── loan_dataset.csv  # Generated training dataset
│
└── frontend/
    ├── index.html            # Applicant chat interface
    ├── index.css             # Applicant UI styles
    ├── bank.html             # Bank risk dashboard
    └── bank.css              # Dashboard styles
```

---

## ⚙️ How the ML Models Work

### Data Generation
Instead of using a public dataset, the system **generates 12,000 synthetic loan applicant records** using realistic statistical distributions:

| Feature | Distribution | Range |
|---|---|---|
| Income | Log-normal (μ=10.8, σ=0.55) | ₹12K – ₹5L |
| Credit Score | Normal (μ=640, σ=90) | 300 – 850 |
| Loan Amount | Log-normal (μ=10.4, σ=0.65) | ₹5K – ₹8L |
| Employment Years | Exponential (scale=5.5) | 0 – 45 yrs |

### Feature Engineering
Two ratio features are derived — standard real-world banking metrics:
```
debt_to_income = (existing_debts + loan_amount × 0.02) / monthly_income
loan_to_income = loan_amount / annual_income
```

### Loan Model Training
```python
Pipeline([
    StandardScaler(),
    GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05
    )
])
# 80/20 train-test split with stratification
# Evaluated on ROC-AUC score
```

### Fraud Model Training
```python
Pipeline([
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=250, max_depth=9,
        class_weight="balanced"   # handles 22% fraud minority class
    )
])
```

---

## 🚀 Getting Started (Windows)

### Prerequisites
- [Python 3.11+](https://python.org/downloads) — check **"Add to PATH"** during install
- [Ollama](https://ollama.com/download) — for local AI (optional but recommended)

### First Time Setup
```
1. Extract the ZIP
2. Double-click SETUP.bat
   → Installs all Python packages
   → Downloads Mistral model (~4GB, one time only)
```

### Run the App
```
Double-click START.bat
```
This automatically:
- Starts Ollama in the background
- Launches the FastAPI backend on port 8000
- Opens both frontend pages in your browser

### Login
| Interface | URL | Credentials |
|---|---|---|
| Applicant App | `frontend/index.html` | No login |
| Bank Dashboard | `frontend/bank.html` | Key: `BANK_SECRET_2024` |

---

## 🚀 Getting Started (Mac / Linux)

```bash
# Clone the repo
git clone https://github.com/yourusername/loanspark.git
cd loanspark/backend

# Install dependencies
pip install -r requirements.txt

# Pull Mistral model
ollama pull mistral

# Start Ollama
ollama serve &

# Start backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Open frontend
open ../frontend/index.html   # Mac
xdg-open ../frontend/index.html  # Linux
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/start-session` | Create new applicant session |
| `GET` | `/next-question` | Fetch next question in flow |
| `POST` | `/submit-answer` | Submit applicant answer |
| `POST` | `/behavior-event` | Log behavioral biometric event |
| `POST` | `/final-decision` | Trigger ML models & get verdict |
| `GET` | `/all-sessions` | Bank: list all sessions |
| `GET` | `/bank-report/{id}` | Bank: full risk report for session |
| `GET` | `/health` | Health check |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend | Python, FastAPI | REST API server |
| ML | scikit-learn | Loan & fraud models |
| Local AI | Mistral via Ollama | Personalized explanations |
| Database | SQLite | Session & answer storage |
| Frontend | HTML, CSS, JS | Applicant & bank interfaces |
| Serialization | joblib | Save/load ML pipelines |

---

## 🔮 Future Improvements

- [ ] Add PostgreSQL support for production deployment
- [ ] Add JWT authentication for bank dashboard
- [ ] Train on real loan datasets (LendingClub, etc.)
- [ ] Add email notification on application completion
- [ ] Deploy backend to Railway / Render
- [ ] Add admin panel to manage questions

---

## 👨‍💻 Author

Built as a college project demonstrating real-world ML, behavioral biometrics, and full-stack fintech application development.

---

<div align="center">

**⚡ LoanSpark** — Built with Python, scikit-learn & Mistral

*If you found this useful, please ⭐ star the repo!*

</div>
