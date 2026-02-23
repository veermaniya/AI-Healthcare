# 🏥 HealthAI — Healthcare Intelligence Platform

> **AI-Powered Clinical Analytics | Django + scikit-learn + Groq LLM**  
> Built for hospitals, researchers, and healthcare professionals

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Django](https://img.shields.io/badge/Django-4.x-green) ![License](https://img.shields.io/badge/License-MIT-yellow) ![Cost](https://img.shields.io/badge/Cost-Free-brightgreen)

---

## 🌟 What Is This?

HealthAI is a **full-stack web application** that turns any healthcare dataset (Excel/CSV/SQL) into actionable clinical intelligence — with zero cost. Upload your data, and instantly get ML predictions, risk alerts, survival curves, patient similarity matching, anomaly detection, and AI-generated clinical narratives.

---

## ✨ Features (16 AI Modules)

| # | Module | Description |
|---|---|---|
| 1 | 📊 **ML Analytics** | Regression, Classification, Clustering with auto-detect |
| 2 | 🧠 **XAI Explainability** | SHAP-style feature contributions & why-narrative |
| 3 | ⚠️ **Risk Engine** | Auto threshold alerts, anomaly detection, risk scores |
| 4 | 📈 **Trend Analysis** | Time-series forecasting with correlation matrix |
| 5 | 👥 **Patient Similarity** | KNN-based similar patient finder + cohort comparison |
| 6 | 🧑‍⚕️ **Patient Dashboard** | Full profile: percentile ranks, risk ring, predictions |
| 7 | 📉 **Survival Analysis** | Kaplan-Meier curves with group stratification |
| 8 | 🎯 **Multi-Target Compare** | Simultaneous predictions across multiple targets |
| 9 | 🔔 **Alert Rules Engine** | Custom clinical threshold rules builder |
| 10 | ⚖️ **Dataset Comparator** | Statistical A/B comparison with Cohen's d |
| 11 | 💡 **Clinical Insights** | LLM-generated narrative summaries (Groq LLaMA) |
| 12 | 🧪 **Data Quality** | Missing data, bias, outlier & class imbalance checks |
| 13 | 🔒 **Privacy & PHI** | PHI detection, masking, anonymisation |
| 14 | 💊 **ICD Coding Assistant** | ICD-10 mapping + drug interaction flags |
| 15 | 📄 **Report Generator** | HTML clinical report export |
| 16 | 🔬 **Medical Imaging** | AI image analysis via Groq vision models |

---

## ⚡ Quick Start

### Option A — One-Click (Windows)
```bash
setup_and_run.bat
```

### Option B — One-Click (Linux/Mac)
```bash
chmod +x run.sh && ./run.sh
```

### Option C — Manual
```bash
# 1. Clone / extract project
cd healthcare_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup database
python manage.py makemigrations healthcare_app
python manage.py migrate

# 4. Create default users
python manage.py create_default_users

# 5. Start server
python manage.py runserver
```

Then open **http://127.0.0.1:8000**

---

## 🔑 Setup: Groq API Key (Free LLM)

1. Go to **https://console.groq.com** — sign up (free)
2. Create an API key (starts with `gsk_...`)
3. Open `healthcare_ai/settings.py`
4. Replace the empty string:

```python
GROQ_API_KEY = 'gsk_your_key_here'   # ← paste here
```

> **Free tier:** 14,400 API requests/day — more than enough for clinical use.  
> The platform works without the key, but AI Insights and ICD Coding features need it.

---

## 👤 Default Login Accounts

| Username | Password | Role |
|---|---|---|
| `admin` | `Admin@1234` | Superuser |
| `doctor` | `Doctor@1234` | Staff |
| `analyst` | `Analyst@1234` | Staff |
| `researcher` | `Research@1234` | Staff |

---

## 📁 Project Structure

```
healthcare_ai/
├── healthcare_ai/                  # Django project config
│   ├── settings.py                 # ← GROQ_API_KEY goes here
│   └── urls.py
│
├── healthcare_app/                 # Main Django app
│   ├── models.py                   # DataSession, AnalysisResult, ChatMessage
│   ├── views.py                    # All 30+ API endpoints (NumpyEncoder safe)
│   ├── urls.py                     # URL routing
│   ├── management/
│   │   └── commands/
│   │       └── create_default_users.py
│   └── templates/
│       └── healthcare_app/
│           └── index.html          # Full single-page UI
│
├── ml_engine/                      # AI/ML Engine (pure Python)
│   ├── __init__.py                 # Exports all engines
│   ├── engine.py                   # HealthcareMLEngine, GroqLLMClient
│   ├── engine_features2.py         # PatientDashboard, Survival, Comparator, etc.
│   ├── engine_extensions.py        # ClinicalInsights, DataQuality, Privacy, Report
│   └── data_loader.py              # Excel/CSV/SQL loaders
│
├── requirements.txt
├── manage.py
├── setup_and_run.bat               # Windows one-click
└── run.sh                          # Linux/Mac one-click
```

---

## 🔌 API Endpoints Reference

### Data Loading
| Endpoint | Method | Description |
|---|---|---|
| `/api/upload/` | POST | Upload Excel / CSV file |
| `/api/upload-temp/` | POST | Upload Dataset B for comparison |
| `/api/connect-sql/` | POST | Connect to SQL database |
| `/api/columns/` | GET | Get session column info |
| `/api/column-stats/` | GET | Min/max/mean per column |
| `/api/clear/` | POST | Clear current session |

### Core ML
| Endpoint | Method | Description |
|---|---|---|
| `/api/run-analysis/` | POST | Run regression / classification / clustering |
| `/api/predict/` | POST | Single-row manual prediction |
| `/api/run-explainability/` | POST | XAI feature contributions |

### Clinical Features
| Endpoint | Method | Description |
|---|---|---|
| `/api/patient-dashboard/` | POST | Full patient profile + risk score |
| `/api/run-risk-engine/` | POST | Threshold alerts + anomaly detection |
| `/api/survival-analysis/` | POST | Kaplan-Meier survival curves |
| `/api/run-trend-analysis/` | POST | Trend + forecasting |
| `/api/run-patient-similarity/` | POST | KNN similar patients + cohort |
| `/api/multi-target-compare/` | POST | Compare multiple target columns |
| `/api/evaluate-alert-rules/` | POST | Evaluate custom clinical rules |
| `/api/suggest-alert-rules/` | POST | Auto-suggest rules from data |
| `/api/compare-datasets/` | POST | Statistical A/B dataset comparison |
| `/api/clinical-insights/` | POST | LLM clinical narrative |
| `/api/data-quality/` | POST | Data quality audit |
| `/api/privacy-check/` | POST | PHI detection scan |
| `/api/anonymise/` | POST | Mask/anonymise PHI columns |
| `/api/clinical-coding/` | POST | ICD-10 + drug interaction flags |
| `/api/generate-report/` | POST | Export HTML clinical report |
| `/api/analyze-image/` | POST | Medical image AI analysis |
| `/api/filter/` | POST | Row-level data filtering |
| `/api/chat/` | POST | LLM chat assistant |

---

## 🔬 ML Models

### Regression
- **Random Forest Regressor** — Best for healthcare tabular data
- **Linear Regression** — Fast, interpretable baseline
- **Gradient Boosting** — Highest accuracy for complex patterns

### Classification
- **Random Forest Classifier** — Handles mixed data well
- **Logistic Regression** — Interpretable, clinical-grade
- **Gradient Boosting Classifier** — Best accuracy

### Unsupervised
- **K-Means Clustering** — Patient subgroup discovery (2-10 clusters)
- **Isolation Forest** — Anomaly / outlier detection

### Distance-based
- **KNN (NearestNeighbors)** — Patient similarity matching

---

## 📊 How To Use

### Step 1 — Upload Data
Click **Upload Excel/CSV** and select your file, or use **Connect SQL** with a connection string.

### Step 2 — Select Columns
- **Click once** → Feature column (blue ✓)
- **Click twice** → Target column (gold ★)
- **Click again** → Deselect

### Step 3 — Run Any Module
Click any tab: Analytics, Risk Engine, Patient Dashboard, Survival Analysis, etc.

### SQL Connection String Formats
```
# SQL Server
mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+17+for+SQL+Server

# PostgreSQL
postgresql://user:pass@localhost:5432/healthcare_db

# MySQL
mysql+pymysql://user:pass@localhost/healthcare_db

# SQLite
sqlite:///path/to/db.sqlite3
```

---

## 💰 Cost Breakdown

| Component | Cost |
|---|---|
| Python + Django | Free (open source) |
| scikit-learn | Free (open source) |
| Groq LLM API | Free (14,400 req/day) |
| SQLite database | Free (built-in) |
| Local hosting | Free |
| **Total** | **$0** |

### Cloud Hosting Options (also free tier)
- **Railway.app** — `railway up`
- **Render.com** — Connect GitHub repo
- **PythonAnywhere** — Upload and configure WSGI

---

## 🔒 Production Security Checklist

```python
# healthcare_ai/settings.py — production changes:
import os

SECRET_KEY  = os.environ['DJANGO_SECRET_KEY']   # Never hardcode
DEBUG       = False
GROQ_API_KEY = os.environ['GROQ_API_KEY']

ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']

# Add to INSTALLED_APPS for static files in production:
# 'whitenoise.middleware.WhiteNoiseMiddleware'
```

Also:
- Enable HTTPS / SSL certificate
- Set `SESSION_COOKIE_SECURE = True`
- Set `CSRF_COOKIE_SECURE = True`
- Use PostgreSQL instead of SQLite for production

---

## 🧪 Testing the Setup

```bash
# Test LLM connection
curl http://127.0.0.1:8000/api/test-llm/

# Test API key
curl http://127.0.0.1:8000/api/debug-key/
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 4.x (Python 3.10+) |
| ML Engine | scikit-learn, pandas, numpy |
| LLM | Groq API (LLaMA 3.3 70B) |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Frontend | Vanilla JS, Chart.js, CSS variables |
| Auth | Django built-in authentication |
| Serialization | Custom NumpyEncoder (int64-safe) |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "Add clinical feature X"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## ⚠️ Medical Disclaimer

> This platform is for **clinical decision support and research only**.  
> It does **not** constitute a medical diagnosis.  
> Always consult a qualified healthcare professional before acting on AI-generated insights.

---

## 📄 License

MIT License — Free for personal, academic, and healthcare charity use.

---

*Built with ❤️ for healthcare professionals and researchers.*  
*IIT Roorkee Project — Healthcare AI Intelligence Platform*
