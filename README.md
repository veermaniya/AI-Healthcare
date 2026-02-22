# 🏥 HealthAI — Free Healthcare Intelligence Platform

> AI-Powered Healthcare Analytics | Free for Charity | LLM + ML + Predictions

---

## 🌟 Features

| Feature | Description |
|---|---|
| 📊 **Excel / CSV Upload** | Upload any `.xlsx`, `.xls`, or `.csv` healthcare dataset |
| 🗄 **SQL Server Connect** | Connect directly to SQL Server, PostgreSQL, MySQL |
| 🎛 **Dynamic Column Selector** | Click any column to select as Feature or Target |
| 📈 **Regression** | Predict continuous values (blood pressure, glucose, BMI) |
| 🏷 **Classification** | Predict categories (diagnosis, risk level, disease type) |
| 🔵 **Clustering** | Group patients by similarity (K-Means) |
| 🎯 **Single Patient Prediction** | Enter values manually and get instant prediction |
| 🤖 **AI Chat Assistant** | Ask natural language questions about your data |
| 📉 **Feature Importance** | See which columns most influence predictions |
| 👁 **Data Preview** | See your dataset before analysis |

---

## ⚡ Quick Start (3 Steps)

### Step 1: Install & Run
```bash
cd healthcare_ai
chmod +x run.sh
./run.sh
```
Or manually:
```bash
pip install -r requirements.txt
python manage.py makemigrations healthcare_app
python manage.py migrate
python manage.py runserver
```

### Step 2: Get Free LLM API Key
1. Go to **https://console.groq.com** (completely free)
2. Sign up and create an API key
3. Open `healthcare_ai/settings.py`
4. Replace `'your-groq-api-key-here'` with your key

### Step 3: Open Browser
```
http://127.0.0.1:8000
```

---

## 📊 How To Use

### Using the Column Selector
- **Click once** on a column = Add as **Feature** (blue ✓)
- **Click twice** on same column = Make it the **Target** (pink ★)
- **Click three times** = Deselect
- You can select **multiple feature columns**

### Analysis Types
| Task | When to Use | Target Column |
|---|---|---|
| **Auto Detect** | Let AI decide | Required |
| **Regression** | Predict a number (glucose level) | Required - numeric |
| **Classification** | Predict a category (disease type) | Required - categorical |
| **Clustering** | Group similar patients | Not needed |

### SQL Server Connection String Examples
```
# SQL Server
mssql+pyodbc://username:password@server_name/database_name?driver=ODBC+Driver+17+for+SQL+Server

# PostgreSQL
postgresql://username:password@localhost:5432/healthcare_db

# MySQL
mysql+pymysql://username:password@localhost/healthcare_db
```

---

## 🤖 LLM Chat Examples

You can ask the AI assistant:
- *"Explain the regression results in simple terms"*
- *"Which patients are at highest risk based on these features?"*
- *"What does the R² score of 0.87 mean?"*
- *"How can I improve my prediction model?"*
- *"What health patterns do you see in this clustering result?"*

---

## 📁 Project Structure

```
healthcare_ai/
├── healthcare_ai/          # Django project config
│   ├── settings.py         # ← Set GROQ_API_KEY here
│   └── urls.py
├── healthcare_app/         # Main Django app
│   ├── models.py           # DataSession, AnalysisResult, ChatMessage
│   ├── views.py            # All API endpoints
│   ├── urls.py             # URL routing
│   └── templates/
│       └── healthcare_app/
│           └── index.html  # Full UI (single file)
├── ml_engine/              # AI/ML Engine
│   ├── engine.py           # HealthcareMLEngine + GroqLLMClient
│   └── data_loader.py      # Excel/CSV/SQL loaders
├── requirements.txt
├── manage.py
├── run.sh                  # One-click launcher
└── sample_healthcare_data.csv  # Test dataset
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload/` | POST | Upload Excel/CSV file |
| `/api/connect-sql/` | POST | Connect to SQL database |
| `/api/columns/` | GET | Get current session columns |
| `/api/run-analysis/` | POST | Run ML analysis |
| `/api/predict/` | POST | Single-row prediction |
| `/api/chat/` | POST | LLM chat message |
| `/api/clear/` | POST | Clear current session |

---

## 🔬 ML Models Included

### Regression Models
- **Random Forest Regressor** (default, best for healthcare data)
- **Linear Regression** (fast, interpretable)
- **Gradient Boosting** (highest accuracy)

### Classification Models
- **Random Forest Classifier** (default)
- **Logistic Regression** (interpretable, fast)

### Clustering
- **K-Means** (configurable clusters 2-10)

---

## 💰 Cost Breakdown (All Free!)

| Component | Cost |
|---|---|
| Django + Python | Free (open source) |
| scikit-learn ML | Free (open source) |
| Groq API (LLM) | Free tier: 14,400 requests/day |
| SQLite Database | Free (included) |
| Hosting (local) | Free |

**Total Cost: $0 — Perfect for healthcare charity!**

For production/cloud hosting:
- **Railway.app** — Free tier available
- **Render.com** — Free tier available
- **PythonAnywhere** — Free tier available

---

## 🔒 Security Notes for Production

1. Change `SECRET_KEY` in settings.py
2. Set `DEBUG = False`
3. Configure proper `ALLOWED_HOSTS`
4. Use environment variables for all sensitive keys
5. Enable HTTPS

```python
# Production settings
SECRET_KEY = os.environ['DJANGO_SECRET_KEY']
DEBUG = False
GROQ_API_KEY = os.environ['GROQ_API_KEY']
```

---

## 📞 Support & Contribution

This is a **free charity project** for global healthcare access.
Feel free to contribute, report issues, and share with healthcare organizations!

---

*Built with ❤️ for healthcare charity. Free forever.*
