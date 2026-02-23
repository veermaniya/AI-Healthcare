# 🚀 HealthAI — Publishing & Deployment Guide

---

## 📋 Pre-Publish Checklist

Before publishing, complete these steps:

- [ ] Remove your real Groq API key from `settings.py`
- [ ] Set `DEBUG = False` (or use environment variable)
- [ ] Ensure `db.sqlite3` is in `.gitignore`
- [ ] Ensure no real patient data in the repo
- [ ] Test a fresh install with `requirements.txt`

---

## 1️⃣ Publish to GitHub

### Step 1 — Prepare settings.py for public repo

Open `healthcare_ai/settings.py` and make these changes before pushing:

```python
import os

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'change-me-in-production')
DEBUG = os.environ.get('DEBUG', 'True') == 'True'
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')   # ← never hardcode key
GROQ_MODEL = 'llama-3.3-70b-versatile'
```

### Step 2 — Initialize Git and push

```bash
cd healthcare_ai

# Initialize repo
git init
git add .
git commit -m "Initial commit — HealthAI Platform"

# Create repo on GitHub (go to github.com → New repository)
# Then connect and push:
git remote add origin https://github.com/YOUR_USERNAME/healthcare-ai.git
git branch -M main
git push -u origin main
```

### Step 3 — Add GitHub repository topics (recommended)
On your GitHub repo page, add these topics:
`django` `healthcare` `machine-learning` `python` `ai` `clinical-analytics` `scikit-learn` `groq` `llm`

---

## 2️⃣ Deploy to Railway.app (Free)

Railway gives you a public URL in under 5 minutes.

### Step 1 — Add Procfile
Create a file named `Procfile` in the root:
```
web: gunicorn healthcare_ai.wsgi --log-file -
```

### Step 2 — Add `gunicorn` to requirements.txt
```
gunicorn>=21.2.0
```

### Step 3 — Deploy
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create project and deploy
railway init
railway up
```

### Step 4 — Set environment variables in Railway dashboard
```
DJANGO_SECRET_KEY = your-very-long-random-secret-key
GROQ_API_KEY      = gsk_your_groq_key_here
DEBUG             = False
ALLOWED_HOSTS     = your-app.railway.app
```

---

## 3️⃣ Deploy to Render.com (Free)

### Step 1 — Create `render.yaml` in project root
```yaml
services:
  - type: web
    name: healthcare-ai
    env: python
    buildCommand: pip install -r requirements.txt && python manage.py migrate
    startCommand: gunicorn healthcare_ai.wsgi:application
    envVars:
      - key: DJANGO_SECRET_KEY
        generateValue: true
      - key: GROQ_API_KEY
        sync: false
      - key: DEBUG
        value: False
```

### Step 2 — Connect GitHub on Render
1. Go to **render.com** → New → Web Service
2. Connect your GitHub repo
3. Render auto-detects `render.yaml`
4. Add `GROQ_API_KEY` in environment variables

---

## 4️⃣ Deploy to PythonAnywhere (Free)

1. Upload project as `.zip` or clone from GitHub
2. Open a Bash console and run:
```bash
pip3.10 install --user -r requirements.txt
python manage.py migrate
python manage.py create_default_users
```
3. Configure WSGI file to point to `healthcare_ai/wsgi.py`
4. Add environment variables in the web app settings

---

## 5️⃣ Production settings.py (Full)

Replace your `healthcare_ai/settings.py` with this production-ready version:

```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Security ──────────────────────────────────
SECRET_KEY   = os.environ.get('DJANGO_SECRET_KEY', 'dev-only-insecure-key-change-me')
DEBUG        = os.environ.get('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# ── Apps ──────────────────────────────────────
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'healthcare_app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'healthcare_ai.urls'
WSGI_APPLICATION = 'healthcare_ai.wsgi.application'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.template.context_processors.debug',
        'django.template.context_processors.request',
        'django.contrib.auth.context_processors.auth',
        'django.contrib.messages.context_processors.messages',
    ]},
}]

# ── Database ──────────────────────────────────
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# ── Static Files ──────────────────────────────
STATIC_URL  = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
MEDIA_URL   = '/media/'
MEDIA_ROOT  = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ── Groq LLM ──────────────────────────────────
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_MODEL   = 'llama-3.3-70b-versatile'

# ── Production Security (when DEBUG=False) ────
if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE    = True
    SECURE_SSL_REDIRECT   = True
    SECURE_HSTS_SECONDS   = 3600
```

---

## 6️⃣ Generate a Strong Secret Key

Run this once to generate a secure key:

```python
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Paste the output as your `DJANGO_SECRET_KEY` environment variable.

---

## 7️⃣ Recommended GitHub Repository Structure

```
healthcare-ai/                    ← repo root
├── README.md                     ← project overview
├── PUBLISH_GUIDE.md              ← this file
├── requirements.txt              ← Python dependencies
├── .gitignore                    ← ignore secrets & data
├── Procfile                      ← Railway / Heroku deploy
├── render.yaml                   ← Render deploy config
├── manage.py
├── setup_and_run.bat             ← Windows launcher
├── run.sh                        ← Linux/Mac launcher
├── healthcare_ai/
│   ├── settings.py               ← uses os.environ for secrets
│   ├── urls.py
│   └── wsgi.py
├── healthcare_app/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── management/commands/
│   │   └── create_default_users.py
│   └── templates/
└── ml_engine/
    ├── __init__.py
    ├── engine.py
    ├── engine_features2.py
    ├── engine_extensions.py
    └── data_loader.py
```

---

## 8️⃣ After Publishing — Share Your Project

**README badges to add at the top of your README:**
```markdown
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Django](https://img.shields.io/badge/Django-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
```

**Demo dataset to include:**  
Include a sample CSV with anonymised/synthetic data so reviewers can test immediately.

---

*HealthAI — IIT Roorkee | Healthcare AI Intelligence Platform*
