#!/bin/bash
# HealthcareAI Platform - Quick Start Script
# Run this script to set up and launch the application

echo "=========================================="
echo "  HealthAI Platform - Setup & Launch"
echo "=========================================="

# Step 1: Install dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip install Django pandas numpy scikit-learn openpyxl xlrd requests sqlalchemy --break-system-packages 2>/dev/null || \
pip install Django pandas numpy scikit-learn openpyxl xlrd requests sqlalchemy

# Step 2: Run migrations
echo ""
echo "[2/4] Setting up database..."
cd "$(dirname "$0")"
python manage.py makemigrations healthcare_app
python manage.py migrate

# Step 3: Create superuser (optional)
echo ""
echo "[3/4] Ready to launch..."

# Step 4: Run server
echo ""
echo "[4/4] Starting Django development server..."
echo ""
echo "  ✅ Open your browser at: http://127.0.0.1:8000"
echo ""
echo "  📋 IMPORTANT: Set your FREE Groq API key for LLM chat:"
echo "     1. Go to https://console.groq.com (free signup)"
echo "     2. Create an API key"
echo "     3. Edit healthcare_ai/settings.py"
echo "     4. Replace 'your-groq-api-key-here' with your key"
echo "     OR set environment variable: export GROQ_API_KEY=your_key_here"
echo ""
echo "  📊 Sample dataset included: sample_healthcare_data.csv"
echo "     Upload it to test all features immediately!"
echo ""
echo "=========================================="

python manage.py runserver 0.0.0.0:8000
