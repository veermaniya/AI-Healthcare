#!/bin/bash
# =========================================================
#  Healthcare AI Platform — Linux/Mac Setup & Run Script
#  Run this ONCE after cloning or deploying the project.
# =========================================================

echo ""
echo "=========================================="
echo "  Healthcare AI — Setup & Launch"
echo "=========================================="

# Step 1: Install dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt --break-system-packages 2>/dev/null || \
pip install -r requirements.txt

# Step 2: Run migrations
echo ""
echo "[2/4] Setting up database..."
cd "$(dirname "$0")"
python manage.py makemigrations
python manage.py migrate

# Step 3: Create default users
echo ""
echo "[3/4] Creating default users in database..."
python manage.py create_default_users

# Step 4: Launch
echo ""
echo "[4/4] Starting server..."
echo ""
echo "  ✅ Open your browser at: http://127.0.0.1:8000"
echo ""
echo "  Default Login Credentials:"
echo "  ------------------------------------------"
echo "   Admin      : admin      / Admin@1234"
echo "   Doctor     : doctor     / Doctor@1234"
echo "   Analyst    : analyst    / Analyst@1234"
echo "   Researcher : researcher / Research@1234"
echo "  ------------------------------------------"
echo ""
echo "  📋 Set your Groq API key in healthcare_ai/settings.py"
echo "  📊 Upload sample_healthcare_data.csv to test features"
echo ""
echo "=========================================="

python manage.py runserver 0.0.0.0:8000
