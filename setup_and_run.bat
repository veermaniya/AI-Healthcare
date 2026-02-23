@echo off
REM =========================================================
REM  Healthcare AI Platform — Windows Setup Script
REM  Run this ONCE after cloning or deploying the project.
REM  It sets up the database and creates all default users.
REM =========================================================

echo.
echo  ==========================================
echo   Healthcare AI — First-Time Setup
echo  ==========================================
echo.

REM Step 1: Install dependencies
echo [1/4] Installing dependencies...
pip install -r requirements.txt
echo.

REM Step 2: Run migrations
echo [2/4] Setting up database (migrations)...
python manage.py makemigrations
python manage.py migrate
echo.

REM Step 3: Create default users
echo [3/4] Creating default users in database...
python manage.py create_default_users
echo.

REM Step 4: Done
echo [4/4] Setup complete!
echo.
echo  ==========================================
echo   Open browser at: http://127.0.0.1:8000
echo  ==========================================
echo.
echo  Default Login Credentials:
echo  ------------------------------------------
echo   Admin      : admin     / Admin@1234
echo   Doctor     : doctor    / Doctor@1234
echo   Analyst    : analyst   / Analyst@1234
echo   Researcher : researcher/ Research@1234
echo  ------------------------------------------
echo.
echo  Now starting the server...
echo.
python manage.py runserver
pause
