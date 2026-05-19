╔══════════════════════════════════════════════════════════╗
║           LoanSpark — AI Loan System                     ║
║           Quick Start Guide (Windows)                    ║
╚══════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FIRST TIME? Do these steps ONCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  STEP 1 — Install Python 3.11+
           https://python.org/downloads
           ⚠️  Check "Add Python to PATH" during install!

  STEP 2 — Install Ollama (optional but recommended)
           https://ollama.com/download

  STEP 3 — Double-click SETUP.bat
           Installs all packages + downloads AI model.
           Takes 5-10 minutes on first run.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EVERY TIME YOU WANT TO RUN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Double-click START.bat
  → Opens both pages automatically in your browser.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LOGIN CREDENTIALS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Bank Dashboard Key:  BANK_SECRET_2024

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FOLDER STRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  LoanSpark/
  ├── START.bat          ← Run this every time
  ├── SETUP.bat          ← Run this once (first time)
  ├── STOP.bat           ← Force stop everything
  ├── README.txt         ← This file
  ├── backend/           ← Python API server
  │   ├── main.py
  │   ├── requirements.txt
  │   ├── models/        ← ML model files
  │   └── data/          ← Training data
  └── frontend/          ← Web pages (open in browser)
      ├── index.html     ← Applicant loan form
      ├── index.css
      ├── bank.html      ← Bank risk dashboard
      └── bank.css

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TROUBLESHOOTING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Problem: "Python not found"
  Fix: Reinstall Python, check "Add to PATH"

  Problem: Port 8000 already in use
  Fix: Run STOP.bat, then START.bat again

  Problem: Bank dashboard shows no data
  Fix: Complete a loan application in index.html first

  Problem: Slow AI responses
  Fix: Normal — local AI (Mistral) takes 5-15 seconds

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
