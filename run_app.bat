@echo off
cd /d "%~dp0"
echo Starting Smart Meal Planner...
echo Please wait while the browser opens...
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
"C:\Users\samat\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run streamlit_app.py --browser.gatherUsageStats false
pause
