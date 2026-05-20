@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo       Finansal Yapay Zeka Baslatiliyor
echo ==================================================

cd /d "%~dp0"

echo [+] Eski Streamlit ve Nginx surecleri temizleniyor...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501') do (
    if not "%%a"=="0" (
        taskkill /f /pid %%a >nul 2>&1
    )
)
taskkill /f /im nginx.exe >nul 2>&1

echo [+] Internet baglantisi kontrol ediliyor...
ping -n 1 google.com >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] UYARI: Internet baglantisi yok gibi gorunuyor.
    echo [!] Online veri cekimi veya ilk model kurulumu sirasinda hata alabilirsiniz.
)

if not exist requirements.txt (
    echo [!!!] requirements.txt bulunamadi.
    pause
    exit /b 1
)

echo [+] Gerekli kutuphaneler yukleniyor...
python -m pip install --upgrade pip
if %errorlevel% neq 0 goto :fail

python -m pip install --user -r requirements.txt
if %errorlevel% neq 0 goto :fail

set PORT=8501

if exist nginx\nginx.exe (
    echo [+] Nginx Guvenlik Katmani baslatiliyor...
    pushd nginx
    start nginx.exe
    popd
    echo [+] Uygulama HTTPS proxy ile baslatiliyor...
    echo [+] Erisim adresi: https://localhost
    start "" "https://localhost"
) else (
    echo [!] UYARI: nginx/nginx.exe bulunamadi. Uygulama dogrudan baslatiliyor.
    echo [+] Erisim adresi: http://127.0.0.1:%PORT%
    start "" "http://127.0.0.1:%PORT%"
)

python -m streamlit run app.py --server.port %PORT% --server.address 127.0.0.1 --browser.gatherUsageStats false
if %errorlevel% neq 0 goto :fail

exit /b 0

:fail
echo.
echo [!!!] Uygulama beklenmedik bir sekilde kapandi veya bagimlilik kurulumu basarisiz oldu.
echo [!] Yukaridaki hata mesajlarini kontrol edin.
pause
exit /b 1
