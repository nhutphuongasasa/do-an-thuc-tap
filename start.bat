@echo off
:: =============================================================================
:: start.bat — Wrapper cho start.ps1 (Windows)
:: Click đúp để chạy, hoặc mở CMD với quyền Administrator.
:: =============================================================================
setlocal

:: Kiểm tra quyền Administrator
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo [!] Script can chay voi quyen Administrator.
    echo     Click chuot phai -> "Run as administrator"
    pause
    exit /b 1
)

:: Chạy PowerShell script
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if %errorLevel% NEQ 0 (
    echo.
    echo [!] Da thoat voi loi. Xem thong bao o tren.
    pause
)
