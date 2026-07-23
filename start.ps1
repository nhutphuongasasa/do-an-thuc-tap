# =============================================================================
# start.ps1 — Khởi động ET-SSL Edge Anomaly Detection Pipeline (Windows)
#
# Yêu cầu:
#   - Python 3.9+ đã cài (thêm vào PATH)
#   - Npcap đã cài: https://npcap.com
#   - Chạy PowerShell với quyền Administrator
#
# Cách dùng:
#   .\start.ps1                      # tự động chọn card mạng
#   .\start.ps1 -Iface "Ethernet"   # chỉ định card mạng
#   .\start.ps1 -ListIfaces          # liệt kê tất cả card mạng
#   .\start.ps1 -Help                # hiển thị trợ giúp
# =============================================================================

param(
    [string]$Iface      = "",
    [string]$Backend    = "onnx",
    [int]   $MaxBatch   = 32,
    [float] $MaxWaitMs  = 200.0,
    [switch]$ListIfaces,
    [switch]$Help
)

# ── Màu sắc ──────────────────────────────────────────────────────────────────
function Write-Color([string]$Text, [string]$Color = "White") {
    Write-Host $Text -ForegroundColor $Color
}
function Write-Ok([string]$Text)   { Write-Host "[OK] $Text" -ForegroundColor Green }
function Write-Warn([string]$Text) { Write-Host "[!]  $Text" -ForegroundColor Yellow }
function Write-Err([string]$Text)  { Write-Host "[X]  $Text" -ForegroundColor Red }
function Write-Info([string]$Text) { Write-Host "     $Text" -ForegroundColor Cyan }

# ── Header ────────────────────────────────────────────────────────────────────
Write-Color "============================================================" Cyan
Write-Color "  ET-SSL Edge Anomaly Detection — Windows Launcher"          Cyan
Write-Color "============================================================" Cyan
Write-Host ""

# ── Help ─────────────────────────────────────────────────────────────────────
if ($Help) {
    Write-Host "Cách dùng: .\start.ps1 [tùy chọn]"
    Write-Host ""
    Write-Host "Tùy chọn:"
    Write-Host "  -Iface <tên>      Card mạng (vd: Ethernet, Wi-Fi). Mặc định: tự động"
    Write-Host "  -Backend <tên>    onnx | pkl. Mặc định: onnx"
    Write-Host "  -MaxBatch <n>     Kích thước batch. Mặc định: 32"
    Write-Host "  -MaxWaitMs <ms>   Thời gian chờ flush batch (ms). Mặc định: 200"
    Write-Host "  -ListIfaces       Liệt kê tất cả card mạng rồi thoát"
    Write-Host "  -Help             Hiển thị trợ giúp này"
    Write-Host ""
    Write-Host "Ví dụ:"
    Write-Host "  .\start.ps1"
    Write-Host "  .\start.ps1 -ListIfaces"
    Write-Host "  .\start.ps1 -Iface Ethernet"
    Write-Host "  .\start.ps1 -Iface Wi-Fi -MaxBatch 16 -MaxWaitMs 100"
    exit 0
}

# ── Kiểm tra quyền Administrator ─────────────────────────────────────────────
$currentPrincipal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Err "Script cần chạy với quyền Administrator để bắt traffic."
    Write-Info "Click chuột phải vào PowerShell → 'Run as administrator'"
    Write-Info "Sau đó chạy lại: .\start.ps1"
    exit 1
}
Write-Ok "Quyền Administrator OK"

# ── Đường dẫn ─────────────────────────────────────────────────────────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Join-Path $ScriptDir "edge-ai-traffic-anomaly"
$VenvDir    = Join-Path $ScriptDir "venv"
$ModelDir   = Join-Path $ProjectDir "model\onnx_models"
$LogDir     = Join-Path $ProjectDir "logs"
$PythonExe  = Join-Path $VenvDir "Scripts\python.exe"
$MainPy     = Join-Path $ProjectDir "main.py"

# ── Kiểm tra Python ───────────────────────────────────────────────────────────
if (Test-Path $PythonExe) {
    $PythonCmd = $PythonExe
    Write-Ok "Dùng Python từ venv: $PythonExe"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = "python"
    Write-Warn "Không tìm thấy venv — dùng Python hệ thống: $(python --version 2>&1)"
    Write-Info "Tạo venv: python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PythonCmd = "python3"
    Write-Warn "Dùng python3: $(python3 --version 2>&1)"
} else {
    Write-Err "Không tìm thấy Python. Cài Python 3.9+ từ https://python.org"
    exit 1
}

# ── Kiểm tra Npcap ────────────────────────────────────────────────────────────
$NpcapPath = "$env:SystemRoot\System32\Npcap"
$WinPcapPath = "$env:SystemRoot\System32\wpcap.dll"
if (-not (Test-Path $NpcapPath) -and -not (Test-Path $WinPcapPath)) {
    Write-Warn "Không phát hiện Npcap. Nếu capture lỗi, hãy cài Npcap từ:"
    Write-Info "  https://npcap.com/#download"
    Write-Info "  Chọn 'WinPcap API-compatible mode' khi cài."
} else {
    Write-Ok "Npcap / WinPcap đã được cài"
}

# ── Kiểm tra project ─────────────────────────────────────────────────────────
if (-not (Test-Path $MainPy)) {
    Write-Err "Không tìm thấy: $MainPy"
    exit 1
}

# ── Kiểm tra model ────────────────────────────────────────────────────────────
if (-not (Test-Path $ModelDir)) {
    Write-Err "Không tìm thấy thư mục model: $ModelDir"
    exit 1
}

$OnnxFile = Get-ChildItem -Path $ModelDir -Filter "*.onnx" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $OnnxFile) {
    Write-Warn "Không tìm thấy file .onnx — thử backend=pkl"
    $Backend = "pkl"
}
Write-Ok "Model dir: $ModelDir (backend=$Backend)"

# ── Tạo thư mục log ──────────────────────────────────────────────────────────
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
Write-Ok "Log dir: $LogDir"

# ── Hiển thị danh sách card mạng rồi thoát nếu yêu cầu ──────────────────────
if ($ListIfaces) {
    Write-Host ""
    Write-Color "  Danh sách card mạng:" Cyan
    & $PythonCmd $MainPy --list-ifaces
    exit 0
}

# ── Tự động phát hiện card mạng nếu không chỉ định ───────────────────────────
if ([string]::IsNullOrWhiteSpace($Iface)) {
    # Lấy adapter đang có kết nối, bỏ qua loopback và virtual
    $SkipNames = @("Loopback", "Teredo", "isatap", "6to4", "VirtualBox", "VMware", "Hyper-V")
    $Adapters  = Get-NetAdapter -ErrorAction SilentlyContinue |
                 Where-Object { $_.Status -eq "Up" } |
                 Where-Object { $Name = $_.Name; -not ($SkipNames | Where-Object { $Name -like "*$_*" }) } |
                 Sort-Object -Property LinkSpeed -Descending

    if ($Adapters) {
        $Iface = $Adapters[0].Name
        Write-Color "  [~] Tự động chọn interface: $Iface" Yellow
    } else {
        Write-Err "Không thể tự động phát hiện card mạng."
        Write-Info "Chạy: .\start.ps1 -ListIfaces"
        Write-Info "Sau đó: .\start.ps1 -Iface <tên>"
        exit 1
    }
}

Write-Ok "Interface: $Iface"

# ── In cấu hình ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Color "  Cấu hình:" Cyan
Write-Host "  • Interface : " -NoNewline; Write-Color $Iface Yellow
Write-Host "  • Backend   : " -NoNewline; Write-Color $Backend Yellow
Write-Host "  • Batch size: " -NoNewline; Write-Color "$MaxBatch flows" Yellow
Write-Host "  • Max wait  : " -NoNewline; Write-Color "$MaxWaitMs ms" Yellow
Write-Host "  • Log dir   : " -NoNewline; Write-Color $LogDir Yellow
Write-Host ""
Write-Color "  Log files:" Cyan
Write-Color "  • $LogDir\flow_decisions.jsonl  — mỗi flow 1 dòng JSON" Yellow
Write-Color "  • $LogDir\stats_summary.json    — tóm tắt mỗi 60 giây" Yellow
Write-Host ""
Write-Color "  Nhấn Ctrl+C để dừng." Green
Write-Color "============================================================" Cyan
Write-Host ""

# ── Chạy pipeline ─────────────────────────────────────────────────────────────
Set-Location $ProjectDir

& $PythonCmd $MainPy `
    --iface      $Iface     `
    --model_dir  $ModelDir  `
    --backend    $Backend   `
    --max_batch  $MaxBatch  `
    --max_wait_ms $MaxWaitMs
