#!/usr/bin/env bash
# ET-SSL Edge AI Traffic Anomaly — Start script
#
# Flow: Traffic thật → Tự detect card mạng chính → Bắt traffic → Flow → Feature → Inference → Alert → Log file
#
# Usage:
#   ./start.sh              # Tự detect card mạng chính + bắt traffic thật + ghi log
#   ./start.sh --iface eth0 # Ép dùng interface cụ thể
#   ./start.sh --pcap f.pcap # Replay file pcap thay vì bắt live
#   ./start.sh check        # Health check nhanh
#   ./start.sh test         # Chạy evaluation
#   ./start.sh pipeline     # Full evaluation suite

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$ROOT_DIR/edge-ai-traffic-anomaly"
VENV_DIR="$ROOT_DIR/venv"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
CAPTURE_PID=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[ET-SSL]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

cleanup() {
    if [[ -n "$CAPTURE_PID" ]] && kill -0 "$CAPTURE_PID" 2>/dev/null; then
        log "Dừng capture (PID $CAPTURE_PID)..."
        kill "$CAPTURE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── Bước 1: Virtual environment ────────────────────────────────────────────────
setup_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Tạo virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    if ! python -c "import torch, scapy, onnxruntime" 2>/dev/null; then
        log "Cài đặt dependencies..."
        pip install -q --upgrade pip
        pip install -q -r "$REQUIREMENTS"
        ok "Dependencies đã cài xong"
    fi
}

# ── Bước 2: Auto-detect card mạng quan trọng nhất ──────────────────────────────
# Ưu tiên: default route → interface UP có IP thật → lỗi
is_virtual_iface() {
    local ifc="$1"
    case "$ifc" in
        lo|lo0|docker*|br-*|veth*|virbr*|tun*|tap*|wg*|zt*|utun*|awdl*|llw*|anpi*|bridge*)
            return 0 ;;
        *)
            return 1 ;;
    esac
}

detect_primary_iface() {
    local ifc=""

    # Cách 1 (ưu tiên nhất): lấy interface gắn với default route — đây là card đang dùng internet thật
    if command -v ip >/dev/null 2>&1; then
        ifc="$(ip route show default 2>/dev/null | awk '/default/ {for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -n1 || true)"
    elif command -v route >/dev/null 2>&1 && [[ "$(uname)" == "Darwin" ]]; then
        ifc="$(route -n get default 2>/dev/null | awk '/interface:/ {print $2}' || true)"
    fi

    if [[ -n "${ifc:-}" ]] && ! is_virtual_iface "$ifc"; then
        echo "$ifc"; return 0
    fi

    # Cách 2 (fallback): quét interface đang UP, có IPv4 thật, không phải ảo
    if command -v ip >/dev/null 2>&1; then
        ip -o link show up 2>/dev/null | awk -F': ' '{print $2}' | while read -r ifc; do
            ifc="${ifc%%@*}"
            if ! is_virtual_iface "$ifc"; then
                if ip -4 addr show "$ifc" 2>/dev/null | grep -q "inet "; then
                    echo "$ifc"; return 0
                fi
            fi
        done
    elif command -v ifconfig >/dev/null 2>&1; then
        ifconfig -l 2>/dev/null | tr ' ' '\n' | while read -r ifc; do
            if ! is_virtual_iface "$ifc"; then
                if ifconfig "$ifc" 2>/dev/null | grep -qE "(status: active|inet )"; then
                    echo "$ifc"; return 0
                fi
            fi
        done
    fi

    return 1
}

show_iface_info() {
    local ifc="$1"
    if command -v ip >/dev/null 2>&1; then
        ip -4 addr show "$ifc" 2>/dev/null | awk '/inet /{print "    IP: "$2}'
    fi
}

# ── Bước 3: Kiểm tra quyền sniff ──────────────────────────────────────────────
# Trả về 0 nếu có thể sniff mà không cần sudo (root hoặc đã có cap_net_raw)
VENV_PYTHON="$VENV_DIR/bin/python"

can_sniff_without_sudo() {
    [[ "$EUID" -eq 0 ]] && return 0
    # Kiểm tra cap_net_raw trên binary python của venv (resolve symlink nếu có)
    if command -v getcap >/dev/null 2>&1 && [[ -f "$VENV_PYTHON" ]]; then
        local real_python
        real_python="$(readlink -f "$VENV_PYTHON")"
        getcap "$real_python" 2>/dev/null | grep -q "cap_net_raw" && return 0
    fi
    return 1
}

# ── Bước 4: Khởi động capture thật + dashboard ─────────────────────────────────
start_capture_and_dashboard() {
    local iface="${1:-}"
    local pcap="${2:-}"

    cd "$PROJECT_DIR"
    mkdir -p logs data/raw data/processed evaluation/results optimization/results

    if [[ -n "$pcap" ]]; then
        # Replay pcap — không cần quyền đặc biệt
        log "Replay PCAP: $pcap"
        python pipeline/capture.py --pcap "$pcap" &
        CAPTURE_PID=$!
        ok "Capture PCAP PID=$CAPTURE_PID"

    elif [[ -n "$iface" ]]; then
        log "Bắt traffic live trên interface: $iface"

        if can_sniff_without_sudo; then
            # Đã có quyền (root hoặc cap_net_raw) — chạy trực tiếp
            "$VENV_PYTHON" pipeline/capture.py --iface "$iface" &
            CAPTURE_PID=$!
        else
            # Thiếu quyền — tự động dùng sudo CHỈ cho subprocess Python
            warn "Thiếu quyền cap_net_raw — tự động dùng sudo cho capture process."
            warn "(Bạn có thể bỏ sudo sau này bằng: sudo setcap cap_net_raw+eip $VENV_PYTHON)"
            if ! command -v sudo >/dev/null 2>&1; then
                fail "Không có sudo. Hãy chạy lại với: sudo ./start.sh"
            fi
            # Dùng đường dẫn tuyệt đối của python trong venv — sudo bảo toàn package scapy
            sudo "$VENV_PYTHON" pipeline/capture.py --iface "$iface" &
            CAPTURE_PID=$!
        fi

        # Xác nhận process còn sống sau 2 giây
        sleep 2
        if ! kill -0 "$CAPTURE_PID" 2>/dev/null; then
            fail "Capture process khởi động thất bại (PID=$CAPTURE_PID). Xem lỗi ở trên."
        fi
        ok "Capture live PID=$CAPTURE_PID → ghi logs/pipeline_state.json + logs/alerts.jsonl"
    fi

    echo ""
    ok "Capture đang chạy — log ghi vào:"
    echo -e "  ${CYAN}$PROJECT_DIR/logs/alerts.jsonl${NC}        (mỗi dòng 1 alert)"
    echo -e "  ${CYAN}$PROJECT_DIR/logs/alerts_summary.json${NC}  (snapshot tổng hợp, load bằng json.load)"
    echo -e "  ${CYAN}$PROJECT_DIR/logs/pipeline_state.json${NC}  (trạng thái pipeline)"
    echo ""
    echo -e "  Nhấn ${YELLOW}Ctrl+C${NC} để dừng"
    echo ""

    # Chờ capture process kết thúc (Ctrl+C sẽ trigger cleanup và thoát)
    wait "$CAPTURE_PID" 2>/dev/null || true
}

# ── Health check ───────────────────────────────────────────────────────────────
health_check() {
    log "Kiểm tra model..."
    cd "$PROJECT_DIR"
    python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from configs.paths import get_model_dir
model_dir = get_model_dir()
required = ["config.json", "encoder_fp32.pt", "encoder_v5.onnx", "mu_norm.npy", "delta.npy"]
missing = [f for f in required if not (model_dir / f).exists()]
if missing:
    raise FileNotFoundError(f"Thiếu file model: {missing}")
print(f"  Model dir: {model_dir}  ✓")
PY
    ok "Model OK"
}

# ── Main ───────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo "=============================================="
    echo "  ET-SSL Edge AI — Traffic Anomaly Detection"
    echo "=============================================="
    echo ""

    command -v python3 >/dev/null 2>&1 || fail "Cần cài Python 3 (python3)"

    setup_venv

    local cmd="${1:-start}"
    shift || true

    case "$cmd" in
        start)
            # Parse args: --iface / --pcap
            local iface="" pcap=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --iface) iface="$2"; shift 2 ;;
                    --pcap)  pcap="$2";  shift 2 ;;
                    *) shift ;;
                esac
            done

            if [[ -n "$pcap" ]]; then
                # Replay pcap — không cần root
                health_check
                start_capture_and_dashboard "" "$pcap"

            elif [[ -n "$iface" ]]; then
                # Interface được chỉ định thủ công
                health_check
                log "Dùng interface được chỉ định: $iface"
                show_iface_info "$iface"
                if [[ "$EUID" -ne 0 ]]; then
                    warn "Bắt traffic live thường cần quyền root/cap_net_raw."
                    warn "Nếu lỗi permission, hãy chạy: sudo ./start.sh --iface $iface"
                fi
                start_capture_and_dashboard "$iface" ""

            else
                # Auto-detect card mạng chính — đây là flow mặc định
                health_check
                log "Đang tự xác định card mạng chính (default route)..."
                if ! iface="$(detect_primary_iface)"; then
                    fail "Không tự detect được card mạng nào. Hãy chỉ định thủ công: ./start.sh --iface <tên_card>"
                fi
                ok "Card mạng chính: $iface"
                show_iface_info "$iface"
                if [[ "$EUID" -ne 0 ]]; then
                    warn "Bắt traffic live cần quyền root/cap_net_raw."
                    warn "→ Chạy lại: sudo ./start.sh"
                    warn "  hoặc cấp capability: sudo setcap cap_net_raw+eip \$(which python3)"
                fi
                start_capture_and_dashboard "$iface" ""
            fi
            ;;

        check|health)
            health_check
            ;;

        test)
            health_check
            cd "$PROJECT_DIR"
            python run_all.py --quick
            ok "Test hoàn tất — xem evaluation/results/full_report.json"
            ;;

        pipeline)
            health_check
            cd "$PROJECT_DIR"
            python run_all.py "$@"
            ok "Pipeline hoàn tất"
            ;;

        dashboard)
            echo ""
            warn "Dashboard Streamlit đã bị gỡ bỏ. Hãy load file JSON để tự làm dashboard:"
            echo "  logs/alerts_summary.json  — json.load() để đọc toàn bộ dữ liệu"
            echo "  logs/alerts.jsonl         — mỗi dòng là 1 event JSON"
            ;;

        *)
            echo "Usage: $0 [start|check|test|pipeline] [args]"
            echo ""
            echo "  start                   Tự detect card mạng + bắt traffic thật + ghi log"
            echo "  start --iface wlan0     Ép dùng interface cụ thể"
            echo "  start --pcap file.pcap  Replay file pcap (không cần root)"
            echo "  check                   Kiểm tra model nhanh"
            echo "  test                    Evaluation quick test"
            echo "  pipeline [--quick]      Full evaluation suite"
            echo ""
            echo "Log file:"
            echo "  logs/alerts.jsonl         — từng alert dưới dạng JSON (append)"
            echo "  logs/alerts_summary.json  — snapshot tổng hợp (load bằng json.load)"
            echo "  logs/pipeline_state.json  — trạng thái pipeline"
            exit 1
            ;;
    esac
}

main "$@"
