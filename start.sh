
set -euo pipefail

# ── Màu sắc terminal ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

# ── Đường dẫn cố định ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/edge-ai-traffic-anomaly"
VENV_DIR="$SCRIPT_DIR/venv"
MODEL_DIR="$PROJECT_DIR/model/onnx_models"
LOG_DIR="$PROJECT_DIR/logs"

# ── Tham số mặc định ─────────────────────────────────────────────────────────
IFACE=""
BACKEND="onnx"
MAX_BATCH=32
MAX_WAIT_MS=200

# ── Parse tham số dòng lệnh ──────────────────────────────────────────────────
LIST_IFACES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iface)         IFACE="$2";       shift 2 ;;
        --backend)       BACKEND="$2";     shift 2 ;;
        --max_batch)     MAX_BATCH="$2";   shift 2 ;;
        --max_wait)      MAX_WAIT_MS="$2"; shift 2 ;;
        --list-ifaces)   LIST_IFACES=true; shift ;;
        --help|-h)
            echo "Cách dùng: sudo bash start.sh [tùy chọn]"
            echo ""
            echo "Tùy chọn:"
            echo "  --iface <tên>       Card mạng (vd: enp62s0, wlp61s0). Mặc định: tự động"
            echo "  --list-ifaces       Liệt kê tất cả card mạng rồi thoát"
            echo "  --backend <tên>     onnx | pkl. Mặc định: onnx"
            echo "  --max_batch <n>     Kích thước batch. Mặc định: 32"
            echo "  --max_wait <ms>     Thời gian chờ flush batch (ms). Mặc định: 200"
            echo ""
            echo "Ví dụ:"
            echo "  sudo bash start.sh"
            echo "  sudo bash start.sh --list-ifaces"
            echo "  sudo bash start.sh --iface enp62s0"
            echo "  sudo bash start.sh --iface wlp61s0 --max_batch 16 --max_wait 100"
            exit 0
            ;;
        *) echo -e "${RED}[!] Tham số không hợp lệ: $1${NC}"; exit 1 ;;
    esac
done

# ── Kiểm tra quyền root ──────────────────────────────────────────────────────
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  ET-SSL Edge Anomaly Detection — Khởi động${NC}"
echo -e "${CYAN}============================================================${NC}"

if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}[!] Script cần chạy với quyền root để mở raw socket.${NC}"
    echo -e "    Chạy lại: ${YELLOW}sudo bash start.sh${NC}"
    exit 1
fi
echo -e "${GREEN}[✓] Quyền root OK${NC}"

# ── Kiểm tra venv ────────────────────────────────────────────────────────────
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo -e "${RED}[!] Không tìm thấy virtual environment tại: $VENV_DIR${NC}"
    echo -e "    Tạo venv: ${YELLOW}python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt${NC}"
    exit 1
fi
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}[✓] Virtual environment: $VENV_DIR${NC}"

# ── Liệt kê card mạng rồi thoát nếu yêu cầu ─────────────────────────────────
if [[ "$LIST_IFACES" == "true" ]]; then
    echo -e "\n${CYAN}  Danh sách card mạng khả dụng:${NC}"
    cd "$PROJECT_DIR"
    python main.py --list-ifaces
    exit 0
fi

# ── Kiểm tra model ───────────────────────────────────────────────────────────
if [[ ! -d "$MODEL_DIR" ]]; then
    echo -e "${RED}[!] Không tìm thấy thư mục model: $MODEL_DIR${NC}"
    exit 1
fi

ONNX_FILE=$(ls "$MODEL_DIR"/*.onnx 2>/dev/null | head -1 || true)
if [[ -z "$ONNX_FILE" ]]; then
    echo -e "${YELLOW}[!] Không tìm thấy file .onnx trong $MODEL_DIR, thử dùng backend=pkl${NC}"
    BACKEND="pkl"
fi
echo -e "${GREEN}[✓] Model dir: $MODEL_DIR (backend=$BACKEND)${NC}"

# ── Tự động phát hiện card mạng nếu chưa chỉ định ───────────────────────────
if [[ -z "$IFACE" ]]; then
    # Ưu tiên: card có default route (ip route)
    IFACE=$(ip route show default 2>/dev/null | awk '/default/ && /dev/ {for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1 || true)

    # Fallback 1: bất kỳ card UP nào, bỏ qua lo/docker/virtual
    if [[ -z "$IFACE" ]]; then
        IFACE=$(ip -o link show up | awk -F': ' '{print $2}' | \
                grep -vE '^(lo|docker|br-|veth|virbr|vmnet|tun|tap)' | head -1 || true)
    fi

    # Fallback 2: dùng Python để lấy interface
    if [[ -z "$IFACE" ]]; then
        cd "$PROJECT_DIR"
        IFACE=$(python -c "
from pipe_line.capture import list_interfaces
ifaces = list_interfaces()
skip = ('lo', 'docker', 'br-', 'veth', 'virbr', 'tun', 'tap')
for i in ifaces:
    if not any(i['name'].startswith(s) for s in skip) and i.get('ips'):
        print(i['name']); break
" 2>/dev/null || true)
    fi

    if [[ -z "$IFACE" ]]; then
        echo -e "${RED}[!] Không thể tự động phát hiện card mạng.${NC}"
        echo -e "    Dùng: ${YELLOW}sudo bash start.sh --list-ifaces${NC}  để xem danh sách"
        echo -e "    Sau đó: ${YELLOW}sudo bash start.sh --iface <tên>${NC}"
        exit 1
    fi
    echo -e "${YELLOW}[~] Tự động chọn interface: $IFACE${NC}"
fi

# ── Kiểm tra interface tồn tại ───────────────────────────────────────────────
if ip link show "$IFACE" &>/dev/null; then
    echo -e "${GREEN}[✓] Interface: $IFACE${NC}"
else
    echo -e "${RED}[!] Interface '$IFACE' không tồn tại.${NC}"
    echo -e "    Dùng: ${YELLOW}sudo bash start.sh --list-ifaces${NC}  để xem danh sách"
    exit 1
fi

# ── Chuẩn bị thư mục log ─────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
echo -e "${GREEN}[✓] Log dir: $LOG_DIR${NC}"

# ── In cấu hình sẽ chạy ──────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}  Cấu hình:${NC}"
echo -e "  • Interface : ${YELLOW}$IFACE${NC}"
echo -e "  • Backend   : ${YELLOW}$BACKEND${NC}"
echo -e "  • Batch size: ${YELLOW}$MAX_BATCH${NC} flows"
echo -e "  • Max wait  : ${YELLOW}$MAX_WAIT_MS${NC} ms"
echo -e "  • Log dir   : ${YELLOW}$LOG_DIR${NC}"
echo ""
echo -e "${CYAN}  Log files:${NC}"
echo -e "  • ${YELLOW}$LOG_DIR/flow_decisions.jsonl${NC}  — mỗi flow 1 dòng JSON"
echo -e "  • ${YELLOW}$LOG_DIR/stats_summary.json${NC}    — tóm tắt mỗi 60 giây"
echo ""
echo -e "${GREEN}  Nhấn Ctrl+C để dừng sạch sẽ.${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# ── Chạy pipeline ────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

exec python main.py \
    --iface      "$IFACE"      \
    --model_dir  "$MODEL_DIR"  \
    --backend    "$BACKEND"    \
    --max_batch  "$MAX_BATCH"  \
    --max_wait_ms "$MAX_WAIT_MS"
