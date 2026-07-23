"""
main.py — Nối 4 module lại bằng threading + queue.Queue.

Sơ đồ luồng:
  capture.py  --[packet_queue]--> flow_tracker.py
                                       |
                                  [flow_queue]
                                       |
                              batch_inference.py
                                       |
                                 [result_queue]
                                       |
                                   logger.py

Tất cả queue có maxsize cứng để tránh tràn RAM trên Pi.
"""

import argparse
import logging
import platform
import signal
import sys
import threading
from queue import Queue
from pathlib import Path

# Thêm thư mục gốc vào path để import module
sys.path.insert(0, str(Path(__file__).parent))

from pipe_line import capture, flow_tracker, batch_inference, logger
from pipe_line.feature_schema import MODEL_FEATURES

IS_WINDOWS = platform.system() == "Windows"

# ── Cấu hình log nội bộ (không phải log flow ra terminal) ──────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def load_engine(model_dir: str, backend: str = "onnx"):
    """
    Load ETSSLInference. Import ở đây để main.py không crash nếu
    onnxruntime chưa cài (giúp test các module khác riêng lẻ).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from model.inference import ETSSLInference
    return ETSSLInference(model_dir, backend=backend)


def print_interfaces():
    """In danh sách card mạng ra terminal theo dạng bảng dễ đọc."""
    ifaces = capture.list_interfaces()
    if not ifaces:
        print("[!] Không tìm thấy card mạng nào.")
        if IS_WINDOWS:
            print("    → Đảm bảo Npcap đã cài: https://npcap.com")
            print("    → Chạy terminal với quyền Administrator.")
        else:
            print("    → Thử chạy lại với: sudo python main.py --list-ifaces")
        return

    width_name = max(len(i["name"]) for i in ifaces) + 2
    width_desc = max((len(i["description"]) for i in ifaces), default=0) + 2

    sep = f"{'─' * (width_name + width_desc + 26)}"
    print(sep)
    print(f"  {'Interface':<{width_name}} {'Description':<{width_desc}} {'IP(s)'}")
    print(sep)
    for idx, i in enumerate(ifaces, 1):
        ips_str = ", ".join(i["ips"]) if i["ips"] else "(không có IP)"
        desc = i["description"] or ""
        print(f"  [{idx:2}] {i['name']:<{width_name - 5}} {desc:<{width_desc}} {ips_str}")
    print(sep)
    print(f"  Tổng: {len(ifaces)} interface\n")
    print("  Cách dùng:")
    if IS_WINDOWS:
        print("    python main.py --iface <tên>")
        print("    Ví dụ: python main.py --iface \"Ethernet\"")
    else:
        print("    sudo python main.py --iface <tên>")
        print("    Ví dụ: sudo python main.py --iface eth0")


def auto_detect_iface() -> str:
    """
    Tự động phát hiện interface mạng phù hợp nhất.
    Ưu tiên card có default route, bỏ qua loopback và virtual interface.
    Hoạt động trên cả Linux và Windows.
    """
    SKIP_PREFIXES = ("lo", "docker", "br-", "veth", "virbr", "vmnet",
                     "vbox", "tun", "tap", "Loopback", "Teredo")

    # ── Linux: đọc default route từ ip route ─────────────────────────────────
    if not IS_WINDOWS:
        try:
            import subprocess
            out = subprocess.check_output(
                ["ip", "route", "show", "default"],
                stderr=subprocess.DEVNULL, text=True
            )
            for line in out.splitlines():
                parts = line.split()
                if "dev" in parts:
                    dev = parts[parts.index("dev") + 1]
                    if not any(dev.startswith(p) for p in SKIP_PREFIXES):
                        return dev
        except Exception:
            pass

    # ── Fallback chung: lấy từ list_interfaces (Scapy/psutil) ────────────────
    ifaces = capture.list_interfaces()
    for iface in ifaces:
        name = iface["name"]
        if any(name.startswith(p) for p in SKIP_PREFIXES):
            continue
        if iface.get("ips"):          # ưu tiên card có IP
            return name

    # Lấy bất kỳ card nào không phải loopback
    for iface in ifaces:
        name = iface["name"]
        if not any(name.startswith(p) for p in SKIP_PREFIXES):
            return name

    return ""


def main():
    parser = argparse.ArgumentParser(
        description="ET-SSL Edge Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Linux
  sudo python main.py --list-ifaces
  sudo python main.py --iface eth0
  sudo python main.py --iface wlan0 --max_batch 16 --max_wait_ms 100

  # Windows (PowerShell Administrator)
  python main.py --list-ifaces
  python main.py --iface "Ethernet"
  python main.py --iface "Wi-Fi" --backend onnx
"""
    )
    parser.add_argument(
        "--iface", default="",
        help="Tên card mạng (vd: eth0, wlan0, 'Ethernet'). "
             "Mặc định: tự động phát hiện. Dùng --list-ifaces để xem danh sách."
    )
    parser.add_argument(
        "--list-ifaces", action="store_true",
        help="Liệt kê tất cả card mạng khả dụng rồi thoát."
    )
    parser.add_argument("--model_dir",   default="model/onnx_models", help="Thư mục chứa ONNX model")
    parser.add_argument("--backend",     default="onnx",  choices=["onnx", "pkl"], help="Backend inference")
    parser.add_argument("--max_batch",   type=int,   default=32,    help="Kích thước batch")
    parser.add_argument("--max_wait_ms", type=float, default=200.0, help="Timeout flush batch (ms)")
    args = parser.parse_args()

    # ── Lệnh --list-ifaces: hiển thị danh sách rồi thoát ──────────────────
    if args.list_ifaces:
        print_interfaces()
        sys.exit(0)

    # ── Tự động phát hiện interface nếu không chỉ định ────────────────────
    iface = args.iface.strip()
    if not iface:
        iface = auto_detect_iface()
        if not iface:
            print("[!] Không thể tự động phát hiện card mạng.")
            print("    Dùng --list-ifaces để xem danh sách, sau đó --iface <tên>.")
            sys.exit(1)
        print(f"[main] Tự động chọn interface: {iface}")

    print(f"[main] Khởi động | iface={iface} | model={args.model_dir} | "
          f"batch={args.max_batch} | wait={args.max_wait_ms}ms")
    if IS_WINDOWS:
        print("[main] Chạy trên Windows — đảm bảo Npcap đã cài và terminal có quyền Administrator.")

    # ── Load model ──────────────────────────────────────────────────────────
    engine = load_engine(args.model_dir, args.backend)

    # ── Queue giữa các tầng (maxsize cứng để tránh OOM trên Pi) ───────────
    packet_queue = Queue(maxsize=1000)   # capture  -> flow_tracker
    flow_queue   = Queue(maxsize=500)    # flow_tracker -> batch_inference
    result_queue = Queue(maxsize=500)    # batch_inference -> logger

    # ── Dict chia sẻ thống kê (thread-safe với GIL, chỉ ghi int/float) ────
    stats = {
        "pkt_total":   0,
        "queue_drops": 0,
        "flow_active": 0,
    }

    stop_event = threading.Event()

    # ── Xử lý tín hiệu dừng (Ctrl+C / SIGTERM) ──────────────────────────────
    def handle_stop(sig, frame):
        print("\n[main] Đang dừng...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_stop)
    # SIGTERM không có trên Windows (bị bỏ qua), chỉ đăng ký trên Linux
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, handle_stop)

    # ── Khởi động 4 thread ──────────────────────────────────────────────────
    threads = [
        threading.Thread(
            target=capture.run,
            args=(iface, packet_queue, stats, stop_event),
            name="capture", daemon=True,
        ),
        threading.Thread(
            target=flow_tracker.run,
            args=(packet_queue, flow_queue, stats, stop_event),
            name="flow_tracker", daemon=True,
        ),
        threading.Thread(
            target=batch_inference.run,
            args=(flow_queue, result_queue, engine, stats, stop_event,
                  args.max_batch, args.max_wait_ms),
            name="batch_inference", daemon=True,
        ),
        threading.Thread(
            target=logger.run,
            args=(result_queue, stats, stop_event, MODEL_FEATURES),
            name="logger", daemon=True,
        ),
    ]

    for t in threads:
        t.start()
        print(f"[main] Thread '{t.name}' đã khởi động.")

    # ── Vòng lặp chính chỉ chờ stop ────────────────────────────────────────
    for t in threads:
        t.join()

    print("[main] Đã dừng hoàn toàn.")


if __name__ == "__main__":
    main()


# =============================================================================
# HƯỚNG DẪN SỬ DỤNG
# =============================================================================
#
# ── Linux ─────────────────────────────────────────────────────────────────────
# 1. Xem danh sách card mạng:
#    sudo python main.py --list-ifaces
#
# 2. Chạy với card mạng cụ thể:
#    sudo python main.py --iface eth0
#    sudo python main.py --iface wlan0 --max_batch 16 --max_wait_ms 100
#
# ── Windows ───────────────────────────────────────────────────────────────────
# 3. Cài Npcap từ https://npcap.com (bắt buộc)
#
# 4. Mở PowerShell / CMD với quyền Administrator
#
# 5. Xem danh sách card mạng:
#    python main.py --list-ifaces
#
# 6. Chạy với card mạng cụ thể (tên có thể có dấu cách):
#    python main.py --iface "Ethernet"
#    python main.py --iface "Wi-Fi"
#    python main.py --iface "Local Area Connection"
#
# ── Chung ─────────────────────────────────────────────────────────────────────
# 7. Chỉnh batch size / wait time:
#    --max_batch 64   : batch lớn hơn = throughput cao hơn, latency cao hơn
#    --max_wait_ms 50 : flush sớm hơn = latency thấp hơn, throughput thấp hơn
#
# 8. Đọc log anomaly bằng jq:
#    cat logs/flow_decisions.jsonl | jq 'select(.is_anomaly == true)'
#    cat logs/flow_decisions.jsonl | jq '[.score] | sort | reverse | .[:10]'
#
# 9. Cấu trúc log flow_decisions.jsonl (mỗi dòng 1 JSON):
#    {
#      "ts": 1721234567.89,       # Unix timestamp
#      "flow_id": "ip-ip-p-p-6", # 5-tuple
#      "score": 0.0234,           # Điểm bất thường
#      "delta_effective": 0.5,    # Ngưỡng Δ hiện tại
#      "is_anomaly": false,
#      "packet_count": 42,
#      "duration_s": 1.23,
#      "top_features": {...}      # 5 đặc trưng có giá trị cao nhất
#    }
# =============================================================================
