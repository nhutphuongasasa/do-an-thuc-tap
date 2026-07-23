"""
watch_anomaly.py — Chạy file này ở MỘT TERMINAL RIÊNG, song song với start.sh.
Nó sẽ tự động đọc logs/flow_decisions.jsonl và CHỈ in ra những flow bị đánh dấu
là bất thường (is_anomaly = true), ngay khi vừa ghi vào file — không cần đọc
log rối mắt của start.sh nữa.

Cách dùng:
  Terminal 1: sudo bash start.sh
  Terminal 2: python watch_anomaly.py

Rồi cứ mở Tor/VPN lên dùng, nhìn Terminal 2 — thấy dòng nào hiện ra là có
anomaly mới, đọc cột "dst_ip" và "dst_port" để biết đó có phải traffic
Tor/VPN bạn vừa tạo hay không.
"""

import json
import time
from pathlib import Path

LOG_PATH = Path("logs/flow_decisions.jsonl")

print("=" * 70)
print("  Đang theo dõi anomaly ... (Ctrl+C để dừng)")
print("  Mở Tor/VPN lên dùng rồi nhìn ở đây.")
print("=" * 70)

# Đợi tới khi file log tồn tại (phòng trường hợp start.sh chưa kịp tạo)
while not LOG_PATH.exists():
    print("Đang chờ start.sh tạo file log...")
    time.sleep(1)

with LOG_PATH.open("r", encoding="utf-8") as f:
    # Nhảy tới cuối file — chỉ quan tâm dòng MỚI từ giờ trở đi
    f.seek(0, 2)

    while True:
        line = f.readline()
        if not line:
            time.sleep(0.3)
            continue

        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not rec.get("is_anomaly"):
            continue

        # flow_id dạng: src_ip-dst_ip-src_port-dst_port-proto
        parts = rec["flow_id"].split("-")
        dst_ip = parts[1] if len(parts) > 1 else "?"
        dst_port = parts[3] if len(parts) > 3 else "?"

        ts_str = time.strftime("%H:%M:%S", time.localtime(rec["ts"]))
        print(
            f"[{ts_str}] 🔴 ANOMALY  dst_ip={dst_ip:<16} dst_port={dst_port:<6} "
            f"score={rec['score']:.4f}  flow_id={rec['flow_id']}"
        )