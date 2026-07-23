"""
flow_tracker.py — Gom gói tin thành Flow theo 5-tuple, đóng flow khi timeout/FIN/RST.

Flow key: (src_ip, dst_ip, src_port, dst_port, proto) — hướng nhỏ-lớn để 2 chiều
khớp cùng 1 key.

Đầu ra: khi flow đóng, đẩy (flow_id, flow_data, feature_vector) vào flow_queue.
"""

import time
import logging
import numpy as np
from queue import Queue, Full
from pipe_line.feature_extractor import extract

log = logging.getLogger("flow_tracker")

FLOW_TIMEOUT_S   = 60.0   # đóng flow nếu không thấy gói tin trong 60s
FLOW_MAX_PKTS    = 2000   # giới hạn để tránh flow lớn bất thường chiếm RAM
CLEANUP_EVERY_N  = 500    # dọn flow hết hạn sau mỗi N gói tin


def _make_key(src_ip, dst_ip, src_port, dst_port, proto):
    # Chuẩn hóa key theo hướng nhỏ-lớn để forward/backward khớp cùng 1 flow
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    lo, hi = (a, b) if a < b else (b, a)
    return (lo[0], hi[0], lo[1], hi[1], proto)


def _is_forward(src_ip, src_port, flow_key):
    # Gói tin là forward nếu src khớp bên "lo" của key
    return (src_ip, src_port) <= (flow_key[1], flow_key[3])


def _close_flow(key, flow, flow_queue: Queue, stats: dict):
    """Tính feature rồi đẩy vào flow_queue. Bỏ qua flow < 2 gói (không đủ thống kê)."""
    if len(flow["packets"]) < 2:
        return

    feat = extract(flow)  # -> np.ndarray (20,)
    flow_id = "{}-{}-{}-{}-{}".format(*key)

    try:
        flow_queue.put_nowait((flow_id, flow, feat))
    except Full:
        stats["queue_drops"] = stats.get("queue_drops", 0) + 1


def run(packet_queue: Queue, flow_queue: Queue, stats: dict, stop_event):
    """
    Vòng lặp chính: đọc packet_queue, cập nhật flow table, đóng flow khi cần.
    """
    flow_table: dict = {}   # key -> flow_data dict
    pkt_since_cleanup = 0

    while not stop_event.is_set() or not packet_queue.empty():
        # Lấy gói tin, chờ tối đa 0.5s để không block mãi khi stop
        try:
            pkt = packet_queue.get(timeout=0.5)
        except Exception:
            continue

        ts, src_ip, dst_ip, src_port, dst_port, proto, pkt_len, header_len, flags = pkt
        key = _make_key(src_ip, dst_ip, src_port, dst_port, proto)
        is_fwd = _is_forward(src_ip, src_port, key)

        if key not in flow_table:
            flow_table[key] = {
                "start_ts":  ts,
                "last_seen": ts,
                "packets":   [],  # list of (ts, size, is_fwd, flags, header_len)
            }

        flow = flow_table[key]
        flow["last_seen"] = ts
        flow["packets"].append((ts, pkt_len, is_fwd, flags, header_len))

        # Đóng flow ngay nếu thấy FIN hoặc RST
        if "F" in flags or "R" in flags:
            _close_flow(key, flow, flow_queue, stats)
            del flow_table[key]

        # Đóng flow nếu đã đạt giới hạn gói (tránh leak RAM)
        elif len(flow["packets"]) >= FLOW_MAX_PKTS:
            _close_flow(key, flow, flow_queue, stats)
            del flow_table[key]

        pkt_since_cleanup += 1
        stats["flow_active"] = len(flow_table)

        # Dọn flow hết hạn (timeout) định kỳ, không cần dọn mỗi gói
        if pkt_since_cleanup >= CLEANUP_EVERY_N:
            pkt_since_cleanup = 0
            now = time.monotonic()
            expired = [k for k, f in flow_table.items()
                       if now - f["last_seen"] > FLOW_TIMEOUT_S]
            for k in expired:
                _close_flow(k, flow_table[k], flow_queue, stats)
                del flow_table[k]

    # Khi stop: flush nốt các flow còn lại trong bảng
    for k, f in flow_table.items():
        _close_flow(k, f, flow_queue, stats)
    log.info("Flow tracker đã dừng. Đã flush %d flow còn lại.", len(flow_table))
