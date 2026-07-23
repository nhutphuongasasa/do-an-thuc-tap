"""
feature_extractor.py — Tính feature vector 20 chiều từ dữ liệu thô của một Flow.

Hàm duy nhất: extract(flow_data) -> np.ndarray shape (20,)
flow_data là dict do flow_tracker.py cung cấp.
"""

import math
import numpy as np
from pipe_line.feature_schema import MODEL_FEATURES


def _safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def _safe_std(lst):
    if len(lst) < 2:
        return 0.0
    m = _safe_mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))


def extract(flow_data: dict) -> np.ndarray:
    """
    Chuyển dict thô của flow thành vector numpy 20 chiều.
    Gọi ngay sau khi flow đóng (timeout/FIN/RST).
    """
    pkts = flow_data["packets"]  # list of (timestamp, size, is_fwd, flags, header_len)
    if not pkts:
        return np.zeros(len(MODEL_FEATURES), dtype=np.float32)

    first_ts = pkts[0][0]
    last_ts  = pkts[-1][0]
    duration = max(last_ts - first_ts, 1e-6)  # tránh chia 0

    fwd = [p for p in pkts if p[2]]   # is_fwd=True
    bwd = [p for p in pkts if not p[2]]

    byte_count     = sum(p[1] for p in pkts)
    fwd_byte_count = sum(p[1] for p in fwd)
    bwd_byte_count = sum(p[1] for p in bwd)

    pkt_count = len(pkts)
    avg_pkt   = byte_count / pkt_count

    # Inter-arrival times (giây)
    all_ts  = [p[0] for p in pkts]
    fwd_ts  = [p[0] for p in fwd]
    bwd_ts  = [p[0] for p in bwd]

    flow_iats = [all_ts[i+1] - all_ts[i] for i in range(len(all_ts)-1)]
    fwd_iats  = [fwd_ts[i+1] - fwd_ts[i]  for i in range(len(fwd_ts)-1)]
    bwd_iats  = [bwd_ts[i+1] - bwd_ts[i]  for i in range(len(bwd_ts)-1)]

    # Đếm cờ TCP
    def count_flag(flag_char):
        return sum(1 for p in pkts if flag_char in str(p[3]))

    values = {
        "duration":             duration,
        "packet_count":         float(pkt_count),
        "byte_count":           float(byte_count),
        "fwd_packet_count":     float(len(fwd)),
        "bwd_packet_count":     float(len(bwd)),
        "fwd_byte_count":       float(fwd_byte_count),
        "bwd_byte_count":       float(bwd_byte_count),
        "avg_packet_size":      avg_pkt,
        "avg_fwd_packet_size":  fwd_byte_count / max(len(fwd), 1),
        "avg_bwd_packet_size":  bwd_byte_count / max(len(bwd), 1),
        "fwd_iat_mean":         _safe_mean(fwd_iats),
        "bwd_iat_mean":         _safe_mean(bwd_iats),
        "flow_iat_mean":        _safe_mean(flow_iats),
        "flow_iat_std":         _safe_std(flow_iats),
        "fwd_header_len":       float(sum(p[4] for p in fwd)),
        "bwd_header_len":       float(sum(p[4] for p in bwd)),
        "syn_flag_count":       float(count_flag("S")),
        "fin_flag_count":       float(count_flag("F")),
        "rst_flag_count":       float(count_flag("R")),
        "ack_flag_count":       float(count_flag("A")),
    }

    return np.array([values[f] for f in MODEL_FEATURES], dtype=np.float32)
