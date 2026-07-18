"""
feature_extractor.py — Trích đặc trưng flow-level (PL, IPI, FD, PC, PM).

Không giải mã payload — chỉ dùng metadata gói tin (timing, length, flags, window).
20 feature thống kê map vào 5 nhóm theo readme:
  PL  — phân phối độ dài gói / payload
  IPI — inter-packet interval (delta time)
  FD  — flow duration (derived)
  PC  — packet count (derived)
  PM  — protocol metadata (TCP flags, init window)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import mode, skew

from data.feature_schema import MODEL_FEATURES

# Nhóm feature theo readme §1
FEATURE_GROUPS = {
    "PL": [
        "payload_bytes_median",
        "fwd_payload_bytes_median",
        "skewness_payload_bytes_delta_len",
        "fwd_min_header_bytes",
        "fwd_segment_size_min",
        "median_bwd_packets_delta_len",
        "mean_packets_delta_len",
        "cov_bwd_payload_bytes_delta_len",
    ],
    "IPI": [
        "median_packets_delta_time",
        "packets_IAT_mean",
        "fwd_packets_IAT_mean",
        "bwd_packets_IAT_mean",
        "mode_packets_delta_time",
        "mean_packets_delta_time",
    ],
    "FD": ["flow_duration_sec"],
    "PC": ["packet_count"],
    "PM": [
        "fwd_init_win_bytes",
        "bwd_init_win_bytes",
        "rst_flag_percentage_in_total",
        "rst_flag_counts",
        "psh_flag_percentage_in_total",
        "ack_flag_percentage_in_total",
    ],
}


@dataclass
class FlowRecord:
    """Buffer gói tin của một flow (5-tuple)."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packets: list[dict[str, Any]] = field(default_factory=list)
    start_time: float | None = None
    last_time: float | None = None
    has_fwd_fin: bool = False
    has_bwd_fin: bool = False
    is_finished: bool = False

    def add_packet(
        self,
        timestamp: float,
        is_forward: bool,
        header_len: int,
        payload_len: int,
        flags: str,
        win_bytes: int,
    ) -> None:
        if self.start_time is None:
            self.start_time = timestamp
        self.last_time = timestamp
        self.packets.append(
            {
                "time": timestamp,
                "is_forward": is_forward,
                "header_len": header_len,
                "payload_len": payload_len,
                "flags": flags,
                "win_bytes": win_bytes,
            }
        )
        # FIN: kết thúc chỉ khi CẢ 2 phía đã gửi FIN (4-way handshake hoàn tất)
        # Tránh tách flow sớm do half-close TCP
        if "F" in flags:
            if is_forward:
                self.has_fwd_fin = True
            else:
                self.has_bwd_fin = True
            if self.has_fwd_fin and self.has_bwd_fin:
                self.is_finished = True
        # RST: KHÔNG set is_finished ngay — post-RST packets (RST-ACK, in-flight ACK)
        # vẫn arrive sau RST và phải được gộp vào cùng flow (RFC 793 §3.4).
        # Flow sẽ tự expire qua flow_timeout_sec (30s idle) — đủ để bắt hết residual.

    @property
    def packet_count(self) -> int:
        return len(self.packets)

    @property
    def flow_duration_sec(self) -> float:
        if self.start_time is None or self.last_time is None:
            return 0.0
        return float(self.last_time - self.start_time)


def extract_flow_features(flow: FlowRecord) -> np.ndarray | None:
    """
    Trích vector 20 feature từ flow record.
    Trả về None nếu flow quá ngắn (< 2 packets).
    """
    if len(flow.packets) < 2:
        return None

    times = np.array([p["time"] for p in flow.packets])
    delta_times = np.diff(times)
    if len(delta_times) == 0:
        delta_times = np.array([0.0])

    fwd_pkts = [p for p in flow.packets if p["is_forward"]]
    bwd_pkts = [p for p in flow.packets if not p["is_forward"]]

    fwd_times = np.array([p["time"] for p in fwd_pkts])
    bwd_times = np.array([p["time"] for p in bwd_pkts])

    fwd_delta_times = np.diff(fwd_times) if len(fwd_times) > 1 else np.array([0.0])
    bwd_delta_times = np.diff(bwd_times) if len(bwd_times) > 1 else np.array([0.0])

    payloads = np.array([p["payload_len"] for p in flow.packets])
    fwd_payloads = np.array([p["payload_len"] for p in fwd_pkts]) if fwd_pkts else np.array([0])
    bwd_payloads = np.array([p["payload_len"] for p in bwd_pkts]) if bwd_pkts else np.array([0])

    delta_payloads = np.diff(payloads) if len(payloads) > 1 else np.array([0.0])
    bwd_delta_payloads = np.diff(bwd_payloads) if len(bwd_payloads) > 1 else np.array([0.0])

    total_pkts = len(flow.packets)
    rst_count = sum(1 for p in flow.packets if "R" in str(p["flags"]))
    psh_count = sum(1 for p in flow.packets if "P" in str(p["flags"]))
    ack_count = sum(1 for p in flow.packets if "A" in str(p["flags"]))

    fwd_header_lens = [p["header_len"] for p in fwd_pkts]
    fwd_min_header_bytes = min(fwd_header_lens) if fwd_header_lens else 0

    fwd_init_win_bytes = fwd_pkts[0]["win_bytes"] if fwd_pkts else 0
    bwd_init_win_bytes = bwd_pkts[0]["win_bytes"] if bwd_pkts else 0

    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    def safe_median(arr):
        return float(np.median(arr)) if len(arr) > 0 else 0.0

    def safe_mode(arr):
        if len(arr) == 0:
            return 0.0
        m = mode(arr, keepdims=True)
        return float(m.mode[0]) if len(m.mode) > 0 else 0.0

    def safe_skew(arr):
        if len(arr) < 2 or np.var(arr) == 0:
            return 0.0
        return float(skew(arr))

    def safe_cov(arr):
        if len(arr) == 0:
            return 0.0
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        if mean_val == 0:
            return 0.0
        # Fix the 2.5M score bug: Coefficient of Variation (std/mean) explodes 
        # when mean approaches zero (common in delta arrays). Clip to [-10, 10].
        cov = float(std_val / mean_val)
        return float(np.clip(cov, -10.0, 10.0))

    features = {
        "median_packets_delta_time": safe_median(delta_times),
        "packets_IAT_mean": safe_mean(delta_times),
        "fwd_init_win_bytes": fwd_init_win_bytes,
        "fwd_packets_IAT_mean": safe_mean(fwd_delta_times),
        "bwd_init_win_bytes": bwd_init_win_bytes,
        "rst_flag_percentage_in_total": rst_count / total_pkts,
        "mode_packets_delta_time": safe_mode(delta_times),
        "bwd_packets_IAT_mean": safe_mean(bwd_delta_times),
        "rst_flag_counts": rst_count,
        "psh_flag_percentage_in_total": psh_count / total_pkts,
        "fwd_min_header_bytes": fwd_min_header_bytes,
        "fwd_segment_size_min": fwd_min_header_bytes,
        "skewness_payload_bytes_delta_len": safe_skew(delta_payloads),
        "payload_bytes_median": safe_median(payloads),
        "median_bwd_packets_delta_len": safe_median(bwd_delta_payloads),
        "fwd_payload_bytes_median": safe_median(fwd_payloads),
        "mean_packets_delta_time": safe_mean(delta_times),
        "cov_bwd_payload_bytes_delta_len": safe_cov(bwd_delta_payloads),
        "mean_packets_delta_len": safe_mean(delta_payloads),
        "ack_flag_percentage_in_total": ack_count / total_pkts,
        # Derived groups FD / PC (metadata, không đưa vào model 20-dim)
        "flow_duration_sec": flow.flow_duration_sec,
        "packet_count": float(total_pkts),
    }

    return np.array([features[k] for k in MODEL_FEATURES], dtype=np.float32)
