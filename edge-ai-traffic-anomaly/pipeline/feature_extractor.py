"""
feature_extractor.py — Trích xuất đặc trưng cấp độ luồng (Flow-level Feature Extraction).

Không thực hiện giải mã payload (Deep Packet Inspection), chỉ dùng siêu dữ liệu gói tin 
(thời gian, độ dài, cờ TCP, kích thước cửa sổ).
20 đặc trưng thống kê được ánh xạ vào 5 nhóm (tham chiếu README):
  PL  — Phân phối độ dài gói tin / payload
  IPI — Khoảng thời gian giữa các gói tin (Inter-packet interval)
  FD  — Thời lượng luồng (Flow duration)
  PC  — Số lượng gói tin (Packet count)
  PM  — Siêu dữ liệu giao thức (Protocol metadata)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import mode, skew

from data.feature_schema import MODEL_FEATURES

# Ngưỡng giới hạn cho hệ số biến thiên (Coefficient of Variation) để tránh bùng nổ giá trị
MAX_COV_CLIP: float = 10.0
MIN_COV_CLIP: float = -10.0

# Phân nhóm đặc trưng theo tài liệu tham khảo
FEATURE_GROUPS: dict[str, list[str]] = {
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
    """Bộ đệm lưu trữ các gói tin thuộc cùng một luồng mạng (được định danh qua 5-tuple)."""

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
        """Thêm gói tin mới vào luồng và cập nhật trạng thái kết nối TCP."""
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
        # Kết thúc luồng chỉ khi CẢ HAI hướng đều đã gửi cờ FIN (hoàn tất 4-way handshake)
        if "F" in flags:
            if is_forward:
                self.has_fwd_fin = True
            else:
                self.has_bwd_fin = True
            if self.has_fwd_fin and self.has_bwd_fin:
                self.is_finished = True

    @property
    def packet_count(self) -> int:
        """Tổng số gói tin hiện có trong luồng."""
        return len(self.packets)

    @property
    def flow_duration_sec(self) -> float:
        """Thời gian tồn tại của luồng (giây)."""
        if self.start_time is None or self.last_time is None:
            return 0.0
        return float(self.last_time - self.start_time)


def _safe_mean(arr: np.ndarray) -> float:
    """Tính trung bình an toàn, chống chia cho mảng rỗng."""
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def _safe_median(arr: np.ndarray) -> float:
    """Tính trung vị an toàn, chống lỗi mảng rỗng."""
    return float(np.median(arr)) if len(arr) > 0 else 0.0


def _safe_mode(arr: np.ndarray) -> float:
    """Tính giá trị phổ biến nhất (mode) an toàn."""
    if len(arr) == 0:
        return 0.0
    m = mode(arr, keepdims=True)
    return float(m.mode[0]) if len(m.mode) > 0 else 0.0


def _safe_skew(arr: np.ndarray) -> float:
    """Tính độ lệch phân phối (skewness) an toàn, tránh lỗi chia cho phương sai bằng 0."""
    if len(arr) < 2 or np.var(arr) == 0:
        return 0.0
    return float(skew(arr))


def _safe_cov(arr: np.ndarray) -> float:
    """
    Tính hệ số biến thiên (Coefficient of Variation - CV) an toàn.
    Cắt gọt giá trị để tránh bùng nổ (CV explosion) khi trung bình tiến dần tới 0.
    """
    if len(arr) == 0:
        return 0.0
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    if mean_val == 0:
        return 0.0
    cov = float(std_val / mean_val)
    return float(np.clip(cov, MIN_COV_CLIP, MAX_COV_CLIP))


def _compute_timing_features(
    flow: FlowRecord, fwd_pkts: list[dict[str, Any]], bwd_pkts: list[dict[str, Any]]
) -> dict[str, float]:
    """Tính toán nhóm đặc trưng khoảng thời gian giữa các gói tin (IPI)."""
    times = np.array([p["time"] for p in flow.packets])
    delta_times = np.diff(times)
    if len(delta_times) == 0:
        delta_times = np.array([0.0])

    fwd_times = np.array([p["time"] for p in fwd_pkts])
    bwd_times = np.array([p["time"] for p in bwd_pkts])

    fwd_delta_times = np.diff(fwd_times) if len(fwd_times) > 1 else np.array([0.0])
    bwd_delta_times = np.diff(bwd_times) if len(bwd_times) > 1 else np.array([0.0])

    return {
        "median_packets_delta_time": _safe_median(delta_times),
        "packets_IAT_mean": _safe_mean(delta_times),
        "fwd_packets_IAT_mean": _safe_mean(fwd_delta_times),
        "bwd_packets_IAT_mean": _safe_mean(bwd_delta_times),
        "mode_packets_delta_time": _safe_mode(delta_times),
        "mean_packets_delta_time": _safe_mean(delta_times),
    }


def _compute_payload_features(
    flow: FlowRecord, fwd_pkts: list[dict[str, Any]], bwd_pkts: list[dict[str, Any]]
) -> dict[str, float]:
    """Tính toán nhóm đặc trưng phân phối kích thước payload (PL)."""
    payloads = np.array([p["payload_len"] for p in flow.packets])
    fwd_payloads = np.array([p["payload_len"] for p in fwd_pkts]) if fwd_pkts else np.array([0])
    bwd_payloads = np.array([p["payload_len"] for p in bwd_pkts]) if bwd_pkts else np.array([0])

    delta_payloads = np.diff(payloads) if len(payloads) > 1 else np.array([0.0])
    bwd_delta_payloads = np.diff(bwd_payloads) if len(bwd_payloads) > 1 else np.array([0.0])

    fwd_header_lens = [p["header_len"] for p in fwd_pkts]
    fwd_min_header_bytes = min(fwd_header_lens) if fwd_header_lens else 0

    return {
        "fwd_min_header_bytes": float(fwd_min_header_bytes),
        "fwd_segment_size_min": float(fwd_min_header_bytes),
        "skewness_payload_bytes_delta_len": _safe_skew(delta_payloads),
        "payload_bytes_median": _safe_median(payloads),
        "median_bwd_packets_delta_len": _safe_median(bwd_delta_payloads),
        "fwd_payload_bytes_median": _safe_median(fwd_payloads),
        "cov_bwd_payload_bytes_delta_len": _safe_cov(bwd_delta_payloads),
        "mean_packets_delta_len": _safe_mean(delta_payloads),
    }


def _compute_protocol_features(
    flow: FlowRecord, fwd_pkts: list[dict[str, Any]], bwd_pkts: list[dict[str, Any]]
) -> dict[str, float]:
    """Tính toán nhóm đặc trưng siêu dữ liệu giao thức (PM)."""
    total_pkts = len(flow.packets)
    rst_count = sum(1 for p in flow.packets if "R" in str(p["flags"]))
    psh_count = sum(1 for p in flow.packets if "P" in str(p["flags"]))
    ack_count = sum(1 for p in flow.packets if "A" in str(p["flags"]))

    fwd_init_win_bytes = fwd_pkts[0]["win_bytes"] if fwd_pkts else 0
    bwd_init_win_bytes = bwd_pkts[0]["win_bytes"] if bwd_pkts else 0

    return {
        "fwd_init_win_bytes": float(fwd_init_win_bytes),
        "bwd_init_win_bytes": float(bwd_init_win_bytes),
        "rst_flag_percentage_in_total": float(rst_count) / total_pkts,
        "rst_flag_counts": float(rst_count),
        "psh_flag_percentage_in_total": float(psh_count) / total_pkts,
        "ack_flag_percentage_in_total": float(ack_count) / total_pkts,
    }


def extract_flow_features(flow: FlowRecord) -> np.ndarray | None:
    """
    Trích xuất vector 20 đặc trưng (features) chuẩn từ một luồng mạng.

    Args:
        flow: Luồng mạng chứa danh sách các gói tin.

    Returns:
        Mảng NumPy chứa vector 20 chiều, hoặc None nếu số lượng gói tin không đủ.
    """
    if len(flow.packets) < 2:
        return None

    fwd_pkts = [p for p in flow.packets if p["is_forward"]]
    bwd_pkts = [p for p in flow.packets if not p["is_forward"]]

    timing_feats = _compute_timing_features(flow, fwd_pkts, bwd_pkts)
    payload_feats = _compute_payload_features(flow, fwd_pkts, bwd_pkts)
    protocol_feats = _compute_protocol_features(flow, fwd_pkts, bwd_pkts)

    # Gộp toàn bộ các nhóm đặc trưng
    features = {**timing_feats, **payload_feats, **protocol_feats}

    # Ánh xạ thành mảng NumPy dựa trên lược đồ chuẩn
    return np.array([features[k] for k in MODEL_FEATURES], dtype=np.float32)
