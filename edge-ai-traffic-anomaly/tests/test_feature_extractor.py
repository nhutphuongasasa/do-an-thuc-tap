"""
test_feature_extractor.py — Bảo vệ tính đúng đắn của extract() trước/sau tối ưu.

Expected values được tính TAY từ định nghĩa của từng feature,
KHÔNG phụ thuộc vào implementation cũ hay mới.

Chạy:
    cd edge-ai-traffic-anomaly
    python -m pytest tests/test_feature_extractor.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Đảm bảo import được từ thư mục project
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipe_line.feature_extractor import extract
from pipe_line.feature_schema import MODEL_FEATURES


# ── Helper: build running_stats từ packet list (mô phỏng flow_tracker) ────────

def _welford_update(count, mean, M2, x):
    """Welford online mean/variance — dùng để xây dựng test fixture."""
    count += 1
    delta = x - mean
    mean += delta / count
    M2 += delta * (x - mean)
    return count, mean, M2


def build_stats(pkts):
    """
    Chuyển list packet (ts, size, is_fwd, flags, header_len) thành
    running_stats dict — cùng thuật toán với flow_tracker._update_stats().
    """
    if not pkts:
        return {}

    stats = {
        "first_ts": pkts[0][0], "last_ts": pkts[0][0],
        "byte_sum": 0,
        # Forward
        "fwd_count": 0, "fwd_byte_sum": 0, "fwd_header_sum": 0,
        "fwd_prev_ts": None, "fwd_iat_count": 0, "fwd_iat_mean": 0.0,
        # Backward
        "bwd_count": 0, "bwd_byte_sum": 0, "bwd_header_sum": 0,
        "bwd_prev_ts": None, "bwd_iat_count": 0, "bwd_iat_mean": 0.0,
        # Flow IAT — cần std nên dùng Welford đầy đủ
        "flow_prev_ts": None,
        "flow_iat_count": 0, "flow_iat_mean": 0.0, "flow_iat_M2": 0.0,
        # TCP flags
        "syn": 0, "fin": 0, "rst": 0, "ack": 0,
    }

    for ts, size, is_fwd, flags, header_len in pkts:
        stats["last_ts"] = ts
        stats["byte_sum"] += size

        # Flow IAT (bỏ qua gói đầu — chưa có prev)
        if stats["flow_prev_ts"] is not None:
            iat = ts - stats["flow_prev_ts"]
            (stats["flow_iat_count"], stats["flow_iat_mean"], stats["flow_iat_M2"]) = \
                _welford_update(stats["flow_iat_count"], stats["flow_iat_mean"],
                                stats["flow_iat_M2"], iat)
        stats["flow_prev_ts"] = ts

        if is_fwd:
            if stats["fwd_prev_ts"] is not None:
                f_iat = ts - stats["fwd_prev_ts"]
                stats["fwd_iat_count"] += 1
                stats["fwd_iat_mean"] += (f_iat - stats["fwd_iat_mean"]) / stats["fwd_iat_count"]
            stats["fwd_prev_ts"] = ts
            stats["fwd_count"] += 1
            stats["fwd_byte_sum"] += size
            stats["fwd_header_sum"] += header_len
        else:
            if stats["bwd_prev_ts"] is not None:
                b_iat = ts - stats["bwd_prev_ts"]
                stats["bwd_iat_count"] += 1
                stats["bwd_iat_mean"] += (b_iat - stats["bwd_iat_mean"]) / stats["bwd_iat_count"]
            stats["bwd_prev_ts"] = ts
            stats["bwd_count"] += 1
            stats["bwd_byte_sum"] += size
            stats["bwd_header_sum"] += header_len

        if "S" in flags: stats["syn"] += 1
        if "F" in flags: stats["fin"] += 1
        if "R" in flags: stats["rst"] += 1
        if "A" in flags: stats["ack"] += 1

    return stats


def make_flow(pkts):
    """Tạo flow_data dict với format mới (stats dict)."""
    return {"stats": build_stats(pkts)}


# ── Test fixtures với expected values tính tay ────────────────────────────────

# Flow 1: Tối thiểu — 2 gói, 1 chiều (fwd only), 1 IAT
# Kiểm tra: bwd=0, flow_iat_std=0 (1 IAT), avg_bwd=0
PKTS_FLOW1 = [
    # (ts,  size, is_fwd, flags, header_len)
    (0.0,  100,  True,  "S",  20),
    (1.0,  200,  True,  "A",  20),
]
EXPECTED_FLOW1 = {
    "duration":            1.0,
    "packet_count":        2.0,
    "byte_count":          300.0,
    "fwd_packet_count":    2.0,
    "bwd_packet_count":    0.0,
    "fwd_byte_count":      300.0,
    "bwd_byte_count":      0.0,
    "avg_packet_size":     150.0,     # 300/2
    "avg_fwd_packet_size": 150.0,     # 300/2
    "avg_bwd_packet_size": 0.0,       # 0/max(0,1)=0
    "fwd_iat_mean":        1.0,       # [1.0] → mean=1.0
    "bwd_iat_mean":        0.0,       # không có bwd IAT
    "flow_iat_mean":       1.0,       # [1.0] → mean=1.0
    "flow_iat_std":        0.0,       # 1 IAT → std=0 (Welford M2/count=0/1=0)
    "fwd_header_len":      40.0,      # 20+20
    "bwd_header_len":      0.0,
    "syn_flag_count":      1.0,       # "S" in "S"
    "fin_flag_count":      0.0,
    "rst_flag_count":      0.0,
    "ack_flag_count":      1.0,       # "A" in "A"
}

# Flow 2: TCP handshake — 5 gói, 2 chiều, có SYN/ACK/FIN
# Kiểm tra: flag counting, fwd/bwd IAT riêng biệt, flow_iat_std
# fwd_iats=[0.2, 0.3] → mean=0.25; bwd_iats=[0.5] → mean=0.5
# flow_iats=[0.1, 0.1, 0.3, 0.1] → mean=0.15, std=sqrt(0.0075)≈0.0866
PKTS_FLOW2 = [
    (0.0,  60,  True,  "S",   40),   # SYN
    (0.1,  60,  False, "SA",  40),   # SYN-ACK
    (0.2,  54,  True,  "A",   20),   # ACK
    (0.5,  200, True,  "A",   20),   # data
    (0.6,  54,  False, "FA",  20),   # FIN-ACK
]
EXPECTED_FLOW2 = {
    "duration":            0.6,
    "packet_count":        5.0,
    "byte_count":          428.0,     # 60+60+54+200+54
    "fwd_packet_count":    3.0,
    "bwd_packet_count":    2.0,
    "fwd_byte_count":      314.0,     # 60+54+200
    "bwd_byte_count":      114.0,     # 60+54
    "avg_packet_size":     428 / 5,   # 85.6
    "avg_fwd_packet_size": 314 / 3,   # 104.666...
    "avg_bwd_packet_size": 57.0,      # 114/2
    "fwd_iat_mean":        0.25,      # (0.2+0.3)/2
    "bwd_iat_mean":        0.5,       # [0.6-0.1=0.5]
    "flow_iat_mean":       0.15,      # (0.1+0.1+0.3+0.1)/4
    "flow_iat_std":        math.sqrt(0.03 / 4),  # sqrt(0.0075)≈0.08660254
    "fwd_header_len":      80.0,      # 40+20+20
    "bwd_header_len":      60.0,      # 40+20
    "syn_flag_count":      2.0,       # "S","SA"
    "fin_flag_count":      1.0,       # "FA"
    "rst_flag_count":      0.0,
    "ack_flag_count":      4.0,       # "SA","A","A","FA"
}

# Flow 3: Kiểm tra Welford accuracy
# flow_iats=[0.5, 1.0, 0.3] → mean=0.6, M2=0.26, std=sqrt(0.26/3)≈0.2944
PKTS_FLOW3 = [
    (0.0,  500, True,  "SA", 40),
    (0.5,  500, False, "A",  20),
    (1.5,  300, True,  "A",  20),
    (1.8,  100, False, "A",  20),
]
EXPECTED_FLOW3 = {
    "duration":            1.8,
    "packet_count":        4.0,
    "byte_count":          1400.0,    # 500+500+300+100
    "fwd_packet_count":    2.0,
    "bwd_packet_count":    2.0,
    "fwd_byte_count":      800.0,     # 500+300
    "bwd_byte_count":      600.0,     # 500+100
    "avg_packet_size":     350.0,     # 1400/4
    "avg_fwd_packet_size": 400.0,     # 800/2
    "avg_bwd_packet_size": 300.0,     # 600/2
    "fwd_iat_mean":        1.5,       # [1.5-0.0=1.5]
    "bwd_iat_mean":        1.3,       # [1.8-0.5=1.3]
    "flow_iat_mean":       0.6,       # (0.5+1.0+0.3)/3
    "flow_iat_std":        math.sqrt(0.26 / 3),  # ≈0.29439202
    "fwd_header_len":      60.0,      # 40+20
    "bwd_header_len":      40.0,      # 20+20
    "syn_flag_count":      1.0,       # "SA"
    "fin_flag_count":      0.0,
    "rst_flag_count":      0.0,
    "ack_flag_count":      4.0,       # "SA","A","A","A"
}


# ── Test helpers ──────────────────────────────────────────────────────────────

ATOL = 1e-5  # tolerance cho float32 so sánh với expected float64


def check_features(pkts, expected, label=""):
    """So sánh extract() với expected values tính tay."""
    flow = make_flow(pkts)
    result = extract(flow)

    assert len(result) == 20, f"{label}: result phải có 20 phần tử, có {len(result)}"
    assert result.dtype == np.float32, f"{label}: dtype phải là float32"

    errors = []
    for i, name in enumerate(MODEL_FEATURES):
        actual = float(result[i])
        exp = float(expected[name])
        if abs(actual - exp) >= ATOL:
            errors.append(f"  [{i}] {name}: expected={exp:.8f}, got={actual:.8f}, diff={abs(actual-exp):.2e}")

    if errors:
        pytest.fail(f"{label} — {len(errors)} feature sai:\n" + "\n".join(errors))


# ── Test cases ────────────────────────────────────────────────────────────────

class TestFeatureExtractor:

    def test_model_features_count(self):
        """MODEL_FEATURES phải có đúng 20 feature."""
        assert len(MODEL_FEATURES) == 20

    def test_output_shape_and_dtype(self):
        """extract() phải trả về ndarray (20,) float32."""
        result = extract(make_flow(PKTS_FLOW2))
        assert result.shape == (20,)
        assert result.dtype == np.float32

    def test_flow1_minimum_unidirectional(self):
        """2 gói, 1 chiều — edge case bwd=0, std=0 khi 1 IAT."""
        check_features(PKTS_FLOW1, EXPECTED_FLOW1, "Flow1")

    def test_flow2_tcp_handshake(self):
        """5 gói, 2 chiều — kiểm tra flag counting, fwd/bwd IAT riêng."""
        check_features(PKTS_FLOW2, EXPECTED_FLOW2, "Flow2")

    def test_flow3_welford_vs_direct_formula(self):
        """4 gói — Welford std phải khớp công thức trực tiếp (atol=1e-5)."""
        check_features(PKTS_FLOW3, EXPECTED_FLOW3, "Flow3")

    def test_no_nan_or_inf(self):
        """Không được có NaN hoặc Inf trong bất kỳ flow nào."""
        for pkts in [PKTS_FLOW1, PKTS_FLOW2, PKTS_FLOW3]:
            result = extract(make_flow(pkts))
            assert not np.any(np.isnan(result)), "Có NaN trong feature vector"
            assert not np.any(np.isinf(result)), "Có Inf trong feature vector"

    def test_no_zero_division_bwd_empty(self):
        """Flow không có gói bwd — avg_bwd_packet_size phải là 0.0, không crash."""
        result = extract(make_flow(PKTS_FLOW1))
        bwd_avg_idx = MODEL_FEATURES.index("avg_bwd_packet_size")
        assert float(result[bwd_avg_idx]) == 0.0

    def test_duration_clamped_when_same_timestamp(self):
        """2 gói cùng timestamp → duration=1e-6, không phải 0."""
        pkts = [(5.0, 100, True, "S", 20), (5.0, 200, True, "A", 20)]
        result = extract(make_flow(pkts))
        dur_idx = MODEL_FEATURES.index("duration")
        assert float(result[dur_idx]) == pytest.approx(1e-6, rel=1e-4)

    def test_rst_flag_triggers(self):
        """Gói RST phải được đếm đúng."""
        pkts = [
            (0.0, 54, True,  "S",  20),
            (0.1, 54, False, "RA", 20),  # RST-ACK
        ]
        result = extract(make_flow(pkts))
        rst_idx = MODEL_FEATURES.index("rst_flag_count")
        ack_idx = MODEL_FEATURES.index("ack_flag_count")
        assert float(result[rst_idx]) == 1.0
        assert float(result[ack_idx]) == 1.0

    def test_feature_order_matches_schema(self):
        """Thứ tự giá trị trong vector phải khớp đúng MODEL_FEATURES."""
        flow = make_flow(PKTS_FLOW2)
        result = extract(flow)
        # Kiểm tra một vài vị trí biết trước
        assert float(result[MODEL_FEATURES.index("byte_count")]) == pytest.approx(428.0)
        assert float(result[MODEL_FEATURES.index("fwd_packet_count")]) == pytest.approx(3.0)
        assert float(result[MODEL_FEATURES.index("syn_flag_count")]) == pytest.approx(2.0)

    def test_flow_iat_std_population_not_sample(self):
        """
        Xác nhận std là population std (chia n), không phải sample std (chia n-1).
        flow_iats=[1.0, 3.0] → mean=2.0, population_std=1.0, sample_std=sqrt(2)≈1.414
        """
        pkts = [
            (0.0, 100, True, "S", 20),
            (1.0, 100, True, "A", 20),  # iat=1.0
            (4.0, 100, True, "A", 20),  # iat=3.0
        ]
        # flow_iats = [1.0, 3.0], mean=2.0
        # population var = ((1-2)²+(3-2)²)/2 = 1.0 → std=1.0
        # sample var = ((1-2)²+(3-2)²)/1 = 2.0 → std=sqrt(2)
        result = extract(make_flow(pkts))
        std_idx = MODEL_FEATURES.index("flow_iat_std")
        assert float(result[std_idx]) == pytest.approx(1.0, abs=ATOL), \
            "flow_iat_std phải là population std (chia n), không phải sample std (chia n-1)"
