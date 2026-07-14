"""
features/feature_extractor.py
================================
Converts a completed Flow (dict, as produced by flow/models.py
to_dict()) into a FeatureVector dict, matching features/feature_schema.py.
"""

import math
from collections import Counter

from features.feature_schema import FEATURE_FIELDS


def _entropy(values: list) -> float:
    """Shannon entropy over packet size distribution (rounded to
    nearest 32 bytes bucket) — a cheap proxy for payload randomness,
    useful for detecting scans/floods with uniform packet sizes vs.
    normal traffic with varied sizes."""
    if not values:
        return 0.0
    buckets = [int(v // 32) for v in values]
    counts = Counter(buckets)
    total = len(buckets)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _inter_arrival_times(timestamps: list) -> list:
    if len(timestamps) < 2:
        return []
    ts = sorted(timestamps)
    return [b - a for a, b in zip(ts, ts[1:])]


class FeatureExtractor:
    def extract(self, flow: dict) -> dict:
        duration = max(flow.get("duration", 0.0), 1e-6)  # avoid div-by-zero
        packet_count = flow.get("packet_count", 0)
        byte_count = flow.get("byte_count", 0)
        sizes = flow.get("packet_sizes", []) or []
        flags = flow.get("tcp_flags_seen", []) or []
        timestamps = flow.get("packet_timestamps", []) or []

        iats = _inter_arrival_times(timestamps)

        feature = {
            "duration": duration,
            "packet_rate": packet_count / duration,
            "byte_rate": byte_count / duration,
            "avg_packet_size": _mean(sizes),
            "std_packet_size": _std(sizes),
            "tcp_flags_diversity": float(len(set(flags))),
            "entropy": _entropy(sizes),
            "avg_iat": _mean(iats),
            "std_iat": _std(iats),
        }

        # carry along identifying/context fields (not fed to the model,
        # but needed downstream for correlation + alerting)
        feature["_meta"] = {
            "src_ip": flow.get("src_ip"),
            "dst_ip": flow.get("dst_ip"),
            "src_port": flow.get("src_port"),
            "dst_port": flow.get("dst_port"),
            "protocol": flow.get("protocol"),
        }

        assert all(f in feature for f in FEATURE_FIELDS)
        return feature