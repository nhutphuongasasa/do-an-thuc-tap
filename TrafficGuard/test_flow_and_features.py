"""
tests/test_flow_and_features.py
==================================
Lightweight unit tests that don't require a live NIC, Suricata, Zeek,
or PostgreSQL — they exercise the pure logic: flow aggregation,
feature extraction, and risk scoring.

Run with: pytest tests/
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.flow_manager import FlowManager
from features.feature_extractor import FeatureExtractor
from correlation.risk_engine import RiskEngine


def make_packet(src_ip="10.0.0.1", dst_ip="10.0.0.2", sport=1111, dport=80,
                 proto="TCP", length=100, flags="S"):
    return {
        "timestamp": time.time(),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": sport,
        "dst_port": dport,
        "protocol": proto,
        "length": length,
        "tcp_flags": flags,
        "ttl": 64,
    }


def test_flow_manager_aggregates_5_tuple():
    fm = FlowManager(idle_timeout=1, hard_timeout=10)
    for _ in range(5):
        fm.process_packet(make_packet())
    assert fm.active_flow_count() == 1

    fm.process_packet(make_packet(dport=443))
    assert fm.active_flow_count() == 2


def test_flow_manager_flushes_idle_flows():
    fm = FlowManager(idle_timeout=0, hard_timeout=10)
    fm.process_packet(make_packet())
    time.sleep(0.05)
    completed = fm.flush_timeout_flows()
    assert len(completed) == 1
    assert fm.active_flow_count() == 0


def test_feature_extractor_produces_expected_fields():
    fm = FlowManager(idle_timeout=0, hard_timeout=10)
    for i in range(10):
        fm.process_packet(make_packet(length=64 + i))
        time.sleep(0.001)
    completed = fm.flush_timeout_flows()
    assert len(completed) == 1

    extractor = FeatureExtractor()
    feature = extractor.extract(completed[0])

    for field in ("duration", "packet_rate", "byte_rate", "avg_packet_size",
                  "std_packet_size", "tcp_flags_diversity", "entropy",
                  "avg_iat", "std_iat"):
        assert field in feature

    assert feature["packet_rate"] >= 0


def test_risk_engine_combines_rule_and_ml_scores():
    engine = RiskEngine(rule_weight=0.6, ml_weight=0.4, low=40, medium=60, high=80)

    rule_event = {"rule_score": 90}
    ml_result = {"confidence": 0.8}

    result = engine.correlate(rule_event, ml_result, context={"src_ip": "1.2.3.4"})
    expected = (90 * 0.6) + (80 * 0.4)
    assert abs(result["risk_score"] - expected) < 0.01
    assert result["severity"] in ("Low", "Medium", "High", "Critical")


def test_risk_engine_handles_missing_evidence():
    engine = RiskEngine(rule_weight=0.6, ml_weight=0.4, low=40, medium=60, high=80)

    only_ml = engine.correlate(None, {"confidence": 0.95}, context={})
    assert only_ml["risk_score"] == 95.0

    only_rule = engine.correlate({"rule_score": 70}, None, context={})
    assert only_rule["risk_score"] == 70.0

    neither = engine.correlate(None, None, context={})
    assert neither["risk_score"] == 0.0


if __name__ == "__main__":
    test_flow_manager_aggregates_5_tuple()
    test_flow_manager_flushes_idle_flows()
    test_feature_extractor_produces_expected_fields()
    test_risk_engine_combines_rule_and_ml_scores()
    test_risk_engine_handles_missing_evidence()
    print("All tests passed.")