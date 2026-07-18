import sys
import numpy as np
import json
sys.path.insert(0, "/home/phuong/Documents/do an thuc tpa tot nghiep/edge-ai-traffic-anomaly")

from pipeline.feature_extractor import FlowRecord, extract_flow_features
from model.inference import ETSSLInference
from configs.paths import get_model_dir
from data.feature_schema import MODEL_FEATURES

engine = ETSSLInference(str(get_model_dir()), backend="onnx")

def create_mock_flow(src_port, dst_ip, n_packets):
    flow = FlowRecord(src_ip="192.168.1.24", dst_ip=dst_ip, src_port=src_port, dst_port=443, protocol="TCP")
    t = 1000.0
    
    # Handshake
    flow.add_packet(t, True, 20, 0, "S", 64240); t += 0.01
    flow.add_packet(t, False, 20, 0, "SA", 64240); t += 0.01
    flow.add_packet(t, True, 20, 0, "A", 64240); t += 0.01
    
    n_data_pkts = max(0, n_packets - 5)
    for i in range(n_data_pkts):
        is_fwd = i % 2 == 0
        plen = 100 if is_fwd else 1400
        flags = "PA" if is_fwd else "A"
        flow.add_packet(t, is_fwd, 20, plen, flags, 64240)
        t += 0.02
        
    # Teardown
    flow.add_packet(t, True, 20, 0, "FA", 64240); t += 0.01
    flow.add_packet(t, False, 20, 0, "A", 64240); t += 0.01
    
    return flow

flows_to_test = [
    (44280, "20.189.172.33", 15),
    (47970, "20.189.172.33", 14),
    (55482, "20.189.172.33", 14),
    (41538, "20.189.172.33", 17),
    (48832, "140.82.114.21", 15) # Github for comparison
]

print(f"{'Flow':<25} | {'Score':<15} | {'Is Anomaly'}")
print("-" * 60)

results = {}
for src_port, dst_ip, n_pkts in flows_to_test:
    flow = create_mock_flow(src_port, dst_ip, n_pkts)
    raw_feats = extract_flow_features(flow)
    if raw_feats is not None:
        pred = engine.predict(raw_feats)
        scaled_feats = engine._scale(raw_feats.reshape(1, -1))
        name = f"{src_port}->{dst_ip} ({n_pkts}p)"
        results[name] = {"raw": raw_feats, "scaled": scaled_feats[0], "score": pred["score"]}
        print(f"{name:<25} | {pred['score']:<15.2f} | {pred['is_anomaly']}")

print("\n--- Feature Comparison (Raw & Scaled) ---")
print(f"{'Idx':<4} {'Feature':<35} | ", end="")
for name in results.keys():
    print(f"{name[:12]:<22} | ", end="")
print()

for i, fname in enumerate(MODEL_FEATURES):
    print(f"{i:<4} {fname[:35]:<35} | ", end="")
    for name, data in results.items():
        raw_val = data['raw'][i]
        scaled_val = data['scaled'][i]
        
        # Check for NaN/Inf
        warn = ""
        if np.isnan(raw_val) or np.isinf(raw_val) or np.isnan(scaled_val) or np.isinf(scaled_val):
            warn = " ⚠️"
            
        print(f"{raw_val:>9.2f}({scaled_val:>7.2f}){warn} | ", end="")
    print()
