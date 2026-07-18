import sys
import numpy as np
from pathlib import Path
import joblib

sys.path.insert(0, "/home/phuong/Documents/do an thuc tpa tot nghiep/edge-ai-traffic-anomaly")

from configs.paths import get_model_dir
from model.inference import ETSSLInference
from pipeline.feature_extractor import extract_flow_features, FlowRecord
from data.feature_schema import MODEL_FEATURES

def create_flow_instance(name, packets_list):
    flow = FlowRecord("192.168.1.24", "20.184.175.1", 49510, 443, "TCP")
    for pkt in packets_list:
        flow.add_packet(
            timestamp=pkt['time'],
            is_forward=pkt['is_forward'],
            header_len=pkt['header_len'],
            payload_len=pkt['payload_len'],
            flags=pkt['flags'],
            win_bytes=pkt['win_bytes']
        )
    return flow

def main():
    engine = ETSSLInference(str(get_model_dir()), backend="onnx")
    scaler = engine.scaler
    
    print("=" * 80)
    print("PHẦN 1: MÔ PHỎNG SỰ PHÂN TÁCH FLOW DO FIN/RST")
    print("=" * 80)
    
    # Giả lập một connection HTTPS bình thường bị split thành 2 flow:
    # Flow 1: 18 gói tin đầu tiên (giao tiếp data + FIN đầu tiên)
    # Flow 2: 2 gói tin tiếp theo (ACK và FIN tiếp theo của teardown)
    
    # Tạo danh sách packet cho Flow 1
    flow1_packets = []
    t = 100.0
    for i in range(9):
        # Forward data
        flow1_packets.append({'time': t, 'is_forward': True, 'header_len': 20, 'payload_len': 150, 'flags': 'PA', 'win_bytes': 64240})
        t += 0.05
        # Backward data
        flow1_packets.append({'time': t, 'is_forward': False, 'header_len': 20, 'payload_len': 1000, 'flags': 'A', 'win_bytes': 64240})
        t += 0.05
    # Thêm FIN từ client vào Flow 1
    flow1_packets.append({'time': t, 'is_forward': True, 'header_len': 20, 'payload_len': 0, 'flags': 'FA', 'win_bytes': 64240})
    
    # Tạo danh sách packet cho Flow 2
    # Do Flow 1 đã bị pop khỏi active_flows, packet tiếp theo của teardown sẽ tạo thành Flow 2
    t += 0.05
    flow2_packets = [
        {'time': t, 'is_forward': False, 'header_len': 20, 'payload_len': 0, 'flags': 'A', 'win_bytes': 64240},
        {'time': t + 0.05, 'is_forward': False, 'header_len': 20, 'payload_len': 0, 'flags': 'FA', 'win_bytes': 64240}
    ]
    
    flow1 = create_flow_instance("Flow 1", flow1_packets)
    flow2 = create_flow_instance("Flow 2", flow2_packets)
    
    feat1 = extract_flow_features(flow1)
    feat2 = extract_flow_features(flow2)
    
    score1 = engine.predict(feat1)
    score2 = engine.predict(feat2)
    
    print(f"Flow 1 (19 packets, data + FIN): Score = {score1['score']:.4f} | Anomaly = {score1['is_anomaly']}")
    print(f"Flow 2 (2 packets, teardown ACKs): Score = {score2['score']:.4f} | Anomaly = {score2['is_anomaly']}")
    
    print("\nSo sánh trực tiếp feature vector (raw) của Flow 1 và Flow 2:")
    print(f"{'Index':<5} {'Feature Name':<35} {'Flow 1 Raw':<15} {'Flow 2 Raw':<15} {'Diff':<15}")
    print("-" * 90)
    for i, name in enumerate(MODEL_FEATURES):
        val1 = feat1[i]
        val2 = feat2[i]
        print(f"{i:<5} {name[:35]:<35} {val1:<15.4f} {val2:<15.4f} {abs(val1-val2):<15.4f}")
        
    print("\n" + "=" * 80)
    print("PHẦN 2: TẠI SAO NHIỀU FLOW KHÁC NHAU CÓ SCORE GẦN GIỐNG NHAU (26000 - 28000)?")
    print("=" * 80)
    
    # Tạo 4 flow HTTPS khác nhau hoàn toàn:
    # Flow A: Microsoft (20 packets, payload trung bình 1200B)
    # Flow B: Github (50 packets, payload trung bình 800B)
    # Flow C: Google (10 packets, payload trung bình 300B)
    # Flow D: CDN (5 packets, payload trung bình 1400B)
    
    flows = {}
    
    # Flow A
    pkts_a = []
    ta = 200.0
    for i in range(10):
        pkts_a.append({'time': ta, 'is_forward': True, 'header_len': 20, 'payload_len': 150, 'flags': 'PA', 'win_bytes': 64240})
        ta += 0.02
        pkts_a.append({'time': ta, 'is_forward': False, 'header_len': 20, 'payload_len': 1200, 'flags': 'A', 'win_bytes': 64240})
        ta += 0.02
    flows['Flow A (Microsoft)'] = pkts_a
    
    # Flow B
    pkts_b = []
    tb = 300.0
    for i in range(25):
        pkts_b.append({'time': tb, 'is_forward': True, 'header_len': 20, 'payload_len': 100, 'flags': 'PA', 'win_bytes': 64240})
        tb += 0.01
        pkts_b.append({'time': tb, 'is_forward': False, 'header_len': 20, 'payload_len': 800, 'flags': 'A', 'win_bytes': 64240})
        tb += 0.01
    flows['Flow B (Github)'] = pkts_b
    
    # Flow C
    pkts_c = []
    tc = 400.0
    for i in range(5):
        pkts_c.append({'time': tc, 'is_forward': True, 'header_len': 20, 'payload_len': 200, 'flags': 'PA', 'win_bytes': 64240})
        tc += 0.08
        pkts_c.append({'time': tc, 'is_forward': False, 'header_len': 20, 'payload_len': 300, 'flags': 'A', 'win_bytes': 64240})
        tc += 0.08
    flows['Flow C (Google)'] = pkts_c
    
    # Flow D
    pkts_d = []
    td = 500.0
    for i in range(3):
        pkts_d.append({'time': td, 'is_forward': True, 'header_len': 20, 'payload_len': 500, 'flags': 'PA', 'win_bytes': 64240})
        td += 0.04
        pkts_d.append({'time': td, 'is_forward': False, 'header_len': 20, 'payload_len': 1400, 'flags': 'A', 'win_bytes': 64240})
        td += 0.04
    flows['Flow D (CDN)'] = pkts_d
    
    features_dict = {}
    scores_dict = {}
    
    for fname, pkts in flows.items():
        flow_obj = create_flow_instance(fname, pkts)
        feat = extract_flow_features(flow_obj)
        features_dict[fname] = feat
        scores_dict[fname] = engine.predict(feat)
        print(f"{fname}: Score = {scores_dict[fname]['score']:.4f} | Anomaly = {scores_dict[fname]['is_anomaly']}")
        
    print("\nSo sánh feature vector (Scaled) của các Flow A, B, C, D:")
    print(f"{'Index':<5} {'Feature Name':<35} {'Flow A':<12} {'Flow B':<12} {'Flow C':<12} {'Flow D':<12}")
    print("-" * 95)
    for i, name in enumerate(MODEL_FEATURES):
        scaled_vals = []
        for fname in flows.keys():
            raw_val = features_dict[fname][i]
            mean = scaler.mean_[i]
            scale = scaler.scale_[i]
            scaled_val = (raw_val - mean) / scale
            scaled_vals.append(scaled_val)
        print(f"{i:<5} {name[:35]:<35} {scaled_vals[0]:<12.4f} {scaled_vals[1]:<12.4f} {scaled_vals[2]:<12.4f} {scaled_vals[3]:<12.4f}")

if __name__ == "__main__":
    main()
