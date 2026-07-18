import sys
import numpy as np
from pathlib import Path
import joblib

sys.path.insert(0, "/home/phuong/Documents/do an thuc tpa tot nghiep/edge-ai-traffic-anomaly")

from configs.paths import get_model_dir
from model.inference import ETSSLInference
from pipeline.flow_aggregator import FlowAggregator
from pipeline.feature_extractor import extract_flow_features, FlowRecord
from data.feature_schema import MODEL_FEATURES
from scapy.all import rdpcap, IP, TCP, UDP

def analyze_pcap(pcap_path):
    print(f"=== PCAP ANALYSIS: {pcap_path} ===")
    engine = ETSSLInference(str(get_model_dir()), backend="onnx")
    scaler = engine.scaler
    
    # Store flows as they are ready
    emitted_flows = []
    
    def on_flow_ready(flow, features):
        emitted_flows.append((flow, features))
        
    aggregator = FlowAggregator(
        flow_timeout=30.0,
        time_window_sec=None,
        on_flow_ready=on_flow_ready
    )
    
    packets = rdpcap(pcap_path)
    print(f"Loaded {len(packets)} packets.")
    
    for pkt in packets:
        if IP not in pkt or (TCP not in pkt and UDP not in pkt):
            continue
        
        protocol = "TCP" if TCP in pkt else "UDP"
        src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
        src_port = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
        dst_port = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport
        timestamp = float(pkt.time)
        
        header_len, payload_len, flags, win_bytes = 0, 0, "", 0
        if TCP in pkt:
            header_len = pkt[TCP].dataofs * 4
            payload_len = len(pkt[TCP].payload)
            flags = str(pkt[TCP].flags)
            win_bytes = pkt[TCP].window
        else:
            header_len = 8
            payload_len = len(pkt[UDP].payload)
            
        aggregator.add_packet(
            src_ip, dst_ip, src_port, dst_port, protocol,
            timestamp, header_len, payload_len, flags, win_bytes
        )
    aggregator.flush_all()
    
    print(f"Emitted {len(emitted_flows)} flows.")
    
    # Check for duplicates by flow_id
    from collections import defaultdict
    flows_by_id = defaultdict(list)
    for flow, feats in emitted_flows:
        flow_id = f"{flow.src_ip}:{flow.src_port}->{flow.dst_ip}:{flow.dst_port}"
        flows_by_id[flow_id].append((flow, feats))
        
    print("\n--- Duplicates Analysis ---")
    dup_found = False
    for flow_id, list_flows in flows_by_id.items():
        if len(list_flows) >= 2:
            dup_found = True
            print(f"\nFlow ID: {flow_id} (found {len(list_flows)} times)")
            for idx, (flow, feats) in enumerate(list_flows):
                score_dict = engine.predict(feats)
                print(f"  Instance {idx+1}: Packets={len(flow.packets)} | Score={score_dict['score']:.4f} | Anomaly={score_dict['is_anomaly']}")
                # Print packets info
                fwd_count = sum(1 for p in flow.packets if p['is_forward'])
                bwd_count = sum(1 for p in flow.packets if not p['is_forward'])
                print(f"    Packets breakdown: Forward={fwd_count}, Backward={bwd_count}")
                print(f"    Flags: {[p['flags'] for p in flow.packets]}")
                # Print some feature values
                print(f"    payload_bytes_median: {feats[MODEL_FEATURES.index('payload_bytes_median')]:.2f}")
                print(f"    mean_packets_delta_len: {feats[MODEL_FEATURES.index('mean_packets_delta_len')]:.2f}")
                
    if not dup_found:
        print("No duplicate flow IDs found in this short PCAP.")

if __name__ == "__main__":
    pcap = "edge-ai-traffic-anomaly/test_clean.pcap"
    analyze_pcap(pcap)
