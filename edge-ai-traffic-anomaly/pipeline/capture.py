"""
capture.py — Bắt gói tin, gom luồng và trích xuất đặc trưng (20 features) sử dụng Scapy.

Lắng nghe traffic từ NIC hoặc đọc file .pcap, gom nhóm các gói tin thành các Flow (5-tuple).
Khi Flow kết thúc (timeout hoặc FIN/RST), tính toán 20 features chuẩn và bơm vào ET-SSL.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import skew, mode
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.inference import ETSSLInference

MODEL_DIR = str(
    Path(__file__).parent.parent.parent / "TrafficGuard/models/edge_ai-20260716T101644Z-1-001/edge_ai"
)

# Các features mục tiêu
MODEL_FEATURES = [
    "median_packets_delta_time",
    "packets_IAT_mean",
    "fwd_init_win_bytes",
    "fwd_packets_IAT_mean",
    "bwd_init_win_bytes",
    "rst_flag_percentage_in_total",
    "mode_packets_delta_time",
    "bwd_packets_IAT_mean",
    "rst_flag_counts",
    "psh_flag_percentage_in_total",
    "fwd_min_header_bytes",
    "fwd_segment_size_min",
    "skewness_payload_bytes_delta_len",
    "payload_bytes_median",
    "median_bwd_packets_delta_len",
    "fwd_payload_bytes_median",
    "mean_packets_delta_time",
    "cov_bwd_payload_bytes_delta_len",
    "mean_packets_delta_len",
    "ack_flag_percentage_in_total",
]

class FlowState:
    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        
        self.packets = []
        self.start_time = None
        self.last_time = None
        self.is_finished = False

    def add_packet(self, pkt, timestamp, is_forward):
        if self.start_time is None:
            self.start_time = timestamp
        self.last_time = timestamp
        
        header_len = 0
        payload_len = 0
        flags = ""
        win_bytes = 0
        
        if TCP in pkt:
            header_len = pkt[TCP].dataofs * 4
            payload_len = len(pkt[TCP].payload)
            flags = pkt[TCP].flags
            win_bytes = pkt[TCP].window
        elif UDP in pkt:
            header_len = 8
            payload_len = len(pkt[UDP].payload)

        self.packets.append({
            "time": timestamp,
            "is_forward": is_forward,
            "header_len": header_len,
            "payload_len": payload_len,
            "flags": flags,
            "win_bytes": win_bytes
        })
        
        # Check FIN or RST to finish flow
        if "F" in str(flags) or "R" in str(flags):
            self.is_finished = True

    def extract_features(self):
        if len(self.packets) < 2:
            return None  # Flow quá ngắn để tính toán
            
        times = np.array([p["time"] for p in self.packets])
        delta_times = np.diff(times)
        if len(delta_times) == 0:
            delta_times = np.array([0.0])
            
        fwd_pkts = [p for p in self.packets if p["is_forward"]]
        bwd_pkts = [p for p in self.packets if not p["is_forward"]]
        
        fwd_times = np.array([p["time"] for p in fwd_pkts])
        bwd_times = np.array([p["time"] for p in bwd_pkts])
        
        fwd_delta_times = np.diff(fwd_times) if len(fwd_times) > 1 else np.array([0.0])
        bwd_delta_times = np.diff(bwd_times) if len(bwd_times) > 1 else np.array([0.0])
        
        payloads = np.array([p["payload_len"] for p in self.packets])
        fwd_payloads = np.array([p["payload_len"] for p in fwd_pkts]) if fwd_pkts else np.array([0])
        bwd_payloads = np.array([p["payload_len"] for p in bwd_pkts]) if bwd_pkts else np.array([0])
        
        delta_payloads = np.diff(payloads) if len(payloads) > 1 else np.array([0.0])
        bwd_delta_payloads = np.diff(bwd_payloads) if len(bwd_payloads) > 1 else np.array([0.0])

        total_pkts = len(self.packets)
        rst_count = sum(1 for p in self.packets if "R" in str(p["flags"]))
        psh_count = sum(1 for p in self.packets if "P" in str(p["flags"]))
        ack_count = sum(1 for p in self.packets if "A" in str(p["flags"]))

        fwd_header_lens = [p["header_len"] for p in fwd_pkts]
        fwd_min_header_bytes = min(fwd_header_lens) if fwd_header_lens else 0
        fwd_segment_size_min = fwd_min_header_bytes # Approximation

        fwd_init_win_bytes = fwd_pkts[0]["win_bytes"] if fwd_pkts else 0
        bwd_init_win_bytes = bwd_pkts[0]["win_bytes"] if bwd_pkts else 0

        # Safe statistics functions
        def safe_mean(arr): return float(np.mean(arr)) if len(arr) > 0 else 0.0
        def safe_median(arr): return float(np.median(arr)) if len(arr) > 0 else 0.0
        def safe_mode(arr): 
            if len(arr) == 0: return 0.0
            m = mode(arr, keepdims=True)
            return float(m.mode[0]) if len(m.mode) > 0 else 0.0
        def safe_skew(arr):
            if len(arr) < 2 or np.var(arr) == 0: return 0.0
            return float(skew(arr))
        def safe_cov(arr):
            if len(arr) == 0: return 0.0
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            return float(std_val / mean_val) if mean_val != 0 else 0.0

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
            "fwd_segment_size_min": fwd_segment_size_min,
            "skewness_payload_bytes_delta_len": safe_skew(delta_payloads),
            "payload_bytes_median": safe_median(payloads),
            "median_bwd_packets_delta_len": safe_median(bwd_delta_payloads),
            "fwd_payload_bytes_median": safe_median(fwd_payloads),
            "mean_packets_delta_time": safe_mean(delta_times),
            "cov_bwd_payload_bytes_delta_len": safe_cov(bwd_delta_payloads),
            "mean_packets_delta_len": safe_mean(delta_payloads),
            "ack_flag_percentage_in_total": ack_count / total_pkts,
        }
        
        return np.array([features[k] for k in MODEL_FEATURES], dtype=np.float32)


class TrafficCapture:
    def __init__(self, engine, flow_timeout=30.0):
        self.engine = engine
        self.flow_timeout = flow_timeout
        self.active_flows = {}
        self.flow_lock = False

    def packet_handler(self, pkt):
        if IP not in pkt:
            return
            
        if TCP not in pkt and UDP not in pkt:
            return
            
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        protocol = "TCP" if TCP in pkt else "UDP"
        
        src_port = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
        dst_port = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport
        
        timestamp = float(pkt.time)
        
        # Determine flow key (direction independent)
        forward_key = (src_ip, dst_ip, src_port, dst_port, protocol)
        backward_key = (dst_ip, src_ip, dst_port, src_port, protocol)
        
        if forward_key in self.active_flows:
            flow_key = forward_key
            is_forward = True
        elif backward_key in self.active_flows:
            flow_key = backward_key
            is_forward = False
        else:
            flow_key = forward_key
            is_forward = True
            self.active_flows[flow_key] = FlowState(src_ip, dst_ip, src_port, dst_port, protocol)
            
        flow = self.active_flows[flow_key]
        flow.add_packet(pkt, timestamp, is_forward)
        
        self.check_timeouts(current_time=timestamp)

    def check_timeouts(self, current_time):
        finished_keys = []
        for key, flow in self.active_flows.items():
            if flow.is_finished or (current_time - flow.last_time > self.flow_timeout):
                finished_keys.append(key)
                
        for key in finished_keys:
            flow = self.active_flows.pop(key)
            self.process_flow(flow)
            
    def process_flow(self, flow):
        feature_vector = flow.extract_features()
        if feature_vector is not None:
            # Reshape to (1, D) for the engine
            X = feature_vector.reshape(1, -1)
            try:
                # Scale data if engine has a scaler
                if self.engine.scaler:
                    X = self.engine.scaler.transform(X).astype(np.float32)
                    
                result = self.engine.predict(X[0])
                is_anomaly = result["is_anomaly"]
                score = result["score"]
                
                status = "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL"
                print(f"[{status}] Flow {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} | Score: {score:.2f}")
            except Exception as e:
                print(f"Error processing flow: {e}")

    def run_live(self, interface=None):
        print(f"🚀 Starting live capture on interface: {interface or 'default'}...")
        sniff(iface=interface, prn=self.packet_handler, store=0)
        
    def run_pcap(self, pcap_file):
        print(f"📂 Reading PCAP file: {pcap_file}...")
        sniff(offline=pcap_file, prn=self.packet_handler, store=0)
        # Process remaining flows
        self.check_timeouts(current_time=float('inf'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", type=str, help="Path to PCAP file (optional)")
    parser.add_argument("--iface", type=str, help="Network interface (optional)")
    args = parser.parse_args()
    
    print("📦 Loading Inference Engine...")
    engine = ETSSLInference(model_dir=MODEL_DIR, backend="onnx")
    
    capture = TrafficCapture(engine=engine, flow_timeout=15.0)
    
    if args.pcap:
        capture.run_pcap(args.pcap)
    else:
        capture.run_live(args.iface)
