import sys
import numpy as np
from pathlib import Path
import joblib

sys.path.insert(0, "/home/phuong/Documents/do an thuc tpa tot nghiep/edge-ai-traffic-anomaly")

from configs.paths import get_model_dir
from model.inference import ETSSLInference
from pipeline.feature_extractor import extract_flow_features, FlowRecord
from data.feature_schema import MODEL_FEATURES

def main():
    engine = ETSSLInference(str(get_model_dir()), backend="onnx")
    scaler = engine.scaler
    
    # Let's create a typical TCP flow
    flow = FlowRecord("192.168.1.24", "20.184.175.1", 49510, 443, "TCP")
    # Add 10 packets, 10ms IAT
    t = 100.0
    for i in range(5):
        flow.add_packet(t, True, 20, 100, "PA", 64240)
        t += 0.01
        flow.add_packet(t, False, 20, 1000, "A", 64240)
        t += 0.01
        
    feat = extract_flow_features(flow)
    
    # Original score
    score_orig = engine.predict(feat)
    print(f"Original features (seconds for IAT): Score = {score_orig['score']:.4f}")
    
    # Print IAT features
    iat_indices = [
        MODEL_FEATURES.index("packets_IAT_mean"),
        MODEL_FEATURES.index("fwd_packets_IAT_mean"),
        MODEL_FEATURES.index("bwd_packets_IAT_mean")
    ]
    
    # Try converting to microseconds (multiply by 1e6)
    feat_us = feat.copy()
    for idx in iat_indices:
        feat_us[idx] = feat[idx] * 1e6
    score_us = engine.predict(feat_us)
    print(f"IAT in MICROSECONDS: Score = {score_us['score']:.4f}")
    
    # Try converting to nanoseconds (multiply by 1e9)
    feat_ns = feat.copy()
    for idx in iat_indices:
        feat_ns[idx] = feat[idx] * 1e9
    score_ns = engine.predict(feat_ns)
    print(f"IAT in NANOSECONDS: Score = {score_ns['score']:.4f}")
    
    # What about other features?
    # Let's see which features contribute most to the reconstruction error
    # Score = ||z - mu_norm||^2
    # Let's see the scaled features for all three cases
    print("\nScaled Feature Vectors comparison:")
    print(f"{'Index':<5} {'Feature Name':<35} {'Seconds':<12} {'Microseconds':<12} {'Nanoseconds':<12}")
    print("-" * 95)
    for i, name in enumerate(MODEL_FEATURES):
        mean = scaler.mean_[i]
        scale = scaler.scale_[i]
        sc_sec = (feat[i] - mean) / scale
        sc_us  = (feat_us[i] - mean) / scale
        sc_ns  = (feat_ns[i] - mean) / scale
        print(f"{i:<5} {name[:35]:<35} {sc_sec:<12.4f} {sc_us:<12.4f} {sc_ns:<12.4f}")

if __name__ == "__main__":
    main()
