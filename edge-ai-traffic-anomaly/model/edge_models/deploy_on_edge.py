
# =====================================================================
# EDGE DEPLOYMENT SCRIPT - CHẠY TRÊN RASPBERRY PI / JETSON
# =====================================================================

import pandas as pd
import numpy as np
import joblib
import json
import time

class EdgeDarknetDetector:
    """Darknet traffic detector cho edge device"""
    
    def __init__(self, model_dir='./models/'):
        print("Loading Edge AI models...")
        self.binary_model = joblib.load(model_dir + 'edge_binary_model.pkl')
        self.scaler = joblib.load(model_dir + 'edge_scaler.pkl')
        self.le_binary = joblib.load(model_dir + 'edge_binary_label_encoder.pkl')
        
        self.multi_model = joblib.load(model_dir + 'edge_multiclass_model.pkl')
        self.le_multi = joblib.load(model_dir + 'edge_multiclass_label_encoder.pkl')
        
        with open(model_dir + 'edge_multiclass_features.json', 'r') as f:
            self.multi_features = json.load(f)['features']
        with open(model_dir + 'edge_schema.json', 'r') as f:
            self.schema = json.load(f)
        self.edge_features = self.schema['edge_features']
        
        print(f"✅ Models loaded! {len(self.edge_features)} features")
    
    def detect(self, features):
        """
        Detect traffic
        
        Args:
            features: list or numpy array of features
        
        Returns:
            dict: {
                'label': 'Encrypted' or 'Non-Encrypted',
                'type': 'VPN/TOR/I2P/FREENET/ZERONET' or 'Unknown',
                'confidence': float
            }
        """
        features = np.array(features).reshape(1, -1)
        
        # Binary
        features_scaled = self.scaler.transform(features)
        proba = self.binary_model.predict_proba(features_scaled)
        pred = self.binary_model.predict(features_scaled)
        confidence = float(np.max(proba))
        label_binary = self.le_binary.inverse_transform(pred)[0]
        
        result = {
            'label': label_binary,
            'type': 'Unknown',
            'confidence': confidence
        }
        
        # Multi-class
        if label_binary == 'Encrypted':
            # Lấy đúng features cho multi-class
            multi_idx = [self.edge_features.index(f) for f in self.multi_features]
            X_multi = features[:, multi_idx]
            pred_multi = self.multi_model.predict(X_multi)[0]
            label_multi = self.le_multi.inverse_transform([pred_multi])[0]
            result['type'] = label_multi
        
        return result
    
    def batch_detect(self, features_list):
        """Batch detection"""
        results = []
        for features in features_list:
            results.append(self.detect(features))
        return results

# =====================================================================
# USAGE EXAMPLE
# =====================================================================
if __name__ == "__main__":
    detector = EdgeDarknetDetector()
    
    # Giả lập dữ liệu từ network capture
    sample_features = [0.5] * len(detector.edge_features)
    
    start = time.time()
    result = detector.detect(sample_features)
    elapsed = (time.time() - start) * 1000
    
    print(f"Result: {result}")
    print(f"Time: {elapsed:.2f} ms")
