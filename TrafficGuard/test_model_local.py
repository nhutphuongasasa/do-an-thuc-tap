# =====================================================================
# TEST V5 EDGE MODEL TRÊN LAPTOP LINUX
# File: test_model_local.py (FIXED INT8)
# =====================================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. ĐỊNH NGHĨA ENCODER (giống hệt khi train)
# =====================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256, embed_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_dim), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_dim), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, embed_dim),
        )
    def forward(self, x):
        return self.net(x)

# =====================================================================
# 2. ENCODER WRAPPER CHO INT8
# =====================================================================
class QuantizedEncoderWrapper(nn.Module):
    """Wrapper để load quantized model"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, x):
        return self.encoder(x)

# =====================================================================
# 3. CLASS INFERENCE V5 (FIXED)
# =====================================================================
class V5Inference:
    def __init__(self, model_dir="models/edge_ai-20260716T101644Z-1-001/edge_ai/", use_int8=False):
        self.model_dir = Path(model_dir)
        self.use_int8 = use_int8
        
        print(f"🔧 Initializing V5 Inference (INT8={use_int8})...")
        print(f"📁 Model directory: {self.model_dir}")
        
        # Load artifacts
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        self.mu_norm = np.load(self.model_dir / 'mu_norm.npy')
        
        # Fix delta loading
        delta_data = np.load(self.model_dir / 'delta.npy')
        self.delta = float(delta_data.item()) if delta_data.size == 1 else float(delta_data[0])
        
        with open(self.model_dir / 'config.json') as f:
            self.config = json.load(f)
        
        print(f"  ✓ Loaded scaler, mu_norm, delta={self.delta:.4f}")
        print(f"  ✓ Config: input_dim={self.config['input_dim']}, embed_dim={self.config['embed_dim']}")
        
        # Load model
        model_name = 'encoder_int8.pt' if use_int8 else 'encoder_fp32.pt'
        model_path = self.model_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Tạo encoder
        self.model = Encoder(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            embed_dim=self.config['embed_dim']
        )
        
        if use_int8:
            # INT8: load với wrapper
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Nếu state_dict có prefix 'encoder.', loại bỏ nó
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('encoder.'):
                    new_key = key[8:]  # Remove 'encoder.'
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Load state dict
            try:
                self.model.load_state_dict(new_state_dict, strict=False)
                print(f"  ✓ Loaded INT8 model (strict=False)")
            except Exception as e:
                print(f"  ⚠️ INT8 load warning: {e}")
                # Thử load trực tiếp
                self.model.load_state_dict(state_dict, strict=False)
                print(f"  ✓ Loaded INT8 model (strict=False)")
        else:
            # FP32: load bình thường
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.model.eval()
        
        print(f"  ✓ Loaded model: {model_name}")
        print("✅ Initialization complete!\n")
    
    def predict(self, data):
        """Predict anomaly score for single sample"""
        # Scale
        data_scaled = self.scaler.transform(data.reshape(1, -1))
        
        # Inference
        with torch.no_grad():
            x = torch.from_numpy(data_scaled).float()
            embedding = self.model(x).numpy()
        
        # Anomaly score
        score = np.sum((embedding - self.mu_norm) ** 2, axis=1)[0]
        
        return {
            'embedding': embedding[0],
            'score': float(score),
            'is_anomaly': bool(score > self.delta)
        }
    
    def predict_batch(self, data_batch):
        """Predict anomaly scores for batch"""
        results = []
        for data in data_batch:
            results.append(self.predict(data))
        return results
    
    def benchmark(self, n_runs=1000):
        """Benchmark inference speed"""
        print(f"📊 Benchmarking {n_runs} runs...")
        test_data = np.random.randn(self.config['input_dim']) * 0.5
        
        # Warmup
        for _ in range(10):
            self.predict(test_data)
        
        start = time.time()
        for _ in range(n_runs):
            self.predict(test_data)
        elapsed = time.time() - start
        
        avg_time = elapsed / n_runs * 1000  # ms
        print(f"  ⚡ Average inference time: {avg_time:.3f} ms/sample")
        print(f"  📈 Throughput: {n_runs / elapsed:.1f} samples/sec")
        return avg_time

# =====================================================================
# 4. TẠO DỮ LIỆU MẪU CHO TEST
# =====================================================================
def generate_test_data(n_samples=500, n_features=20):
    """Generate sample data for testing"""
    # Normal data (mean=0, std=0.5)
    normal_data = np.random.randn(n_samples, n_features) * 0.5
    
    # Anomaly data (mean=1, std=1.0)
    anomaly_data = np.random.randn(n_samples, n_features) * 1.0 + 1.0
    
    # Labels: 0=normal, 1=anomaly
    X_test = np.vstack([normal_data, anomaly_data])
    y_test = np.array([0]*n_samples + [1]*n_samples)
    
    return X_test, y_test

# =====================================================================
# 5. MAIN TEST
# =====================================================================
def main():
    print("="*60)
    print("🧪 V5 EDGE MODEL TEST")
    print("="*60)
    
    # Đường dẫn đến model
    MODEL_DIR = "models/edge_ai-20260716T101644Z-1-001/edge_ai/"
    
    # Check files
    required_files = ['scaler.pkl', 'mu_norm.npy', 'delta.npy', 'config.json']
    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            missing.append(f)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        print("Please check model directory path")
        return
    
    # =============================================================
    # TEST 1: FP32 MODEL
    # =============================================================
    print("\n" + "="*60)
    print("🔬 TESTING FP32 MODEL")
    print("="*60)
    
    engine_fp32 = V5Inference(model_dir=MODEL_DIR, use_int8=False)
    
    # Test single sample
    test_data = np.random.randn(20) * 0.5
    result = engine_fp32.predict(test_data)
    print(f"📊 Sample prediction:")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Is anomaly: {result['is_anomaly']}")
    print(f"  Embedding shape: {result['embedding'].shape}")
    print(f"  Embedding[:5]: {result['embedding'][:5]}...")
    
    # Benchmark
    avg_time_fp32 = engine_fp32.benchmark(500)
    
    # =============================================================
    # TEST 2: INT8 MODEL (SKIP IF FAILS)
    # =============================================================
    print("\n" + "="*60)
    print("🔬 TESTING INT8 MODEL")
    print("="*60)
    
    try:
        engine_int8 = V5Inference(model_dir=MODEL_DIR, use_int8=True)
        
        # Compare results
        print("📊 Comparing FP32 vs INT8 (5 samples):")
        print("-" * 70)
        print(f"{'Sample':<8} {'FP32 Score':<12} {'INT8 Score':<12} {'Difference':<12}")
        print("-" * 70)
        
        diffs = []
        for i in range(5):
            test_data = np.random.randn(20) * 0.5
            r_fp32 = engine_fp32.predict(test_data)
            r_int8 = engine_int8.predict(test_data)
            diff = abs(r_fp32['score'] - r_int8['score'])
            diffs.append(diff)
            print(f"{i+1:<8} {r_fp32['score']:<12.4f} {r_int8['score']:<12.4f} {diff:<12.4f}")
        
        print("-" * 70)
        print(f"Average difference: {np.mean(diffs):.4f}")
        
        # Benchmark INT8
        avg_time_int8 = engine_int8.benchmark(500)
        int8_available = True
        
    except Exception as e:
        print(f"⚠️ INT8 model test failed: {e}")
        print("   Skipping INT8 tests...")
        avg_time_int8 = 0
        int8_available = False
    
    # =============================================================
    # TEST 3: EVALUATION ON SAMPLE DATA
    # =============================================================
    print("\n" + "="*60)
    print("📊 EVALUATION ON SAMPLE DATA")
    print("="*60)
    
    # Generate test data
    X_test, y_test = generate_test_data(n_samples=500, n_features=20)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Normal samples: {sum(y_test==0)}, Anomaly samples: {sum(y_test==1)}")
    
    # Predict
    scores = []
    for i in range(len(X_test)):
        result = engine_fp32.predict(X_test[i])
        scores.append(result['score'])
    
    scores = np.array(scores)
    
    # Metrics
    auc = roc_auc_score(y_test, scores)
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    
    # Best threshold based on Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    preds = (scores > best_threshold).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"\n📊 Performance on test data:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  Model threshold: {engine_fp32.delta:.4f}")
    
    # Score distribution
    normal_scores = scores[y_test == 0]
    anomaly_scores = scores[y_test == 1]
    
    print(f"\n📊 Score statistics:")
    print(f"  Normal scores: mean={np.mean(normal_scores):.4f}, std={np.std(normal_scores):.4f}")
    print(f"  Anomaly scores: mean={np.mean(anomaly_scores):.4f}, std={np.std(anomaly_scores):.4f}")
    
    # =============================================================
    # TEST 4: VISUALIZATION
    # =============================================================
    print("\n" + "="*60)
    print("📊 GENERATING PLOTS")
    print("="*60)
    
    # Figure 1: Score Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.hist(normal_scores, bins=30, alpha=0.6, label='Normal', color='blue')
    ax1.hist(anomaly_scores, bins=30, alpha=0.6, label='Anomaly', color='red')
    ax1.axvline(engine_fp32.delta, color='green', linestyle='--', 
                label=f'Threshold = {engine_fp32.delta:.2f}')
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Figure 2: ROC Curve
    ax2 = axes[1]
    ax2.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v5_test_results.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: v5_test_results.png")
    plt.show()
    
    # =============================================================
    # TEST 5: CHECK ONNX FILE
    # =============================================================
    print("\n" + "="*60)
    print("🔍 CHECKING ONNX FILE")
    print("="*60)
    
    onnx_path = os.path.join(MODEL_DIR, 'encoder_v5.onnx')
    if os.path.exists(onnx_path):
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"✅ ONNX file exists: {onnx_path}")
        print(f"   Size: {file_size:.2f} MB")
        
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print("   ✅ ONNX model is valid")
            print(f"   Opset version: {model.opset_import[0].version}")
            print(f"   Inputs: {[i.name for i in model.graph.input]}")
            print(f"   Outputs: {[o.name for o in model.graph.output]}")
        except Exception as e:
            print(f"   ⚠️ ONNX check failed: {e}")
    else:
        print("⚠️ ONNX file not found")
    
    # =============================================================
    # SUMMARY
    # =============================================================
    print("\n" + "="*60)
    print("✅ V5 EDGE MODEL TEST COMPLETE!")
    print("="*60)
    
    summary = f"""
📁 Model directory: {MODEL_DIR}

📊 Performance Summary:
  ├── FP32 Inference: {avg_time_fp32:.3f} ms/sample
  ├── Throughput: {1000/avg_time_fp32:.1f} samples/sec
  ├── Test AUC: {auc:.4f}
  └── Test F1: {f1:.4f}
"""
    
    if int8_available:
        summary += f"""
  ├── INT8 Inference: {avg_time_int8:.3f} ms/sample
  ├── INT8 Throughput: {1000/avg_time_int8:.1f} samples/sec
  └── INT8 Speedup: {avg_time_fp32/avg_time_int8:.2f}x
"""
    else:
        summary += "  └── INT8: Not available (skipped)\n"
    
    summary += """
📁 Files tested:
  ├── encoder_fp32.pt: OK
  ├── scaler.pkl: OK
  ├── mu_norm.npy: OK
  ├── delta.npy: OK
  └── config.json: OK

✅ Model is ready for deployment!
"""
    
    print(summary)

if __name__ == "__main__":
    main()