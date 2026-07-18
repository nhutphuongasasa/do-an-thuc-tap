"""
compare_before_after.py — So sánh model gốc vs. đã quantize/prune.

Tái hiện phần "Model Optimization Analysis" trong báo cáo:
- Size reduction (KB)
- Latency improvement (ms)  
- Accuracy preservation (F1, AUC)
- Output drift (mean absolute diff in embedding)

Output: bảng so sánh + biểu đồ radar/bar.

Usage:
    python optimization/compare_before_after.py --dataset synthetic
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.inference import ETSSLInference
from model.quantize import run_quantization
from optimization.benchmark import benchmark_backend, ResourceMonitor
from evaluation.metrics import compute_binary_metrics, compute_auc_roc

from configs.paths import get_model_dir

MODEL_DIR = str(get_model_dir())


def compare_all(
    dataset_name: str = "synthetic",
    n_benchmark_runs: int = 500,
    plot: bool = True,
) -> dict:
    """
    So sánh đầy đủ FP32 → INT8 → ONNX.

    1. Benchmark latency/throughput
    2. Đo accuracy trên dataset
    3. Đo output drift (embedding diff)
    4. In bảng tổng hợp
    """
    print("=" * 65)
    print("⚡ Model Optimization Comparison: FP32 vs INT8 vs ONNX")
    print("=" * 65)

    # Load test data
    if dataset_name == "synthetic":
        from data.preprocess import generate_synthetic
        from sklearn.preprocessing import StandardScaler

        print("\n🎲 Loading synthetic test data...")
        X, y = generate_synthetic(n_samples=5000, random_seed=99)
        X_test = X[4000:]
        y_test = y[4000:]
        X_train_n = X[y == 0][:2000]
        scaler = StandardScaler().fit(X_train_n)
        X_test_s = scaler.transform(X_test).astype(np.float32)
    else:
        data_dir = Path(__file__).parent.parent / f"data/processed/{dataset_name}"
        X_test_s = np.load(data_dir / "X_test.npy")
        y_test   = np.load(data_dir / "y_test.npy")
        scaler = None

    print(f"  Test samples: {len(X_test_s):,} | Anomaly: {sum(y_test==1):,}")

    # Model file sizes
    model_dir = Path(MODEL_DIR)
    sizes_kb = {
        "fp32": (model_dir / "encoder_fp32.pt").stat().st_size / 1024,
        "int8": (model_dir / "encoder_int8.pt").stat().st_size / 1024,
        "onnx": ((model_dir / "encoder_v5.onnx").stat().st_size +
                 (model_dir / "encoder_v5.onnx.data").stat().st_size) / 1024
                if (model_dir / "encoder_v5.onnx.data").exists()
                else (model_dir / "encoder_v5.onnx").stat().st_size / 1024,
    }

    monitor = ResourceMonitor()
    results = {}

    for backend in ["fp32", "int8", "onnx"]:
        print(f"\n{'─'*50}")
        print(f"🔧 Testing {backend.upper()}...")

        try:
            engine = ETSSLInference(MODEL_DIR, backend=backend)
            if scaler and engine.scaler is None:
                engine.scaler = scaler

            # Benchmark latency
            bm = engine.benchmark(n_runs=n_benchmark_runs)

            # Predict + metrics
            preds_raw = engine.predict_batch(X_test_s)
            scores = np.array([r["score"] for r in preds_raw])
            preds  = np.array([int(r["is_anomaly"]) for r in preds_raw])
            metrics = compute_binary_metrics(y_test, preds)
            auc, _, _, _ = compute_auc_roc(y_test, scores)

            results[backend] = {
                "size_kb":        round(sizes_kb[backend], 1),
                "mean_latency_ms":bm["mean_ms"],
                "p95_latency_ms": bm["p95_ms"],
                "throughput_fps": bm["throughput_fps"],
                "accuracy":       metrics["accuracy"],
                "precision":      metrics["precision"],
                "recall":         metrics["recall"],
                "f1":             metrics["f1"],
                "fpr":            metrics["fpr"],
                "auc":            round(auc, 4),
            }

            print(f"  Size: {sizes_kb[backend]:.1f} KB | "
                  f"Latency: {bm['mean_ms']:.3f}ms | "
                  f"F1: {metrics['f1']:.4f} | AUC: {auc:.4f}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[backend] = {"error": str(e)}

    # --- Output drift: FP32 vs INT8 ---
    print("\n🔬 Computing embedding drift FP32 vs INT8...")
    try:
        eng_fp32 = ETSSLInference(MODEL_DIR, backend="fp32")
        eng_int8 = ETSSLInference(MODEL_DIR, backend="int8")
        if scaler:
            eng_fp32.scaler = scaler
            eng_int8.scaler = scaler

        n_drift = min(200, len(X_test_s))
        sample = X_test_s[:n_drift]
        z_fp32 = np.array([r["embedding"] for r in eng_fp32.predict_batch(sample)])
        z_int8 = np.array([r["embedding"] for r in eng_int8.predict_batch(sample)])

        drift = np.abs(z_fp32 - z_int8)
        drift_result = {
            "mean_abs_diff":    round(float(drift.mean()), 6),
            "max_abs_diff":     round(float(drift.max()), 6),
            "mean_rel_diff_pct":round(float((drift / (np.abs(z_fp32) + 1e-8)).mean() * 100), 3),
        }
        print(f"  Mean |Δz|: {drift_result['mean_abs_diff']:.6f} | "
              f"Relative: {drift_result['mean_rel_diff_pct']:.3f}%")
    except Exception as e:
        drift_result = {"error": str(e)}
        print(f"  ⚠️  Drift computation failed: {e}")

    # --- Print summary table ---
    print(f"\n{'='*75}")
    print(f"  OPTIMIZATION SUMMARY (Tái hiện phần 'Edge Efficiency' của báo cáo)")
    print(f"{'='*75}")
    print(f"  {'Metric':<22} {'FP32':>10} {'INT8':>10} {'ONNX':>10} {'Unit'}")
    print(f"  {'─'*65}")

    rows = [
        ("Model size",       "size_kb",         "KB"),
        ("Mean latency",     "mean_latency_ms",  "ms"),
        ("P95 latency",      "p95_latency_ms",   "ms"),
        ("Throughput",       "throughput_fps",   "fps"),
        ("Accuracy",         "accuracy",         ""),
        ("F1 Score",         "f1",               ""),
        ("AUC-ROC",          "auc",              ""),
        ("FPR",              "fpr",              ""),
    ]

    for label, key, unit in rows:
        vals = []
        for backend in ["fp32", "int8", "onnx"]:
            if backend in results and "error" not in results[backend]:
                v = results[backend].get(key, "—")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}" if key in ("accuracy","f1","auc","fpr")
                                else f"{v:.2f}")
                else:
                    vals.append(str(v))
            else:
                vals.append("ERR")
        print(f"  {label:<22} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}  {unit}")

    # Relative improvements vs FP32
    if "fp32" in results and "error" not in results["fp32"]:
        fp32 = results["fp32"]
        print(f"\n  Relative vs FP32 baseline:")
        for backend in ["int8", "onnx"]:
            if backend in results and "error" not in results[backend]:
                r = results[backend]
                size_red = (1 - r["size_kb"]/fp32["size_kb"]) * 100
                speed_up = fp32["mean_latency_ms"] / r["mean_latency_ms"]
                f1_diff = (r["f1"] - fp32["f1"]) * 100
                print(f"    {backend.upper()}: size ↓{size_red:.1f}% | "
                      f"speed ↑{speed_up:.2f}x | F1 diff {f1_diff:+.2f}%")

    print(f"\n  📄 Paper reference (server GPU):")
    print(f"     Latency 15–25ms | Throughput 1500–1900 fps | Accuracy 96.8%")
    print(f"  ℹ️  Laptop CPU benchmark dùng để chứng minh relative improvement,")
    print(f"     không so sánh trực tiếp số tuyệt đối với paper.")
    print(f"{'='*75}")

    # Save
    out = {
        "models": results,
        "embedding_drift_fp32_vs_int8": drift_result,
        "dataset": dataset_name,
    }
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "optimization_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n💾 Results saved: {out_path}")

    if plot:
        _plot_comparison(results)

    return out


def _plot_comparison(results: dict):
    """Vẽ bar chart so sánh latency, size, accuracy."""
    try:
        import matplotlib.pyplot as plt

        backends = [b for b in ["fp32", "int8", "onnx"] if b in results and "error" not in results[b]]
        colors = {"fp32": "#2196F3", "int8": "#4CAF50", "onnx": "#FF9800"}

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Model Optimization: FP32 vs INT8 vs ONNX", fontsize=13, fontweight="bold")

        # Latency
        ax = axes[0]
        vals = [results[b]["mean_latency_ms"] for b in backends]
        bars = ax.bar(backends, vals, color=[colors[b] for b in backends], alpha=0.85)
        ax.set_title("Mean Inference Latency")
        ax.set_ylabel("ms / sample")
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.grid(axis="y", alpha=0.3)

        # Model size
        ax = axes[1]
        vals = [results[b]["size_kb"] for b in backends]
        bars = ax.bar(backends, vals, color=[colors[b] for b in backends], alpha=0.85)
        ax.set_title("Model File Size")
        ax.set_ylabel("KB")
        ax.bar_label(bars, fmt="%.0f", padding=3)
        ax.grid(axis="y", alpha=0.3)

        # F1 score
        ax = axes[2]
        vals = [results[b]["f1"] for b in backends]
        bars = ax.bar(backends, vals, color=[colors[b] for b in backends], alpha=0.85)
        ax.set_title("F1 Score (anomaly detection)")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1.1)
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = Path(__file__).parent / "results/optimization_comparison.png"
        out_path.parent.mkdir(exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Plot saved: {out_path}")
        plt.show()
    except ImportError:
        print("  ⚠️  matplotlib not available")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare FP32 vs INT8 vs ONNX")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--n_runs", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    compare_all(dataset_name=args.dataset, n_benchmark_runs=args.n_runs, plot=not args.no_plot)
