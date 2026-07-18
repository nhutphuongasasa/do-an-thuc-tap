"""
metrics.py — Detection performance metrics cho ET-SSL.

Tái hiện cấu trúc Bảng 4, 5 trong bài báo Sattar et al. 2025,
nhưng đo trên laptop với dataset của bạn.

Usage:
    python evaluation/metrics.py --dataset unsw_nb15 --backend onnx
    python evaluation/metrics.py --dataset synthetic
"""

import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.inference import ETSSLInference

from configs.paths import get_model_dir

MODEL_DIR = str(get_model_dir())
PROCESSED_DIR = Path(__file__).parent.parent / "data/processed"


# =====================================================================
# Core metric functions
# =====================================================================
def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Tính Accuracy, Precision, Recall (TPR), FPR, F1.

    Args:
        y_true: binary ground truth (0=normal, 1=anomaly)
        y_pred: binary predictions (0=normal, 1=anomaly)
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)   # TPR
    fpr       = fp / (fp + tn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),    # TPR
        "fpr":       round(fpr, 4),
        "f1":        round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def compute_auc_roc(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tính AUC-ROC và trả về fpr/tpr/thresholds cho vẽ ROC curve.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return float(auc), fpr, tpr, thresholds


def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray, method: str = "youden") -> float:
    """
    Tìm ngưỡng tối ưu từ validation set.

    Methods:
        "youden": J = TPR - FPR (Youden's J statistic)
        "f1": maximize F1 score
    """
    from sklearn.metrics import roc_curve, f1_score

    fpr, tpr, thresholds = roc_curve(y_true, scores)

    if method == "youden":
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
    elif method == "f1":
        f1s = []
        for t in thresholds:
            preds = (scores >= t).astype(int)
            f1s.append(f1_score(y_true, preds, zero_division=0))
        best_idx = np.argmax(f1s)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(thresholds[best_idx])


def print_metrics_table(metrics: dict, title: str = "Detection Performance"):
    """In bảng metrics theo cấu trúc giống bài báo (Bảng 4)."""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  {'Metric':<20} {'Value':>10}  {'Paper (ref)':>12}")
    print(f"  {'-'*50}")
    paper_ref = {
        "accuracy":  "96.8%",
        "precision": "—",
        "recall":    "92.7%",
        "fpr":       "1.2%",
        "f1":        "—",
    }
    for key in ["accuracy", "precision", "recall", "fpr", "f1"]:
        if key in metrics:
            val = metrics[key]
            pct = f"{val*100:.2f}%"
            ref = paper_ref.get(key, "—")
            print(f"  {key.capitalize():<20} {pct:>10}  {ref:>12}")
    if "auc" in metrics:
        print(f"  {'AUC-ROC':<20} {metrics['auc']:>10.4f}  {'—':>12}")

    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics['tp']:,}  FP={metrics['fp']:,}")
    print(f"    FN={metrics['fn']:,}  TN={metrics['tn']:,}")
    print(f"{'='*55}")


# =====================================================================
# Main evaluation pipeline
# =====================================================================
def evaluate(
    dataset_name: str,
    backend: str = "onnx",
    delta_override: float = None,
    tune_threshold: bool = True,
    plot: bool = True,
) -> dict:
    """
    Chạy full evaluation pipeline.

    Args:
        dataset_name: "unsw_nb15" | "cic_darknet2020" | "synthetic"
        backend: "fp32" | "int8" | "onnx"
        delta_override: ghi đè ngưỡng δ nếu muốn
        tune_threshold: tự tìm ngưỡng tối ưu từ val set
        plot: vẽ ROC curve

    Returns:
        dict kết quả đầy đủ
    """
    print("=" * 60)
    print(f"🔬 ET-SSL Evaluation — Dataset: {dataset_name} | Backend: {backend}")
    print("=" * 60)

    # 1. Load model
    print("\n📦 Loading inference engine...")
    engine = ETSSLInference(
        model_dir=MODEL_DIR,
        backend=backend,
        delta_override=delta_override,
    )

    # 2. Load data
    data_dir = PROCESSED_DIR / dataset_name
    if dataset_name == "synthetic":
        from data.preprocess import generate_synthetic, split_and_save
        print("\n🎲 Generating synthetic test data...")
        X, y = generate_synthetic(n_samples=5000)
        X_train_n = X[y == 0][:3000]

        # Fit scaler nếu chưa có
        if engine.scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(X_train_n)
            engine.scaler = scaler
            print("  ⚠️  Scaler not found — fitted on synthetic normal data")

        # Split
        X_val = X[3500:4000]
        y_val = y[3500:4000]
        X_test = X[4000:]
        y_test = y[4000:]
    else:
        if not (data_dir / "X_test.npy").exists():
            raise FileNotFoundError(
                f"Processed data not found: {data_dir}\n"
                f"Run: python data/preprocess.py --dataset {dataset_name} --data_path ..."
            )
        print(f"\n📂 Loading processed data from: {data_dir}")
        X_val  = np.load(data_dir / "X_val.npy")
        y_val  = np.load(data_dir / "y_val.npy")
        X_test = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test.npy")

    print(f"  Val:  {len(X_val):,} samples | Test: {len(X_test):,} samples")

    # 3. Tune threshold trên val set nếu yêu cầu
    if tune_threshold:
        print("\n🔧 Tuning threshold on validation set...")
        val_results = engine.predict_batch(X_val)
        val_scores = np.array([r["score"] for r in val_results])

        optimal_delta = find_optimal_threshold(y_val, val_scores, method="youden")
        engine.update_delta(optimal_delta)
        print(f"  Optimal δ (Youden): {optimal_delta:.4f} (original: {engine.delta:.4f})")

    # 4. Predict trên test set
    print(f"\n⚡ Running inference on {len(X_test):,} test samples...")
    t0 = time.perf_counter()
    test_results = engine.predict_batch(X_test)
    elapsed = time.perf_counter() - t0

    scores = np.array([r["score"] for r in test_results])
    preds  = np.array([int(r["is_anomaly"]) for r in test_results])

    # 5. Compute metrics
    metrics = compute_binary_metrics(y_test, preds)
    auc, fpr_curve, tpr_curve, thresholds = compute_auc_roc(y_test, scores)
    metrics["auc"] = round(auc, 4)
    metrics["total_time_sec"] = round(elapsed, 3)
    metrics["avg_latency_ms"] = round(elapsed / len(X_test) * 1000, 3)
    metrics["throughput_fps"] = round(len(X_test) / elapsed, 1)
    metrics["delta_used"] = engine.delta
    metrics["dataset"] = dataset_name
    metrics["backend"] = backend

    print_metrics_table(metrics, title=f"Test Set Results — {dataset_name}")
    print(f"\n⚡ Throughput: {metrics['throughput_fps']:.1f} samples/sec")
    print(f"   Avg latency: {metrics['avg_latency_ms']:.3f} ms/sample")

    # 6. Plot (nếu yêu cầu)
    if plot:
        _plot_results(scores, y_test, fpr_curve, tpr_curve, auc, engine.delta, dataset_name)

    # 7. Lưu kết quả
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"metrics_{dataset_name}_{backend}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n💾 Saved: {out_path}")

    return metrics


def _plot_results(
    scores, y_true, fpr_curve, tpr_curve, auc, delta, dataset_name
):
    """Vẽ Score Distribution và ROC Curve."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"ET-SSL Evaluation — {dataset_name}", fontsize=13, fontweight="bold")

        # Score distribution
        ax = axes[0]
        normal_s = scores[y_true == 0]
        attack_s = scores[y_true == 1]
        ax.hist(normal_s, bins=50, alpha=0.6, color="#2196F3", label=f"Normal (n={len(normal_s):,})")
        ax.hist(attack_s, bins=50, alpha=0.6, color="#F44336", label=f"Anomaly (n={len(attack_s):,})")
        ax.axvline(delta, color="#4CAF50", lw=2, linestyle="--", label=f"δ = {delta:.2f}")
        ax.set_xlabel("Anomaly Score S(x) = ||z - μ_norm||²")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution")
        ax.legend()
        ax.grid(alpha=0.3)

        # ROC Curve
        ax = axes[1]
        ax.plot(fpr_curve, tpr_curve, lw=2, color="#9C27B0", label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.5)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out_path = Path(__file__).parent / f"results/roc_{dataset_name}.png"
        out_path.parent.mkdir(exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Plot saved: {out_path}")
        plt.show()
    except ImportError:
        print("  ⚠️  matplotlib not installed — skipping plots")


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ET-SSL detection performance")
    parser.add_argument("--dataset", default="synthetic",
                        choices=["synthetic", "unsw_nb15", "cic_darknet2020"])
    parser.add_argument("--backend", default="onnx",
                        choices=["fp32", "int8", "onnx"])
    parser.add_argument("--delta", type=float, default=None, help="Override delta threshold")
    parser.add_argument("--no-tune", action="store_true", help="Skip threshold tuning on val set")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")

    args = parser.parse_args()
    evaluate(
        dataset_name=args.dataset,
        backend=args.backend,
        delta_override=args.delta,
        tune_threshold=not args.no_tune,
        plot=not args.no_plot,
    )
