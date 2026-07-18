"""
zero_day_test.py — Kiểm tra phát hiện zero-day attacks.

Tái hiện Bảng 5 trong bài báo Sattar et al. 2025:
Model không thấy attack pattern này trong training,
nhưng có phát hiện được không?

Với dataset thật: tách một số attack category ra khỏi training.
Với synthetic: tạo attack pattern hoàn toàn mới (kiểu phân phối khác).

Usage:
    python evaluation/zero_day_test.py
    python evaluation/zero_day_test.py --dataset unsw_nb15
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.inference import ETSSLInference
from evaluation.metrics import compute_binary_metrics, compute_auc_roc, print_metrics_table

from configs.paths import get_model_dir

MODEL_DIR = str(get_model_dir())


def generate_zero_day_patterns(n_each: int = 200, input_dim: int = 20) -> dict:
    """
    Tạo các pattern zero-day khác nhau:
    1. High-volume: nhiều packet nhỏ (DDoS-like)
    2. Low-slow: ít packet nhưng payload lớn (slow-read attack)
    3. Evasion: traffic gần giống normal nhưng có padding ngẫu nhiên
    4. Timing: bất thường về IAT (burst rồi dừng hẳn)
    """
    rng = np.random.default_rng(123)

    patterns = {
        "high_volume_ddos": rng.normal([5.0, 0.1] + [0.0] * 18, [1.0, 0.05] + [0.5] * 18, (n_each, input_dim)).astype(np.float32),
        "low_slow": rng.normal([-2.0, 5.0] + [0.0] * 18, [0.3, 2.0] + [0.5] * 18, (n_each, input_dim)).astype(np.float32),
        "evasion": rng.normal([0.5] * input_dim, [2.0] * input_dim, (n_each, input_dim)).astype(np.float32),
        "timing_burst": rng.exponential(3.0, (n_each, input_dim)).astype(np.float32),
    }
    return patterns


def run_zero_day_test(
    dataset_name: str = "synthetic",
    backend: str = "onnx",
    n_zero_day: int = 200,
) -> dict:
    """
    Zero-day detection test.

    Strategy:
    1. Load engine với model đã train (không thấy zero-day attacks)
    2. Chạy normal test data → tính FP baseline
    3. Chạy zero-day attack patterns → tính TPR

    Returns:
        dict kết quả theo từng pattern
    """
    print("=" * 60)
    print(f"🛡️  Zero-Day Detection Test — {dataset_name}")
    print("=" * 60)

    # Load engine
    engine = ETSSLInference(MODEL_DIR, backend=backend)

    # Load/generate test data
    if dataset_name == "synthetic":
        from data.preprocess import generate_synthetic
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        # Normal traffic (đã thấy trong training)
        X_normal = rng.normal(0.0, 1.0, (500, 20)).astype(np.float32)
        # Standard attacks (đã thấy trong training)
        X_known = rng.normal(3.0, 1.5, (200, 20)).astype(np.float32)

        # Fit scaler nếu cần
        if engine.scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(X_normal)
            engine.scaler = scaler

        # Zero-day patterns (không thấy trong training)
        zd_patterns = generate_zero_day_patterns(n_each=n_zero_day)

    else:
        # Với dataset thật: cần tách một attack category ra
        # (simplified: dùng last 20% của test set như "zero-day")
        data_dir = Path(__file__).parent.parent / f"data/processed/{dataset_name}"
        X_test = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test.npy")

        X_normal = X_test[y_test == 0][:500]
        X_known  = X_test[y_test == 1][:200]
        # Dùng scaled data trực tiếp làm zero-day proxy
        zd_patterns = {
            "unseen_attacks": X_test[y_test == 1][200:200+n_zero_day]
        }

    # 1. Baseline FP rate trên normal traffic
    print("\n📊 Baseline: Normal traffic (should be ≤ δ)...")
    normal_results = engine.predict_batch(X_normal)
    normal_scores = np.array([r["score"] for r in normal_results])
    normal_preds  = np.array([int(r["is_anomaly"]) for r in normal_results])
    fp_rate = normal_preds.mean()
    print(f"  FP rate on normal: {fp_rate*100:.2f}% (target: < {5.0:.1f}%)")
    print(f"  Score: mean={normal_scores.mean():.2f} ± {normal_scores.std():.2f}")

    # 2. Known attacks detection
    print("\n🔴 Known attacks (đã thấy trong training distribution)...")
    known_results = engine.predict_batch(X_known)
    known_preds = np.array([int(r["is_anomaly"]) for r in known_results])
    known_tpr = known_preds.mean()
    print(f"  TPR: {known_tpr*100:.2f}%")

    # 3. Zero-day detection
    print("\n🆕 Zero-day attack patterns:")
    print(f"  {'Pattern':<25} {'TPR':>8} {'Mean Score':>12} {'Verdict'}")
    print(f"  {'─'*55}")

    pattern_results = {}
    for name, X_zd in zd_patterns.items():
        zd_res = engine.predict_batch(X_zd)
        zd_scores = np.array([r["score"] for r in zd_res])
        zd_preds  = np.array([int(r["is_anomaly"]) for r in zd_res])
        tpr = zd_preds.mean()
        verdict = "✅ Detected" if tpr >= 0.7 else ("⚠️ Partial" if tpr >= 0.4 else "❌ Missed")

        pattern_results[name] = {
            "tpr": round(float(tpr), 4),
            "mean_score": round(float(zd_scores.mean()), 2),
            "std_score": round(float(zd_scores.std()), 2),
            "n_samples": len(X_zd),
        }
        print(f"  {name:<25} {tpr*100:>7.2f}% {zd_scores.mean():>12.2f}  {verdict}")

    # 4. Summary
    all_tprs = [v["tpr"] for v in pattern_results.values()]
    avg_zd_tpr = np.mean(all_tprs)

    print(f"\n{'='*60}")
    print(f"  Zero-Day Summary:")
    print(f"  ├── Avg zero-day TPR: {avg_zd_tpr*100:.2f}%")
    print(f"  ├── Normal FP rate:   {fp_rate*100:.2f}%")
    print(f"  └── Known attack TPR: {known_tpr*100:.2f}%")
    print(f"\n  📄 Paper reference (Bảng 5):")
    print(f"     Zero-day TPR ~90% | FPR ~5%")
    print(f"{'='*60}")

    results = {
        "dataset": dataset_name,
        "baseline_fp_rate": round(float(fp_rate), 4),
        "known_attack_tpr": round(float(known_tpr), 4),
        "zero_day_avg_tpr": round(float(avg_zd_tpr), 4),
        "patterns": pattern_results,
        "delta_used": engine.delta,
    }

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"zero_day_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved: {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--backend", default="onnx")
    parser.add_argument("--n_zero_day", type=int, default=200)
    args = parser.parse_args()

    run_zero_day_test(args.dataset, args.backend, args.n_zero_day)
