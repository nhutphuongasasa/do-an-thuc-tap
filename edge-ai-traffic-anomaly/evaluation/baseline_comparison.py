"""
baseline_comparison.py — So sánh ET-SSL vs. baseline methods.

Tái hiện Bảng 8 trong bài báo: ET-SSL vs Random Forest vs K-Means.
Dùng cùng feature set và test set.

Usage:
    python evaluation/baseline_comparison.py --dataset synthetic
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.inference import ETSSLInference
from evaluation.metrics import compute_binary_metrics, compute_auc_roc

from configs.paths import get_model_dir

MODEL_DIR = str(get_model_dir())


def run_comparison(dataset_name: str = "synthetic") -> dict:
    """
    So sánh 3 phương pháp:
    1. ET-SSL (self-supervised, model đã có)
    2. Random Forest (supervised, train trên labeled data)
    3. K-Means (unsupervised, cluster-based)
    """
    print("=" * 65)
    print(f"📊 Baseline Comparison — {dataset_name}")
    print("   (Tái hiện Bảng 8 trong Sattar et al. 2025)")
    print("=" * 65)

    # Load data
    if dataset_name == "synthetic":
        from data.preprocess import generate_synthetic
        from sklearn.preprocessing import StandardScaler

        X, y = generate_synthetic(n_samples=8000, random_seed=7)
        X_train, y_train = X[:5000], y[:5000]
        X_test,  y_test  = X[5000:], y[5000:]

        scaler = StandardScaler().fit(X_train[y_train == 0])
        X_train_s = scaler.transform(X_train).astype(np.float32)
        X_test_s  = scaler.transform(X_test).astype(np.float32)
    else:
        data_dir = Path(__file__).parent.parent / f"data/processed/{dataset_name}"
        X_train_s = np.load(data_dir / "X_train.npy")
        y_train   = np.load(data_dir / "y_train.npy")
        X_test_s  = np.load(data_dir / "X_test.npy")
        y_test    = np.load(data_dir / "y_test.npy")
        scaler    = None

    results = {}

    # ---------------------------------------------------------------
    # 1. ET-SSL (model đã có)
    # ---------------------------------------------------------------
    print("\n🔬 ET-SSL (self-supervised)...")
    try:
        engine = ETSSLInference(MODEL_DIR, backend="onnx")
        if scaler and not engine.scaler:
            engine.scaler = scaler

        preds_raw = engine.predict_batch(X_test_s)
        scores    = np.array([r["score"] for r in preds_raw])
        preds     = np.array([int(r["is_anomaly"]) for r in preds_raw])

        m = compute_binary_metrics(y_test, preds)
        auc, _, _, _ = compute_auc_roc(y_test, scores)

        results["ET-SSL"] = {**m, "auc": round(auc, 4), "requires_labels": False}
        print(f"  F1={m['f1']:.4f} | AUC={auc:.4f} | Acc={m['accuracy']:.4f}")
    except Exception as e:
        results["ET-SSL"] = {"error": str(e)}
        print(f"  ❌ {e}")

    # ---------------------------------------------------------------
    # 2. Random Forest (supervised)
    # ---------------------------------------------------------------
    print("\n🌳 Random Forest (supervised)...")
    try:
        from sklearn.ensemble import RandomForestClassifier

        # Train chỉ trên normal + labeled attacks (supervised cần nhãn)
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(X_train_s, y_train)
        preds_rf = rf.predict(X_test_s)
        scores_rf = rf.predict_proba(X_test_s)[:, 1]

        m = compute_binary_metrics(y_test, preds_rf)
        auc, _, _, _ = compute_auc_roc(y_test, scores_rf)

        results["Random Forest"] = {**m, "auc": round(auc, 4), "requires_labels": True}
        print(f"  F1={m['f1']:.4f} | AUC={auc:.4f} | Acc={m['accuracy']:.4f}")
    except Exception as e:
        results["Random Forest"] = {"error": str(e)}
        print(f"  ❌ {e}")

    # ---------------------------------------------------------------
    # 3. K-Means (unsupervised)
    # ---------------------------------------------------------------
    print("\n🔵 K-Means (unsupervised, k=2)...")
    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        # Train chỉ trên normal (như ET-SSL)
        X_train_normal = X_train_s[y_train == 0]
        kmeans.fit(X_train_normal)

        # Score = distance to nearest centroid
        dists = kmeans.transform(X_test_s)
        dist_to_nearest = dists.min(axis=1)

        # Threshold = 95th percentile của train normal
        train_dists = kmeans.transform(X_train_normal).min(axis=1)
        threshold = np.percentile(train_dists, 95)

        preds_km = (dist_to_nearest > threshold).astype(int)
        m = compute_binary_metrics(y_test, preds_km)
        auc, _, _, _ = compute_auc_roc(y_test, dist_to_nearest)

        results["K-Means"] = {**m, "auc": round(auc, 4), "requires_labels": False}
        print(f"  F1={m['f1']:.4f} | AUC={auc:.4f} | Acc={m['accuracy']:.4f}")
    except Exception as e:
        results["K-Means"] = {"error": str(e)}
        print(f"  ❌ {e}")

    # ---------------------------------------------------------------
    # Summary Table
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE (Bảng 8 in paper)")
    print(f"{'='*70}")
    print(f"  {'Method':<18} {'Acc':>7} {'F1':>7} {'AUC':>7} {'FPR':>7} {'Labels?':>9}")
    print(f"  {'─'*58}")

    for method, r in results.items():
        if "error" in r:
            print(f"  {method:<18} ERROR")
            continue
        labeled = "✅ Yes" if r["requires_labels"] else "❌ No"
        print(f"  {method:<18} {r['accuracy']*100:>6.2f}% {r['f1']:>7.4f} "
              f"{r['auc']:>7.4f} {r['fpr']*100:>6.2f}%  {labeled}")

    print(f"\n  📄 Paper reference (Bảng 8, CIC-Darknet2020):")
    print(f"     ET-SSL: Acc=96.8% | RF (supervised): Acc=94.2% | K-Means: Acc=78.5%")
    print(f"  ℹ️  ET-SSL không cần nhãn (self-supervised) — lợi thế thực tế lớn")
    print(f"{'='*70}")

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"baseline_comparison_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset_name, "results": results}, f, indent=2, default=str)
    print(f"\n💾 Saved: {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic")
    args = parser.parse_args()
    run_comparison(args.dataset)
