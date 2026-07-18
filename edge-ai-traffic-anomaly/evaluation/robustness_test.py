"""
robustness_test.py — Robustness / evasion test (readme §5, Bảng 12).

Test model với traffic có obfuscation: padding, timing randomization, volume shift.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.paths import get_model_dir
from evaluation.metrics import compute_binary_metrics, print_metrics_table
from model.inference import ETSSLInference


def apply_evasion(X: np.ndarray, pattern: str, rng: np.random.Generator) -> np.ndarray:
    """Biến đổi feature vector mô phỏng evasion."""
    X = X.copy()
    if pattern == "padding":
        X[:, [12, 14, 15]] *= rng.uniform(1.5, 3.0, size=(len(X), 3))
    elif pattern == "timing_random":
        X[:, [0, 1, 6, 16]] += rng.normal(0, 2.0, size=(len(X), 4))
    elif pattern == "volume_shift":
        X += rng.normal(1.5, 0.8, size=X.shape)
    elif pattern == "combined":
        X = apply_evasion(X, "padding", rng)
        X = apply_evasion(X, "timing_random", rng)
    return X.astype(np.float32)


def run_robustness_test(
    dataset_name: str = "synthetic",
    backend: str = "onnx",
    n_samples: int = 500,
) -> dict:
    from data.preprocess import generate_synthetic

    model_dir = get_model_dir()
    engine = ETSSLInference(str(model_dir), backend=backend)

    if dataset_name == "synthetic":
        X, y = generate_synthetic(n_samples=n_samples * 2, anomaly_ratio=0.2)
    else:
        data_dir = Path(f"data/processed/{dataset_name}")
        X = np.load(data_dir / "X_test.npy")
        y = np.load(data_dir / "y_test.npy")
        n_samples = min(n_samples, len(X))
        X, y = X[:n_samples], y[:n_samples]

    if engine.scaler:
        X = engine.scaler.transform(X).astype(np.float32)

    rng = np.random.default_rng(42)
    patterns = ["baseline", "padding", "timing_random", "volume_shift", "combined"]
    results = {}

    print("=" * 60)
    print("🛡️  Robustness / Evasion Test (readme Bảng 12)")
    print("=" * 60)

    for pattern in patterns:
        X_test = X if pattern == "baseline" else apply_evasion(X, pattern, rng)
        preds = engine.predict_batch(X_test)
        y_pred = np.array([int(r["is_anomaly"]) for r in preds])
        m = compute_binary_metrics(y, y_pred)
        results[pattern] = m
        print(f"\n  Pattern: {pattern}")
        print(f"    Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | FPR={m['fpr']:.4f}")

    out = Path("evaluation/results") / f"robustness_{dataset_name}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Saved: {out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--backend", default="onnx")
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()
    run_robustness_test(args.dataset, args.backend, args.n_samples)
