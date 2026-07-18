"""
drift_simulation.py — Giả lập traffic drift và test incremental learning.

Chia dataset thành nhiều "đợt" theo thời gian, giả lập traffic drift.
Đo model có thích nghi tốt với incremental learning không.

Tái hiện phần evaluation Giai đoạn 6 + Bảng 11 của bài báo.

Usage:
    python evaluation/drift_simulation.py
    python evaluation/drift_simulation.py --n_epochs 10 --alpha 0.95
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.inference import ETSSLInference
from pipeline.incremental_learner import IncrementalLearner, simulate_drift
from evaluation.metrics import compute_binary_metrics

from configs.paths import get_model_dir

MODEL_DIR = str(get_model_dir())


def run_drift_simulation(
    n_epochs: int = 5,
    n_flows_per_epoch: int = 1000,
    alpha: float = 0.99,
    drift_magnitude: float = 2.0,
    anomaly_ratio: float = 0.15,
    compare_with_static: bool = True,
) -> dict:
    """
    So sánh: model với incremental learning vs. model tĩnh (không cập nhật μ_norm).

    Args:
        n_epochs: số đợt traffic
        n_flows_per_epoch: số flow mỗi đợt
        alpha: decay factor cho EMA update
        drift_magnitude: mức độ drift tổng cộng qua toàn bộ epochs
        compare_with_static: so sánh với static model không update

    Returns:
        dict kết quả so sánh
    """
    print("=" * 65)
    print("🌊 Drift Simulation — Incremental Learning Evaluation")
    print(f"   {n_epochs} epochs × {n_flows_per_epoch} flows | α={alpha} | drift={drift_magnitude}")
    print("=" * 65)

    # Load engine
    initial_mu = np.load(Path(MODEL_DIR) / "mu_norm.npy")
    engine_inc = ETSSLInference(MODEL_DIR, backend="onnx")
    if compare_with_static:
        engine_static = ETSSLInference(MODEL_DIR, backend="onnx")

    rng = np.random.default_rng(42)
    input_dim = 20

    # Learner cho incremental model
    learner = IncrementalLearner(
        initial_mu=initial_mu,
        alpha=alpha,
        update_interval=50,
        min_batch=20,
        rollback_window=10,
    )

    history_inc = []
    history_static = []

    print(f"\n  {'Epoch':<6} {'Drift':>7} {'Inc.Acc':>9} {'Inc.F1':>8} {'Sta.Acc':>9} {'Sta.F1':>8} {'μ drift':>9}")
    print(f"  {'─'*60}")

    for epoch in range(n_epochs):
        # Traffic shift tăng dần
        current_drift = drift_magnitude * ((epoch + 1) / n_epochs)

        n_attack = int(n_flows_per_epoch * anomaly_ratio)
        n_normal = n_flows_per_epoch - n_attack

        # Generate drifted traffic
        X_norm = rng.normal(current_drift, 1.0, (n_normal, input_dim)).astype(np.float32)
        X_atk  = rng.normal(current_drift + 3.5, 1.5, (n_attack, input_dim)).astype(np.float32)

        X_epoch = np.vstack([X_norm, X_atk])
        y_epoch = np.array([0] * n_normal + [1] * n_attack)

        # Shuffle
        idx = rng.permutation(len(X_epoch))
        X_epoch, y_epoch = X_epoch[idx], y_epoch[idx]

        # Scale nếu có scaler
        if engine_inc.scaler:
            X_s = engine_inc.scaler.transform(X_epoch).astype(np.float32)
        else:
            X_s = X_epoch

        # --- Incremental model ---
        old_mu = learner.mu_norm.copy()
        preds_inc = engine_inc.predict_batch(X_s)
        embs_inc  = np.array([r["embedding"] for r in preds_inc])
        is_anom   = np.array([r["is_anomaly"] for r in preds_inc])
        scores    = np.array([r["score"] for r in preds_inc])

        # Update learner
        for i in range(len(embs_inc)):
            learner.observe(embs_inc[i], bool(is_anom[i]), float(scores[i]))
        learner.force_update()
        engine_inc.update_mu_norm(learner.mu_norm.astype(np.float32))

        mu_drift = float(np.linalg.norm(learner.mu_norm - old_mu))

        m_inc = compute_binary_metrics(y_epoch, is_anom.astype(int))
        history_inc.append({
            "epoch": epoch + 1,
            "drift": round(current_drift, 3),
            "mu_drift": round(mu_drift, 6),
            **{k: v for k, v in m_inc.items() if not isinstance(v, int)},
        })

        # --- Static model ---
        m_sta = {}
        if compare_with_static:
            preds_sta = engine_static.predict_batch(X_s)
            is_anom_s = np.array([int(r["is_anomaly"]) for r in preds_sta])
            m_sta = compute_binary_metrics(y_epoch, is_anom_s)
            history_static.append({
                "epoch": epoch + 1,
                "drift": round(current_drift, 3),
                **{k: v for k, v in m_sta.items() if not isinstance(v, int)},
            })

        acc_i = m_inc.get("accuracy", 0)
        f1_i  = m_inc.get("f1", 0)
        acc_s = m_sta.get("accuracy", 0) if m_sta else 0
        f1_s  = m_sta.get("f1", 0) if m_sta else 0

        print(f"  {epoch+1:<6} {current_drift:>7.2f} {acc_i*100:>8.2f}% {f1_i:>8.4f} "
              f"{acc_s*100:>8.2f}% {f1_s:>8.4f} {mu_drift:>9.6f}")

    # Summary
    final_inc = history_inc[-1]
    final_sta = history_static[-1] if history_static else {}

    print(f"\n{'='*65}")
    print(f"  Final Epoch Results:")
    print(f"  Incremental: Acc={final_inc['accuracy']*100:.2f}% | F1={final_inc['f1']:.4f}")
    if final_sta:
        print(f"  Static:      Acc={final_sta['accuracy']*100:.2f}% | F1={final_sta['f1']:.4f}")
        acc_gain = (final_inc["accuracy"] - final_sta["accuracy"]) * 100
        f1_gain  = (final_inc["f1"] - final_sta["f1"])
        print(f"  Gain from incremental: Acc {acc_gain:+.2f}% | F1 {f1_gain:+.4f}")
    print(f"\n  μ_norm history: {learner.stats['history_size']} checkpoints saved")
    print(f"{'='*65}")

    # Plot
    _plot_drift(history_inc, history_static)

    results = {
        "config": {"n_epochs": n_epochs, "alpha": alpha, "drift_magnitude": drift_magnitude},
        "incremental": history_inc,
        "static": history_static,
        "final_mu_total_drift": round(float(np.linalg.norm(learner.mu_norm - initial_mu)), 6),
    }

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "drift_simulation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved: {out_path}")
    return results


def _plot_drift(history_inc, history_static):
    try:
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history_inc]
        f1_inc = [h["f1"] for h in history_inc]
        acc_inc = [h["accuracy"] for h in history_inc]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Incremental Learning vs Static — Drift Adaptation", fontsize=13, fontweight="bold")

        for ax, metric, label in zip(axes, ["f1", "accuracy"], ["F1 Score", "Accuracy"]):
            vals_inc = [h[metric] for h in history_inc]
            ax.plot(epochs, vals_inc, "o-", color="#2196F3", lw=2, label="Incremental")
            if history_static:
                vals_sta = [h[metric] for h in history_static]
                ax.plot(epochs, vals_sta, "s--", color="#F44336", lw=2, label="Static (no update)")
            ax.set_xlabel("Epoch (traffic drift ↑)")
            ax.set_ylabel(label)
            ax.set_title(f"{label} over Epochs")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        out_path = Path(__file__).parent / "results/drift_simulation.png"
        out_path.parent.mkdir(exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Plot saved: {out_path}")
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--n_flows", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--drift", type=float, default=2.0)
    args = parser.parse_args()

    run_drift_simulation(
        n_epochs=args.n_epochs,
        n_flows_per_epoch=args.n_flows,
        alpha=args.alpha,
        drift_magnitude=args.drift,
    )
